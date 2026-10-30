"""Global cache manager (Plan.md C5).

This component enforces the central product contract:

* the persistent weight cache never exceeds the configured hard limit,
  including in-flight reservations;
* the optional managed-store cap covers payload, metadata, logs and temporary
  files;
* only verified data is ever served;
* blocks are pinned while a backend holds a lease on them;
* concurrent requests for the same immutable block share one transfer.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable

from ..config import StorageConfig
from ..errors import (
    CacheBudgetExceeded,
    IntegrityFailure,
    StreamAgentError,
    WorkingSetExceedsBudget,
)
from ..telemetry import RunTelemetry
from .budget import BudgetMixin
from .accounting import build_status, directory_bytes, model_used_bytes, unique_bytes
from .db import CacheEntry, MetadataDB
from .eviction import select_candidates
from .recovery import recover_cache, verify_cache
from .store import BlockStore, content_hash
from .types import BlockRequest, CacheStatus, Reservation, canonical_block_key

Fetcher = Callable[[int, int], Awaitable[bytes]]


class _BlockVanished(Exception):
    """Internal signal: the cached block disappeared during a read."""


class CacheManager(BudgetMixin):
    """Owns the global budget, the block store and the metadata database."""

    def __init__(
        self,
        storage: StorageConfig,
        db: MetadataDB,
        store: BlockStore,
        *,
        telemetry: RunTelemetry | None = None,
        run_id: str = "",
        quota_policy: str | None = None,
    ) -> None:
        self.storage = storage
        self.db = db
        self.store = store
        self.telemetry = telemetry
        self.run_id = run_id
        self.quota_policy = quota_policy or storage.quota_policy

        self._entries: dict[str, CacheEntry] = {}
        self._pins: dict[str, int] = {}
        self._in_use: dict[str, int] = {}
        self._hash_in_use: dict[str, int] = {}
        self._promoting: set[str] = set()
        self._io_lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Future[bytes]] = {}
        self._reserved: dict[str, Reservation] = {}
        self._model_quotas: dict[str, tuple[int, str]] = {}  # alias -> (bytes, mode)
        self._lock = asyncio.Lock()
        self._capacity_changed = asyncio.Event()
        self._used_bytes = 0
        self._reserved_bytes = 0
        self._evictions: list[dict[str, object]] = []
        #: Content hashes that were evicted at least once.  A later promotion of
        #: the same content is a re-fetch caused by cache thrashing.
        self._evicted_content_hashes: dict[str, int] = {}

    # -- setup -------------------------------------------------------------
    def set_model_quota(self, alias: str, quota_bytes: int, mode: str | None = None) -> None:
        self._model_quotas[alias] = (int(quota_bytes), mode or self.quota_policy)

    def model_quota(self, alias: str) -> tuple[int, str]:
        return self._model_quotas.get(alias, (0, self.quota_policy))

    def set_weight_limit(self, limit_bytes: int) -> None:
        """Change the hard payload limit (used by the cache sweep)."""
        from dataclasses import replace as _replace

        self.storage = _replace(self.storage, weight_cache_limit_bytes=int(limit_bytes))

    async def recover(self) -> dict[str, int]:
        """Reconstruct the cache state after a crash.

        Orphan partials are removed, entries whose physical block vanished are
        dropped, and pins are reset because sessions do not survive a restart.
        """
        report = await recover_cache(self.storage, self.db, self.store)
        for entry in await asyncio.to_thread(self.db.list_entries):
            self._entries[entry.block_key] = entry
            entry.pin_count = 0
        self.db.reset_pins()
        self._used_bytes = self._compute_unique_bytes()
        return report

    def _compute_unique_bytes(self) -> int:
        return unique_bytes(self._entries.values())

    # -- accounting --------------------------------------------------------
    @property
    def used_weight_bytes(self) -> int:
        return self._used_bytes

    @property
    def reserved_weight_bytes(self) -> int:
        return self._reserved_bytes

    def free_weight_bytes(self) -> int:
        return max(0, self.storage.weight_cache_limit_bytes - self._used_bytes - self._reserved_bytes)

    def model_used_bytes(self, alias: str) -> int:
        """Reference weighted attribution of physical bytes to a model."""
        return model_used_bytes(self._entries.values(), alias)

    def _directory_bytes(self, directory: Path) -> int:
        return directory_bytes(directory)

    def managed_store_bytes(self) -> int:
        return (
            self._used_bytes
            + self._reserved_bytes
            + self._directory_bytes(self.storage.metadata_root)
            + self._directory_bytes(self.storage.log_root)
            + self._directory_bytes(self.storage.tmp_root)
        )

    def status(self) -> CacheStatus:
        return build_status(
            storage=self.storage,
            entries=self._entries,
            used_bytes=self._used_bytes,
            reserved_bytes=self._reserved_bytes,
            pinned_count=sum(1 for count in self._pins.values() if count > 0),
            quotas=self._model_quotas,
        )

    # -- pins --------------------------------------------------------------
    def has_block(self, block_key: str) -> bool:
        """Return True when the block is present and verified in the weight cache."""
        entry = self._entries.get(block_key)
        return entry is not None and entry.verified

    async def pin(self, block_key: str) -> None:
        self._pins[block_key] = self._pins.get(block_key, 0) + 1
        entry = self._entries.get(block_key)
        if entry is not None:
            entry.pin_count = self._pins[block_key]
            await asyncio.to_thread(self.db.upsert_entry, entry)

    async def unpin(self, block_key: str) -> None:
        count = self._pins.get(block_key, 0) - 1
        if count <= 0:
            self._pins.pop(block_key, None)
        else:
            self._pins[block_key] = count
        entry = self._entries.get(block_key)
        if entry is not None:
            entry.pin_count = max(0, self._pins.get(block_key, 0))
            await asyncio.to_thread(self.db.upsert_entry, entry)

    def is_pinned(self, block_key: str) -> bool:
        return self._pins.get(block_key, 0) > 0

    # -- reservation and admission ----------------------------------------
    async def reserve(
        self,
        model_alias: str,
        num_bytes: int,
        *,
        wait_timeout: float = 30.0,
    ) -> Reservation:
        """Reserve capacity for a transfer before any byte is written."""
        if num_bytes < 0:
            raise CacheBudgetExceeded("cannot reserve a negative number of bytes")
        deadline = time.monotonic() + wait_timeout
        while True:
            async with self._lock:
                freed = await self._try_make_room(model_alias, num_bytes)
                if freed:
                    reservation = Reservation(
                        reservation_id=f"res-{time.time_ns()}-{id(self)}",
                        model_alias=model_alias,
                        num_bytes=num_bytes,
                    )
                    self._reserved[reservation.reservation_id] = reservation
                    self._reserved_bytes += num_bytes
                    self._capacity_changed.clear()
                    if self.telemetry is not None:
                        self.telemetry.storage.reserved_weight_bytes = self._reserved_bytes
                        self.telemetry.storage.peak_weight_bytes = max(
                            self.telemetry.storage.peak_weight_bytes,
                            self._used_bytes + self._reserved_bytes,
                        )
                    return reservation
                pinned_bytes = self._pinned_bytes()
                blocked = self._would_exceed_forever(num_bytes, pinned_bytes)
            if blocked:
                raise WorkingSetExceedsBudget(
                    "the pinned working set does not fit into the configured cache budget",
                    details={
                        "pinned_bytes": pinned_bytes,
                        "weight_cache_limit_bytes": self.storage.weight_cache_limit_bytes,
                        "requested_bytes": num_bytes,
                    },
                )
            if not self._reserved:
                # ``_try_make_room`` already failed (eviction exhausted) and no
                # outstanding reservation can release capacity.  Waiting would
                # deadlock, so fail fast instead.
                raise CacheBudgetExceeded(
                    "cache budget exhausted and no capacity will be released",
                    details={
                        "used_bytes": self._used_bytes,
                        "reserved_bytes": self._reserved_bytes,
                        "weight_cache_limit_bytes": self.storage.weight_cache_limit_bytes,
                        "requested_bytes": num_bytes,
                    },
                )
            if time.monotonic() > deadline:
                raise CacheBudgetExceeded(
                    "no cache capacity became available within the wait budget",
                    details={
                        "requested_bytes": num_bytes,
                        "used_bytes": self._used_bytes,
                        "reserved_bytes": self._reserved_bytes,
                        "limit_bytes": self.storage.weight_cache_limit_bytes,
                    },
                )
            self._capacity_changed.clear()
            try:
                await asyncio.wait_for(self._capacity_changed.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

    async def _try_make_room(self, model_alias: str, num_bytes: int) -> bool:
        """Try to make room; returns True when the reservation was granted."""
        if num_bytes == 0:
            return True
        if not self._fits_global(num_bytes):
            if not await self._evict_for_global(num_bytes):
                return False
        if not self._fits_model(model_alias, num_bytes):
            if not await self._evict_for_model(model_alias, num_bytes):
                return False
        return True

    # -- eviction ----------------------------------------------------------
    def _eviction_candidates(self, alias: str | None = None) -> list[CacheEntry]:
        return select_candidates(
            self._entries,
            pinned={key for key, count in self._pins.items() if count > 0},
            in_use=self._in_use,
            inflight=set(self._inflight),
            alias=alias,
        )

    async def _evict_for_global(self, num_bytes: int) -> bool:
        needed = self._used_bytes + self._reserved_bytes + num_bytes - self.storage.weight_cache_limit_bytes
        limit = self.storage.managed_store_limit_bytes
        if limit is not None:
            needed = max(needed, self.managed_store_bytes() + num_bytes - limit)
        if needed <= 0:
            return True
        for entry in self._eviction_candidates():
            await self._evict(entry, reason="global_pressure")
            needed -= entry.size
            if needed <= 0:
                return True
        return False

    async def _evict_for_model(self, alias: str, num_bytes: int) -> bool:
        quota, mode = self.model_quota(alias)
        if mode != "fixed" or quota <= 0:
            return True
        needed = self.model_used_bytes(alias) + num_bytes - quota
        if needed <= 0:
            return True
        for entry in self._eviction_candidates(alias):
            await self._evict(entry, reason="model_quota")
            needed -= entry.size
            if needed <= 0:
                return True
        return False

    async def _evict(self, entry: CacheEntry, *, reason: str) -> None:
        async with self._io_lock:
            await asyncio.to_thread(self.db.delete_entry, entry.block_key)
            self._entries.pop(entry.block_key, None)
            # Only drop the physical block when no other entry references it,
            # no promotion is writing it and no reader is using it.
            still_referenced = any(
                other.content_hash == entry.content_hash
                for other in self._entries.values()
                if other.block_key != entry.block_key
            )
            freed = 0
            digest = entry.content_hash
            if (
                not still_referenced
                and digest not in self._promoting
                and self._hash_in_use.get(digest, 0) == 0
            ):
                freed = await asyncio.to_thread(self.store.delete, digest)
            self._evicted_content_hashes[digest] = (
                self._evicted_content_hashes.get(digest, 0) + 1
            )
            self._used_bytes = self._compute_unique_bytes()
        record = {
            "block_key": entry.block_key,
            "reason": reason,
            "bytes": freed,
            "model": entry.model_alias,
            "path": entry.path,
            "ts": time.time(),
        }
        self._evictions.append(record)
        if self.telemetry is not None:
            self.telemetry.cache.evictions += 1
            self.telemetry.cache.evicted_bytes += freed
        await asyncio.to_thread(
            self.db.record_event,
            "eviction",
            run_id=self.run_id,
            model_alias=entry.model_alias,
            num_bytes=freed,
            detail={"reason": reason, "path": entry.path},
        )

    def eviction_log(self) -> list[dict[str, object]]:
        return list(self._evictions)

    # -- fetching ----------------------------------------------------------
    async def fetch_block(
        self,
        request: BlockRequest,
        fetcher: Fetcher,
        *,
        wait_timeout: float = 30.0,
    ) -> bytes:
        """Return the verified bytes of a block, fetching them if necessary."""
        key = request.key
        for attempt in range(2):
            existing = self._entries.get(key)
            if existing is not None and existing.verified:
                # Mark the entry as in use *before* any await so that a
                # concurrent eviction cannot remove the physical block while
                # it is read.
                digest = existing.content_hash
                self._in_use[key] = self._in_use.get(key, 0) + 1
                self._hash_in_use[digest] = self._hash_in_use.get(digest, 0) + 1
                try:
                    await self._touch(key)
                    if request.model_alias and request.model_alias not in existing.refs:
                        # A second model read this shared block.  Record the
                        # reference so per-model attribution and quota
                        # accounting include it; otherwise the block is
                        # charged only to its creator.
                        existing.refs.add(request.model_alias)
                        await asyncio.to_thread(self.db.upsert_entry, existing)
                    if self.telemetry is not None:
                        self.telemetry.cache.hits += 1
                        self.telemetry.cache.reused_bytes += existing.size
                    return await self._read_existing(existing, request)
                except _BlockVanished:
                    continue
                finally:
                    remaining = self._in_use.get(key, 1) - 1
                    if remaining <= 0:
                        self._in_use.pop(key, None)
                    else:
                        self._in_use[key] = remaining
                    hash_remaining = self._hash_in_use.get(digest, 1) - 1
                    if hash_remaining <= 0:
                        self._hash_in_use.pop(digest, None)
                    else:
                        self._hash_in_use[digest] = hash_remaining

        inflight = self._inflight.get(key)
        if inflight is not None:
            if self.telemetry is not None:
                self.telemetry.cache.single_flight_hits += 1
            return await asyncio.shield(inflight)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()
        self._inflight[key] = future
        try:
            data = await self._fetch_and_promote(request, fetcher, wait_timeout=wait_timeout)
            future.set_result(data)
            return data
        except BaseException as exc:  # noqa: BLE001 - propagate the real cause
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)
            self._capacity_changed.set()

    async def _fetch_and_promote(
        self,
        request: BlockRequest,
        fetcher: Fetcher,
        *,
        wait_timeout: float,
    ) -> bytes:
        reservation = await self.reserve(request.model_alias, request.size, wait_timeout=wait_timeout)
        try:
            data = await fetcher(request.start, request.end)
        except BaseException:
            await self.release(reservation)
            raise
        if len(data) != request.size:
            await self.release(reservation)
            raise IntegrityFailure(
                f"source returned {len(data)} bytes for a {request.size} byte block of "
                f"{request.path}",
                details={"path": request.path, "expected": request.size, "actual": len(data)},
            )
        digest = content_hash(data)
        try:
            entry = await self._promote(request, data, digest)
        except BaseException:
            # The promotion failed: no byte reached the cache, so the
            # reservation has to be given back.
            await self.release(reservation)
            raise
        await self.release(reservation, committed=entry.size)
        if self.telemetry is not None:
            self.telemetry.cache.misses += 1
            self.telemetry.cache.promotions += 1
            self.telemetry.cache.cache_write_bytes += len(data)
            self.telemetry.record_download(len(data), kind="weight")
            if self._evicted_content_hashes.get(digest, 0) > 0:
                # The block had been evicted before and had to be fetched again.
                self.telemetry.cache.refetched_bytes += len(data)
        return data

    async def _promote(self, request: BlockRequest, data: bytes, digest: str) -> CacheEntry:
        self._promoting.add(digest)
        try:
            async with self._io_lock:
                stored = await asyncio.to_thread(
                    self.store.write, data, tmp_dir=self.storage.tmp_root
                )
                entry = CacheEntry(
                    block_key=request.key,
                    content_hash=stored.content_hash,
                    repo=request.repo,
                    commit=request.commit,
                    path=request.path,
                    start=request.start,
                    end=request.end,
                    size=len(data),
                    expected_size=request.size,
                    verified=True,
                    phys_path=str(stored.path),
                    model_alias=request.model_alias or "unassigned",
                    refs={request.model_alias} if request.model_alias else set(),
                    last_access=time.time(),
                    access_count=1,
                    created_at=time.time(),
                )
                existing = self._entries.get(entry.block_key)
                if existing is not None:
                    entry.refs |= existing.refs
                    entry.access_count = existing.access_count + 1
                if request.model_alias:
                    entry.refs.add(request.model_alias)
                await asyncio.to_thread(self.db.upsert_entry, entry)
                self._entries[entry.block_key] = entry
                self._used_bytes = self._compute_unique_bytes()
            if self.telemetry is not None:
                self.telemetry.storage.current_weight_bytes = self._used_bytes
                self.telemetry.storage.peak_weight_bytes = max(
                    self.telemetry.storage.peak_weight_bytes,
                    self._used_bytes + self._reserved_bytes,
                )
            return entry
        finally:
            self._promoting.discard(digest)

    async def _read_existing(self, entry: CacheEntry, request: BlockRequest) -> bytes:
        if self.storage.verify_before_use:
            ok = await asyncio.to_thread(
                self.store.verify, entry.content_hash, expected_size=entry.size
            )
            if not ok:
                if self.telemetry is not None:
                    self.telemetry.cache.integrity_failures += 1
                exists = await asyncio.to_thread(self.store.exists, entry.content_hash)
                await asyncio.to_thread(self.db.delete_entry, entry.block_key)
                self._entries.pop(entry.block_key, None)
                self._used_bytes = self._compute_unique_bytes()
                if not exists:
                    # The block was removed underneath us (a concurrent
                    # eviction).  Drop it and fetch it again.
                    raise _BlockVanished(entry.block_key)
                raise IntegrityFailure(
                    f"cached block for {entry.path} failed verification and was dropped",
                    details={"path": entry.path, "block_key": entry.block_key},
                )
        return await asyncio.to_thread(self.store.read, entry.content_hash)

    async def _touch(self, block_key: str) -> None:
        entry = self._entries.get(block_key)
        if entry is None:
            return
        entry.last_access = time.time()
        entry.access_count += 1
        await asyncio.to_thread(self.db.upsert_entry, entry)

    async def release(self, reservation: Reservation, *, committed: int = 0) -> None:
        if reservation.released:
            return
        reservation.released = True
        self._reserved.pop(reservation.reservation_id, None)
        self._reserved_bytes = max(0, self._reserved_bytes - reservation.num_bytes)
        if self.telemetry is not None:
            self.telemetry.storage.reserved_weight_bytes = self._reserved_bytes
        self._capacity_changed.set()

    # -- maintenance -------------------------------------------------------
    async def verify_all(self) -> dict[str, int]:
        """Re-verify every cached block and repair deterministic inconsistencies."""
        report = await verify_cache(self.db, self.store)
        for entry in list(self._entries.values()):
            if entry.block_key not in {e.block_key for e in await asyncio.to_thread(self.db.list_entries)}:
                self._entries.pop(entry.block_key, None)
        self._used_bytes = self._compute_unique_bytes()
        await asyncio.to_thread(self.db.record_event, "cache_verify", run_id=self.run_id,
                                num_bytes=self._used_bytes, detail=report)
        return report

    async def evict_model(self, alias: str, *, reason: str = "manual") -> dict[str, int]:
        """Remove every eligible block of a model (never touches pinned blocks)."""
        removed = 0
        freed = 0
        for entry in self._eviction_candidates(alias):
            size = entry.size
            await self._evict(entry, reason=reason)
            removed += 1
            freed += size
        return {"entries": removed, "bytes": freed}

    async def evict_block(self, block_key: str, *, reason: str = "manual") -> bool:
        entry = self._entries.get(block_key)
        if entry is None or self.is_pinned(block_key):
            return False
        await self._evict(entry, reason=reason)
        return True


__all__ = [
    "BlockRequest",
    "CacheManager",
    "CacheStatus",
    "Reservation",
    "canonical_block_key",
    "StreamAgentError",
]
