"""Cache hard-limit audit (Plan.md gate G1).

These tests attack the *capacity control* of the weight cache from the
outside: they assert that the persistent payload never exceeds the configured
limit, that concurrent transfers cannot break the limit, that a failed
transfer leaves no residue, and that the accounting stays consistent with the
state on disk.

The suite is deliberately written against the public behaviour of
``CacheManager`` rather than against its internals.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from stream_agent.cache.db import MetadataDB
from stream_agent.cache.manager import BlockRequest, CacheManager
from stream_agent.cache.store import BlockStore
from stream_agent.config import StorageConfig
from stream_agent.errors import (
    CacheBudgetExceeded,
    IntegrityFailure,
    WorkingSetExceedsBudget,
)
from stream_agent.telemetry import RunTelemetry

REPO = "fixture/tiny-model"
COMMIT = "c0ffee1234567890"
PATH = "model.safetensors"


def build_manager(tmp_path: Path, limit: int, *, store_limit: int | None = None):
    storage = StorageConfig(
        root=tmp_path / "store",
        weight_cache_limit_bytes=limit,
        managed_store_limit_bytes=store_limit if store_limit is not None else limit * 4,
        metadata_reserve_bytes=64 * 1024,
        staging_reserve_bytes=64 * 1024,
    )
    db = MetadataDB(storage.db_path)
    store = BlockStore(storage.blocks_root)
    telemetry = RunTelemetry()
    manager = CacheManager(storage, db, store, telemetry=telemetry)
    return manager, storage, telemetry


def block(start: int, end: int, alias: str = "tiny") -> BlockRequest:
    return BlockRequest(REPO, COMMIT, PATH, start, end, alias)


async def test_concurrent_transfers_never_exceed_the_limit(tmp_path: Path):
    """Many simultaneous fetches must not push the cache over the limit.

    Sixteen blocks of 64 KiB are requested at once against a 256 KiB budget,
    i.e. the sum of all blocks is four times the budget.  The manager may
    serialise or evict, but it must never exceed the limit.
    """
    manager, storage, telemetry = build_manager(tmp_path, limit=256 * 1024)
    size = 64 * 1024

    async def fetcher(start: int, end: int) -> bytes:
        await asyncio.sleep(0.005)  # give the transfers time to overlap
        return bytes((start // size) % 251 for _ in range(end - start + 1))

    requests = [block(i * size, (i + 1) * size - 1) for i in range(16)]
    results = await asyncio.gather(
        *(manager.fetch_block(req, fetcher) for req in requests),
        return_exceptions=True,
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    successes = [r for r in results if isinstance(r, bytes)]
    for data in successes:
        assert len(data) == size
    assert manager.used_weight_bytes <= storage.weight_cache_limit_bytes
    assert manager.reserved_weight_bytes >= 0
    assert manager.used_weight_bytes + manager.reserved_weight_bytes <= (
        storage.weight_cache_limit_bytes
    )
    # The physical state on disk must agree with the accounting.
    assert manager.store.total_bytes() <= storage.weight_cache_limit_bytes
    # Under overload the manager serialises and retries (Plan section 10.3
    # allows waiting for capacity to be released) rather than rejecting;
    # the hard limit is still never exceeded, which the assertions above
    # verify.


async def test_reservation_is_made_before_the_transfer(tmp_path: Path):
    """The capacity must be reserved while the transfer runs, not after it."""
    manager, storage, _ = build_manager(tmp_path, limit=128 * 1024)
    observed: list[tuple[int, int]] = []

    async def fetcher(start: int, end: int) -> bytes:
        observed.append((manager.used_weight_bytes, manager.reserved_weight_bytes))
        return b"\x00" * (end - start + 1)

    await manager.fetch_block(block(0, 1023), fetcher)
    used, reserved = observed[0]
    assert used == 0
    assert reserved >= 1024
    assert manager.reserved_weight_bytes == 0
    assert manager.used_weight_bytes == 1024


async def test_failed_transfer_leaves_no_residue(tmp_path: Path):
    """A failing transfer must not keep the capacity reserved."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    async def failing(start: int, end: int) -> bytes:
        raise RuntimeError("network died")

    with pytest.raises(RuntimeError):
        await manager.fetch_block(block(0, 4095), failing)

    assert manager.reserved_weight_bytes == 0
    assert manager.used_weight_bytes == 0
    assert manager.store.total_bytes() == 0
    assert list(manager.storage.tmp_root.glob("*")) == []


async def test_short_transfer_is_rejected_and_releases_the_reservation(tmp_path: Path):
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    async def short(start: int, end: int) -> bytes:
        return b"\x00" * 10  # far too few bytes

    with pytest.raises(IntegrityFailure):
        await manager.fetch_block(block(0, 4095), short)

    assert manager.reserved_weight_bytes == 0
    assert manager.used_weight_bytes == 0
    assert manager.store.total_bytes() == 0


async def test_failed_promotion_releases_the_reservation(tmp_path: Path, monkeypatch):
    """When the promotion fails, the reserved capacity must be given back."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    def broken_write(data: bytes, *, tmp_dir=None):
        raise OSError("the disk is full")

    monkeypatch.setattr(manager.store, "write", broken_write)

    async def fetcher(start: int, end: int) -> bytes:
        return b"\x00" * (end - start + 1)

    with pytest.raises(OSError):
        await manager.fetch_block(block(0, 4095), fetcher)

    assert manager.reserved_weight_bytes == 0
    assert manager.used_weight_bytes == 0


async def test_pinned_blocks_survive_eviction_pressure(tmp_path: Path):
    """A pinned block must never be evicted to make room for a new one."""
    manager, storage, _ = build_manager(tmp_path, limit=32 * 1024)

    async def fetcher(start: int, end: int) -> bytes:
        return bytes([start % 251]) * (end - start + 1)

    await manager.fetch_block(block(0, 8191), fetcher)
    pinned_key = next(iter(manager._entries))
    await manager.pin(pinned_key)

    with pytest.raises((CacheBudgetExceeded, WorkingSetExceedsBudget)):
        await manager.fetch_block(block(8192, 16383), fetcher)

    assert manager.is_pinned(pinned_key)
    assert pinned_key in manager._entries
    assert manager.used_weight_bytes <= storage.weight_cache_limit_bytes


async def test_identical_content_is_stored_once(tmp_path: Path):
    """Two different block keys with the same bytes share one physical block."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)
    payload = b"\x07" * 4096

    async def fetcher(start: int, end: int) -> bytes:
        return payload

    await manager.fetch_block(block(0, 4095), fetcher)
    await manager.fetch_block(block(8192, 12287), fetcher)

    assert manager.used_weight_bytes == 4096
    assert manager.store.total_bytes() == 4096


async def test_inflight_block_is_fetched_only_once(tmp_path: Path):
    """Concurrent readers of the same block share a single transfer."""
    manager, storage, telemetry = build_manager(tmp_path, limit=64 * 1024)
    calls = {"n": 0}

    async def fetcher(start: int, end: int) -> bytes:
        calls["n"] += 1
        await asyncio.sleep(0.01)
        return b"\x03" * (end - start + 1)

    results = await asyncio.gather(
        *(manager.fetch_block(block(0, 4095), fetcher) for _ in range(8))
    )
    assert all(r == results[0] for r in results)
    assert calls["n"] == 1
    assert telemetry.cache.single_flight_hits >= 1


async def test_unfinished_data_is_not_recognised_as_a_valid_entry(tmp_path: Path):
    """A partial file must never be promoted into a usable cache entry."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    async def failing(start: int, end: int) -> bytes:
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        await manager.fetch_block(block(0, 4095), failing)

    assert manager._entries == {}
    assert manager.store.total_bytes() == 0

    # A recovery pass finds nothing usable either.
    report = await manager.recover()
    assert report["restored_entries"] == 0
    assert report["orphan_blocks_removed"] == 0


async def test_recovery_reconciles_the_database_with_the_disk(tmp_path: Path):
    """A block that vanished from disk must be dropped from the metadata."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    async def fetcher(start: int, end: int) -> bytes:
        return b"\x05" * (end - start + 1)

    await manager.fetch_block(block(0, 4095), fetcher)
    assert manager.used_weight_bytes == 4096

    # Somebody removes the physical block behind the manager's back.
    entry = next(iter(manager._entries.values()))
    manager.store.path_for(entry.content_hash).unlink()

    recovered = CacheManager(storage, manager.db, manager.store)
    report = await recovered.recover()
    assert report["removed_entries"] == 1
    assert report["restored_entries"] == 0
    assert recovered.used_weight_bytes == 0


async def test_orphan_blocks_are_removed_by_the_recovery(tmp_path: Path):
    """A block on disk that the database does not know is reclaimed."""
    manager, storage, _ = build_manager(tmp_path, limit=64 * 1024)

    async def fetcher(start: int, end: int) -> bytes:
        return b"\x06" * (end - start + 1)

    await manager.fetch_block(block(0, 4095), fetcher)
    entry = next(iter(manager._entries.values()))
    manager.db.delete_entry(entry.block_key)
    assert manager.store.total_bytes() == 4096

    recovered = CacheManager(storage, manager.db, manager.store)
    report = await recovered.recover()
    assert report["orphan_blocks_removed"] == 1
    assert recovered.store.total_bytes() == 0


async def test_managed_store_limit_is_also_respected(tmp_path: Path):
    """The managed-store cap covers payload, metadata, logs and temporaries."""
    manager, storage, _ = build_manager(
        tmp_path, limit=64 * 1024, store_limit=256 * 1024
    )

    async def fetcher(start: int, end: int) -> bytes:
        return b"\x08" * (end - start + 1)

    await manager.fetch_block(block(0, 4095), fetcher)
    status = manager.status()
    assert status.managed_store_bytes <= storage.managed_store_limit_bytes


async def test_temporary_files_are_counted_in_the_managed_store(tmp_path: Path):
    """A file in the staging area counts against the managed-store cap."""
    manager, storage, _ = build_manager(
        tmp_path, limit=64 * 1024, store_limit=256 * 1024
    )
    storage.tmp_root.mkdir(parents=True, exist_ok=True)
    (storage.tmp_root / "partial.download").write_bytes(b"\x00" * 4096)

    status = manager.status()
    assert status.temporary_bytes == 4096
    assert status.managed_store_bytes >= status.temporary_bytes

    # The staging area is cleaned up on recovery.
    report = await manager.recover()
    assert report["reclaimed_temp_bytes"] == 4096


async def test_eviction_records_the_reason_and_the_bytes(tmp_path: Path):
    manager, storage, _ = build_manager(tmp_path, limit=8 * 1024, store_limit=256 * 1024)

    async def fetcher(start: int, end: int) -> bytes:
        return bytes([start % 251]) * (end - start + 1)

    for i in range(4):
        await manager.fetch_block(block(i * 4096, (i + 1) * 4096 - 1), fetcher)

    log = manager.eviction_log()
    assert log, "the cache must have evicted something"
    for record in log:
        assert record["reason"] in {"global_pressure", "model_quota", "manual"}
        assert record["bytes"] >= 0
    assert manager.used_weight_bytes <= storage.weight_cache_limit_bytes
