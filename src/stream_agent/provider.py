"""WeightProvider: the boundary between the inference backend and storage
(Plan.md C4).

A backend never asks for "bytes at an offset".  It asks for a semantic unit --
a tensor, an aligned tensor slice or an expert -- and the provider maps that
unit onto validated byte ranges of a pinned revision.

The provider is responsible for

* mapping semantic keys onto the layout,
* splitting a request into cache blocks,
* coalescing/parallelising the transfers within the cache budget,
* single-flight de-duplication of concurrent requests,
* keeping host staging buffers bounded,
* reporting the time spent waiting for weights separately from compute time.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .cache.manager import BlockRequest, CacheManager
from .errors import BackendCapabilityMissing, ModelFormatUnsupported
from .layout.model_map import ModelLayout
from .sources.base import SourceAdapter
from .telemetry import RunTelemetry


@dataclass(frozen=True)
class TensorKey:
    tensor: str


@dataclass(frozen=True)
class TensorSliceKey:
    tensor: str
    start: int  # first element (flattened)
    end: int  # exclusive


@dataclass(frozen=True)
class ExpertKey:
    tensor: str
    expert_id: int


WeightKey = TensorKey | TensorSliceKey | ExpertKey


@dataclass
class TensorBuffer:
    """Verified weight bytes handed to a backend."""

    data: bytes
    dtype: str
    shape: tuple[int, ...]
    shard: str = ""

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass
class PrefetchHandle:
    prefetch_id: str
    keys: tuple[WeightKey, ...] = ()
    cancelled: bool = False
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    def cancel(self) -> None:
        self.cancelled = True
        for task in self.tasks:
            if not task.done():
                task.cancel()


@runtime_checkable
class WeightProvider(Protocol):
    """The interface a backend may rely on."""

    async def read_tensor(self, key: TensorKey) -> TensorBuffer: ...

    async def read_slice(self, key: TensorSliceKey) -> TensorBuffer: ...

    async def read_expert(self, key: ExpertKey) -> TensorBuffer: ...

    async def prefetch(
        self, keys: Sequence[WeightKey], priority: int = 0
    ) -> PrefetchHandle: ...

    async def cancel_prefetch(self, handle: PrefetchHandle) -> None: ...


class RemoteWeightProvider:
    """WeightProvider implementation backed by a remote source and the cache."""

    def __init__(
        self,
        *,
        source: SourceAdapter,
        layout: ModelLayout,
        cache: CacheManager,
        model_alias: str = "",
        block_size_bytes: int = 8 * 1024 * 1024,
        telemetry: RunTelemetry | None = None,
        max_parallel_blocks: int = 4,
        metadata_root: Path | None = None,
    ) -> None:
        self.source = source
        self.layout = layout
        self.cache = cache
        self.model_alias = model_alias
        self.block_size_bytes = max(1, int(block_size_bytes))
        self.telemetry = telemetry
        self.max_parallel_blocks = max(1, int(max_parallel_blocks))
        self.metadata_root = Path(metadata_root) if metadata_root else None
        self._prefetched: dict[str, int] = {}
        self._weight_wait_ms = 0.0
        self._lock = asyncio.Lock()

    # -- accounting --------------------------------------------------------
    @property
    def weight_wait_ms(self) -> float:
        return self._weight_wait_ms

    def reset_wait_timer(self) -> float:
        value = self._weight_wait_ms
        self._weight_wait_ms = 0.0
        return value

    # -- block helpers -----------------------------------------------------
    def _block_bounds(self, start: int, end: int, file_size: int) -> list[tuple[int, int]]:
        """Split ``[start, end]`` into file aligned cache blocks."""
        blocks: list[tuple[int, int]] = []
        cursor = start
        while cursor <= end:
            block_start = (cursor // self.block_size_bytes) * self.block_size_bytes
            block_end = min(block_start + self.block_size_bytes - 1, max(file_size - 1, end))
            if block_end < cursor:
                block_end = end
            blocks.append((block_start, block_end))
            if block_end >= end:
                break
            cursor = block_end + 1
        return blocks

    async def _fetch_block(self, repo: str, commit: str, path: str, start: int, end: int) -> bytes:
        data = await self.source.fetch_range(repo, commit, path, start, end)
        if self.telemetry is not None:
            self.telemetry.network.range_requests += 1
        return data

    async def _read_range(
        self,
        path: str,
        start: int,
        end: int,
        *,
        file_size: int | None = None,
        demand: bool = True,
    ) -> bytes:
        """Read the inclusive byte range ``[start, end]`` of one artifact file."""
        if end < start:
            raise ModelFormatUnsupported(f"invalid byte range [{start}, {end}]")
        repo = self.layout.repo
        commit = self.layout.commit
        shard = self.layout.shards.get(path)
        size = file_size or (shard.size if shard else None)
        if size is None:
            size = self.layout.metadata_files.get(path)
        if size is None:
            raise ModelFormatUnsupported(
                f"the size of {path} is unknown; a bounded range cannot be planned"
            )
        blocks = self._block_bounds(start, end, size)
        sem = asyncio.Semaphore(self.max_parallel_blocks)
        results: dict[tuple[int, int], bytes] = {}

        async def load(block_start: int, block_end: int) -> None:
            request = BlockRequest(
                repo=repo,
                commit=commit,
                path=path,
                start=block_start,
                end=block_end,
                model_alias=self.model_alias,
            )

            async def fetcher(bs: int, be: int) -> bytes:
                return await self._fetch_block(repo, commit, path, bs, be)

            async with sem:
                data = await self.cache.fetch_block(request, fetcher)
            results[(block_start, block_end)] = data
            if demand:
                prefetched = self._prefetched.pop(request.key, None)
                if prefetched is not None and self.telemetry is not None:
                    self.telemetry.cache.useful_prefetched_bytes += len(data)

        started = time.perf_counter()
        await asyncio.gather(*(load(bs, be) for bs, be in blocks))
        self._weight_wait_ms += (time.perf_counter() - started) * 1000.0

        chunks: list[bytes] = []
        for block_start, block_end in blocks:
            data = results[(block_start, block_end)]
            offset = max(0, start - block_start)
            length = min(end, block_end) - max(start, block_start) + 1
            chunks.append(data[offset : offset + length])
        return b"".join(chunks)

    # -- semantic reads ----------------------------------------------------
    async def read_tensor(self, key: TensorKey) -> TensorBuffer:
        ref = self.layout.tensor(key.tensor)
        data = await self._read_range(ref.shard, ref.data_start, ref.data_end - 1)
        return TensorBuffer(data=data, dtype=ref.dtype, shape=ref.shape, shard=ref.shard)

    async def read_slice(self, key: TensorSliceKey) -> TensorBuffer:
        ref = self.layout.tensor(key.tensor)
        start, end = ref.byte_range_for_elements(key.start, key.end)
        data = await self._read_range(ref.shard, start, end - 1)
        return TensorBuffer(data=data, dtype=ref.dtype, shape=ref.shape, shard=ref.shard)

    async def read_expert(self, key: ExpertKey) -> TensorBuffer:
        ref = self.layout.tensor(key.tensor)
        dim, index, count = self.layout.expert_element_range(key.tensor, key.expert_id)
        start, end = ref.byte_range_for_dim_slice(dim, index, count)
        data = await self._read_range(ref.shard, start, end - 1)
        return TensorBuffer(data=data, dtype=ref.dtype, shape=ref.shape, shard=ref.shard)

    async def read_file(self, path: str, start: int, end: int) -> bytes:
        """Read a raw byte range of a repository file.

        Metadata files (config, tokenizer, shard index) are kept in their own
        store so that they are not confused with the weight payload budget
        (Plan.md section 8.4).
        """
        if self.metadata_root is not None:
            target = self.metadata_root / self.layout.commit / path.replace("/", "__")
            if target.is_file():
                data = await asyncio.to_thread(target.read_bytes)
                if len(data) > end:
                    return data[start : end + 1]
            size = self.layout.metadata_files.get(path)
            if size is None:
                size = end + 1
            payload = await self.source.fetch_range(
                self.layout.repo, self.layout.commit, path, 0, size - 1
            )
            if self.telemetry is not None:
                self.telemetry.network.range_requests += 1
                self.telemetry.record_download(len(payload), kind="metadata")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_bytes, payload)
            except OSError:  # pragma: no cover - a read only metadata area is fine
                pass
            return payload[start : end + 1]
        return await self._read_range(path, start, end)

    # -- prefetch ----------------------------------------------------------
    async def prefetch(
        self, keys: Sequence[WeightKey], priority: int = 0
    ) -> PrefetchHandle:
        """Warm the cache for the given units.

        Prefetching is a performance optimisation only; failures are recorded
        but never propagated to the caller.
        """
        handle = PrefetchHandle(prefetch_id=f"pf-{time.time_ns()}", keys=tuple(keys))
        for key in keys:
            task = asyncio.create_task(self._prefetch_one(key))
            handle.tasks.append(task)
        return handle

    async def _prefetch_one(self, key: WeightKey) -> None:
        try:
            path, start, end = self._range_for(key)
            blocks = self._blocks_for(path, start, end)
            for block_start, block_end in blocks:
                block_key = self._block_key_for(path, block_start, block_end)
                self._prefetched[block_key] = block_end - block_start + 1
                if self.telemetry is not None:
                    self.telemetry.cache.prefetched_bytes += block_end - block_start + 1
            await self._read_range(path, start, end, demand=False)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            raise
        except Exception:
            if self.telemetry is not None:
                self.telemetry.notes.append("prefetch failed for one unit")
            return

    def _range_for(self, key: WeightKey) -> tuple[str, int, int]:
        if isinstance(key, TensorKey):
            ref = self.layout.tensor(key.tensor)
            return ref.shard, ref.data_start, ref.data_end - 1
        if isinstance(key, TensorSliceKey):
            ref = self.layout.tensor(key.tensor)
            start, end = ref.byte_range_for_elements(key.start, key.end)
            return ref.shard, start, end - 1
        if isinstance(key, ExpertKey):
            ref = self.layout.tensor(key.tensor)
            dim, index, count = self.layout.expert_element_range(key.tensor, key.expert_id)
            start, end = ref.byte_range_for_dim_slice(dim, index, count)
            return ref.shard, start, end - 1
        raise BackendCapabilityMissing(f"unsupported prefetch key type {type(key).__name__}")

    def _blocks_for(self, path: str, start: int, end: int) -> list[tuple[int, int]]:
        shard = self.layout.shards.get(path)
        size = shard.size if shard else self.layout.metadata_files.get(path)
        if size is None:
            return [(start, end)]
        return self._block_bounds(start, end, size)

    def _block_key_for(self, path: str, start: int, end: int) -> str:
        from .cache.manager import canonical_block_key

        return canonical_block_key(self.layout.repo, self.layout.commit, path, start, end)

    async def cancel_prefetch(self, handle: PrefetchHandle) -> None:
        handle.cancel()
        if handle.tasks:
            await asyncio.gather(*handle.tasks, return_exceptions=True)

    # -- diagnostics -------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "repo": self.layout.repo,
            "commit": self.layout.commit,
            "block_size_bytes": self.block_size_bytes,
            "tensors": len(self.layout.tensors),
            "weight_wait_ms": round(self._weight_wait_ms, 3),
        }


__all__ = [
    "ExpertKey",
    "PrefetchHandle",
    "RemoteWeightProvider",
    "TensorBuffer",
    "TensorKey",
    "TensorSliceKey",
    "WeightKey",
    "WeightProvider",
]
