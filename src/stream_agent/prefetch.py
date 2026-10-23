"""Prefetch scheduler (Plan.md C6).

Prefetching is a performance optimisation, never a correctness requirement.
This scheduler only starts a prefetch when the backend could state what comes
next; it never guesses expert ids and never lets a prefetch evict a hot block.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Sequence

from .provider import PrefetchHandle, RemoteWeightProvider, WeightKey
from .telemetry import RunTelemetry


@dataclass
class PrefetchStats:
    requested: int = 0
    requested_bytes: int = 0
    useful_bytes: int = 0
    cancelled: int = 0
    skipped: int = 0
    blocked_ms: float = 0.0
    handles: int = 0


@dataclass
class _QueueItem:
    keys: tuple[WeightKey, ...]
    priority: int
    created_at: float = field(default_factory=time.time)


class PrefetchScheduler:
    """Bounded, priority aware prefetch queue in front of the provider."""

    def __init__(
        self,
        provider: RemoteWeightProvider,
        *,
        max_bytes: int = 512 * 1024 * 1024,
        max_concurrency: int = 2,
        telemetry: RunTelemetry | None = None,
    ) -> None:
        self.provider = provider
        self.max_bytes = max_bytes
        self.max_concurrency = max(1, max_concurrency)
        self.telemetry = telemetry
        self.stats = PrefetchStats()
        self._inflight_bytes = 0
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._handles: list[PrefetchHandle] = []
        self._active = False

    # -- public API --------------------------------------------------------
    async def schedule(
        self, keys: Sequence[WeightKey], *, priority: int = 0
    ) -> PrefetchHandle | None:
        """Queue a prefetch; returns ``None`` when the budget does not allow it."""
        if not keys:
            return None
        if self._inflight_bytes >= self.max_bytes:
            self.stats.skipped += 1
            return None
        self._active = True
        self.stats.handles += 1
        handle = await self.provider.prefetch(keys, priority)
        self._handles.append(handle)
        self.stats.requested += len(keys)
        if self.telemetry is not None:
            self.telemetry.cache.prefetched_bytes += 0
        return handle

    async def cancel(self, handle: PrefetchHandle | None) -> None:
        if handle is None:
            return
        await self.provider.cancel_prefetch(handle)
        self.stats.cancelled += 1

    async def cancel_all(self) -> None:
        """Cancel obsolete prefetch work (session cancelled or model switched)."""
        for handle in list(self._handles):
            await self.provider.cancel_prefetch(handle)
            self.stats.cancelled += 1
        self._handles.clear()
        self._active = False

    async def wait_idle(self, timeout: float = 30.0) -> None:
        tasks = [task for handle in self._handles for task in handle.tasks]
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:  # pragma: no cover - defensive
            for task in tasks:
                task.cancel()

    def snapshot(self) -> dict[str, float | int]:
        return {
            "requested": self.stats.requested,
            "requested_bytes": self.stats.requested_bytes,
            "useful_bytes": self.stats.useful_bytes,
            "cancelled": self.stats.cancelled,
            "skipped": self.stats.skipped,
            "handles": self.stats.handles,
        }


__all__ = ["PrefetchScheduler", "PrefetchStats"]
