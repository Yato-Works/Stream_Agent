"""Eviction policy for the weight cache (Plan.md C5).

The v0.1 policy is deliberately simple and deterministic: least recently used
blocks are evicted first, blocks that are pinned or in use are never
evicted, and ties are broken by the block key so that two runs with the same
trace produce the same result.
"""

from __future__ import annotations

from .db import CacheEntry


def select_candidates(
    entries: dict[str, CacheEntry],
    *,
    pinned: set[str],
    in_use: dict[str, int],
    inflight: set[str],
    alias: str | None = None,
) -> list[CacheEntry]:
    """Return the eviction candidates, oldest effective use first."""
    candidates = [
        entry
        for entry in entries.values()
        if entry.block_key not in pinned
        and in_use.get(entry.block_key, 0) == 0
        and entry.block_key not in inflight
        and entry.verified
        and (alias is None or alias in entry.refs)
    ]
    candidates.sort(key=lambda e: (e.last_access, e.block_key))
    return candidates


__all__ = ["select_candidates"]
