"""Storage accounting for the weight cache (Plan.md C5 / 10.4).

Accounting is always based on the *unique physical blocks* that are actually
stored.  Per-model attribution is reference weighted: a block that several
models share is charged to each of them proportionally, with the remainder
assigned to the model that created it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..config import StorageConfig
from .db import CacheEntry
from .types import CacheStatus


def directory_bytes(directory: Path) -> int:
    """Total size of the files below ``directory``."""
    if not directory.is_dir():
        return 0
    total = 0
    for item in directory.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:  # pragma: no cover - race with deletion
                continue
    return total


def unique_bytes(entries: Iterable[CacheEntry]) -> int:
    """Physical bytes held by the verified blocks (identical content once)."""
    seen: dict[str, int] = {}
    for entry in entries:
        if entry.verified:
            seen[entry.content_hash] = entry.size
    return sum(seen.values())


def model_used_bytes(entries: Iterable[CacheEntry], alias: str) -> int:
    """Reference weighted attribution of physical bytes to one model."""
    total = 0
    for entry in entries:
        if not entry.verified or alias not in entry.refs:
            continue
        share, remainder = divmod(entry.size, max(1, len(entry.refs)))
        total += share
        if entry.model_alias == alias:
            total += remainder
    return total


def build_status(
    *,
    storage: StorageConfig,
    entries: dict[str, CacheEntry],
    used_bytes: int,
    reserved_bytes: int,
    pinned_count: int,
    quotas: dict[str, tuple[int, str]],
) -> CacheStatus:
    aliases = set(quotas) | {entry.model_alias for entry in entries.values()}
    per_model: dict[str, dict[str, int]] = {}
    for alias in aliases:
        quota, mode = quotas.get(alias, (0, storage.quota_policy))
        per_model[alias] = {
            "quota_bytes": quota,
            "quota_mode": mode,
            "used_bytes": model_used_bytes(entries.values(), alias),
            "entry_count": sum(1 for entry in entries.values() if alias in entry.refs),
        }
    managed = (
        used_bytes
        + reserved_bytes
        + directory_bytes(storage.metadata_root)
        + directory_bytes(storage.log_root)
        + directory_bytes(storage.tmp_root)
    )
    return CacheStatus(
        weight_cache_limit_bytes=storage.weight_cache_limit_bytes,
        managed_store_limit_bytes=storage.managed_store_limit_bytes,
        used_weight_bytes=used_bytes,
        reserved_weight_bytes=reserved_bytes,
        free_weight_bytes=max(
            0, storage.weight_cache_limit_bytes - used_bytes - reserved_bytes
        ),
        managed_store_bytes=managed,
        metadata_bytes=directory_bytes(storage.metadata_root),
        log_bytes=directory_bytes(storage.log_root),
        temporary_bytes=directory_bytes(storage.tmp_root),
        entry_count=len(entries),
        pinned_count=pinned_count,
        per_model=per_model,
    )


__all__ = [
    "build_status",
    "directory_bytes",
    "model_used_bytes",
    "unique_bytes",
]
