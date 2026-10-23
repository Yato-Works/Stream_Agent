"""Value types used by the cache manager (Plan.md C5)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


def canonical_block_key(repo: str, commit: str, path: str, start: int, end: int) -> str:
    """Stable identity of a byte range inside a pinned revision."""
    material = f"{repo}\n{commit}\n{path}\n{start}\n{end}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass
class BlockRequest:
    repo: str
    commit: str
    path: str
    start: int
    end: int  # inclusive
    model_alias: str = ""

    @property
    def size(self) -> int:
        return max(0, self.end - self.start + 1)

    @property
    def key(self) -> str:
        return canonical_block_key(self.repo, self.commit, self.path, self.start, self.end)


@dataclass
class Reservation:
    """Capacity reserved *before* a transfer starts."""

    reservation_id: str
    model_alias: str
    num_bytes: int
    created_at: float = field(default_factory=time.time)
    committed: bool = False
    released: bool = False


@dataclass
class CacheStatus:
    weight_cache_limit_bytes: int
    managed_store_limit_bytes: int | None
    used_weight_bytes: int
    reserved_weight_bytes: int
    free_weight_bytes: int
    managed_store_bytes: int
    metadata_bytes: int
    log_bytes: int
    temporary_bytes: int
    entry_count: int
    pinned_count: int
    per_model: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "weight_cache_limit_bytes": self.weight_cache_limit_bytes,
            "managed_store_limit_bytes": self.managed_store_limit_bytes,
            "used_weight_bytes": self.used_weight_bytes,
            "reserved_weight_bytes": self.reserved_weight_bytes,
            "free_weight_bytes": self.free_weight_bytes,
            "managed_store_bytes": self.managed_store_bytes,
            "metadata_bytes": self.metadata_bytes,
            "log_bytes": self.log_bytes,
            "temporary_bytes": self.temporary_bytes,
            "entry_count": self.entry_count,
            "pinned_count": self.pinned_count,
            "per_model": self.per_model,
        }


__all__ = [
    "BlockRequest",
    "CacheStatus",
    "Reservation",
    "canonical_block_key",
]
