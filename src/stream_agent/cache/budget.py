"""Budget admission rules for the weight cache (Plan.md C5 / 10.3).

The rules are kept separate from the cache bookkeeping so that the admission
policy can be read (and tested) on its own:

* capacity is reserved **before** a transfer starts;
* the global hard limit always wins over a per-model quota;
* a per-model quota in ``fixed`` mode is a hard cap for that model;
* a per-model quota in ``elastic`` mode is a target that may borrow unused
  capacity until the global limit is reached;
* when nothing can be freed the caller gets a deterministic error instead of a
  deadlock.
"""

from __future__ import annotations

from typing import Protocol


class BudgetState(Protocol):
    """The state the admission rules need from the cache manager."""

    storage: object
    _used_bytes: int
    _reserved_bytes: int
    _pins: dict
    _entries: dict

    def model_quota(self, alias: str) -> tuple[int, str]: ...
    def model_used_bytes(self, alias: str) -> int: ...
    def _pinned_bytes(self) -> int: ...


class BudgetMixin:
    """Admission rules for :class:`stream_agent.cache.manager.CacheManager`."""

    def _fits_global(self, num_bytes: int) -> bool:
        storage = self.storage  # type: ignore[attr-defined]
        if self._used_bytes + self._reserved_bytes + num_bytes > storage.weight_cache_limit_bytes:  # type: ignore[attr-defined]
            return False
        limit = storage.managed_store_limit_bytes  # type: ignore[attr-defined]
        if limit is None:
            return True
        projected = self.managed_store_bytes() + num_bytes  # type: ignore[attr-defined]
        return projected <= limit

    def _fits_model(self, alias: str, num_bytes: int) -> bool:
        quota, mode = self.model_quota(alias)  # type: ignore[attr-defined]
        if mode != "fixed" or quota <= 0:
            return True
        return self.model_used_bytes(alias) + num_bytes <= quota  # type: ignore[attr-defined]

    def _pinned_bytes(self) -> int:
        total = 0
        for key, count in self._pins.items():  # type: ignore[attr-defined]
            if count > 0:
                entry = self._entries.get(key)  # type: ignore[attr-defined]
                if entry is not None:
                    total += entry.size
        return total

    def _would_exceed_forever(self, num_bytes: int, pinned_bytes: int) -> bool:
        """True when even a completely empty cache could not satisfy the request."""
        limit = self.storage.weight_cache_limit_bytes  # type: ignore[attr-defined]
        if num_bytes > limit:
            return True
        if pinned_bytes + num_bytes > limit:
            return True
        managed = self.storage.managed_store_limit_bytes  # type: ignore[attr-defined]
        if managed is not None and num_bytes > managed:
            return True
        return False


__all__ = ["BudgetMixin", "BudgetState"]
