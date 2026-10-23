"""Source adapter boundary (Plan.md C2).

The cache manager and the inference backends must never learn anything about
Hugging Face, Xet, S3 or any other repository protocol.  They only know about
this interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RepoFile:
    """One file inside a repository revision."""

    path: str
    size: int | None = None
    sha256: str | None = None
    xet_hash: str | None = None
    is_lfs: bool = False

    @property
    def has_content_identity(self) -> bool:
        return bool(self.sha256)


@dataclass
class RevisionInfo:
    """Immutable identity of a repository revision."""

    repo: str
    commit: str
    files: dict[str, RepoFile] = field(default_factory=dict)
    license_id: str | None = None
    license_url: str | None = None
    gated: bool | None = None
    requires_custom_code: bool = False
    card_data: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def file(self, path: str) -> RepoFile | None:
        return self.files.get(path)

    @property
    def total_weight_bytes(self) -> int:
        return sum(f.size or 0 for f in self.files.values())


@dataclass(frozen=True)
class RangeSupport:
    """Result of a capability probe for partial access."""

    supported: bool
    reason: str = ""
    probe_start: int = 0
    probe_end: int = 0
    content_range: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "probe_start": self.probe_start,
            "probe_end": self.probe_end,
            "content_range": self.content_range,
        }


@runtime_checkable
class SourceAdapter(Protocol):
    """The only way Stream Agent is allowed to touch model artifacts."""

    name: str

    async def resolve_revision(self, repo: str, ref: str = "main") -> RevisionInfo:
        """Resolve a human readable ref into an immutable commit identity."""

    async def fetch_range(
        self,
        repo: str,
        commit: str,
        path: str,
        start: int,
        end: int,
    ) -> bytes:
        """Return bytes ``[start, end]`` (inclusive) of one repository file."""

    async def probe_range_support(
        self,
        repo: str,
        commit: str,
        path: str,
        *,
        file_size: int | None = None,
    ) -> RangeSupport:
        """Check whether partial access actually works for this path."""

    async def aclose(self) -> None:
        """Release network resources."""


class BoundedSemaphore:
    """Small helper used by adapters to bound parallel range requests."""

    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(max(1, limit))

    async def __aenter__(self) -> "BoundedSemaphore":
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._sem.release()


__all__ = [
    "RepoFile",
    "RevisionInfo",
    "RangeSupport",
    "SourceAdapter",
    "BoundedSemaphore",
]
