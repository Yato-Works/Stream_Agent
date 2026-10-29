"""Local filesystem and in-memory source adapters.

These back two legitimate use cases:

* the *fully local artifact baseline* required by Plan.md section 16.4, and
* deterministic tests that must not depend on network access.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from ..errors import (
    RangeMismatchError,
    RangeUnsupportedError,
    SourceForbiddenError,
    SourceNotFoundError,
)
from .base import RangeSupport, RepoFile, RevisionInfo


class LocalDirectorySource:
    """Serves model artifacts from a directory tree.

    Layout: ``<root>/<owner>/<repo>/<commit>/<path>``.
    """

    name = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _repo_dir(self, repo: str) -> Path:
        return self.root.joinpath(*repo.split("/"))

    def _file_path(self, repo: str, commit: str, path: str) -> Path:
        base = (self._repo_dir(repo) / commit).resolve()
        candidate = (base / path).resolve()
        # Path traversal guard: a repository path may never escape the repo dir.
        try:
            candidate.relative_to(base)
        except ValueError:
            raise SourceForbiddenError(
                f"repository path escapes the managed root: {path!r}"
            ) from None
        return candidate

    async def resolve_revision(self, repo: str, ref: str = "main") -> RevisionInfo:
        base = self._repo_dir(repo) / ref
        if not base.is_dir():
            raise SourceNotFoundError(f"local revision {repo}@{ref} does not exist")
        files: dict[str, RepoFile] = {}
        for item in sorted(base.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(base).as_posix()
            files[relative] = RepoFile(path=relative, size=item.stat().st_size)
        return RevisionInfo(repo=repo, commit=ref, files=files, source=self.name)

    async def list_files(self, repo: str, commit: str) -> RevisionInfo:
        return await self.resolve_revision(repo, commit)

    async def fetch_range(
        self,
        repo: str,
        commit: str,
        path: str,
        start: int,
        end: int,
    ) -> bytes:
        if end < start:
            raise RangeMismatchError(f"invalid range [{start}, {end}]")
        target = self._file_path(repo, commit, path)
        if not target.is_file():
            raise SourceNotFoundError(f"{path} not found in {repo}@{commit}")

        def read_window() -> bytes:
            size = target.stat().st_size
            if end >= size:
                raise RangeMismatchError(
                    f"range [{start}, {end}] exceeds file size {size} for {path}",
                    details={"size": size},
                )
            with target.open("rb") as handle:
                # Seek to the requested window.  Reading the complete file
                # first would make the local baseline O(file size) per block
                # and would defeat the whole point of range access.
                handle.seek(start)
                return handle.read(end - start + 1)

        return await asyncio.to_thread(read_window)

    async def probe_range_support(
        self,
        repo: str,
        commit: str,
        path: str,
        *,
        file_size: int | None = None,
    ) -> RangeSupport:
        try:
            data = await self.fetch_range(repo, commit, path, 0, 63)
        except SourceNotFoundError:
            return RangeSupport(supported=False, reason="file not found")
        except RangeMismatchError as exc:
            return RangeSupport(supported=False, reason=str(exc))
        return RangeSupport(supported=len(data) > 0, reason="local range read")

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None


class MemorySource:
    """A tiny in-memory source used by the test-suite and the fixture backend."""

    name = "memory"

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, dict[str, bytes]]] = {}
        self.licenses: dict[str, str | None] = {}
        self.request_log: list[tuple[str, str, str, int, int]] = []
        self.fail_next: dict[str, Exception] = {}

    def add_revision(
        self,
        repo: str,
        commit: str,
        files: dict[str, bytes],
        *,
        license_id: str | None = None,
    ) -> None:
        self.repos.setdefault(repo, {})[commit] = dict(files)
        self.licenses[f"{repo}@{commit}"] = license_id

    async def resolve_revision(self, repo: str, ref: str = "main") -> RevisionInfo:
        revisions = self.repos.get(repo, {})
        commit = ref if ref in revisions else next(iter(revisions), None)
        if commit is None:
            raise SourceNotFoundError(f"unknown repository {repo}")
        files = {
            name: RepoFile(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in revisions[commit].items()
        }
        return RevisionInfo(
            repo=repo,
            commit=commit,
            files=files,
            license_id=self.licenses.get(f"{repo}@{commit}"),
            source=self.name,
        )

    async def list_files(self, repo: str, commit: str) -> RevisionInfo:
        return await self.resolve_revision(repo, commit)

    async def fetch_range(
        self,
        repo: str,
        commit: str,
        path: str,
        start: int,
        end: int,
    ) -> bytes:
        self.request_log.append((repo, commit, path, start, end))
        failure = self.fail_next.pop(path, None)
        if failure is not None:
            raise failure
        data = self.repos.get(repo, {}).get(commit, {}).get(path)
        if data is None:
            raise SourceNotFoundError(f"{path} not found in {repo}@{commit}")
        if end >= len(data):
            raise RangeMismatchError(
                f"range [{start}, {end}] exceeds file size {len(data)} for {path}"
            )
        return data[start : end + 1]

    async def probe_range_support(
        self,
        repo: str,
        commit: str,
        path: str,
        *,
        file_size: int | None = None,
    ) -> RangeSupport:
        try:
            await self.fetch_range(repo, commit, path, 0, 63)
        except RangeUnsupportedError as exc:
            return RangeSupport(supported=False, reason=str(exc))
        except Exception as exc:
            return RangeSupport(supported=False, reason=str(exc))
        return RangeSupport(supported=True, reason="memory range read")

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None


__all__ = ["LocalDirectorySource", "MemorySource"]
