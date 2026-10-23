"""Content-addressed weight block store (Plan.md C5).

Blocks are stored by the SHA-256 of their content, so identical content is
physically stored once.  Writes are atomic: data is written to a temporary
file, flushed, and then renamed into place, so a crash can never leave a
partially written block visible as a cache entry.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredBlock:
    content_hash: str
    path: Path
    size: int


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlockStore:
    """Filesystem backed store of verified weight blocks."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def path_for(self, digest: str) -> Path:
        if len(digest) < 4:
            raise ValueError("invalid content hash")
        return self.root / digest[:2] / f"{digest}.blk"

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    # -- writing -----------------------------------------------------------
    def write(self, data: bytes, *, tmp_dir: Path | None = None) -> StoredBlock:
        """Write ``data`` and atomically promote it into the store."""
        digest = content_hash(data)
        target = self.path_for(digest)
        if target.is_file() and target.stat().st_size == len(data):
            return StoredBlock(digest, target, len(data))
        target.parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tmp_dir) if tmp_dir is not None else target.parent
        directory.mkdir(parents=True, exist_ok=True)
        tmp_path = directory / f".incoming-{digest}-{os.getpid()}-{id(data)}.tmp"
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return StoredBlock(digest, target, len(data))

    # -- reading -----------------------------------------------------------
    def read(self, digest: str, *, start: int = 0, end: int | None = None) -> bytes:
        path = self.path_for(digest)
        if not path.is_file():
            raise FileNotFoundError(f"block {digest} is not present in the store")
        with open(path, "rb") as handle:
            if start:
                handle.seek(start)
            if end is None:
                return handle.read()
            return handle.read(end - start)

    def size(self, digest: str) -> int | None:
        path = self.path_for(digest)
        if not path.is_file():
            return None
        return path.stat().st_size

    def verify(self, digest: str, *, expected_size: int | None = None) -> bool:
        """Verify the stored bytes still hash to ``digest``."""
        path = self.path_for(digest)
        try:
            if not path.is_file():
                return False
            if expected_size is not None and path.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        digestor = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digestor.update(chunk)
        except OSError:
            return False
        return digestor.hexdigest() == digest

    # -- removal -----------------------------------------------------------
    def delete(self, digest: str) -> int:
        path = self.path_for(digest)
        if not path.is_file():
            return 0
        size = path.stat().st_size
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A concurrent reader may still hold the file open (Windows).
            # The block stays on disk and is reclaimed by a later sweep.
            return 0
        return size

    # -- maintenance -------------------------------------------------------
    def iter_blocks(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.rglob("*.blk") if p.is_file())

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.iter_blocks())

    def clear_tmp(self, tmp_dir: Path) -> int:
        """Remove leftover partial downloads; returns the reclaimed bytes."""
        if not tmp_dir.is_dir():
            return 0
        reclaimed = 0
        for item in tmp_dir.iterdir():
            if item.is_file():
                reclaimed += item.stat().st_size
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        return reclaimed


__all__ = ["BlockStore", "StoredBlock", "content_hash"]
