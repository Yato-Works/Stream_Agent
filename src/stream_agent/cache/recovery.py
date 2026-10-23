"""Crash recovery and consistency repair for the weight cache (Plan.md C5).

After a restart the cache state is reconstructed from the manifests: entries
whose physical block vanished are dropped, orphan blocks and leftover partial
downloads are removed.  Only deterministic inconsistencies are repaired.
"""

from __future__ import annotations

from pathlib import Path

from ..config import StorageConfig
from .db import CacheEntry, MetadataDB
from .store import BlockStore


async def recover_cache(
    storage: StorageConfig,
    db: MetadataDB,
    store: BlockStore,
) -> dict[str, int]:
    """Rebuild the cache state after a crash."""
    reclaimed_tmp = await _run(store.clear_tmp, storage.tmp_root)
    removed = 0
    restored = 0
    for entry in await _run(db.list_entries):
        path = Path(entry.phys_path)
        try:
            present = path.is_file() and path.stat().st_size == entry.size
        except OSError:
            present = False
        if not present:
            await _run(db.delete_entry, entry.block_key)
            removed += 1
            continue
        restored += 1

    referenced = {entry.content_hash for entry in await _run(db.list_entries)}
    orphans = 0
    for block in await _run(store.iter_blocks):
        if block.stem not in referenced:
            await _run(store.delete, block.stem)
            orphans += 1

    return {
        "reclaimed_temp_bytes": reclaimed_tmp,
        "removed_entries": removed,
        "restored_entries": restored,
        "orphan_blocks_removed": orphans,
    }


async def verify_cache(db: MetadataDB, store: BlockStore) -> dict[str, int]:
    """Re-verify every cached block and remove the inconsistent ones."""
    checked = 0
    corrupted = 0
    missing = 0
    for entry in await _run(db.list_entries):
        checked += 1
        path = Path(entry.phys_path)
        if not path.is_file():
            missing += 1
            await _run(db.delete_entry, entry.block_key)
            continue
        ok = await _run(store.verify, entry.content_hash, expected_size=entry.size)
        if not ok:
            corrupted += 1
            await _run(db.delete_entry, entry.block_key)
            await _run(store.delete, entry.content_hash)
    return {"checked": checked, "corrupted": corrupted, "missing": missing}


async def _run(func, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


__all__ = ["recover_cache", "verify_cache", "CacheEntry"]
