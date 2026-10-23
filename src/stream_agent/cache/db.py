"""Metadata database and usage ledger (Plan.md C5 / C9).

SQLite is used in WAL mode.  Every database operation is executed through
``asyncio.to_thread`` by the callers so the event loop is never blocked by
metadata work (Plan.md C4 requires this explicitly).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class CacheEntry:
    """One cached weight block."""

    block_key: str
    content_hash: str
    repo: str
    commit: str
    path: str
    start: int
    end: int
    size: int
    expected_size: int
    verified: bool
    phys_path: str
    model_alias: str
    refs: set[str] = field(default_factory=set)
    last_access: float = 0.0
    access_count: int = 0
    pin_count: int = 0
    created_at: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "block_key": self.block_key,
            "content_hash": self.content_hash,
            "repo": self.repo,
            "commit_sha": self.commit,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "expected_size": self.expected_size,
            "verified": 1 if self.verified else 0,
            "phys_path": self.phys_path,
            "model_alias": self.model_alias,
            "refs": json.dumps(sorted(self.refs)),
            "last_access": self.last_access,
            "access_count": self.access_count,
            "pin_count": self.pin_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CacheEntry":
        try:
            refs = set(json.loads(row["refs"]))
        except (json.JSONDecodeError, TypeError):
            refs = set()
        return cls(
            block_key=row["block_key"],
            content_hash=row["content_hash"],
            repo=row["repo"],
            commit=row["commit_sha"],
            path=row["path"],
            start=int(row["start"]),
            end=int(row["end"]),
            size=int(row["size"]),
            expected_size=int(row["expected_size"]),
            verified=bool(row["verified"]),
            phys_path=row["phys_path"],
            model_alias=row["model_alias"],
            refs=refs,
            last_access=float(row["last_access"]),
            access_count=int(row["access_count"]),
            pin_count=int(row["pin_count"]),
            created_at=float(row["created_at"]),
        )


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS models (
    alias              TEXT PRIMARY KEY,
    repo               TEXT NOT NULL,
    revision           TEXT NOT NULL,
    artifact_format    TEXT NOT NULL DEFAULT 'safetensors',
    backend            TEXT NOT NULL DEFAULT '',
    quota_bytes        INTEGER NOT NULL DEFAULT 0,
    quota_mode         TEXT NOT NULL DEFAULT 'fixed',
    license_review     TEXT NOT NULL DEFAULT 'pending',
    allow_remote_code  INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL,
    metadata_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entries (
    block_key      TEXT PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    repo           TEXT NOT NULL,
    commit_sha     TEXT NOT NULL,
    path           TEXT NOT NULL,
    start          INTEGER NOT NULL,
    end            INTEGER NOT NULL,
    size           INTEGER NOT NULL,
    expected_size  INTEGER NOT NULL,
    verified       INTEGER NOT NULL DEFAULT 0,
    phys_path      TEXT NOT NULL,
    model_alias    TEXT NOT NULL,
    refs           TEXT NOT NULL DEFAULT '[]',
    last_access    REAL NOT NULL DEFAULT 0,
    access_count   INTEGER NOT NULL DEFAULT 0,
    pin_count      INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_entries_model ON entries(model_alias);
CREATE INDEX IF NOT EXISTS idx_entries_content ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_entries_lru ON entries(last_access);

CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    run_id      TEXT,
    event       TEXT NOT NULL,
    model_alias TEXT,
    bytes       INTEGER NOT NULL DEFAULT 0,
    detail      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_usage_run ON usage_events(run_id);
"""


class MetadataDB:
    """Thin, thread-safe wrapper around the metadata database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- models ------------------------------------------------------------
    def upsert_model(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO models (alias, repo, revision, artifact_format, backend, quota_bytes,
                                    quota_mode, license_review, allow_remote_code, created_at,
                                    metadata_json)
                VALUES (:alias, :repo, :revision, :artifact_format, :backend, :quota_bytes,
                        :quota_mode, :license_review, :allow_remote_code, :created_at,
                        :metadata_json)
                ON CONFLICT(alias) DO UPDATE SET
                    repo=excluded.repo,
                    revision=excluded.revision,
                    artifact_format=excluded.artifact_format,
                    backend=excluded.backend,
                    quota_bytes=excluded.quota_bytes,
                    quota_mode=excluded.quota_mode,
                    license_review=excluded.license_review,
                    allow_remote_code=excluded.allow_remote_code,
                    metadata_json=excluded.metadata_json
                """,
                {
                    "alias": record["alias"],
                    "repo": record["repo"],
                    "revision": record["revision"],
                    "artifact_format": record.get("artifact_format", "safetensors"),
                    "backend": record.get("backend", ""),
                    "quota_bytes": int(record.get("quota_bytes", 0)),
                    "quota_mode": record.get("quota_mode", "fixed"),
                    "license_review": record.get("license_review", "pending"),
                    "allow_remote_code": 1 if record.get("allow_remote_code") else 0,
                    "created_at": float(record.get("created_at", time.time())),
                    "metadata_json": json.dumps(record.get("metadata", {})),
                },
            )
            self._conn.commit()

    def get_model(self, alias: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE alias = ?", (alias,)).fetchone()
        return dict(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM models ORDER BY alias").fetchall()
        return [dict(row) for row in rows]

    def remove_model(self, alias: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM models WHERE alias = ?", (alias,))
            self._conn.commit()

    # -- entries -----------------------------------------------------------
    def upsert_entry(self, entry: CacheEntry) -> None:
        row = entry.to_row()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entries (block_key, content_hash, repo, commit_sha, path, start, end, size,
                                     expected_size, verified, phys_path, model_alias, refs,
                                     last_access, access_count, pin_count, created_at)
                VALUES (:block_key, :content_hash, :repo, :commit_sha, :path, :start, :end, :size,
                        :expected_size, :verified, :phys_path, :model_alias, :refs,
                        :last_access, :access_count, :pin_count, :created_at)
                ON CONFLICT(block_key) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    verified=excluded.verified,
                    phys_path=excluded.phys_path,
                    model_alias=excluded.model_alias,
                    refs=excluded.refs,
                    last_access=excluded.last_access,
                    access_count=excluded.access_count,
                    pin_count=excluded.pin_count,
                    size=excluded.size                """,
                row,
            )
            self._conn.commit()

    def delete_entry(self, block_key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries WHERE block_key = ?", (block_key,))
            self._conn.commit()

    def get_entry(self, block_key: str) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entries WHERE block_key = ?", (block_key,)
            ).fetchone()
        return CacheEntry.from_row(row) if row else None

    def list_entries(self) -> list[CacheEntry]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM entries").fetchall()
        return [CacheEntry.from_row(row) for row in rows]

    # -- usage ledger ------------------------------------------------------
    def record_event(
        self,
        event: str,
        *,
        run_id: str | None = None,
        model_alias: str | None = None,
        num_bytes: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events (ts, run_id, event, model_alias, bytes, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    run_id,
                    event,
                    model_alias,
                    int(num_bytes),
                    json.dumps(detail or {}),
                ),
            )
            self._conn.commit()

    def usage_events(self, run_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            if run_id:
                rows = self._conn.execute(
                    "SELECT * FROM usage_events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM usage_events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["detail"] = json.loads(item.get("detail") or "{}")
            except json.JSONDecodeError:
                item["detail"] = {}
            out.append(item)
        return out

    def usage_totals(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT event, COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes
                FROM usage_events GROUP BY event
                """
            ).fetchall()
        return {r["event"]: {"count": r["count"], "bytes": r["bytes"]} for r in row}

    # -- maintenance -------------------------------------------------------
    def reset_pins(self) -> None:
        """Sessions do not survive a restart, so no block can still be pinned."""
        with self._lock:
            self._conn.execute("UPDATE entries SET pin_count = 0")
            self._conn.commit()


def entries_to_json(entries: Iterable[CacheEntry]) -> str:
    return json.dumps([e.to_row() for e in entries], indent=2)


__all__ = ["CacheEntry", "MetadataDB", "entries_to_json"]
