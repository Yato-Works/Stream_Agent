"""Configuration model for Stream Agent (Plan.md C0 / section 13).

This module holds the configuration *objects* and the validation rules.  The
TOML reader and writer live in :mod:`stream_agent.config_file`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigInvalid
from .sizes import GIB, KIB, MIB, format_size, parse_size

QUOTA_POLICIES = ("fixed", "elastic")
EVICTION_POLICIES = ("lru",)
LICENSE_REVIEW_STATES = ("pending", "required", "accepted", "rejected")
FORMATS = ("safetensors", "gguf")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8789
    max_concurrent_requests: int = 1
    request_timeout_seconds: int = 3600
    allow_remote_bind: bool = False


@dataclass(frozen=True)
class StorageConfig:
    root: Path = Path.home() / ".stream-agent"
    weight_cache_limit_bytes: int = 18 * GIB
    managed_store_limit_bytes: int | None = 20 * GIB
    metadata_reserve_bytes: int = 256 * MIB
    staging_reserve_bytes: int = 768 * MIB
    quota_policy: str = "fixed"
    eviction_policy: str = "lru"
    verify_before_use: bool = True
    block_size_bytes: int = 8 * 1024 * 1024

    @property
    def blocks_root(self) -> Path:
        return self.root / "blocks"

    @property
    def tmp_root(self) -> Path:
        return self.root / "tmp"

    @property
    def metadata_root(self) -> Path:
        return self.root / "metadata"

    @property
    def log_root(self) -> Path:
        return self.root / "logs"

    @property
    def db_path(self) -> Path:
        return self.metadata_root / "stream_agent.sqlite3"


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for the Hugging Face source adapter."""

    enabled: bool = True
    token_env: str = "HF_TOKEN"
    max_parallel_ranges: int = 4
    request_timeout_seconds: int = 120
    retry_limit: int = 4
    user_agent: str = "stream-agent/0.1"

    def resolve_token(self) -> str | None:
        import os

        token = os.environ.get(self.token_env)
        return token.strip() if token and token.strip() else None


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    repo: str
    revision: str
    format: str = "safetensors"
    backend: str = ""
    weight_cache_quota_bytes: int = 0
    quota_mode: str = "fixed"
    license_review: str = "pending"
    allow_remote_code: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pinned(self) -> bool:
        import re

        return bool(self.revision) and not re.search(
            r"<[^>]*>|PINNED|TODO|FIXME|PLACEHOLDER", self.revision, re.IGNORECASE
        )


@dataclass(frozen=True)
class Config:
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    path: Path | None = None

    def model(self, alias: str) -> ModelConfig:
        try:
            return self.models[alias]
        except KeyError as exc:
            raise ConfigInvalid(
                f"unknown model alias {alias!r}",
                details={"known": sorted(self.models)},
            ) from exc

    def with_models(self, models: Iterable[ModelConfig]) -> "Config":
        return replace(self, models={m.alias: m for m in models})


def validate_config(config: Config) -> None:
    """Fail fast on a configuration that cannot satisfy the design invariants."""
    storage = config.storage
    if storage.weight_cache_limit_bytes <= 0:
        raise ConfigInvalid("storage.weight_cache_limit_bytes must be positive")
    if storage.managed_store_limit_bytes is not None:
        if storage.managed_store_limit_bytes <= 0:
            raise ConfigInvalid("storage.managed_store_limit_bytes must be positive")
        overhead = storage.metadata_reserve_bytes + storage.staging_reserve_bytes
        if storage.weight_cache_limit_bytes > storage.managed_store_limit_bytes:
            raise ConfigInvalid(
                "storage.weight_cache_limit_bytes exceeds storage.managed_store_limit_bytes; "
                "the payload budget must leave room for metadata and staging"
            )
        if storage.weight_cache_limit_bytes + overhead > storage.managed_store_limit_bytes:
            raise ConfigInvalid(
                "storage.weight_cache_limit_bytes plus metadata/staging reserve exceeds "
                "storage.managed_store_limit_bytes"
            )
        if storage.managed_store_limit_bytes - storage.weight_cache_limit_bytes < overhead:
            raise ConfigInvalid(
                "managed store limit does not leave enough headroom for the configured "
                "metadata and staging reserves"
            )
    if storage.quota_policy == "fixed":
        total_fixed = sum(
            m.weight_cache_quota_bytes for m in config.models.values() if m.quota_mode == "fixed"
        )
        if total_fixed > storage.weight_cache_limit_bytes:
            raise ConfigInvalid(
                "sum of fixed per-model quotas exceeds the global weight cache limit",
                details={
                    "quota_total_bytes": total_fixed,
                    "weight_cache_limit_bytes": storage.weight_cache_limit_bytes,
                },
            )


__all__ = [
    "Config",
    "StorageConfig",
    "ServerConfig",
    "SourceConfig",
    "ModelConfig",
    "QUOTA_POLICIES",
    "EVICTION_POLICIES",
    "LICENSE_REVIEW_STATES",
    "FORMATS",
    "validate_config",
    "parse_size",
    "format_size",
    "GIB",
    "MIB",
    "KIB",
]
