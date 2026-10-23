"""TOML reader and writer for the Stream Agent configuration (Plan.md C0)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import (
    Config,
    EVICTION_POLICIES,
    FORMATS,
    LICENSE_REVIEW_STATES,
    ModelConfig,
    QUOTA_POLICIES,
    ServerConfig,
    SourceConfig,
    StorageConfig,
    validate_config,
)
from .errors import ConfigInvalid
from .sizes import GIB, MIB, parse_size

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_NAME = "stream-agent.toml"


def _get_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigInvalid(f"[{key}] must be a table")
    return value


def _limit_from_toml(
    table: dict[str, Any],
    *,
    gib_key: str,
    bytes_key: str,
    field_name: str,
    default: int,
) -> int:
    """Read a limit that may be given in GiB (``*_gib``) or in raw bytes."""
    if gib_key in table:
        raw = table[gib_key]
        if isinstance(raw, str):
            return parse_size(raw, field_name=field_name)
        return int(float(raw) * GIB)
    if bytes_key in table:
        return parse_size(table[bytes_key], field_name=field_name)
    return default


def _storage_from_toml(table: dict[str, Any], *, root_override: Path | None) -> StorageConfig:
    defaults = StorageConfig()
    root = table.get("root", str(defaults.root))
    root_path = Path(os.path.expanduser(str(root)))
    if root_override is not None:
        root_path = root_override

    weight_bytes = _limit_from_toml(
        table,
        gib_key="weight_cache_limit_gib",
        bytes_key="weight_cache_limit_bytes",
        field_name="storage.weight_cache_limit",
        default=defaults.weight_cache_limit_bytes,
    )
    managed_bytes = _limit_from_toml(
        table,
        gib_key="managed_store_limit_gib",
        bytes_key="managed_store_limit_bytes",
        field_name="storage.managed_store_limit",
        default=defaults.managed_store_limit_bytes
        if defaults.managed_store_limit_bytes is not None
        else 0,
    )
    metadata_reserve = (
        int(float(table["metadata_reserve_mib"]) * MIB)
        if "metadata_reserve_mib" in table
        else _limit_from_toml(
            table,
            gib_key="metadata_reserve_gib",
            bytes_key="metadata_reserve_bytes",
            field_name="storage.metadata_reserve",
            default=defaults.metadata_reserve_bytes,
        )
    )
    staging_reserve = (
        int(float(table["staging_reserve_mib"]) * MIB)
        if "staging_reserve_mib" in table
        else _limit_from_toml(
            table,
            gib_key="staging_reserve_gib",
            bytes_key="staging_reserve_bytes",
            field_name="storage.staging_reserve",
            default=defaults.staging_reserve_bytes,
        )
    )

    quota_policy = str(table.get("quota_policy", defaults.quota_policy)).lower()
    if quota_policy not in QUOTA_POLICIES:
        raise ConfigInvalid(
            f"storage.quota_policy must be one of {QUOTA_POLICIES}, got {quota_policy!r}"
        )
    eviction_policy = str(table.get("eviction_policy", defaults.eviction_policy)).lower()
    if eviction_policy not in EVICTION_POLICIES:
        raise ConfigInvalid(
            f"storage.eviction_policy must be one of {EVICTION_POLICIES}, got {eviction_policy!r}"
        )
    block_size = int(table.get("block_size_bytes", defaults.block_size_bytes))
    if block_size <= 0:
        raise ConfigInvalid("storage.block_size_bytes must be positive")

    return StorageConfig(
        root=root_path,
        weight_cache_limit_bytes=weight_bytes,
        managed_store_limit_bytes=managed_bytes,
        metadata_reserve_bytes=metadata_reserve,
        staging_reserve_bytes=staging_reserve,
        quota_policy=quota_policy,
        eviction_policy=eviction_policy,
        verify_before_use=bool(table.get("verify_before_use", defaults.verify_before_use)),
        block_size_bytes=block_size,
    )


def _server_from_toml(table: dict[str, Any]) -> ServerConfig:
    defaults = ServerConfig()
    host = str(table.get("host", defaults.host))
    allow_remote = bool(table.get("allow_remote_bind", defaults.allow_remote_bind))
    if allow_remote and host in ("127.0.0.1", "localhost", "::1"):
        raise ConfigInvalid(
            "server.allow_remote_bind is true but server.host is a loopback address; "
            "set an explicit non-loopback host if you really mean to expose the endpoint"
        )
    return ServerConfig(
        host=host,
        port=int(table.get("port", defaults.port)),
        max_concurrent_requests=int(table.get("max_concurrent_requests", defaults.max_concurrent_requests)),
        request_timeout_seconds=int(
            table.get("request_timeout_seconds", defaults.request_timeout_seconds)
        ),
        allow_remote_bind=allow_remote,
    )


def _source_from_toml(table: dict[str, Any]) -> SourceConfig:
    defaults = SourceConfig()
    return SourceConfig(
        enabled=bool(table.get("enabled", defaults.enabled)),
        token_env=str(table.get("token_env", defaults.token_env)),
        max_parallel_ranges=int(table.get("max_parallel_ranges", defaults.max_parallel_ranges)),
        request_timeout_seconds=int(table.get("request_timeout_seconds", defaults.request_timeout_seconds)),
        retry_limit=int(table.get("retry_limit", defaults.retry_limit)),
    )


def _models_from_toml(table: dict[str, Any], *, storage: StorageConfig) -> dict[str, ModelConfig]:
    models: dict[str, ModelConfig] = {}
    for alias, raw in table.items():
        if not isinstance(raw, dict):
            raise ConfigInvalid(f"[models.{alias}] must be a table")
        repo = str(raw.get("repo", "")).strip()
        if not repo:
            raise ConfigInvalid(f"[models.{alias}] is missing 'repo'")
        revision = str(raw.get("revision", "")).strip()
        fmt = str(raw.get("format", "safetensors")).lower()
        if fmt not in FORMATS:
            raise ConfigInvalid(
                f"[models.{alias}] format {fmt!r} is not supported; expected one of {FORMATS}"
            )
        quota_raw = raw.get("weight_cache_quota_gib", raw.get("weight_cache_quota_bytes"))
        quota = (
            parse_size(quota_raw, field_name=f"models.{alias}.weight_cache_quota")
            if quota_raw is not None
            else 0
        )
        quota_mode = str(raw.get("quota_mode", storage.quota_policy)).lower()
        if quota_mode not in QUOTA_POLICIES:
            raise ConfigInvalid(
                f"[models.{alias}] quota_mode must be one of {QUOTA_POLICIES}, got {quota_mode!r}"
            )
        review = str(raw.get("license_review", "pending")).lower()
        if review not in LICENSE_REVIEW_STATES:
            raise ConfigInvalid(
                f"[models.{alias}] license_review must be one of {LICENSE_REVIEW_STATES}, got {review!r}"
            )
        known = {
            "repo",
            "revision",
            "format",
            "backend",
            "weight_cache_quota_gib",
            "weight_cache_quota_bytes",
            "quota_mode",
            "license_review",
            "allow_remote_code",
        }
        extra = {k: v for k, v in raw.items() if k not in known}
        models[alias] = ModelConfig(
            alias=alias,
            repo=repo,
            revision=revision,
            format=fmt,
            backend=str(raw.get("backend", "")),
            weight_cache_quota_bytes=quota,
            quota_mode=quota_mode,
            license_review=review,
            allow_remote_code=bool(raw.get("allow_remote_code", False)),
            extra=extra,
        )
    return models


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root_override: Path | None = None,
) -> Config:
    """Load and validate a TOML configuration file.

    When ``path`` is None the default locations are searched; a missing default
    file is not an error (the built-in defaults are used).
    """
    config_path: Path | None = None
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigInvalid(f"configuration file not found: {config_path}")
    else:
        candidates = [
            Path.cwd() / DEFAULT_CONFIG_NAME,
            Path(os.path.expanduser("~")) / ".stream-agent" / DEFAULT_CONFIG_NAME,
            Path(os.path.expanduser("~")) / ".config" / "stream-agent" / DEFAULT_CONFIG_NAME,
        ]
        for candidate in candidates:
            if candidate.is_file():
                config_path = candidate
                break

    data: dict[str, Any] = {}
    if config_path is not None:
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigInvalid(f"invalid TOML in {config_path}: {exc}") from exc

    storage = _storage_from_toml(_get_table(data, "storage"), root_override=root_override)
    server = _server_from_toml(_get_table(data, "server"))
    source = _source_from_toml(_get_table(_get_table(data, "source"), "huggingface"))
    models = _models_from_toml(_get_table(data, "models"), storage=storage)

    config = Config(storage=storage, server=server, source=source, models=models, path=config_path)
    validate_config(config)
    return config


def _toml_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_config(config: Config, path: str | os.PathLike[str]) -> Path:
    """Serialise a configuration to TOML (used by `config init` and tests)."""
    storage = config.storage
    lines: list[str] = []
    lines.append("[server]")
    lines.append(f"host = {_toml_string(config.server.host)}")
    lines.append(f"port = {config.server.port}")
    lines.append(f"max_concurrent_requests = {config.server.max_concurrent_requests}")
    lines.append(f"request_timeout_seconds = {config.server.request_timeout_seconds}")
    lines.append(f"allow_remote_bind = {'true' if config.server.allow_remote_bind else 'false'}")
    lines.append("")
    lines.append("[storage]")
    lines.append(f"root = {_toml_string(str(storage.root))}")
    lines.append(f"weight_cache_limit_gib = {storage.weight_cache_limit_bytes / GIB:g}")
    if storage.managed_store_limit_bytes is None:
        lines.append("# managed_store_limit_gib = (unset)")
    else:
        lines.append(f"managed_store_limit_gib = {storage.managed_store_limit_bytes / GIB:g}")
    lines.append(f"metadata_reserve_mib = {storage.metadata_reserve_bytes / MIB:g}")
    lines.append(f"staging_reserve_mib = {storage.staging_reserve_bytes / MIB:g}")
    lines.append(f"quota_policy = {_toml_string(storage.quota_policy)}")
    lines.append(f"eviction_policy = {_toml_string(storage.eviction_policy)}")
    lines.append(f"verify_before_use = {'true' if storage.verify_before_use else 'false'}")
    lines.append(f"block_size_bytes = {storage.block_size_bytes}")
    lines.append("")
    lines.append("[source.huggingface]")
    lines.append(f"enabled = {'true' if config.source.enabled else 'false'}")
    lines.append(f"token_env = {_toml_string(config.source.token_env)}")
    lines.append(f"max_parallel_ranges = {config.source.max_parallel_ranges}")
    lines.append(f"request_timeout_seconds = {config.source.request_timeout_seconds}")
    lines.append(f"retry_limit = {config.source.retry_limit}")
    for alias, model in sorted(config.models.items()):
        lines.append("")
        lines.append(f"[models.{alias}]")
        lines.append(f"repo = {_toml_string(model.repo)}")
        lines.append(f"revision = {_toml_string(model.revision)}")
        lines.append(f"format = {_toml_string(model.format)}")
        if model.backend:
            lines.append(f"backend = {_toml_string(model.backend)}")
        lines.append(f"weight_cache_quota_gib = {model.weight_cache_quota_bytes / GIB:g}")
        lines.append(f"quota_mode = {_toml_string(model.quota_mode)}")
        lines.append(f"license_review = {_toml_string(model.license_review)}")
        lines.append(f"allow_remote_code = {'true' if model.allow_remote_code else 'false'}")
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_CONFIG_NAME",
    "load_config",
    "write_config",
]
