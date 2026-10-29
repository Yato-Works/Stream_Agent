"""Configuration and error contract tests (Plan.md C0 / section 15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from stream_agent.config import (
    Config,
    ModelConfig,
    StorageConfig,
    validate_config,
)
from stream_agent.config_file import load_config, write_config
from stream_agent.errors import (
    ConfigInvalid,
    ErrorCode,
    StreamAgentError,
)
from stream_agent.sizes import GIB, format_size, parse_size


def test_parse_size_supports_binary_and_decimal_units():
    assert parse_size(1024) == 1024
    assert parse_size("1KiB") == 1024
    assert parse_size("1MiB") == 1024**2
    assert parse_size("1GiB") == GIB
    assert parse_size("1GB") == 1000**3
    with pytest.raises(ConfigInvalid):
        parse_size("seven")


def test_format_size_is_display_only():
    assert format_size(GIB) == "1.00 GiB"
    assert format_size(0) == "0 B"


def test_quota_sum_may_not_exceed_the_global_limit():
    config = Config(
        storage=StorageConfig(
            root=Path("x"), weight_cache_limit_bytes=10 * GIB, managed_store_limit_bytes=None
        ),
        models={
            "a": ModelConfig(alias="a", repo="x/y", revision="abc", weight_cache_quota_bytes=6 * GIB),
            "b": ModelConfig(alias="b", repo="x/z", revision="abc", weight_cache_quota_bytes=6 * GIB),
        },
    )
    with pytest.raises(ConfigInvalid):
        validate_config(config)


def test_fixed_quotas_are_validated_even_under_a_shared_default_policy():
    config = Config(
        storage=StorageConfig(
            root=Path("x"),
            weight_cache_limit_bytes=10 * GIB,
            managed_store_limit_bytes=None,
            quota_policy="elastic",
        ),
        models={
            "a": ModelConfig(
                alias="a", repo="x/y", revision="abc",
                weight_cache_quota_bytes=6 * GIB, quota_mode="fixed",
            ),
            "b": ModelConfig(
                alias="b", repo="x/z", revision="abc",
                weight_cache_quota_bytes=6 * GIB, quota_mode="fixed",
            ),
            "c": ModelConfig(
                alias="c", repo="x/w", revision="abc",
                weight_cache_quota_bytes=99 * GIB, quota_mode="elastic",
            ),
        },
    )
    with pytest.raises(ConfigInvalid):
        validate_config(config)


def test_weight_cache_limit_must_leave_room_inside_the_store_limit():
    config = Config(
        storage=StorageConfig(
            root=Path("x"),
            weight_cache_limit_bytes=20 * GIB,
            managed_store_limit_bytes=20 * GIB,
            metadata_reserve_bytes=256 * 1024**2,
            staging_reserve_bytes=768 * 1024**2,
        )
    )
    with pytest.raises(ConfigInvalid):
        validate_config(config)


def test_round_trip_of_a_written_configuration(tmp_path: Path):
    config = Config(
        storage=StorageConfig(
            root=tmp_path / "store",
            weight_cache_limit_bytes=12 * GIB,
            managed_store_limit_bytes=16 * GIB,
            metadata_reserve_bytes=128 * 1024**2,
            staging_reserve_bytes=256 * 1024**2,
        ),
        models={
            "demo": ModelConfig(
                alias="demo",
                repo="acme/demo",
                revision="0123456789abcdef",
                backend="transformers",
                weight_cache_quota_bytes=4 * GIB,
            )
        },
    )
    path = write_config(config, tmp_path / "stream-agent.toml")
    loaded = load_config(path)
    assert loaded.storage.weight_cache_limit_bytes == 12 * GIB
    assert loaded.models["demo"].repo == "acme/demo"
    assert loaded.models["demo"].is_pinned


def test_placeholder_revisions_are_not_pinned():
    model = ModelConfig(alias="a", repo="x/y", revision="<PINNED_COMMIT_SHA>")
    assert not model.is_pinned


def test_error_codes_are_stable():
    assert ErrorCode.CACHE_BUDGET_EXCEEDED == "CACHE_BUDGET_EXCEEDED"
    assert ErrorCode.RANGE_UNSUPPORTED == "RANGE_UNSUPPORTED"
    error = StreamAgentError("boom", details={"a": 1})
    assert error.to_dict() == {
        "code": "STREAM_AGENT_ERROR",
        "message": "boom",
        "details": {"a": 1},
    }


def test_unknown_model_alias_is_rejected(tmp_path: Path):
    config = Config(storage=StorageConfig(root=tmp_path))
    with pytest.raises(ConfigInvalid):
        config.model("nope")
