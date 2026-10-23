"""Shared pytest fixtures."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from stream_agent.config import Config, ModelConfig, ServerConfig, SourceConfig, StorageConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider
from stream_agent.sources.local import MemorySource
from stream_agent.telemetry import RunTelemetry

REPO = "fixture/tiny-model"
COMMIT = "c0ffee1234567890"


def make_safetensors(tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> bytes:
    """Build a small but valid safetensors file."""
    header: dict[str, object] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payloads.append(data)
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payloads)


@pytest.fixture()
def fixture_source() -> MemorySource:
    source = MemorySource()
    tensors = {
        "model.embed_tokens.weight": ("F32", (8, 4), b"\x00" * 128),
        "model.layers.0.self_attn.q_proj.weight": ("F32", (4, 4), b"\x01" * 64),
        "model.layers.0.mlp.down_proj.weight": ("F32", (4, 4), b"\x02" * 64),
        "model.layers.1.self_attn.q_proj.weight": ("F32", (4, 4), b"\x03" * 64),
        "lm_head.weight": ("F32", (8, 4), b"\x04" * 128),
    }
    payload = make_safetensors(tensors)
    index = {
        "metadata": {"total_size": len(payload)},
        "weight_map": {
            "model.embed_tokens.weight": "model.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
        },
    }
    source.add_revision(
        REPO,
        COMMIT,
        {
            "model.safetensors": payload,
            "model.safetensors.index.json": json.dumps(index).encode("utf-8"),
            "config.json": json.dumps({"model_type": "fixture"}).encode("utf-8"),
        },
        license_id="mit",
    )
    return source


@pytest.fixture()
def storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        root=tmp_path / "store",
        weight_cache_limit_bytes=256 * 1024,
        managed_store_limit_bytes=1024 * 1024,
        metadata_reserve_bytes=64 * 1024,
        staging_reserve_bytes=64 * 1024,
    )


@pytest.fixture()
def config(storage: StorageConfig) -> Config:
    return Config(
        storage=storage,
        server=ServerConfig(),
        source=SourceConfig(),
        models={
            "tiny": ModelConfig(
                alias="tiny",
                repo=REPO,
                revision=COMMIT,
                format="safetensors",
                backend="fixture",
                weight_cache_quota_bytes=128 * 1024,
            )
        },
    )


@pytest.fixture()
async def runtime(config: Config, fixture_source: MemorySource):
    from stream_agent.runtime import StreamAgentRuntime

    rt = StreamAgentRuntime(config, source=fixture_source, telemetry=RunTelemetry())
    await rt.start()
    try:
        yield rt
    finally:
        await rt.close()


@pytest.fixture()
async def provider(runtime, fixture_source):
    revision = await fixture_source.resolve_revision(REPO, COMMIT)
    layout = await build_layout(fixture_source, revision)
    return RemoteWeightProvider(
        source=fixture_source,
        layout=layout,
        cache=runtime.cache,
        model_alias="tiny",
        block_size_bytes=64,
        telemetry=runtime.telemetry,
    )


__all__ = ["make_safetensors", "REPO", "COMMIT"]
