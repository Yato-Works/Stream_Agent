"""Layout parsing tests (Plan.md C3 / 9.2)."""

from __future__ import annotations

import json
import struct

import pytest

from stream_agent.errors import ModelFormatUnsupported
from stream_agent.layout.gguf import parse_gguf_metadata
from stream_agent.layout.model_map import build_layout
from stream_agent.layout.safetensors import (
    parse_header_length,
    parse_safetensors_header,
    parse_shard_index,
)

from .conftest import COMMIT, REPO, make_safetensors


def test_header_length_is_decoded_little_endian():
    assert parse_header_length(struct.pack("<Q", 42) + b"") == 42


def test_implausible_header_length_is_rejected():
    with pytest.raises(ModelFormatUnsupported):
        parse_header_length(struct.pack("<Q", 0))
    with pytest.raises(ModelFormatUnsupported):
        parse_header_length(struct.pack("<Q", 10**12))


def test_offsets_are_validated_against_the_file_size():
    payload = make_safetensors({"a": ("F32", (2, 2), b"\x00" * 16)})
    header_length = struct.unpack("<Q", payload[:8])[0]
    header = payload[8 : 8 + header_length]
    with pytest.raises(ModelFormatUnsupported):
        parse_safetensors_header(header, file_size=8, shard="x.safetensors")


def test_declared_shape_must_match_the_payload():
    header = json.dumps(
        {"a": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 16]}}
    ).encode("utf-8")
    with pytest.raises(ModelFormatUnsupported):
        parse_safetensors_header(header, file_size=1024, shard="x.safetensors")


def test_unknown_dtype_is_rejected():
    header = json.dumps(
        {"a": {"dtype": "Q9", "shape": [2], "data_offsets": [0, 2]}}
    ).encode("utf-8")
    with pytest.raises(ModelFormatUnsupported):
        parse_safetensors_header(header, file_size=1024, shard="x.safetensors")


def test_overlapping_tensors_are_rejected():
    header = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        }
    ).encode("utf-8")
    with pytest.raises(ModelFormatUnsupported):
        parse_safetensors_header(header, file_size=1024, shard="x.safetensors")


def test_shard_index_rejects_invalid_entries():
    with pytest.raises(ModelFormatUnsupported):
        parse_shard_index(json.dumps({"weight_map": {"a": 42}}))
    with pytest.raises(ModelFormatUnsupported):
        parse_shard_index(json.dumps({"weight_map": {}}))


async def test_layout_rejects_an_index_that_references_absent_shards(fixture_source):
    fixture_source.repos[REPO][COMMIT]["model.safetensors.index.json"] = json.dumps(
        {"weight_map": {"model.embed_tokens.weight": "other.safetensors"}}
    ).encode("utf-8")
    revision = await fixture_source.resolve_revision(REPO, COMMIT)
    with pytest.raises(ModelFormatUnsupported):
        await build_layout(fixture_source, revision)


async def test_layout_reports_tensor_byte_ranges(fixture_source):
    revision = await fixture_source.resolve_revision(REPO, COMMIT)
    layout = await build_layout(fixture_source, revision)
    ref = layout.tensor("model.embed_tokens.weight")
    assert ref.data_start > 0
    assert ref.data_end - ref.data_start == 128
    assert layout.shard_of("model.embed_tokens.weight") == "model.safetensors"


def test_gguf_metadata_is_parsed():
    import struct

    def gguf_string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    payload = b"GGUF"
    payload += struct.pack("<I", 3)  # version
    payload += struct.pack("<Q", 1)  # tensor count
    payload += struct.pack("<Q", 1)  # kv count
    payload += gguf_string("general.alignment")
    payload += struct.pack("<I", 4)  # uint32
    payload += struct.pack("<I", 32)
    payload += gguf_string("blk.0.weight")
    payload += struct.pack("<I", 1)  # dims
    payload += struct.pack("<Q", 4)
    payload += struct.pack("<I", 0)  # type F32
    payload += struct.pack("<Q", 0)  # offset
    metadata = parse_gguf_metadata(payload, file_size=1024)
    assert metadata.metadata["general.alignment"] == 32
    assert metadata.tensors[0].name == "blk.0.weight"
    assert metadata.tensors[0].dims == (4,)
import struct


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make_gguf() -> bytes:
    payload = b"GGUF"
    payload += struct.pack("<I", 3)  # version
    payload += struct.pack("<Q", 1)  # tensor count
    payload += struct.pack("<Q", 1)  # kv count
    payload += gguf_string("general.alignment")
    payload += struct.pack("<I", 4)  # uint32
    payload += struct.pack("<I", 32)
    payload += gguf_string("w")
    payload += struct.pack("<I", 1)  # dims
    payload += struct.pack("<Q", 4)
    payload += struct.pack("<I", 0)  # F32
    payload += struct.pack("<Q", 0)  # offset
    data_offset = (len(payload) + 31) // 32 * 32
    payload += b"\x00" * (data_offset - len(payload))
    payload += b"\x00" * 16
    return payload


async def test_gguf_layout_reports_tensor_byte_ranges():
    from stream_agent.sources.local import MemorySource

    source = MemorySource()
    payload = make_gguf()
    source.add_revision("gguf/fixture", "rev", {"model.gguf": payload})
    revision = await source.resolve_revision("gguf/fixture", "rev")
    layout = await build_layout(source, revision, artifact_format="gguf")
    ref = layout.tensor("w")
    assert ref.data_end - ref.data_start == 16
    assert layout.shard_of("w") == "model.gguf"
