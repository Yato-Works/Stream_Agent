"""GGUF metadata parsing (Plan.md C3 / 9.3).

GGUF stores a header with a key/value metadata section followed by a tensor
directory with byte offsets, which makes selective reads possible in
principle.  Most inference backends expect the file to be locally
addressable (mmap), so remote-backed GGUF execution still requires a tested
reader path; this module provides the metadata side of that work.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..errors import ModelFormatUnsupported
from ..sources.base import RevisionInfo, SourceAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model_map import ModelLayout

GGUF_MAGIC = b"GGUF"

#: ggml tensor type identifiers and their element sizes in bytes.
GGML_TYPE_SIZES: dict[int, int] = {
    0: 4,   # F32
    1: 2,   # F16
    2: 28,  # Q4_0 (block of 32)
    3: 32,  # Q4_1 (block of 32)
    6: 20,  # Q5_0 (block of 32)
    7: 24,  # Q5_1 (block of 32)
    8: 34,  # Q8_0 (block of 32)
    9: 36,  # Q8_1 (block of 32)
    10: 84, # Q2_K (block of 256)
    11: 110, # Q3_K
    12: 144, # Q4_K
    13: 176, # Q5_K
    14: 210, # Q6_K
    15: 292, # Q8_K
    16: 66, # IQ2_XXS
    17: 74, # IQ2_XS
    18: 98, # IQ3_XXS
    19: 50, # IQ1_S
    20: 110, # IQ4_NL
    21: 82, # IQ3_S
    22: 136, # I8
    23: 52, # I16
    24: 56, # I32
    25: 64, # I64
    26: 78, # F64
    27: 96, # IQ1_M
    28: 74, # BF16
}

GGUF_TYPE_SIZES: dict[int, int] = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    8: 0,   # STRING (variable)
    9: 0,   # ARRAY (variable)
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        if self.pos + count > len(self.data):
            raise ModelFormatUnsupported(
                "GGUF metadata is truncated",
                details={"needed": count, "available": len(self.data) - self.pos},
            )
        chunk = self.data[self.pos : self.pos + count]
        self.pos += count
        return chunk

    def u8(self) -> int:
        return struct.unpack("<B", self.take(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 64 * 1024 * 1024:
            raise ModelFormatUnsupported("implausible GGUF string length")
        raw = self.take(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelFormatUnsupported("GGUF string is not valid UTF-8") from exc


@dataclass
class GGUFTensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int  # relative to the tensor data section

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.dims:
            count *= dim
        return count


@dataclass
class GGUFMetadata:
    version: int
    tensor_count: int
    kv_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[GGUFTensorInfo] = field(default_factory=list)
    data_offset: int = 0
    file_size: int = 0

    def tensor(self, name: str) -> GGUFTensorInfo:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise ModelFormatUnsupported(f"GGUF tensor {name!r} is not present in the file")


def _read_value(reader: _Reader, value_type: int) -> Any:
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        element_type = reader.u32()
        count = reader.u64()
        if count > 10_000_000:
            raise ModelFormatUnsupported("implausible GGUF array length")
        return [_read_value(reader, element_type) for _ in range(count)]
    size = GGUF_TYPE_SIZES.get(value_type)
    if size is None:
        raise ModelFormatUnsupported(f"unknown GGUF metadata value type {value_type}")
    raw = reader.take(size)
    fmt = {1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[value_type]
    return struct.unpack(fmt, raw)[0]


def parse_gguf_metadata(payload: bytes, *, file_size: int | None = None) -> GGUFMetadata:
    """Parse the GGUF header of a file."""
    reader = _Reader(payload)
    magic = reader.take(4)
    if magic != GGUF_MAGIC:
        raise ModelFormatUnsupported("file does not start with the GGUF magic")
    version = reader.u32()
    if version not in (2, 3):
        raise ModelFormatUnsupported(f"unsupported GGUF version {version}")
    tensor_count = reader.u64()
    kv_count = reader.u64()
    if tensor_count > 10_000_000 or kv_count > 1_000_000:
        raise ModelFormatUnsupported("implausible GGUF header counts")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.u32()
        metadata[key] = _read_value(reader, value_type)

    tensors: list[GGUFTensorInfo] = []
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        if n_dims > 8:
            raise ModelFormatUnsupported(f"GGUF tensor {name!r} has too many dimensions")
        dims = tuple(reader.u64() for _ in range(n_dims))
        ggml_type = reader.u32()
        offset = reader.u64()
        tensors.append(GGUFTensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))

    alignment = int(metadata.get("general.alignment", 32))
    data_offset = (reader.pos + alignment - 1) // alignment * alignment
    return GGUFMetadata(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        metadata=metadata,
        tensors=tensors,
        data_offset=data_offset,
        file_size=file_size or 0,
    )


async def build_gguf_layout(
    source: SourceAdapter,
    revision: RevisionInfo,
    *,
    max_concurrency: int = 4,
) -> "ModelLayout":
    from .model_map import ModelLayout

    candidates = sorted(p for p in revision.files if p.endswith(".gguf"))
    if not candidates:
        raise ModelFormatUnsupported(f"repository {revision.repo} contains no GGUF file")
    path = candidates[0]
    info = revision.files[path]
    size = info.size or 0
    if size <= 0:
        raise ModelFormatUnsupported(f"the source did not report a size for {path}")

    # The header is small in practice; cap the range we are willing to read.
    header_bytes = await source.fetch_range(revision.repo, revision.commit, path, 0, min(size, 8 * 1024 * 1024) - 1)
    metadata = parse_gguf_metadata(header_bytes, file_size=size)

    layout = ModelLayout(
        repo=revision.repo,
        commit=revision.commit,
        artifact_format="gguf",
        total_weight_bytes=size,
    )
    layout.shards[path] = _gguf_shard(path, size, metadata)
    for tensor in metadata.tensors:
        layout.tensor_shard[tensor.name] = path
    return layout


def _gguf_shard(path: str, size: int, metadata: GGUFMetadata):
    from .model_map import ShardInfo

    return ShardInfo(path=path, size=size, header_length=metadata.data_offset, header=None)


__all__ = [
    "GGUFTensorInfo",
    "GGUFMetadata",
    "GGML_TYPE_SIZES",
    "parse_gguf_metadata",
    "build_gguf_layout",
]
