"""Safetensors layout parsing (Plan.md C3 / 9.2).

The parser is deliberately strict: model metadata is untrusted input.  A
malformed header, an out-of-bounds offset or an index that references an
absent shard must fail loudly instead of producing a wrong byte range.

File layout::

    [ 8 bytes : little endian u64 header length N ]
    [ N bytes : JSON header ]
    [ data    : tensor payloads, addressed by relative offsets ]

For sharded repositories an additional ``model.safetensors.index.json`` maps
tensor names onto shard file names.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from ..errors import ModelFormatUnsupported

HEADER_LENGTH_BYTES = 8

#: Bytes per element for the dtypes Stream Agent understands.
DTYPE_SIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: Dtypes whose payload is not a plain dense array of the given element size.
QUANTIZED_DTYPES = frozenset(
    {
        "I4", "U4", "I4_BF16", "I4_F16", "F4", "I8", "U8",
    }
)


@dataclass(frozen=True)
class TensorRef:
    """A tensor and where its bytes live inside a concrete artifact."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_start: int  # absolute byte offset inside the shard file
    data_end: int  # absolute byte offset (exclusive)
    file_size: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def num_elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def element_size(self) -> int:
        return DTYPE_SIZES.get(self.dtype.upper(), 0)

    @property
    def is_quantized(self) -> bool:
        return self.dtype.upper() in QUANTIZED_DTYPES

    def byte_range_for_elements(self, start: int, end: int) -> tuple[int, int]:
        """Absolute file byte range for the flattened element range ``[start, end)``.

        The range is aligned to the element size of the dtype; quantized
        payloads may only be sliced on their own packing grid and are rejected
        unless the requested window matches the packing unit exactly.
        """
        total = self.num_elements
        if start < 0 or end < start or end > total:
            raise ModelFormatUnsupported(
                f"element range [{start}, {end}) is outside tensor {self.name!r} "
                f"with {total} elements",
                details={"tensor": self.name, "elements": total},
            )
        size = self.element_size
        if size == 0:
            raise ModelFormatUnsupported(
                f"tensor {self.name!r} has unsupported dtype {self.dtype!r} for slicing",
                details={"tensor": self.name, "dtype": self.dtype},
            )
        if self.is_quantized:
            raise ModelFormatUnsupported(
                f"tensor {self.name!r} uses a quantized packing ({self.dtype!r}); "
                "slicing requires a model-specific packing map",
                details={"tensor": self.name, "dtype": self.dtype},
            )
        byte_start = self.data_start + start * size
        byte_end = self.data_start + end * size
        return byte_start, byte_end

    def byte_range_for_dim_slice(self, dim: int, index: int, count: int = 1) -> tuple[int, int]:
        """Absolute file byte range for ``count`` elements along one dimension.

        Used for expert level access, where a backend declares that experts
        are packed along a specific dimension of a tensor.
        """
        if dim < 0 or dim >= len(self.shape):
            raise ModelFormatUnsupported(
                f"tensor {self.name!r} has no dimension {dim}",
                details={"tensor": self.name, "dim": dim},
            )
        size = self.element_size
        if size == 0:
            raise ModelFormatUnsupported(
                f"tensor {self.name!r} has unsupported dtype {self.dtype!r} for slicing",
                details={"tensor": self.name, "dtype": self.dtype},
            )
        if self.is_quantized:
            raise ModelFormatUnsupported(
                f"tensor {self.name!r} uses a quantized packing ({self.dtype!r}); "
                "dimension slicing requires a model-specific packing map",
                details={"tensor": self.name, "dtype": self.dtype},
            )
        if index < 0 or index + count > self.shape[dim]:
            raise ModelFormatUnsupported(
                f"slice [{index}, {index + count}) is outside dimension {dim} of {self.name!r}",
                details={"tensor": self.name, "dim": dim, "shape": self.shape},
            )
        stride = 1
        for dim_size in self.shape[dim + 1 :]:
            stride *= dim_size
        byte_start = self.data_start + index * stride * size
        byte_end = byte_start + count * stride * size
        return byte_start, byte_end


@dataclass
class SafetensorsHeader:
    """Parsed header of a single safetensors file."""

    file_size: int
    tensors: dict[str, TensorRef]
    metadata: dict[str, str]

    def tensor(self, name: str) -> TensorRef:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ModelFormatUnsupported(
                f"tensor {name!r} is not present in the artifact",
                details={"tensor": name},
            ) from exc


def _validate_relative_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ModelFormatUnsupported(
            f"repository path {path!r} escapes the repository root",
            details={"path": path},
        )
    return path


def parse_safetensors_header(header_bytes: bytes, *, file_size: int, shard: str) -> SafetensorsHeader:
    """Parse and validate a safetensors header block."""
    try:
        payload = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFormatUnsupported(
            f"safetensors header of {shard} is not valid JSON: {exc}",
            details={"shard": shard},
        ) from exc
    if not isinstance(payload, dict):
        raise ModelFormatUnsupported(
            f"safetensors header of {shard} must be a JSON object",
            details={"shard": shard},
        )

    metadata = payload.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        raise ModelFormatUnsupported(
            f"safetensors __metadata__ of {shard} must be an object",
            details={"shard": shard},
        )

    data_start = HEADER_LENGTH_BYTES + len(header_bytes)
    tensors: dict[str, TensorRef] = {}
    occupied: list[tuple[int, int, str]] = []

    for name, spec in payload.items():
        if not isinstance(spec, dict):
            raise ModelFormatUnsupported(
                f"tensor entry {name!r} in {shard} must be an object",
                details={"shard": shard, "tensor": name},
            )
        dtype = str(spec.get("dtype", "")).upper()
        if dtype not in DTYPE_SIZES:
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} has unsupported dtype {dtype!r}",
                details={"shard": shard, "tensor": name, "dtype": dtype},
            )
        raw_shape = spec.get("shape")
        if not isinstance(raw_shape, list) or any(
            (not isinstance(d, int)) or d < 0 for d in raw_shape
        ):
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} has an invalid shape {raw_shape!r}",
                details={"shard": shard, "tensor": name},
            )
        shape = tuple(int(d) for d in raw_shape)
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
        ):
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} has invalid data_offsets {offsets!r}",
                details={"shard": shard, "tensor": name},
            )
        rel_start, rel_end = int(offsets[0]), int(offsets[1])
        if rel_start < 0 or rel_end < rel_start:
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} has an impossible offset range",
                details={"shard": shard, "tensor": name},
            )
        absolute_start = data_start + rel_start
        absolute_end = data_start + rel_end
        if absolute_end > file_size:
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} extends past the end of the file",
                details={"shard": shard, "tensor": name, "file_size": file_size},
            )
        expected = 1
        for dim in shape:
            expected *= dim
        element_size = DTYPE_SIZES[dtype]
        if expected * element_size != rel_end - rel_start:
            raise ModelFormatUnsupported(
                f"tensor {name!r} in {shard} declares {expected} elements of {dtype} "
                f"but occupies {rel_end - rel_start} bytes",
                details={"shard": shard, "tensor": name},
            )
        for other_start, other_end, other_name in occupied:
            if absolute_start < other_end and other_start < absolute_end:
                raise ModelFormatUnsupported(
                    f"tensor {name!r} in {shard} overlaps tensor {other_name!r}",
                    details={"shard": shard, "tensor": name},
                )
        occupied.append((absolute_start, absolute_end, name))
        tensors[name] = TensorRef(
            name=name,
            dtype=dtype,
            shape=shape,
            shard=shard,
            data_start=absolute_start,
            data_end=absolute_end,
            file_size=file_size,
        )

    return SafetensorsHeader(file_size=file_size, tensors=tensors, metadata=metadata)


def parse_header_length(prefix: bytes) -> int:
    """Decode the 8 byte little-endian header length prefix."""
    if len(prefix) < HEADER_LENGTH_BYTES:
        raise ModelFormatUnsupported(
            "not enough bytes to read the safetensors header length",
            details={"got": len(prefix)},
        )
    (length,) = struct.unpack("<Q", prefix[:HEADER_LENGTH_BYTES])
    if length <= 0 or length > 100 * 1024 * 1024:
        raise ModelFormatUnsupported(
            f"implausible safetensors header length: {length}",
            details={"header_length": length},
        )
    return length


def parse_shard_index(payload: bytes | str) -> dict[str, str]:
    """Parse ``model.safetensors.index.json`` into a tensor -> shard mapping."""
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelFormatUnsupported("shard index is not valid UTF-8") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelFormatUnsupported(f"shard index is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelFormatUnsupported("shard index must be a JSON object")
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelFormatUnsupported("shard index has no usable weight_map")
    result: dict[str, str] = {}
    for tensor, shard in weight_map.items():
        if not isinstance(shard, str) or not shard:
            raise ModelFormatUnsupported(
                f"shard index entry for {tensor!r} does not name a shard file"
            )
        result[str(tensor)] = _validate_relative_path(shard)
    return result


def known_dtypes() -> Iterable[str]:
    return tuple(DTYPE_SIZES)


__all__ = [
    "DTYPE_SIZES",
    "HEADER_LENGTH_BYTES",
    "SafetensorsHeader",
    "TensorRef",
    "parse_safetensors_header",
    "parse_header_length",
    "parse_shard_index",
    "known_dtypes",
]
