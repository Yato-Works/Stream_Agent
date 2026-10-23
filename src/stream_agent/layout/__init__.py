"""Model layout / weight map components (Plan.md C3)."""

from .model_map import (
    ExpertPacking,
    ModelLayout,
    ShardInfo,
    build_layout,
    build_safetensors_layout,
    describe_files,
)
from .safetensors import (
    DTYPE_SIZES,
    SafetensorsHeader,
    TensorRef,
    parse_header_length,
    parse_safetensors_header,
    parse_shard_index,
)

__all__ = [
    "DTYPE_SIZES",
    "ExpertPacking",
    "ModelLayout",
    "SafetensorsHeader",
    "ShardInfo",
    "TensorRef",
    "build_layout",
    "build_safetensors_layout",
    "describe_files",
    "parse_header_length",
    "parse_safetensors_header",
    "parse_shard_index",
]
