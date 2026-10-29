"""Model layout / weight map (Plan.md C3).

Maps model concepts (tensors, tensor slices, experts) onto byte ranges of a
pinned artifact.  The layout is built from *remote metadata only*: header
bytes and the shard index are fetched as small ranges, never a full weight
file.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..errors import ModelFormatUnsupported, SourceNotFoundError
from ..sources.base import RevisionInfo, SourceAdapter
from .safetensors import (
    HEADER_LENGTH_BYTES,
    SafetensorsHeader,
    TensorRef,
    parse_header_length,
    parse_safetensors_header,
    parse_shard_index,
)
from .safetensors import _validate_relative_path

WEIGHT_SUFFIXES = (".safetensors", ".gguf")
INDEX_CANDIDATES = (
    "model.safetensors.index.json",
    "model.gguf.index.json",
)

#: How many bytes of a shard are read to obtain its header in one request.
HEADER_PROBE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ShardInfo:
    path: str
    size: int
    header_length: int | None = None
    header: SafetensorsHeader | None = None


@dataclass(frozen=True)
class ExpertPacking:
    """Declared packing of experts inside one tensor.

    Backends declare this explicitly; the cache manager must never guess
    expert ids from file names (Plan.md C6).

    ``rows_per_expert`` is normally ``1`` (one expert per index along ``dim``,
    as in Qwen-style ``experts.gate_up_proj``).  Fused layouts that place
    several rows per expert can declare it explicitly; when omitted it is
    inferred from the tensor shape.
    """

    tensor: str
    dim: int
    num_experts: int
    rows_per_expert: int | None = None

    def slice_for(
        self, expert_id: int, shape: Sequence[int] | None = None
    ) -> tuple[int, int, int]:
        """Return ``(dim, index, count)`` for an expert id."""
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ModelFormatUnsupported(
                f"expert id {expert_id} is outside the declared packing "
                f"(0..{self.num_experts - 1})",
                details={"expert": expert_id, "num_experts": self.num_experts},
            )
        rows = self.rows_per_expert
        if rows is None:
            if shape is None:
                rows = 1
            else:
                if self.dim < 0 or self.dim >= len(shape):
                    raise ModelFormatUnsupported(
                        f"expert packing for {self.tensor!r} names dimension {self.dim}, "
                        f"but the tensor has shape {tuple(shape)}",
                        details={"tensor": self.tensor, "dim": self.dim},
                    )
                if shape[self.dim] % self.num_experts:
                    raise ModelFormatUnsupported(
                        f"dimension {self.dim} of {self.tensor!r} holds {shape[self.dim]} "
                        f"rows, which is not divisible by {self.num_experts} experts",
                        details={
                            "tensor": self.tensor,
                            "dim": self.dim,
                            "shape": tuple(shape),
                        },
                    )
                rows = shape[self.dim] // self.num_experts
        if rows <= 0:
            raise ModelFormatUnsupported(
                f"expert packing for {self.tensor!r} has no rows per expert",
                details={"tensor": self.tensor, "rows_per_expert": rows},
            )
        index = expert_id * rows
        return self.dim, index, rows


@dataclass
class ModelLayout:
    """Everything the runtime knows about where the bytes of a model live."""

    repo: str
    commit: str
    artifact_format: str
    shards: dict[str, ShardInfo] = field(default_factory=dict)
    tensors: dict[str, TensorRef] = field(default_factory=dict)
    tensor_shard: dict[str, str] = field(default_factory=dict)
    auxiliary_files: dict[str, int] = field(default_factory=dict)
    metadata_files: dict[str, int] = field(default_factory=dict)
    expert_packing: dict[str, ExpertPacking] = field(default_factory=dict)
    total_weight_bytes: int = 0

    # -- lookups -----------------------------------------------------------
    def tensor(self, name: str) -> TensorRef:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ModelFormatUnsupported(
                f"tensor {name!r} is not present in the layout",
                details={"tensor": name},
            ) from exc

    def has_tensor(self, name: str) -> bool:
        return name in self.tensors

    def shard_of(self, tensor_name: str) -> str:
        try:
            return self.tensor_shard[tensor_name]
        except KeyError as exc:
            raise ModelFormatUnsupported(
                f"tensor {tensor_name!r} has no shard mapping",
                details={"tensor": tensor_name},
            ) from exc

    def tensors_in_shard(self, shard: str) -> list[str]:
        return [name for name, path in self.tensor_shard.items() if path == shard]

    def register_expert_packing(self, packing: ExpertPacking) -> None:
        self.expert_packing[packing.tensor] = packing

    def expert_element_range(self, tensor_name: str, expert_id: int) -> tuple[int, int, int]:
        packing = self.expert_packing.get(tensor_name)
        if packing is None:
            raise ModelFormatUnsupported(
                f"no expert packing has been declared for tensor {tensor_name!r}",
                details={"tensor": tensor_name},
            )
        return packing.slice_for(expert_id, self.tensor(tensor_name).shape)

    # -- accounting --------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "commit": self.commit,
            "artifact_format": self.artifact_format,
            "shards": len(self.shards),
            "tensors": len(self.tensors),
            "total_weight_bytes": self.total_weight_bytes,
            "auxiliary_files": dict(self.auxiliary_files),
            "metadata_files": dict(self.metadata_files),
        }


def _classify_files(revision: RevisionInfo) -> tuple[list[str], dict[str, int], dict[str, int]]:
    weight_files: list[str] = []
    auxiliary: dict[str, int] = {}
    metadata: dict[str, int] = {}
    for path, info in revision.files.items():
        size = info.size or 0
        if path.endswith(".safetensors"):
            weight_files.append(path)
        elif path.endswith(".gguf"):
            weight_files.append(path)
        elif path.endswith(".json") or path.endswith(".txt") or path.endswith(".jinja"):
            metadata[path] = size
        elif path.endswith(".bin") or path.endswith(".pt") or path.endswith(".pth"):
            auxiliary[path] = size
        else:
            metadata[path] = size
    weight_files.sort()
    return weight_files, auxiliary, metadata


async def _read_range(source: SourceAdapter, repo: str, commit: str, path: str, start: int, end: int) -> bytes:
    return await source.fetch_range(repo, commit, path, start, end)


async def build_safetensors_layout(
    source: SourceAdapter,
    revision: RevisionInfo,
    *,
    max_concurrency: int = 4,
) -> ModelLayout:
    """Build a layout for a safetensors repository.

    Only small ranges (the 8 byte prefix and the JSON header) are fetched.
    """
    weight_files, auxiliary, metadata = _classify_files(revision)
    if not weight_files:
        raise ModelFormatUnsupported(
            f"repository {revision.repo}@{revision.commit} contains no safetensors weight file",
            details={"repo": revision.repo},
        )
    for path in weight_files:
        _validate_relative_path(path)

    layout = ModelLayout(
        repo=revision.repo,
        commit=revision.commit,
        artifact_format="safetensors",
        auxiliary_files=auxiliary,
        metadata_files=metadata,
    )

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def load_shard(path: str) -> ShardInfo:
        info = revision.files[path]
        size = info.size
        if size is None:
            raise ModelFormatUnsupported(
                f"the source did not report a size for {path}; range access cannot be validated"
            )
        async with sem:
            # One request is enough for most artifacts: the 8 byte length
            # prefix and the JSON header usually share the first chunk.
            probe_len = min(size, HEADER_PROBE_BYTES)
            prefix = await _read_range(
                source, revision.repo, revision.commit, path, 0, probe_len - 1
            )
            header_length = parse_header_length(prefix)
            if header_length + HEADER_LENGTH_BYTES > size:
                raise ModelFormatUnsupported(
                    f"header of {path} does not fit into the file",
                    details={"shard": path, "header_length": header_length, "size": size},
                )
            if len(prefix) >= HEADER_LENGTH_BYTES + header_length:
                header_bytes = prefix[HEADER_LENGTH_BYTES : HEADER_LENGTH_BYTES + header_length]
            else:
                header_bytes = await _read_range(
                    source,
                    revision.repo,
                    revision.commit,
                    path,
                    HEADER_LENGTH_BYTES,
                    HEADER_LENGTH_BYTES + header_length - 1,
                )
        header = parse_safetensors_header(header_bytes, file_size=size, shard=path)
        return ShardInfo(path=path, size=size, header_length=header_length, header=header)

    results = await asyncio.gather(*(load_shard(path) for path in weight_files))
    for shard in results:
        layout.shards[shard.path] = shard
        layout.total_weight_bytes += shard.size
        assert shard.header is not None
        for name, ref in shard.header.tensors.items():
            layout.tensors[name] = ref
            layout.tensor_shard[name] = shard.path

    index_path = next((p for p in INDEX_CANDIDATES if p in revision.files), None)
    if index_path is not None:
        index_info = revision.files[index_path]
        size = index_info.size or 0
        if size > 64 * 1024 * 1024:
            raise ModelFormatUnsupported(
                "shard index is implausibly large; refusing to parse it",
                details={"index": index_path, "size": size},
            )
        raw = await _read_range(source, revision.repo, revision.commit, index_path, 0, size - 1)
        weight_map = parse_shard_index(raw)
        missing = sorted({shard for shard in weight_map.values() if shard not in layout.shards})
        if missing:
            raise ModelFormatUnsupported(
                "the shard index references files that are not in the revision",
                details={"missing": missing},
            )
        unknown = sorted(set(weight_map) - set(layout.tensors))
        if unknown:
            raise ModelFormatUnsupported(
                "the shard index lists tensors that the shards do not contain",
                details={"tensors": unknown[:20], "count": len(unknown)},
            )
        for tensor, shard in weight_map.items():
            if layout.tensor_shard.get(tensor) != shard:
                raise ModelFormatUnsupported(
                    f"the shard index maps {tensor!r} to {shard!r} but the shard contains it elsewhere",
                    details={"tensor": tensor, "index_shard": shard},
                )
        layout.metadata_files[index_path] = size

    return layout


async def build_layout(
    source: SourceAdapter,
    revision: RevisionInfo,
    *,
    artifact_format: str = "safetensors",
    max_concurrency: int = 4,
) -> ModelLayout:
    if artifact_format == "safetensors":
        return await build_safetensors_layout(source, revision, max_concurrency=max_concurrency)
    if artifact_format == "gguf":
        from .gguf import build_gguf_layout

        return await build_gguf_layout(source, revision, max_concurrency=max_concurrency)
    raise ModelFormatUnsupported(f"unsupported artifact format {artifact_format!r}")


def describe_files(revision: RevisionInfo, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """List repository files with their sizes (used by `model add`/`doctor`)."""
    selected = list(names) if names is not None else sorted(revision.files)
    out: list[dict[str, Any]] = []
    for name in selected:
        info = revision.files.get(name)
        if info is None:
            continue
        out.append(
            {
                "path": name,
                "size": info.size,
                "sha256": info.sha256,
                "xet": bool(info.xet_hash),
            }
        )
    return out


def load_json_bytes(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFormatUnsupported(f"payload is not valid JSON: {exc}") from exc


__all__ = [
    "ExpertPacking",
    "ModelLayout",
    "ShardInfo",
    "build_layout",
    "build_safetensors_layout",
    "describe_files",
    "load_json_bytes",
    "SourceNotFoundError",
    "Sequence",
]
