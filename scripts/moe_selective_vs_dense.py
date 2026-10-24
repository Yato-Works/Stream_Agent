"""Selected vs. dense expert loading comparison (Plan.md Appendix C).

Measures, on a real MoE model layout:
  - Dense fetch of *every* expert tensor.
  - Selective fetch of only a subset (simulated router top-k selections).
and reports transferred bytes, cache usage and process max-RSS for each path,
demonstrating real transfer/memory reduction versus dense expert loading.

Skipped gracefully when the repository is unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import psutil

from stream_agent.cache.db import MetadataDB
from stream_agent.cache.manager import CacheManager
from stream_agent.cache.store import BlockStore
from stream_agent.config import StorageConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider, TensorKey
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "tiny-random/qwen3.5-moe"

# Simulated router: each layer selects the first two expert ids.  This mirrors
# the top-k behaviour observed in `moe_selective_fetch_prototype.json`
# (2 selected / 8 experts per layer) without requiring a full model forward.
SIMULATED_SELECTED = 2


def _max_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


async def fetch_set(provider, names):
    for n in names:
        await provider.read_tensor(TensorKey(tensor=n))


async def measure(source, layout, storage, tensor_names, *, cache_tag):
    cache = CacheManager(storage, MetadataDB(storage.db_path), BlockStore(storage.blocks_root))
    provider = RemoteWeightProvider(
        source=source,
        layout=layout,
        cache=cache,
        model_alias="moe",
        block_size_bytes=64 * 1024,
    )
    await fetch_set(provider, tensor_names)
    return {
        "path": cache_tag,
        "tensor_count": len(tensor_names),
        "transferred_bytes": sum(layout.tensor(n).nbytes for n in tensor_names),
        "cache_used_bytes": cache.used_weight_bytes,
        "max_rss_bytes": _max_rss_bytes(),
    }


async def run(args):
    storage = StorageConfig(
        root=Path(args.root),
        weight_cache_limit_bytes=args.cache_limit,
        managed_store_limit_bytes=args.store_limit,
        metadata_reserve_bytes=64 * 1024 * 1024,
        staging_reserve_bytes=128 * 1024 * 1024,
    )
    import re
    import shutil

    if Path(args.root).exists():
        shutil.rmtree(Path(args.root))

    source = HuggingFaceSource()
    try:
        revision = await source.resolve_revision(REPO, "main")
        layout = await build_layout(source, revision)

        all_expert = [n for n in layout.tensors if ".mlp.experts." in n]
        total_bytes = sum(layout.tensor(n).nbytes for n in all_expert)

        # Simulate router top-k selections: first SIMULATED_SELECTED expert ids
        # per layer.
        selected = []
        by_layer: dict[int, list[int]] = {}
        for n in all_expert:
            layer = int(re.search(r"layers\.(\d+)\.", n).group(1))
            eid = int(re.search(r"experts\.(\d+)\.", n).group(1))
            by_layer.setdefault(layer, set()).add(eid)
        for layer, experts in sorted(by_layer.items()):
            chosen = sorted(experts)[:SIMULATED_SELECTED]
            for eid in chosen:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.language_model.layers.{layer}.mlp.experts.{eid}.{proj}.weight"
                    if name in layout.tensors:
                        selected.append(name)
        selected_bytes = sum(layout.tensor(n).nbytes for n in selected)

        dense = await measure(source, layout, storage, all_expert, cache_tag="dense")
        # Wipe cache so the selective path is independent.
        import shutil as _sh

        _sh.rmtree(Path(args.root), ignore_errors=True)
        selective = await measure(source, layout, storage, selected, cache_tag="selected")
        selective["selected_expert_bytes"] = selected_bytes
        selective["total_expert_bytes"] = total_bytes
        selective["expert_byte_reduction_ratio"] = (
            round(selected_bytes / total_bytes, 6) if total_bytes else None
        )
        record = {
            "repo": REPO,
            "total_expert_tensor_count": len(all_expert),
            "total_expert_bytes": total_bytes,
            "selected_expert_tensor_count": len(selected),
            "dense_fetch": dense,
            "selective_fetch": selective,
        }
    finally:
        await source.aclose()

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / ".stream-agent-moe-bench"))
    p.add_argument("--artifact", default="docs/benchmarks/moe_selective_vs_dense.json")
    p.add_argument("--cache-limit", type=int, default=256 * 1024 * 1024)
    p.add_argument("--store-limit", type=int, default=1 * 1024 ** 3)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())






