from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

from stream_agent.cache.db import MetadataDB
from stream_agent.cache.manager import CacheManager
from stream_agent.cache.store import BlockStore
from stream_agent.config import StorageConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider, TensorKey
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "tiny-random/qwen3.5-moe"


async def run(args):
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    storage = StorageConfig(
        root=root,
        weight_cache_limit_bytes=args.cache_limit,
        managed_store_limit_bytes=args.store_limit,
        metadata_reserve_bytes=64 * 1024 * 1024,
        staging_reserve_bytes=128 * 1024 * 1024,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model.eval()

    selected_ids = {}

    def make_hook(idx):
        def hook(module, args, output):
            indices = output[2].detach().cpu().tolist()
            for row in indices:
                for expert_id in row:
                    selected_ids[idx].add(int(expert_id))
        return hook

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        selected_ids[layer_idx] = set()
        hooks.append(layer.mlp.gate.register_forward_hook(make_hook(layer_idx)))

    inputs = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    for handle in hooks:
        handle.remove()

    source = HuggingFaceSource()
    try:
        revision = await source.resolve_revision(REPO, "main")
        layout = await build_layout(source, revision)

        cache = CacheManager(storage, MetadataDB(storage.db_path), BlockStore(storage.blocks_root))
        provider = RemoteWeightProvider(
            source=source,
            layout=layout,
            cache=cache,
            model_alias="moe",
            block_size_bytes=64 * 1024,
        )

        total_expert_tensors = 0
        total_expert_bytes = 0
        selected_tensor_count = 0
        selected_bytes = 0
        fetched = []
        for layer_idx, expert_ids in selected_ids.items():
            for expert_id in sorted(expert_ids):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj}.weight"
                    if name not in layout.tensors:
                        continue
                    await provider.read_tensor(TensorKey(tensor=name))
                    selected_tensor_count += 1
                    selected_bytes += layout.tensor(name).nbytes
                    fetched.append(name)

        for name in layout.tensors:
            if ".mlp.experts." in name:
                total_expert_tensors += 1
                total_expert_bytes += layout.tensor(name).nbytes

        record = {
            "repo": REPO,
            "selected_experts": {str(k): sorted(v) for k, v in selected_ids.items()},
            "selected_expert_tensor_count": selected_tensor_count,
            "selected_expert_bytes": selected_bytes,
            "total_expert_tensor_count": total_expert_tensors,
            "total_expert_bytes": total_expert_bytes,
            "expert_byte_reduction_ratio": round(selected_bytes / total_expert_bytes, 6) if total_expert_bytes else None,
            "cache_used_bytes": cache.used_weight_bytes,
            "fetched_sample": fetched[:10],
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
    p.add_argument("--root", default=str(Path.home() / ".stream-agent-moe"))
    p.add_argument("--artifact", default="docs/benchmarks/moe_selective_fetch_prototype.json")
    p.add_argument("--cache-limit", type=int, default=256 * 1024 * 1024)
    p.add_argument("--store-limit", type=int, default=1 * 1024 ** 3)
    p.add_argument("--max-tokens", type=int, default=4)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
PY