"""GGUF cache-budget compliance check.

Fetches a GGUF artifact through Stream Agent's bounded cache (whole-file fetch,
not tensor streaming) and runs local inference.  Records cache usage vs. budget
and pass/fail for budgets that fit and that do not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

from stream_agent.config import Config, ServerConfig, SourceConfig, StorageConfig
from stream_agent.layout.gguf import build_gguf_layout
from stream_agent.provider import RemoteWeightProvider
from stream_agent.runtime import StreamAgentRuntime

REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
FILE = "SmolLM2-135M-Instruct-IQ3_M.gguf"
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
FILE_SIZE = 90213472
CONFIG_PATH = Path("C:/Users/smily/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json")


async def run_budget(cache_limit: int, root: Path) -> dict:
    storage = StorageConfig(
        root=root,
        weight_cache_limit_bytes=cache_limit,
        managed_store_limit_bytes=cache_limit,
        metadata_reserve_bytes=16 * 1024 * 1024,
        staging_reserve_bytes=16 * 1024 * 1024,
        block_size_bytes=8 * 1024 * 1024,
    )
    config = Config(storage=storage, server=ServerConfig(), source=SourceConfig(), models={})
    runtime = StreamAgentRuntime(config)
    await runtime.start()
    rec: dict = {"cache_limit_bytes": cache_limit, "file_size_bytes": FILE_SIZE}
    try:
        src = runtime.source
        rev = await src.list_files(REPO, "main")
        layout = await build_gguf_layout(src, rev)
        provider = RemoteWeightProvider(
            source=src,
            layout=layout,
            cache=runtime.cache,
            model_alias="gguf",
            block_size_bytes=storage.block_size_bytes,
            telemetry=runtime.telemetry,
            metadata_root=None,
        )
        shard = sorted(layout.shards)[0]
        size = layout.shards[shard].size
        rec["shard"] = shard
        rec["shard_size"] = size
        started = time.perf_counter()
        try:
            data = await provider.read_file(shard, 0, size - 1)
            rec["fetch_seconds"] = round(time.perf_counter() - started, 3)
            rec["fetched_bytes"] = len(data)
            rec["fetch_ok"] = True
        except Exception as e:
            rec["fetch_seconds"] = round(time.perf_counter() - started, 3)
            rec["fetch_ok"] = False
            rec["fetch_error"] = str(e)
            data = None

        if data is not None:
            tmpdir = Path(tempfile.mkdtemp())
            (tmpdir / "config.json").write_bytes(CONFIG_PATH.read_bytes())
            (tmpdir / "model.gguf").write_bytes(data)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t0 = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(str(tmpdir), gguf_file="model.gguf", device_map="cpu")
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            rec["load_seconds"] = round(time.perf_counter() - t0, 3)
            inputs = tok("Hello", return_tensors="pt")
            t1 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=4)
            rec["gen_seconds"] = round(time.perf_counter() - t1, 3)
            rec["output"] = tok.decode(out[0])
            shutil.rmtree(tmpdir, ignore_errors=True)

        status = runtime.cache.status()
        rec["cache_used_bytes"] = status.used_weight_bytes
        rec["within_budget"] = status.used_weight_bytes <= status.weight_cache_limit_bytes
        rec["downloaded_bytes"] = runtime.telemetry.network.downloaded_bytes
        rec["range_requests"] = runtime.telemetry.network.range_requests
        rec["full_object_requests"] = runtime.telemetry.network.full_object_requests
        rec["cache_hits"] = runtime.telemetry.cache.hits
        rec["cache_misses"] = runtime.telemetry.cache.misses
    finally:
        await runtime.close()
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--budgets", default="67108864,104857600")
    args = ap.parse_args()
    out = {}
    for b in args.budgets.split(","):
        b = int(b)
        root = Path(args.root) / f"budget_{b}"
        root.mkdir(parents=True, exist_ok=True)
        print(f"=== budget {b} ===")
        rec = await run_budget(b, root)
        out[str(b)] = rec
        print(json.dumps(rec, indent=2))
    Path("docs/benchmarks/gguf-cache-budget.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    asyncio.run(main())
