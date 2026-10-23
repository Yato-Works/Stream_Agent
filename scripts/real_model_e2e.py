"""End-to-end experiment: fetch a real public checkpoint and generate with it.

The script records every step of the path that the design asks for:

1. pinned revision catalog resolution
2. file and tensor map construction
3. bounded range retrieval
4. bounded cache write
5. weight loading through the real backend
6. token generation with the real model
7. cache usage and cumulative transfer accounting

It never downloads a complete weight file: the artifact is only read through
validated ``Range`` requests.  The transferred byte counter is compared with
the artifact size so that an implicit full download would be visible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from stream_agent.config import (
    Config,
    ModelConfig,
    ServerConfig,
    SourceConfig,
    StorageConfig,
)
from stream_agent.runtime import StreamAgentRuntime

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
LICENSE = "apache-2.0"


async def run(args: argparse.Namespace) -> int:
    storage = StorageConfig(
        root=Path(args.root),
        weight_cache_limit_bytes=args.cache_limit,
        managed_store_limit_bytes=args.store_limit,
        metadata_reserve_bytes=64 * 1024 * 1024,
        staging_reserve_bytes=128 * 1024 * 1024,
    )
    config = Config(
        storage=storage,
        server=ServerConfig(),
        source=SourceConfig(),
        models={
            "smol": ModelConfig(
                alias="smol",
                repo=REPO,
                revision=REVISION,
                format="safetensors",
                backend="transformers",
            )
        },
    )
    runtime = StreamAgentRuntime(config)
    await runtime.start()
    record: dict[str, object] = {
        "repo": REPO,
        "revision": REVISION,
        "license": LICENSE,
        "device": args.device,
        "dtype": args.dtype,
        "materialize": args.materialize,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "weight_cache_limit_bytes": args.cache_limit,
    }
    try:
        await runtime.register_model(
            alias="smol",
            repo=REPO,
            ref=REVISION,
            artifact_format="safetensors",
            backend="transformers",
            quota_bytes=args.cache_limit,
            license_review="accepted",
            probe=True,
        )
        record["registered"] = True

        started = time.perf_counter()
        session = await runtime.open_session(
            "smol",
            materialize=args.materialize,
            device=args.device,
            dtype=args.dtype,
        )
        record["load_seconds"] = round(time.perf_counter() - started, 3)
        record["tensors"] = len(session.layout.tensors)
        record["total_weight_bytes"] = session.layout.total_weight_bytes
        record["shards"] = {p: s.size for p, s in session.layout.shards.items()}

        started = time.perf_counter()
        result = await session.generate(args.prompt, max_tokens=args.max_tokens)
        record["generation_seconds"] = round(time.perf_counter() - started, 3)
        record["text"] = result.text
        record["generated_tokens"] = result.generated_tokens
        record["prompt_tokens"] = result.prompt_tokens
        record["weight_wait_ms"] = round(result.weight_wait_ms, 3)
        record["load_weight_wait_ms"] = round(result.load_weight_wait_ms, 3)
        record["finish_reason"] = result.finish_reason

        status = runtime.cache.status()
        record["cache_used_bytes"] = status.used_weight_bytes
        record["cache_limit_bytes"] = status.weight_cache_limit_bytes
        record["managed_store_bytes"] = status.managed_store_bytes
        record["downloaded_bytes"] = runtime.telemetry.network.downloaded_bytes
        record["range_requests"] = runtime.telemetry.network.range_requests
        record["full_object_requests"] = runtime.telemetry.network.full_object_requests
        record["cache_hits"] = runtime.telemetry.cache.hits
        record["cache_misses"] = runtime.telemetry.cache.misses
        artifact_bytes = record.get("total_weight_bytes") or 0
        record["artifact_bytes"] = artifact_bytes
        record["downloaded_fraction_of_artifact"] = (
            round(record["downloaded_bytes"] / artifact_bytes, 4) if artifact_bytes else None
        )
        await session.close()
    except Exception as exc:  # noqa: BLE001 - the failure itself is the result
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_details"] = getattr(exc, "details", {})
        record["ok"] = False
    finally:
        await runtime.close()

    print(json.dumps(record, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0 if record.get("ok", True) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.home() / ".stream-agent"))
    parser.add_argument("--cache-limit", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--store-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--materialize", default="eager", choices=["eager", "lazy"])
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
