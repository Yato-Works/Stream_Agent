from __future__ import annotations
import argparse, asyncio, json, shutil, sys, time
from pathlib import Path
from stream_agent.config import Config, ModelConfig, ServerConfig, SourceConfig, StorageConfig
from stream_agent.runtime import StreamAgentRuntime

REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"


async def run(args):
    root = Path(args.root)
    cold_cache = not root.exists()
    if args.cold and root.exists():
        shutil.rmtree(root)
        cold_cache = True

    storage = StorageConfig(
        root=root,
        weight_cache_limit_bytes=args.cache_limit,
        managed_store_limit_bytes=args.store_limit,
        metadata_reserve_bytes=64 * 1024 * 1024,
        staging_reserve_bytes=128 * 1024 * 1024,
    )
    config = Config(
        storage=storage,
        server=ServerConfig(),
        source=SourceConfig(),
        models={"g": ModelConfig(alias="g", repo=REPO, revision="main", format="gguf", backend="gguf")},
    )
    runtime = StreamAgentRuntime(config)
    await runtime.start()
    record = {"cold_cache": cold_cache, "repo": REPO, "ref": "main"}
    try:
        await runtime.register_model(
            alias="g", repo=REPO, ref="main", artifact_format="gguf",
            backend="gguf", quota_bytes=args.cache_limit, license_review="accepted", probe=True,
        )
        record["registered"] = True

        started = time.perf_counter()
        session = await runtime.open_session("g", materialize="eager", device=args.device, dtype=args.dtype)
        record["load_seconds"] = round(time.perf_counter() - started, 3)
        record["tensors"] = len(session.layout.tensors)
        shard = next(iter(session.layout.shards.values()))
        record["shard_path"] = shard.path
        record["shard_size_bytes"] = shard.size

        started = time.perf_counter()
        result = await session.generate(args.prompt, max_tokens=args.max_tokens)
        record["generation_seconds"] = round(time.perf_counter() - started, 3)
        record["text"] = result.text
        record["generated_tokens"] = result.generated_tokens
        record["prompt_tokens"] = result.prompt_tokens

        status = runtime.cache.status()
        network = runtime.telemetry.network
        record["cache_used_bytes"] = status.used_weight_bytes
        record["cache_reserved_bytes"] = status.reserved_weight_bytes
        record["cache_limit_bytes"] = args.cache_limit
        record["cache_budget_ok"] = status.used_weight_bytes <= args.cache_limit
        record["downloaded_bytes"] = network.downloaded_bytes
        record["weight_downloaded_bytes"] = network.weight_downloaded_bytes
        record["metadata_downloaded_bytes"] = network.metadata_downloaded_bytes
        record["range_requests"] = network.range_requests
        record["full_object_requests"] = network.full_object_requests
        record["range_request_samples"] = network.range_request_samples
        record["range_validation_ok"] = (
            network.full_object_requests == 0
            and all(sample.get("status") == 206 for sample in network.range_request_samples)
            and all(sample.get("content_range") is not None for sample in network.range_request_samples)
        )
        record["cache_hits"] = runtime.telemetry.cache.hits
        record["cache_misses"] = runtime.telemetry.cache.misses
        record["integrity_failures"] = runtime.telemetry.cache.integrity_failures

        await session.close()
        record["ok"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_details"] = getattr(exc, "details", {})
        record["ok"] = False
    finally:
        await runtime.close()

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record.get("ok", False) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / ".stream-agent-gguf"))
    p.add_argument("--artifact", default="docs/benchmarks/gguf_backend_e2e.json")
    p.add_argument("--cold", action="store_true", default=True)
    p.add_argument("--cache-limit", type=int, default=512 * 1024 * 1024)
    p.add_argument("--store-limit", type=int, default=2 * 1024 ** 3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
