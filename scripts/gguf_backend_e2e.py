from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path
from stream_agent.config import Config, ModelConfig, ServerConfig, SourceConfig, StorageConfig
from stream_agent.runtime import StreamAgentRuntime

REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
REVISION = None  # resolve latest? runtime may require ref

async def run(args):
    storage = StorageConfig(root=Path(args.root), weight_cache_limit_bytes=args.cache_limit, managed_store_limit_bytes=args.store_limit, metadata_reserve_bytes=64*1024*1024, staging_reserve_bytes=128*1024*1024)
    config = Config(storage=storage, server=ServerConfig(), source=SourceConfig(), models={"g": ModelConfig(alias="g", repo=REPO, revision="main", format="gguf", backend="gguf")})
    runtime = StreamAgentRuntime(config)
    await runtime.start()
    record={}
    try:
        # list revision via source? Use runtime register_model with ref main
        await runtime.register_model(alias="g", repo=REPO, ref="main", artifact_format="gguf", backend="gguf", quota_bytes=args.cache_limit, license_review="accepted", probe=True)
        record["registered"]=True
        started=time.perf_counter()
        session = await runtime.open_session("g", materialize="eager", device=args.device, dtype=args.dtype)
        record["load_seconds"]=round(time.perf_counter()-started,3)
        record["tensors"]=len(session.layout.tensors)
        started=time.perf_counter()
        result = await session.generate(args.prompt, max_tokens=args.max_tokens)
        record["generation_seconds"]=round(time.perf_counter()-started,3)
        record["text"]=result.text
        record["generated_tokens"]=result.generated_tokens
        status=runtime.cache.status()
        record["cache_used_bytes"]=status.used_weight_bytes
        record["downloaded_bytes"]=runtime.telemetry.network.downloaded_bytes
        record["range_requests"]=runtime.telemetry.network.range_requests
        record["full_object_requests"]=runtime.telemetry.network.full_object_requests
        await session.close()
    except Exception as exc:
        record["error"]=f"{type(exc).__name__}: {exc}"
        record["error_details"]=getattr(exc,"details",{})
        record["ok"]=False
    finally:
        await runtime.close()
    print(json.dumps(record, indent=2))
    return 0 if record.get("ok", True) else 1

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home()/".stream-agent-gguf"))
    p.add_argument("--cache-limit", type=int, default=512*1024*1024)
    p.add_argument("--store-limit", type=int, default=2*1024**3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-tokens", type=int, default=8)
    args=p.parse_args()
    return asyncio.run(run(args))

if __name__=="__main__":
    sys.exit(main())