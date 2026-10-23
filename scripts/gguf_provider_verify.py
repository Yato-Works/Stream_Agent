from __future__ import annotations
import asyncio
from pathlib import Path
from stream_agent.config import Config, ServerConfig, SourceConfig, StorageConfig
from stream_agent.sources.hf_hub import HuggingFaceSource
from stream_agent.layout.gguf import build_gguf_layout
from stream_agent.provider import RemoteWeightProvider
from stream_agent.runtime import StreamAgentRuntime

REPO="bartowski/SmolLM2-135M-Instruct-GGUF"
LOCAL=Path("C:/Users/smily/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-IQ3_M.gguf")

async def main():
    storage=StorageConfig(root=Path("C:/tmp/ggufprov"), weight_cache_limit_bytes=256*1024*1024, managed_store_limit_bytes=256*1024*1024, metadata_reserve_bytes=16*1024*1024, staging_reserve_bytes=16*1024*1024, block_size_bytes=8*1024*1024)
    config=Config(storage=storage, server=ServerConfig(), source=SourceConfig(), models={})
    runtime=StreamAgentRuntime(config)
    await runtime.start()
    try:
        src=runtime.source
        rev=await src.list_files(REPO,"main")
        layout=await build_gguf_layout(src,rev)
        provider=RemoteWeightProvider(source=src, layout=layout, cache=runtime.cache, model_alias="g", block_size_bytes=storage.block_size_bytes, telemetry=runtime.telemetry)
        name=list(layout.tensors)[0]
        buf=await provider.read_tensor(__import__("stream_agent.provider",fromlist=["TensorKey"]).TensorKey(name))
        data=LOCAL.read_bytes()
        ref=layout.tensors[name]
        print("name",name,"buf",len(buf.data),"expected",ref.data_end-ref.data_start,"match",buf.data==data[ref.data_start:ref.data_end])
    finally:
        await runtime.close()

asyncio.run(main())