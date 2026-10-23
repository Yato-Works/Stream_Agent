from __future__ import annotations
import asyncio
from pathlib import Path
from stream_agent.sources.hf_hub import HuggingFaceSource
from stream_agent.layout.gguf import build_gguf_layout

REPO="bartowski/SmolLM2-135M-Instruct-GGUF"
LOCAL=Path("C:/Users/smily/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-IQ3_M.gguf")

async def main():
    s=HuggingFaceSource()
    rev=await s.list_files(REPO,"main")
    layout=await build_gguf_layout(s,rev)
    data=LOCAL.read_bytes()
    ok=True; bad=0
    for name,ref in layout.tensors.items():
        remote=await s.fetch_range(REPO,rev.commit,ref.shard,ref.data_start,ref.data_end-1)
        if remote!=data[ref.data_start:ref.data_end]:
            ok=False; bad+=1
            if bad<5: print("MISMATCH",name)
    print("total",len(layout.tensors),"bad",bad,"all_ok",ok)

asyncio.run(main())