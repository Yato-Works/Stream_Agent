"""Build the weight map of a real repository without downloading any weights."""

import asyncio
import sys
import time

from stream_agent.config import SourceConfig
from stream_agent.sources.hf_hub import HuggingFaceSource
from stream_agent.layout.model_map import build_layout

REPOS = [
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
    ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"),
]


async def main() -> int:
    source = HuggingFaceSource(SourceConfig())
    try:
        for repo, commit in REPOS:
            started = time.perf_counter()
            try:
                revision = await source.list_files(repo, commit)
                layout = await build_layout(source, revision)
                elapsed = (time.perf_counter() - started) * 1000
                print(
                    f"{repo}\n"
                    f"  commit:        {commit}\n"
                    f"  license:       {revision.license_id or '(unknown)'}  gated={revision.gated}\n"
                    f"  shards:        {len(layout.shards)}\n"
                    f"  tensors:       {len(layout.tensors)}\n"
                    f"  weight bytes:  {layout.total_weight_bytes:,}\n"
                    f"  layout built in {elapsed:.0f} ms (header ranges only)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{repo}\n  FAILED: {type(exc).__name__}: {exc}")
            print()
    finally:
        await source.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
