"""Compare Stream Agent generation with an ordinary pinned Transformers load.

This opt-in integration check downloads/caches the model through Transformers
for the reference run and through Stream Agent's bounded range cache. Run it
explicitly; it is intentionally excluded from the fast offline CI suite.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stream_agent.config import Config, ModelConfig, StorageConfig
from stream_agent.runtime import StreamAgentRuntime

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


class PeakRssSampler:
    """Sample process resident memory during one Stream Agent run."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            try:
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def reference_generate(prompt: str, max_tokens: int, device: str) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REVISION, torch_dtype=torch.float32, trust_remote_code=False
    ).to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    token_ids = output[0, inputs["input_ids"].shape[1] :].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"token_ids": token_ids, "text": text}


async def streamed_generate(args: argparse.Namespace) -> dict[str, object]:
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    rss_sampler = PeakRssSampler()
    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="stream-agent-parity-"))
    cache_limit = args.cache_limit
    store_limit = max(args.store_limit, cache_limit + 512 * 1024**2)
    config = Config(
        storage=StorageConfig(
            root=root,
            weight_cache_limit_bytes=cache_limit,
            managed_store_limit_bytes=store_limit,
            metadata_reserve_bytes=128 * 1024**2,
            staging_reserve_bytes=256 * 1024**2,
        ),
        models={
            "smol": ModelConfig(
                alias="smol",
                repo=REPO,
                revision=REVISION,
                format="safetensors",
                backend="transformers",
                weight_cache_quota_bytes=cache_limit,
                license_review="accepted",
            )
        },
    )
    runtime = StreamAgentRuntime(config)
    with rss_sampler:
        started = time.perf_counter()
        await runtime.start()
        runtime_start_seconds = time.perf_counter() - started
        try:
            await runtime.register_model(
                alias="smol", repo=REPO, ref=REVISION, artifact_format="safetensors",
                backend="transformers", quota_bytes=cache_limit,
                license_review="accepted", probe=True,
            )
            started = time.perf_counter()
            session = await runtime.open_session(
                "smol", backend_id="transformers", materialize=args.materialize,
                device=args.device, dtype="float32",
                extra_options={"prefetch_layers": args.prefetch_layers},
            )
            load_seconds = time.perf_counter() - started
            try:
                started = time.perf_counter()
                result = await session.generate(
                    args.prompt, max_tokens=args.max_tokens, temperature=0.0
                )
                generation_seconds = time.perf_counter() - started
                return {
                    "text": result.text,
                    "generated_tokens": result.generated_tokens,
                    "materialize": args.materialize,
                    "prefetch_layers": args.prefetch_layers,
                    "runtime_start_seconds": round(runtime_start_seconds, 3),
                    "load_seconds": round(load_seconds, 3),
                    "generation_seconds": round(generation_seconds, 3),
                    "generation_weight_wait_ms": round(result.weight_wait_ms, 3),
                    "prefetch_wait_ms": round(session.provider.prefetch_wait_ms, 3),
                    "prefetch": session.backend._handle["materializer"].prefetch_scheduler.snapshot()
                    if session.backend._handle["materializer"].prefetch_scheduler
                    else None,
                    "prefetched_bytes": runtime.telemetry.cache.prefetched_bytes,
                    "useful_prefetched_bytes": runtime.telemetry.cache.useful_prefetched_bytes,
                    "materialized_weight_bytes": (
                        session.backend._handle["materializer"].materialized_bytes
                    ),
                    "layer_profile": (
                        session.backend._handle["materializer"].layer_profile
                    ),
                    "revision": session.revision,
                    "artifact_bytes": session.layout.total_weight_bytes,
                    "cache_used_bytes": runtime.cache.status().used_weight_bytes,
                    "cache_limit_bytes": runtime.cache.status().weight_cache_limit_bytes,
                    "peak_rss_bytes": rss_sampler.peak_bytes,
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated(args.device)
                        if args.device.startswith("cuda") else None
                    ),
                    "downloaded_bytes": runtime.telemetry.network.downloaded_bytes,
                    "range_requests": runtime.telemetry.network.range_requests,
                    "full_object_requests": runtime.telemetry.network.full_object_requests,
                    "cache_hits": runtime.telemetry.cache.hits,
                    "cache_misses": runtime.telemetry.cache.misses,
                    "cache_evictions": runtime.telemetry.cache.evictions,
                    "refetched_bytes": runtime.telemetry.cache.refetched_bytes,
                }
            finally:
                await session.close()
        finally:
            await runtime.close()


async def run(args: argparse.Namespace) -> int:
    reference = await asyncio.to_thread(reference_generate, args.prompt, args.max_tokens, args.device)
    streamed = await streamed_generate(args)
    # Text is the stable public result contract. Token counts are measured by
    # each engine; decoding then re-encoding is not a reliable token-ID test.
    result = {
        "repo": REPO,
        "revision": REVISION,
        "device": args.device,
        "materialize": args.materialize,
        "prefetch_layers": args.prefetch_layers,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "reference_text": reference["text"],
        "stream_agent_text": streamed["text"],
        "reference_tokens": len(reference["token_ids"]),
        "stream_agent_tokens": streamed["generated_tokens"],
        "text_match": reference["text"] == streamed["text"],
        "stream_agent": {key: value for key, value in streamed.items() if key not in {"text", "generated_tokens"}},
        "ok": reference["text"] == streamed["text"]
        and streamed["cache_used_bytes"] <= streamed["cache_limit_bytes"]
        and streamed["full_object_requests"] == 0
        and streamed["materialize"] == args.materialize,
        "elapsed_unix_seconds": round(time.time(), 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="", help="managed cache path; defaults to a fresh temp directory")
    parser.add_argument("--cache-limit", type=int, default=128 * 1024**2)
    parser.add_argument("--store-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--materialize", choices=("lazy", "eager"), default="lazy")
    parser.add_argument(
        "--prefetch-layers", action="store_true",
        help="prefetch the next decoder layer during lazy inference",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--out", default="")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
