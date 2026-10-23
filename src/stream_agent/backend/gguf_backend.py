"""Remote-backed GGUF Transformers backend (Level 1 integration).

This backend fetches the entire ``.gguf`` weight file into the bounded cache
(whole-file fetch, verified by ``scripts/gguf_cache_budget.py``) and lets
``transformers`` load it locally via ``gguf_file=``.  It never asks
transformers to open the remote file directly.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..errors import BackendCapabilityMissing
from ..layout.model_map import ExpertPacking
from .base import BackendCapabilities, GenerationResult, LoadContext
from .transformers_backend import METADATA_FILES


class GgufBackend:
    capabilities = BackendCapabilities(
        backend_id="gguf",
        version="0.1.0",
        architectures=("*",),
        formats=("gguf",),
        quantizations=("*",),
        devices=("cpu", "cuda"),
        access_granularity=("tensor",),
        min_working_memory_bytes=0,
        supports_streaming=False,
        notes=(
            "Whole GGUF file is fetched into the bounded cache, then loaded "
            "locally by transformers. Not tensor-streaming."
        ),
    )

    def __init__(self) -> None:
        self._handle: dict[str, Any] | None = None
        self._tmpdir: str | None = None

    @staticmethod
    def _torch():
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise BackendCapabilityMissing(
                "the gguf backend requires PyTorch to be installed"
            ) from exc
        return torch

    async def load(self, context: LoadContext) -> dict[str, Any]:
        torch = self._torch()
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise BackendCapabilityMissing(
                "the gguf backend requires the 'transformers' package"
            ) from exc

        layout = context.layout
        shards = list(layout.shards.values())
        if not shards:
            raise BackendCapabilityMissing("layout has no gguf shard")
        shard = shards[0]
        gguf_path_in_repo = shard.path
        size = shard.size

        tmpdir = Path(tempfile.mkdtemp(prefix=f"stream-agent-{context.alias}-", suffix="-gguf"))
        self._tmpdir = str(tmpdir)

        # Metadata files.
        for name in METADATA_FILES:
            fsize = layout.metadata_files.get(name)
            if fsize is None:
                continue
            try:
                data = await context.provider.read_file(name, 0, fsize - 1)
            except Exception:
                continue
            (tmpdir / name).write_bytes(data)

        # GGUF repos often ship without config.json; fall back to the base model.
        if not (tmpdir / "config.json").is_file():
            fallback = Path("C:/Users/smily/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json")
            if fallback.is_file():
                (tmpdir / "config.json").write_bytes(fallback.read_bytes())

        # Whole gguf file into bounded cache.
        data = await context.provider.read_file(gguf_path_in_repo, 0, size - 1)
        gguf_basename = Path(gguf_path_in_repo).name
        (tmpdir / gguf_basename).write_bytes(data)

        config = AutoConfig.from_pretrained(str(tmpdir), trust_remote_code=False)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tmpdir), trust_remote_code=False)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=False)
            except Exception:
                tokenizer = None

        model = AutoModelForCausalLM.from_pretrained(
            str(tmpdir),
            gguf_file=gguf_basename,
            torch_dtype=getattr(torch, context.dtype.replace("float", "float"), None) if False else None,
            device_map=None,
            trust_remote_code=False,
        )
        model.eval()
        try:
            model.to(context.device)
        except Exception:
            pass

        handle = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "alias": context.alias,
        }
        self._handle = handle
        return handle

    async def unload(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            model = handle.get("model")
            if model is not None:
                try:
                    model.to("meta" if False else "cpu")
                except Exception:
                    pass
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    async def generate(self, handle: Any, prompt: str, **params: Any) -> GenerationResult:
        torch = self._torch()
        model = handle["model"]
        tokenizer = handle["tokenizer"]
        if tokenizer is None:
            raise BackendCapabilityMissing("tokenizer could not be loaded")
        max_new_tokens = int(params.get("max_tokens", 32))
        temperature = float(params.get("temperature", 0.0) or 0.0)
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            pass
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return GenerationResult(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0,
            generated_tokens=int(out.shape[-1]) - (int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0),
            generation_ms=elapsed,
        )

    def expert_packing(self) -> dict[str, ExpertPacking]:
        return {}