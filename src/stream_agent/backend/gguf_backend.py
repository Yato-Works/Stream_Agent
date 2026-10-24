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

        # Whole gguf file into bounded cache.
        data = await context.provider.read_file(gguf_path_in_repo, 0, size - 1)
        gguf_basename = Path(gguf_path_in_repo).name
        (tmpdir / gguf_basename).write_bytes(data)

        _DTYPE_TO_TORCH = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "float64": "float64",
        }
        torch_dtype_name = _DTYPE_TO_TORCH.get(context.dtype)
        torch_dtype = getattr(torch, torch_dtype_name, None) if torch_dtype_name else None

        config = AutoConfig.from_pretrained(
            str(tmpdir), gguf_file=gguf_basename, trust_remote_code=False
        )
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tmpdir), gguf_file=gguf_basename, trust_remote_code=False
            )
        except Exception:
            tokenizer = None

        model = AutoModelForCausalLM.from_pretrained(
            str(tmpdir),
            gguf_file=gguf_basename,
            torch_dtype=torch_dtype,
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