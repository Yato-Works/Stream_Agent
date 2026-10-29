"""Deterministic fixture backend.

The fixture backend is used by the test-suite and by ``stream-agent bench``
when a real model is not available.  It exercises the complete remote-backed
path -- it reads tensors through the ``WeightProvider`` (so blocks are really
fetched, verified and cached) and produces deterministic output.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from ..layout.model_map import ExpertPacking
from ..provider import TensorKey
from .base import BackendCapabilities, GenerationResult, LoadContext


class FixtureBackend:
    """A backend whose weights are real bytes from the artifact."""

    capabilities = BackendCapabilities(
        backend_id="fixture",
        version="0.1.0",
        architectures=("*",),
        formats=("safetensors", "gguf"),
        quantizations=("unquantized",),
        devices=("cpu",),
        access_granularity=("tensor", "tensor_slice"),
        min_working_memory_bytes=0,
        # The fixture computes the whole completion in one call; it does not
        # emit tokens incrementally, so claiming streaming would be false.
        supports_streaming=False,
        notes="Deterministic backend used for tests and plumbing benchmarks.",
    )

    def __init__(self) -> None:
        self._handle: dict[str, Any] | None = None

    async def load(self, context: LoadContext) -> dict[str, Any]:
        names = sorted(context.layout.tensors)
        digest = hashlib.sha256()
        loaded: list[str] = []
        for name in names:
            buffer = await context.provider.read_tensor(TensorKey(tensor=name))
            digest.update(name.encode("utf-8"))
            digest.update(buffer.data)
            loaded.append(name)
        self._handle = {
            "digest": digest.hexdigest(),
            "tensors": loaded,
            "alias": context.alias,
            "revision": context.revision,
        }
        return self._handle

    async def unload(self) -> None:
        self._handle = None

    async def generate(self, handle: Any, prompt: str, **params: Any) -> GenerationResult:
        max_tokens = int(params.get("max_tokens", 16))
        started = time.perf_counter()
        digest = hashlib.sha256(
            f"{handle['digest']}|{prompt}".encode("utf-8")
        ).hexdigest()
        tokens = [
            f"{digest[i * 4 : i * 4 + 4]}"
            for i in range(max_tokens)
        ]
        text = " ".join(tokens)
        elapsed = (time.perf_counter() - started) * 1000.0
        return GenerationResult(
            text=text,
            prompt_tokens=max(1, len(prompt.split())),
            generated_tokens=max_tokens,
            generation_ms=elapsed,
            finish_reason="stop",
        )

    def expert_packing(self) -> dict[str, ExpertPacking]:
        return {}


__all__ = ["FixtureBackend"]
