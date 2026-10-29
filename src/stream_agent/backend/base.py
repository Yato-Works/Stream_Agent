"""Inference backend adapter boundary (Plan.md C7).

A backend declares what it can do (architecture, format, quantization,
device, access granularity) and owns tensor assembly, kernel invocation and
the KV/recurrent state.  It asks the runtime for weights through the
``WeightProvider`` interface and never talks to the cache directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..layout.model_map import ExpertPacking, ModelLayout
from ..provider import RemoteWeightProvider
from ..telemetry import RunTelemetry

#: Granularities a backend may declare.
GRANULARITIES = ("tensor", "tensor_slice", "layer", "expert")


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    version: str
    architectures: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("safetensors",)
    quantizations: tuple[str, ...] = ("unquantized",)
    devices: tuple[str, ...] = ("cpu",)
    access_granularity: tuple[str, ...] = ("tensor",)
    min_working_memory_bytes: int = 0
    requires_custom_code: bool = False
    supports_experts: bool = False
    supports_streaming: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "version": self.version,
            "architectures": list(self.architectures),
            "formats": list(self.formats),
            "quantizations": list(self.quantizations),
            "devices": list(self.devices),
            "access_granularity": list(self.access_granularity),
            "min_working_memory_bytes": self.min_working_memory_bytes,
            "requires_custom_code": self.requires_custom_code,
            "supports_experts": self.supports_experts,
            "supports_streaming": self.supports_streaming,
            "notes": self.notes,
        }


@dataclass
class LoadContext:
    """Everything a backend needs to open a model."""

    alias: str
    repo: str
    revision: str
    layout: ModelLayout
    provider: RemoteWeightProvider
    telemetry: RunTelemetry | None = None
    device: str = "cpu"
    dtype: str = "float32"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    #: Time to the first generated token, or ``None`` when the backend produces
    #: the whole completion in one blocking call and therefore cannot observe
    #: it.  Reporting the total generation time here would fabricate a metric.
    ttft_ms: float | None = None
    generation_ms: float = 0.0
    weight_wait_ms: float = 0.0
    finish_reason: str = "stop"
    load_weight_wait_ms: float = 0.0


@runtime_checkable
class InferenceBackend(Protocol):
    capabilities: BackendCapabilities

    async def load(self, context: LoadContext) -> Any:
        """Prepare the model; returns an opaque handle used by ``generate``."""

    async def unload(self) -> None:
        """Release accelerator and host memory."""

    async def generate(self, handle: Any, prompt: str, **params: Any) -> GenerationResult:
        """Run a bounded generation."""

    def expert_packing(self) -> dict[str, ExpertPacking]:
        """Expert packings the backend declares (never guessed by the cache)."""
        return {}


__all__ = [
    "BackendCapabilities",
    "GenerationResult",
    "GRANULARITIES",
    "InferenceBackend",
    "LoadContext",
]
