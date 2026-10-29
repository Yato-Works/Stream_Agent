"""Selective MoE expert materialisation (Plan.md C7).

Lazy-load path of the Transformers backend only fetches the tensors a layer
needs before that layer runs.  Selective expert materialisation goes one step
further for Mixture-of-Experts layers: instead of fetching the *whole* stacked
expert parameter ``mlp.experts.gate_up_proj`` (which holds every expert), it
waits for the router to pick the top-K experts and then fetches *only* those
expert rows through the provider's ``read_expert`` interface.

The materialiser is backend-agnostic.  Each supported MoE architecture
registers an :class:`ExpertSpec` describing how its packed checkpoint tensors
map onto the module parameters, and the backend consults the spec from its
lazy hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..layout.model_map import ExpertPacking
from ..provider import ExpertKey, TensorBuffer


@dataclass(frozen=True)
class ExpertSpec:
    """How a MoE checkpoint packs experts for one projection.

    Attributes:
        tensor: The artifact tensor name, e.g. ``model.layers.0.mlp.experts.gate_up_proj``.
        dim: Axis along which experts are stacked (0 for ``gate_up_proj`` /
            ``down_proj`` in Qwen-style packed checkpoints).
        num_experts: Total number of experts declared on ``tensor``.
        proj: Projection name this spec describes, used only for diagnostics.
    """

    tensor: str
    dim: int
    num_experts: int
    proj: str = "unknown"


@dataclass
class MaterialisedExperts:
    """Result of a selective fetch: raw bytes per tensor and the expert ids.

    The bytes are still in the checkpoint dtype/shape; the backend reinterprets
    them exactly as :meth:`TransformersBackend._materialize_names` does for a
    full tensor.
    """

    per_tensor: dict[str, TensorBuffer]
    selected: dict[str, tuple[int, ...]]
    fetched_bytes: int = 0
    total_bytes: int = 0

    @property
    def saved_bytes(self) -> int:
        return self.total_bytes - self.fetched_bytes


@dataclass
class SelectiveExpertMaterializer:
    """Fetch only router-selected experts from the weight provider.

    The materialiser never guesses an expert layout.  Every tensor it reads
    must already have an :class:`~stream_agent.layout.model_map.ExpertPacking`
    declared on the layout; that packing is normally registered by the backend
    through :meth:`InferenceBackend.expert_packing` so the cache and provider
    stay the single source of truth.
    """

    provider: Any  # WeightProvider
    layout: Any  # ModelLayout
    specs: dict[str, ExpertSpec] = field(default_factory=dict)

    def register_spec(self, spec: ExpertSpec) -> None:
        """Declare how one architecture packs its experts."""
        self.specs[spec.tensor] = spec
        # Keep the layout packing in sync so provider.read_expert can resolve
        # the slice for this expert.
        self.layout.register_expert_packing(
            ExpertPacking(tensor=spec.tensor, dim=spec.dim, num_experts=spec.num_experts)
        )

    async def materialize(
        self, selected: Mapping[str, Iterable[int]]
    ) -> MaterialisedExperts:
        """Fetch the requested experts for every registered spec.

        Args:
            selected: Mapping of tensor name -> iterable of expert ids that the
                router selected for the current batch.
        """
        per_tensor: dict[str, TensorBuffer] = {}
        selected_map: dict[str, tuple[int, ...]] = {}
        fetched_bytes = 0
        total_bytes = 0
        for tensor, spec in self.specs.items():
            ref = self.layout.tensor(spec.tensor)
            total_bytes += ref.nbytes
            ids = sorted({int(e) for e in selected.get(tensor, ())})
            if not ids:
                continue
            for expert_id in ids:
                buffer = await self.provider.read_expert(
                    ExpertKey(tensor=spec.tensor, expert_id=expert_id)
                )
                per_tensor[f"{spec.tensor}::{expert_id}"] = buffer
                fetched_bytes += buffer.nbytes
            selected_map[spec.tensor] = tuple(ids)
        return MaterialisedExperts(
            per_tensor=per_tensor,
            selected=selected_map,
            fetched_bytes=fetched_bytes,
            total_bytes=total_bytes,
        )


__all__ = ["ExpertSpec", "MaterialisedExperts", "SelectiveExpertMaterializer"]