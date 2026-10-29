"""Selective expert materialiser tests (Plan.md C7 selective MoE).

Validates that the materialiser only fetches the router-selected expert rows
from a *packed* layout, leaving unselected experts untouched, and that the
transferred bytes are a strict fraction of the dense tensor size.
"""

from __future__ import annotations

import pytest

from stream_agent.backend.selective_experts import (
    ExpertSpec,
    SelectiveExpertMaterializer,
)
from stream_agent.layout.model_map import ExpertPacking
from stream_agent.provider import ExpertKey


async def test_selective_fetch_transfers_only_selected_experts(provider):
    provider.layout.register_expert_packing(
        ExpertPacking(tensor="model.embed_tokens.weight", dim=0, num_experts=8)
    )
    ref = provider.layout.tensor("model.embed_tokens.weight")
    per_expert_bytes = ref.nbytes // 8

    materializer = SelectiveExpertMaterializer(
        provider=provider, layout=provider.layout
    )
    materializer.register_spec(
        ExpertSpec(tensor="model.embed_tokens.weight", dim=0, num_experts=8)
    )

    result = await materializer.materialize(
        {"model.embed_tokens.weight": (1, 4)}
    )

    assert result.selected == {"model.embed_tokens.weight": (1, 4)}
    assert result.total_bytes == ref.nbytes
    assert result.fetched_bytes == 2 * per_expert_bytes
    # Strict reduction: 2 of 8 experts selected.
    assert result.saved_bytes == 6 * per_expert_bytes
    assert result.fetched_bytes < result.total_bytes

    # Each fetched buffer is exactly one expert slice.
    keys = list(result.per_tensor)
    assert len(keys) == 2

    # Reading an *unselected* expert still works independently through the
    # provider API but was never requested by the materialiser.
    fetched_all = set()
    for key in keys:
        _, expert_id = key.rsplit("::", 1)
        fetched_all.add(int(expert_id))
    assert fetched_all == {1, 4}


async def test_selective_fetch_with_no_selection_fetches_nothing(provider):
    provider.layout.register_expert_packing(
        ExpertPacking(tensor="model.embed_tokens.weight", dim=0, num_experts=4)
    )
    materializer = SelectiveExpertMaterializer(
        provider=provider, layout=provider.layout
    )
    materializer.register_spec(
        ExpertSpec(tensor="model.embed_tokens.weight", dim=0, num_experts=4)
    )

    result = await materializer.materialize({"model.embed_tokens.weight": ()})

    assert result.fetched_bytes == 0
    assert result.selected == {}
    assert result.per_tensor == {}


async def test_register_spec_declares_layout_packing(provider):
    materializer = SelectiveExpertMaterializer(
        provider=provider, layout=provider.layout
    )
    materializer.register_spec(
        ExpertSpec(tensor="model.embed_tokens.weight", dim=0, num_experts=8)
    )
    ref = provider.layout.tensor("model.embed_tokens.weight")
    expert_buffer = await provider.read_expert(
        ExpertKey(tensor="model.embed_tokens.weight", expert_id=7)
    )
    assert len(expert_buffer.data) == ref.nbytes // 8