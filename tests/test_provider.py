"""WeightProvider, expert packing and prefetch tests (Plan.md C4 / C6)."""

from __future__ import annotations


import pytest

from stream_agent.errors import ModelFormatUnsupported
from stream_agent.layout.model_map import ExpertPacking
from stream_agent.prefetch import PrefetchScheduler
from stream_agent.provider import (
    ExpertKey,
    PrefetchHandle,
    TensorKey,
    TensorSliceKey,
    WeightProvider,
)



async def test_provider_satisfies_the_protocol(provider):
    assert isinstance(provider, WeightProvider)


async def test_unknown_tensor_raises_a_typed_error(provider):
    with pytest.raises(ModelFormatUnsupported):
        await provider.read_tensor(TensorKey(tensor="does.not.exist"))


async def test_slice_outside_the_tensor_is_rejected(provider):
    with pytest.raises(ModelFormatUnsupported):
        await provider.read_slice(
            TensorSliceKey(tensor="model.embed_tokens.weight", start=0, end=9999)
        )


async def test_expert_read_requires_a_declared_packing(provider):
    with pytest.raises(ModelFormatUnsupported):
        await provider.read_expert(ExpertKey(tensor="model.embed_tokens.weight", expert_id=0))


async def test_declared_expert_packing_yields_the_expert_slice(provider):
    ref = provider.layout.tensor("model.embed_tokens.weight")
    provider.layout.register_expert_packing(
        ExpertPacking(tensor="model.embed_tokens.weight", dim=0, num_experts=8)
    )
    buffer = await provider.read_expert(
        ExpertKey(tensor="model.embed_tokens.weight", expert_id=2)
    )
    assert len(buffer.data) == ref.nbytes // 8
    # The returned shape must describe the slice, not the full packed tensor.
    assert buffer.shape == (1, ref.shape[1])


async def test_expert_packing_handles_merged_rows_per_expert(provider):
    # 8 rows / 4 experts = 2 rows per expert, not a single row.
    provider.layout.register_expert_packing(
        ExpertPacking(tensor="model.embed_tokens.weight", dim=0, num_experts=4)
    )
    buffer = await provider.read_expert(
        ExpertKey(tensor="model.embed_tokens.weight", expert_id=1)
    )
    assert buffer.shape == (2, 4)
    assert len(buffer.data) == 2 * 4 * 4


async def test_tensor_slice_reports_the_slice_shape(provider):
    ref = provider.layout.tensor("lm_head.weight")
    buffer = await provider.read_slice(
        TensorSliceKey(tensor="lm_head.weight", start=0, end=8)
    )
    assert len(buffer.data) == 32
    assert buffer.shape == (8,)
    assert buffer.shape != ref.shape


async def test_read_file_routes_a_weight_shard_through_the_cache(provider):
    before = provider.cache.used_weight_bytes
    data = await provider.read_file("model.safetensors", 0, 127)
    assert len(data) == 128
    assert provider.cache.used_weight_bytes > before


async def test_read_file_keeps_metadata_out_of_the_weight_cache(
    provider, fixture_source, tmp_path
):
    from stream_agent.provider import RemoteWeightProvider

    metadata_provider = RemoteWeightProvider(
        source=fixture_source,
        layout=provider.layout,
        cache=provider.cache,
        model_alias="tiny",
        block_size_bytes=64,
        telemetry=provider.telemetry,
        metadata_root=tmp_path / "metadata",
    )
    before = provider.cache.used_weight_bytes
    data = await metadata_provider.read_file("config.json", 0, 9)
    assert len(data) == 10
    assert provider.cache.used_weight_bytes == before
    assert provider.telemetry.network.metadata_downloaded_bytes >= 10


async def test_expert_id_outside_the_packing_is_rejected(provider):
    provider.layout.register_expert_packing(
        ExpertPacking(tensor="model.embed_tokens.weight", dim=0, num_experts=4)
    )
    with pytest.raises(ModelFormatUnsupported):
        await provider.read_expert(
            ExpertKey(tensor="model.embed_tokens.weight", expert_id=9)
        )


async def test_prefetch_warms_the_cache_and_is_counted(provider, fixture_source):
    fixture_source.request_log.clear()
    scheduler = PrefetchScheduler(provider)
    handle = await scheduler.schedule([TensorKey(tensor="lm_head.weight")])
    assert isinstance(handle, PrefetchHandle)
    await scheduler.wait_idle()
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    assert provider.telemetry.cache.useful_prefetched_bytes > 0
    await scheduler.cancel_all()


async def test_prefetch_respects_its_budget(provider):
    scheduler = PrefetchScheduler(provider, max_bytes=0)
    handle = await scheduler.schedule([TensorKey(tensor="lm_head.weight")])
    assert handle is None
    assert scheduler.stats.skipped == 1


async def test_prefetch_can_be_cancelled(provider):
    scheduler = PrefetchScheduler(provider)
    handle = await scheduler.schedule(
        [TensorKey(tensor="model.embed_tokens.weight"), TensorKey(tensor="lm_head.weight")]
    )
    await scheduler.cancel(handle)
    assert handle.cancelled
    assert scheduler.stats.cancelled == 1


async def test_weight_wait_is_measured_separately(provider):
    provider.reset_wait_timer()
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    assert provider.weight_wait_ms >= 0
