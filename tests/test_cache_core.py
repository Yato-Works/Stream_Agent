"""Stage A/B tests: range integrity and cache invariants (Plan.md section 16.2)."""

from __future__ import annotations

import asyncio

import pytest

from stream_agent.errors import (
    CacheBudgetExceeded,
    IntegrityFailure,
    WorkingSetExceedsBudget,
)
from stream_agent.provider import TensorKey, TensorSliceKey

from .conftest import COMMIT, REPO


async def test_layout_is_built_from_remote_metadata_only(provider):
    layout = provider.layout
    assert set(layout.tensors) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    }
    # Only the header and the index were read: no full weight file download.
    sizes = [end - start + 1 for _, _, _, start, end in provider.source.request_log]
    assert max(sizes) < 4096


async def test_tensor_roundtrip_matches_source_bytes(provider, fixture_source):
    ref = provider.layout.tensor("model.embed_tokens.weight")
    buffer = await provider.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    reference = await fixture_source.fetch_range(
        REPO, COMMIT, "model.safetensors", ref.data_start, ref.data_end - 1
    )
    assert buffer.data == reference
    assert buffer.dtype == "F32"
    assert buffer.shape == (8, 4)


async def test_second_read_is_a_cache_hit(provider, fixture_source):
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    before = len(fixture_source.request_log)
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    assert len(fixture_source.request_log) == before
    assert provider.telemetry.cache.hits >= 1


async def test_slice_read_returns_the_requested_elements(provider):
    buffer = await provider.read_slice(
        TensorSliceKey(tensor="model.embed_tokens.weight", start=0, end=2)
    )
    assert len(buffer.data) == 2 * 4  # two F32 elements


async def test_concurrent_requests_share_one_transfer(provider, fixture_source):
    fixture_source.request_log.clear()
    results = await asyncio.gather(
        *[
            provider.read_tensor(TensorKey(tensor="lm_head.weight"))
            for _ in range(6)
        ]
    )
    assert len({bytes(r.data) for r in results}) == 1
    # Six concurrent readers share one transfer per cache block.
    distinct = {(path, start, end) for _, _, path, start, end in fixture_source.request_log}
    assert len(fixture_source.request_log) == len(distinct) <= 3


async def test_budget_is_never_exceeded(provider, runtime):
    runtime.cache.storage = runtime.cache.storage.__class__(
        **{**runtime.cache.storage.__dict__, "weight_cache_limit_bytes": 32}
    )
    # A single indivisible block no longer fits into the budget at all.
    with pytest.raises(WorkingSetExceedsBudget):
        await provider.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    assert runtime.cache.used_weight_bytes <= 32


async def test_pinned_blocks_are_never_evicted(provider, runtime):
    runtime.cache.storage = runtime.cache.storage.__class__(
        **{**runtime.cache.storage.__dict__, "weight_cache_limit_bytes": 128}
    )
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    for key in list(runtime.cache._entries):
        await runtime.cache.pin(key)
    # The cache is full and nothing can be evicted any more.
    with pytest.raises((CacheBudgetExceeded, WorkingSetExceedsBudget)):
        await provider.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    assert runtime.cache.used_weight_bytes <= 128
    assert all(runtime.cache.is_pinned(k) for k in runtime.cache._entries)


async def test_corrupt_block_is_never_served(runtime, fixture_source, provider):
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    entry = next(iter(runtime.cache._entries.values()))
    block_path = runtime.store.path_for(entry.content_hash)
    block_path.write_bytes(b"corrupted")
    with pytest.raises(IntegrityFailure):
        await provider.read_tensor(TensorKey(tensor="lm_head.weight"))


async def test_eviction_reclaims_space_under_pressure(runtime, fixture_source):
    from stream_agent.layout.model_map import build_layout

    revision = await fixture_source.resolve_revision(REPO, COMMIT)
    layout = await build_layout(fixture_source, revision)
    from stream_agent.provider import RemoteWeightProvider

    provider = RemoteWeightProvider(
        source=fixture_source,
        layout=layout,
        cache=runtime.cache,
        model_alias="tiny",
        block_size_bytes=64,
    )
    runtime.cache.storage = runtime.cache.storage.__class__(
        **{**runtime.cache.storage.__dict__, "weight_cache_limit_bytes": 128}
    )
    for name in sorted(layout.tensors):
        await provider.read_tensor(TensorKey(tensor=name))
    assert runtime.cache.used_weight_bytes <= 128
    assert runtime.cache.eviction_log()


async def test_verify_all_drops_corrupted_blocks(provider, runtime):
    await provider.read_tensor(TensorKey(tensor="lm_head.weight"))
    entries = list(runtime.cache._entries.values())
    runtime.store.path_for(entries[0].content_hash).write_bytes(b"corrupted")
    report = await runtime.cache.verify_all()
    assert report["checked"] == len(entries)
    assert report["corrupted"] == 1
    assert runtime.cache.used_weight_bytes < 64 * 3


async def test_crash_recovery_rebuilds_the_cache(runtime, fixture_source):
    from stream_agent.cache.manager import BlockRequest, CacheManager

    manager = runtime.cache

    async def fetcher(start: int, end: int) -> bytes:
        return await fixture_source.fetch_range(
            REPO, COMMIT, "model.safetensors", start, end
        )

    await manager.fetch_block(
        BlockRequest(REPO, COMMIT, "model.safetensors", 0, 63, "tiny"), fetcher
    )
    assert manager.used_weight_bytes == 64

    recovered = CacheManager(
        runtime.config.storage,
        runtime.db,
        runtime.store,
    )
    report = await recovered.recover()
    assert report["restored_entries"] == 1
    assert recovered.used_weight_bytes == 64
