"""Multi model accounting and quota policy tests (Plan.md gates G4 / 10.2)."""

from __future__ import annotations


import pytest

from stream_agent.errors import CacheBudgetExceeded, WorkingSetExceedsBudget
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider, TensorKey

from .conftest import COMMIT, REPO, make_safetensors


@pytest.fixture()
def two_models(fixture_source):
    """A second model that shares bytes with the first one."""
    payload = make_safetensors(
        {
            "model.embed_tokens.weight": ("F32", (8, 4), b"\x00" * 128),
            "model.layers.0.self_attn.q_proj.weight": ("F32", (4, 4), b"\x01" * 64),
            "model.layers.0.mlp.down_proj.weight": ("F32", (4, 4), b"\x02" * 64),
            "model.layers.1.self_attn.q_proj.weight": ("F32", (4, 4), b"\x03" * 64),
            "lm_head.weight": ("F32", (8, 4), b"\x04" * 128),
        }
    )
    fixture_source.add_revision(
        "fixture/other-model",
        "feedface000000000",
        {"model.safetensors": payload},
        license_id="mit",
    )
    return fixture_source


async def provider_for(source, layout, cache, alias):
    return RemoteWeightProvider(
        source=source,
        layout=layout,
        cache=cache,
        model_alias=alias,
        block_size_bytes=64,
    )


async def test_two_models_share_one_global_budget(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    second = await build_layout(
        two_models, await two_models.resolve_revision("fixture/other-model", "feedface000000000")
    )
    runtime.cache.set_model_quota("a", 4096, "fixed")
    runtime.cache.set_model_quota("b", 4096, "fixed")
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    provider_b = await provider_for(two_models, second, runtime.cache, "b")

    for name in sorted(first.tensors):
        await provider_a.read_tensor(TensorKey(tensor=name))
    for name in sorted(second.tensors):
        await provider_b.read_tensor(TensorKey(tensor=name))

    status = runtime.cache.status()
    assert status.used_weight_bytes <= status.weight_cache_limit_bytes
    assert set(status.per_model) >= {"a", "b"}
    assert status.per_model["a"]["used_bytes"] > 0
    assert status.per_model["b"]["used_bytes"] > 0


async def test_identical_content_is_stored_once(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    second = await build_layout(
        two_models, await two_models.resolve_revision("fixture/other-model", "feedface000000000")
    )
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    provider_b = await provider_for(two_models, second, runtime.cache, "b")
    await provider_a.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    used_after_first = runtime.cache.used_weight_bytes
    await provider_b.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    # The same bytes are not stored twice.
    assert runtime.cache.used_weight_bytes == used_after_first
    assert runtime.cache.model_used_bytes("a") > 0
    assert runtime.cache.model_used_bytes("b") > 0


async def test_cache_hit_adds_the_reading_model_reference(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    provider_b = await provider_for(two_models, first, runtime.cache, "b")
    await provider_a.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    assert runtime.cache.model_used_bytes("b") == 0

    await provider_b.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    # Every reused block must now be attributed to b as well; otherwise b can
    # read a shared cache for free and its quota is never enforced.
    assert runtime.cache.model_used_bytes("b") > 0
    assert all("b" in entry.refs for entry in runtime.cache._entries.values())


async def test_fixed_quota_blocks_a_model_that_is_too_large(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    # The quota is smaller than one indivisible cache block.
    runtime.cache.set_model_quota("a", 32, "fixed")
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    with pytest.raises((CacheBudgetExceeded, WorkingSetExceedsBudget)):
        await provider_a.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    assert runtime.cache.model_used_bytes("a") <= 32


async def test_elastic_quota_allows_borrowing(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    runtime.cache.set_model_quota("a", 64, "elastic")
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    buffer = await provider_a.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    assert buffer.data
    # The global hard limit still holds.
    assert runtime.cache.used_weight_bytes <= runtime.cache.status().weight_cache_limit_bytes


async def test_eviction_log_records_the_reason(two_models, runtime):
    first = await build_layout(two_models, await two_models.resolve_revision(REPO, COMMIT))
    runtime.cache.storage = runtime.cache.storage.__class__(
        **{**runtime.cache.storage.__dict__, "weight_cache_limit_bytes": 128}
    )
    provider_a = await provider_for(two_models, first, runtime.cache, "a")
    for name in sorted(first.tensors):
        await provider_a.read_tensor(TensorKey(tensor=name))
    log = runtime.cache.eviction_log()
    assert log
    assert log[0]["reason"] in {"global_pressure", "model_quota", "manual", "bench_reset"}
    assert log[0]["bytes"] >= 0
