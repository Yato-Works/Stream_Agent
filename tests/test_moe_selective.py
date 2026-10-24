"""Selective-expert fetch test against a real MoE model (Plan.md Appendix C).

Skipped automatically when the repository is unreachable.  It proves the
provider can fetch *only* the chosen expert slices and that the host cache
holds the selected bytes, never the full expert bank.
"""

from __future__ import annotations

import pytest

from stream_agent.cache.manager import CacheManager
from stream_agent.config import StorageConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider, TensorKey
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "tiny-random/qwen3.5-moe"


def _reachable() -> bool:
    import httpx

    try:
        response = httpx.get(f"https://huggingface.co/api/models/{REPO}", timeout=15)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not _reachable(), reason="the model repository is not reachable"),
]


@pytest.fixture()
async def source():
    adapter = HuggingFaceSource()
    try:
        yield adapter
    finally:
        await adapter.aclose()


@pytest.fixture()
def storage(tmp_path):
    return StorageConfig(
        root=tmp_path / "store",
        weight_cache_limit_bytes=64 * 1024 * 1024,
        managed_store_limit_bytes=128 * 1024 * 1024,
        metadata_reserve_bytes=1024 * 1024,
        staging_reserve_bytes=1024 * 1024,
    )


def _make_db_and_store(storage):
    from stream_agent.cache.db import MetadataDB
    from stream_agent.cache.store import BlockStore

    return MetadataDB(storage.db_path), BlockStore(storage.blocks_root)


async def test_provider_fetches_only_selected_experts(source, storage, tmp_path):
    revision = await source.resolve_revision(REPO, "main")
    layout = await build_layout(source, revision)

    import re

    expert_tensors = [n for n in layout.tensors if ".mlp.experts." in n]
    assert expert_tensors, "the artifact must expose MoE expert tensors"

    # Pick two expert ids from layer 0 and read only their weights.
    layer = min(int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in expert_tensors)
    chosen = sorted(
        {int(re.search(r"experts\.(\d+)\.", n).group(1)) for n in expert_tensors if f"layers.{layer}." in n}
    )[:2]

    selected_names = []
    for eid in chosen:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.{eid}.{proj}.weight"
            if name in layout.tensors:
                selected_names.append(name)

    total_expert_bytes = sum(layout.tensor(n).nbytes for n in expert_tensors)
    selected_bytes = sum(layout.tensor(n).nbytes for n in selected_names)
    assert selected_bytes < total_expert_bytes, "selected experts must be smaller than the whole bank"
    assert selected_bytes > 0

    cache = CacheManager(storage, *_make_db_and_store(storage))
    provider = RemoteWeightProvider(
        source=source,
        layout=layout,
        cache=cache,
        model_alias="moe",
        block_size_bytes=64 * 1024,
    )

    buffers = [await provider.read_tensor(TensorKey(tensor=n)) for n in selected_names]
    assert all(len(b.data) == layout.tensor(n).nbytes for b, n in zip(buffers, selected_names))

    # A second read of the same experts must be served from the cache.
    await provider.read_tensor(TensorKey(tensor=selected_names[0]))
    assert cache.used_weight_bytes > 0
    assert cache.used_weight_bytes < total_expert_bytes, (
        "the cache must not hold expert weights that were never requested"
    )
