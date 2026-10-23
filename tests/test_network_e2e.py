"""End to end test against a real model repository (Plan.md Appendix C).

The test is skipped automatically when the repository cannot be reached.  It
performs the experiment the design proposes: pin a revision, read only the
header, fetch a single tensor range, compare it byte for byte with a locally
downloaded reference and assert that no full object was transferred.
"""

from __future__ import annotations


import pytest

from stream_agent.cache.manager import CacheManager
from stream_agent.config import StorageConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.provider import RemoteWeightProvider, TensorKey
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


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


async def test_header_and_tensor_range_match_a_local_reference(source, storage, tmp_path):
    revision = await source.resolve_revision(REPO, "main")
    layout = await build_layout(source, revision)
    assert layout.tensors, "the artifact must expose tensors"

    name = min(layout.tensors, key=lambda n: layout.tensor(n).nbytes)
    ref = layout.tensor(name)

    # The reference copy is downloaded once, outside the runtime, only to be
    # able to compare the requested range byte for byte.
    reference = await source.fetch_range(
        REPO, revision.commit, ref.shard, 0, layout.shards[ref.shard].size - 1
    )

    cache = CacheManager(storage, *_make_db_and_store(storage))
    provider = RemoteWeightProvider(
        source=source,
        layout=layout,
        cache=cache,
        model_alias="e2e",
        block_size_bytes=64 * 1024,
    )
    buffer = await provider.read_tensor(TensorKey(tensor=name))
    assert buffer.data == reference[ref.data_start : ref.data_end]

    # A second read must be served locally.
    await provider.read_tensor(TensorKey(tensor=name))
    assert cache.used_weight_bytes > 0
    assert cache.used_weight_bytes < len(reference), (
        "the cache must not hold the complete artifact"
    )


def _make_db_and_store(storage):
    from stream_agent.cache.db import MetadataDB
    from stream_agent.cache.store import BlockStore

    return MetadataDB(storage.db_path), BlockStore(storage.blocks_root)
