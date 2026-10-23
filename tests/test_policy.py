"""Policy tests: remote code, licence review, credential hygiene and
revision isolation (Plan.md §14 / §19)."""

from __future__ import annotations


import pytest

from stream_agent.catalog import ModelCatalog
from stream_agent.errors import LicenseReviewRequired, RemoteCodeBlocked

from .conftest import COMMIT, REPO


async def test_license_review_required_blocks_inference(runtime):
    runtime.db.upsert_model(
        {
            "alias": "gated",
            "repo": REPO,
            "revision": COMMIT,
            "artifact_format": "safetensors",
            "backend": "fixture",
            "quota_bytes": 1024 * 1024,
            "quota_mode": "fixed",
            "license_review": "required",
            "allow_remote_code": False,
        }
    )
    record = runtime.catalog.registered("gated")
    with pytest.raises(LicenseReviewRequired):
        ModelCatalog.assert_inference_allowed(record)


async def test_pending_license_review_does_not_block(runtime):
    runtime.db.upsert_model(
        {
            "alias": "pending",
            "repo": REPO,
            "revision": COMMIT,
            "artifact_format": "safetensors",
            "backend": "fixture",
            "quota_bytes": 1024 * 1024,
            "quota_mode": "fixed",
            "license_review": "pending",
            "allow_remote_code": False,
        }
    )
    record = runtime.catalog.registered("pending")
    ModelCatalog.assert_inference_allowed(record)


async def test_remote_code_is_blocked_by_default(runtime):
    runtime.db.upsert_model(
        {
            "alias": "custom",
            "repo": REPO,
            "revision": COMMIT,
            "artifact_format": "safetensors",
            "backend": "fixture",
            "quota_bytes": 1024 * 1024,
            "quota_mode": "fixed",
            "license_review": "accepted",
            "allow_remote_code": False,
            "metadata": {"requires_custom_code": True},
        }
    )
    record = runtime.catalog.registered("custom")
    with pytest.raises(RemoteCodeBlocked):
        ModelCatalog.assert_inference_allowed(record)
    # An explicit opt-in is honoured.
    ModelCatalog.assert_inference_allowed(record, allow_remote_code=True)


async def test_credentials_never_reach_the_metadata_database(runtime, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")
    runtime.db.upsert_model(
        {
            "alias": "tiny",
            "repo": REPO,
            "revision": COMMIT,
            "artifact_format": "safetensors",
            "backend": "fixture",
            "quota_bytes": 1024 * 1024,
            "quota_mode": "fixed",
            "license_review": "pending",
            "allow_remote_code": False,
        }
    )
    runtime.db.record_event("probe", detail={"note": "no credentials here"})
    raw = runtime.storage.db_path.read_bytes()
    assert b"super-secret-token" not in raw
    assert b"Authorization" not in raw


async def test_two_revisions_never_share_a_block(runtime, fixture_source):
    """The same file content in two revisions must stay in separate blocks."""
    other_commit = "9999999999999999"
    payload = fixture_source.repos[REPO][COMMIT]["model.safetensors"]
    fixture_source.add_revision(REPO, other_commit, {"model.safetensors": payload})

    from stream_agent.layout.model_map import build_layout
    from stream_agent.provider import RemoteWeightProvider, TensorKey

    first = await build_layout(fixture_source, await fixture_source.resolve_revision(REPO, COMMIT))
    second = await build_layout(
        fixture_source, await fixture_source.resolve_revision(REPO, other_commit)
    )
    provider_a = RemoteWeightProvider(
        source=fixture_source, layout=first, cache=runtime.cache, model_alias="a"
    )
    provider_b = RemoteWeightProvider(
        source=fixture_source, layout=second, cache=runtime.cache, model_alias="b"
    )
    await provider_a.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))
    await provider_b.read_tensor(TensorKey(tensor="model.embed_tokens.weight"))

    commits = {entry.commit for entry in runtime.cache._entries.values()}
    assert commits == {COMMIT, other_commit}
    keys = {entry.block_key for entry in runtime.cache._entries.values()}
    assert len(keys) == len(runtime.cache._entries)
