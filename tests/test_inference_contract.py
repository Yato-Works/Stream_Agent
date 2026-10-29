"""Honesty contracts for generation results (Plan.md section 16.3)."""

from __future__ import annotations

import pytest

from .conftest import COMMIT, REPO


@pytest.fixture()
def registered(runtime):
    runtime.db.upsert_model(
        {
            "alias": "tiny",
            "repo": REPO,
            "revision": COMMIT,
            "artifact_format": "safetensors",
            "backend": "fixture",
            "quota_bytes": 128 * 1024,
            "quota_mode": "fixed",
            "license_review": "pending",
            "allow_remote_code": False,
        }
    )
    runtime.cache.set_model_quota("tiny", 128 * 1024, "fixed")
    return runtime


async def test_non_streaming_backend_does_not_fabricate_ttft(registered):
    session = await registered.open_session("tiny")
    try:
        result = await session.generate("hello", max_tokens=3)
    finally:
        await session.close()
    # The fixture generates the completion in one call.  Reporting the total
    # generation time as TTFT would be a different metric under the same name.
    assert result.ttft_ms is None
    assert result.generation_ms >= 0


async def test_stop_sequence_is_honoured_in_the_returned_text(registered):
    session = await registered.open_session("tiny")
    try:
        baseline = await session.generate("hello", max_tokens=4)
        tokens = baseline.text.split(" ")
        assert len(tokens) >= 3
        marker = tokens[1]

        stopped = await session.generate("hello", max_tokens=4, stop=marker)
    finally:
        await session.close()
    assert stopped.text == baseline.text[: baseline.text.find(marker)]
    assert stopped.finish_reason == "stop"
    # Stop sequences truncate text but do not undo compute.
    assert stopped.generated_tokens == baseline.generated_tokens
