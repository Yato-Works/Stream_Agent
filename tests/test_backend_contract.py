"""Backend capability and registration contracts."""

from __future__ import annotations

import pytest

from stream_agent.backend.base import BackendCapabilities, LoadContext
from stream_agent.backend.registry import create_backend, register_backend
from stream_agent.errors import BackendCapabilityMissing


@pytest.fixture()
def registered(runtime):
    from .conftest import COMMIT, REPO

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
    return runtime


def test_capabilities_require_both_format_and_device():
    caps = BackendCapabilities(
        backend_id="example", version="1", formats=("safetensors",), devices=("cpu",)
    )
    assert caps.supports(artifact_format="safetensors", device="cpu")
    assert not caps.supports(artifact_format="gguf", device="cpu")
    assert not caps.supports(artifact_format="safetensors", device="cuda")
    cuda_caps = BackendCapabilities(
        backend_id="cuda-example", version="1", devices=("cuda",)
    )
    assert cuda_caps.supports(artifact_format="safetensors", device="cuda:0")


def test_load_context_reads_backend_options_with_default():
    context = LoadContext("a", "r", "rev", None, None, options={"x": 3})  # type: ignore[arg-type]
    assert context.option("x") == 3
    assert context.option("missing", "fallback") == "fallback"


def test_backend_registry_creates_fresh_instances():
    class Backend:
        capabilities = BackendCapabilities(backend_id="test-contract", version="1")

        def __init__(self):
            self.state = []

    register_backend("test-contract", Backend)
    first = create_backend("test-contract")
    second = create_backend("test-contract")
    assert first is not second
    assert first.state is not second.state


def test_backend_registry_unknown_id_lists_available_backends():
    with pytest.raises(KeyError, match="available backends"):
        create_backend("does-not-exist")


@pytest.mark.asyncio
async def test_runtime_rejects_device_not_declared_by_backend(registered, monkeypatch):
    class CpuOnlyBackend:
        capabilities = BackendCapabilities(
            backend_id="cpu-only-contract", version="1", formats=("safetensors",), devices=("cpu",)
        )

        def expert_packing(self):
            return {}

    monkeypatch.setattr("stream_agent.runtime.create_backend", lambda _: CpuOnlyBackend())
    with pytest.raises(BackendCapabilityMissing, match="device 'cuda'"):
        await registered.open_session("tiny", backend_id="cpu-only-contract", device="cuda")
