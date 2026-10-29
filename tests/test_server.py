"""Local endpoint, job API and MCP adapter tests (Plan.md C8 / 12.2-12.4)."""

from __future__ import annotations


import pytest

from stream_agent.jobs import JobController
from stream_agent.mcp import MCPAdapter
from stream_agent.server import ModelController, build_app, check_bind_policy

from .conftest import COMMIT, REPO


@pytest.fixture()
def registered(runtime):
    runtime.register_model_sync = None
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


@pytest.fixture()
def client(registered):
    from fastapi.testclient import TestClient

    controller = ModelController(registered, model_alias="tiny")
    app = build_app(registered, controller)
    with TestClient(app) as test_client:
        test_client.controller = controller
        yield test_client


def test_health_reports_a_local_service(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["remote_inference"] == "disabled"


def test_models_endpoint_lists_registered_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["id"] == "tiny"
    assert "fields" in payload["capabilities"]
    # The fixture backend does not emit tokens incrementally, so ``stream``
    # must not be advertised.
    assert payload["capabilities"]["streaming"] == "unsupported"
    assert "stream" not in payload["capabilities"]["fields"]
    assert "stop" in payload["capabilities"]["fields"]


def test_chat_completion_returns_text(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"]
    assert payload["usage"]["completion_tokens"] == 4


def test_unknown_model_is_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "other", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"


def test_streaming_is_refused_when_unsupported(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BACKEND_CAPABILITY_MISSING"


def test_metrics_endpoint_reports_storage(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "cache" in payload and "network" in payload


def test_job_lifecycle(client):
    response = client.post("/v1/jobs", json={"prompt": "review this", "max_tokens": 4})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["state"] in ("completed", "failed", "cancelled"):
            break
        import time

        time.sleep(0.05)
    assert status["state"] == "completed"
    assert status["result"]
    cancel = client.post(f"/v1/jobs/{job_id}/cancel")
    assert cancel.status_code == 200


def test_bind_policy_refuses_a_public_interface():
    from stream_agent.errors import StreamAgentError

    with pytest.raises(StreamAgentError):
        check_bind_policy("0.0.0.0", allow_remote_bind=False)
    check_bind_policy("0.0.0.0", allow_remote_bind=True)


async def test_mcp_adapter_lists_and_calls_tools(registered):
    controller = ModelController(registered, model_alias="tiny")
    jobs = JobController()
    adapter = MCPAdapter(controller, jobs)
    init = await adapter.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["name"] == "stream-agent"
    listed = await adapter.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == {
        "stream_agent_submit_review",
        "stream_agent_job_status",
        "stream_agent_cancel",
    }
    called = await adapter.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "stream_agent_submit_review", "arguments": {"prompt": "hi"}},
        }
    )
    assert "job_id" in called["result"]["content"][0]["text"]
    await jobs.aclose()
    await controller.close()
