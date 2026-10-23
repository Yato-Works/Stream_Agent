"""Local model server / worker controller (Plan.md C8).

The endpoint is loopback-only by default, serves one selected model per
worker and never executes shell commands or touches a code workspace.  It
exposes a small, OpenAI-compatible surface for chat completions plus a job
API for slow background work.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .errors import (
    BackendCapabilityMissing,
    StreamAgentError,
    UnknownModel,
)
from .jobs import JobController
from .runtime import ModelSession, StreamAgentRuntime

try:  # pragma: no cover - the endpoint is an optional extra
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]

MAX_REQUEST_BYTES = 1 * 1024 * 1024


class ModelController:
    """Owns the active session and the concurrency/deadline policy."""

    def __init__(
        self,
        runtime: StreamAgentRuntime,
        *,
        model_alias: str,
        max_concurrent_requests: int = 1,
        request_timeout_seconds: int = 3600,
        max_tokens: int = 2048,
    ) -> None:
        self.runtime = runtime
        self.model_alias = model_alias
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.request_timeout_seconds = request_timeout_seconds
        self.max_tokens = max_tokens
        self._session: ModelSession | None = None
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self._lock = asyncio.Lock()
        self.started_at = time.time()
        self.request_count = 0

    async def session(self) -> ModelSession:
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    self._session = await self.runtime.open_session(self.model_alias)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def generate(self, prompt: str, **params: Any) -> dict[str, Any]:
        async with self._semaphore:
            session = await self.session()
            try:
                result = await asyncio.wait_for(
                    session.generate(prompt, **params),
                    timeout=self.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise StreamAgentError(
                    "the request exceeded the configured deadline",
                    details={"timeout_seconds": self.request_timeout_seconds},
                ) from exc
            self.request_count += 1
            return {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "ttft_ms": result.ttft_ms,
                "generation_ms": result.generation_ms,
                "weight_wait_ms": result.weight_wait_ms,
                "finish_reason": result.finish_reason,
            }

    def describe(self) -> dict[str, Any]:
        session = self._session
        return {
            "model": self.model_alias,
            "ready": session is not None,
            "repo": session.repo if session else "",
            "revision": session.revision if session else "",
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "requests": self.request_count,
            "max_concurrent_requests": self.max_concurrent_requests,
        }


def _error_payload(exc: StreamAgentError) -> dict[str, Any]:
    return {"error": exc.to_dict()}


def build_app(runtime: StreamAgentRuntime, controller: ModelController):
    """Create the FastAPI application for the local endpoint."""
    if not FASTAPI_AVAILABLE:
        raise BackendCapabilityMissing(
            "the local endpoint requires FastAPI; install stream-agent[server]"
        )

    app = FastAPI(title="Stream Agent", version="0.1.0", docs_url=None, redoc_url=None)
    jobs = JobController(result_dir=runtime.storage.log_root / "jobs")
    app.state.controller = controller
    app.state.jobs = jobs

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "stream-agent",
            "uptime_seconds": round(time.time() - controller.started_at, 3),
            "model": controller.describe(),
            "remote_inference": "disabled",
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        from .backend.registry import available_backends

        entries = []
        for record in runtime.db.list_models() if runtime.db else []:
            entries.append(
                {
                    "id": record["alias"],
                    "object": "model",
                    "owned_by": record["repo"],
                    "revision": record["revision"],
                    "format": record["artifact_format"],
                    "backend": record["backend"],
                    "license_review": record["license_review"],
                }
            )
        return {
            "object": "list",
            "data": entries,
            "capabilities": {
                "backends": available_backends(),
                "max_concurrent_requests": controller.max_concurrent_requests,
                "streaming": "token" if _supports_streaming(controller) else "unsupported",
                "fields": [
                    "model",
                    "messages",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                    "stream",
                ],
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "CONFIG_INVALID", "message": "request body too large"}},
            )
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "CONFIG_INVALID", "message": "invalid JSON body"}},
            )

        model_name = payload.get("model") or controller.model_alias
        if model_name != controller.model_alias:
            return JSONResponse(
                status_code=404,
                content=_error_payload(UnknownModel(
                    f"this worker serves {controller.model_alias!r}, not {model_name!r}"
                )),
            )
        messages = payload.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "CONFIG_INVALID", "message": "messages are required"}},
            )
        prompt = "\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, dict)
        )
        max_tokens = min(int(payload.get("max_tokens", 256)), controller.max_tokens)
        params = {
            "max_tokens": max_tokens,
            "temperature": float(payload.get("temperature", 0.0) or 0.0),
            "top_p": payload.get("top_p"),
            "stop": payload.get("stop"),
        }
        stream = bool(payload.get("stream"))

        if stream and not _supports_streaming(controller):
            return JSONResponse(
                status_code=400,
                content=_error_payload(
                    BackendCapabilityMissing(
                        "the selected backend does not support token streaming"
                    )
                ),
            )

        try:
            result = await controller.generate(prompt, **params)
        except StreamAgentError as exc:
            return JSONResponse(status_code=400, content=_error_payload(exc))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
            )

        completion_id = f"chatcmpl-{int(time.time()*1000)}"
        if stream:
            return StreamingResponse(
                _stream_chunks(completion_id, model_name, result["text"]),
                media_type="text/event-stream",
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["generated_tokens"],
            },
            "stream_agent": {
                "weight_wait_ms": result["weight_wait_ms"],
                "generation_ms": result["generation_ms"],
            },
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        telemetry = runtime.telemetry
        return {
            "run_id": telemetry.run_id,
            "model": controller.describe(),
            "network": {
                "downloaded_bytes_total": telemetry.network.downloaded_bytes,
                "range_requests": telemetry.network.range_requests,
                "retries": telemetry.network.retries,
                "source_errors": telemetry.network.source_errors,
            },
            "cache": {
                "hits": telemetry.cache.hits,
                "misses": telemetry.cache.misses,
                "evictions": telemetry.cache.evictions,
                "evicted_bytes": telemetry.cache.evicted_bytes,
            },
            "storage": runtime.cache.status().to_dict() if runtime.cache else {},
        }

    @app.post("/v1/jobs")
    async def submit_job(request: Request):
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "CONFIG_INVALID", "message": "request body too large"}},
            )
        payload = json.loads(raw or b"{}")
        prompt = payload.get("prompt") or payload.get("input") or ""
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "CONFIG_INVALID", "message": "prompt is required"}},
            )
        params = {
            "max_tokens": min(int(payload.get("max_tokens", 512)), controller.max_tokens),
            "temperature": float(payload.get("temperature", 0.0) or 0.0),
        }

        async def runner(job: Any) -> str:
            result = await controller.generate(str(prompt), **params)
            job.weight_wait_ms = result["weight_wait_ms"]
            job.generated_tokens = result["generated_tokens"]
            return result["text"]

        job = jobs.submit(model=controller.model_alias, prompt=str(prompt), runner=runner)
        return {"job_id": job.job_id, "state": job.state, "model": job.model}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "UNKNOWN_MODEL", "message": "unknown job id"}},
            )
        return job.to_dict()

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        ok = await jobs.cancel(job_id)
        if not ok:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "UNKNOWN_MODEL", "message": "unknown job id"}},
            )
        return {"job_id": job_id, "state": "cancelled"}

    return app


def _supports_streaming(controller: ModelController) -> bool:
    session = controller._session
    if session is None:
        return False
    capabilities = getattr(session.backend, "capabilities", None)
    return bool(getattr(capabilities, "supports_streaming", False))


async def _stream_chunks(completion_id: str, model: str, text: str):
    """Deliver a completed generation as server sent events."""
    created = int(time.time())
    for piece in _split_text(text):
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _split_text(text: str, size: int = 64) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def check_bind_policy(host: str, allow_remote_bind: bool) -> None:
    """Refuse to expose the endpoint beyond loopback without an explicit opt-in."""
    loopback = {"127.0.0.1", "localhost", "::1"}
    if host not in loopback and not allow_remote_bind:
        raise StreamAgentError(
            "refusing to bind a non-loopback address without server.allow_remote_bind",
            details={"host": host},
        )


__all__ = ["ModelController", "build_app", "check_bind_policy"]
