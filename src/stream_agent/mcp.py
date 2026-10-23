"""Minimal MCP adapter for agent frameworks (Plan.md section 12.4).

The adapter exposes three tools and nothing else.  It deliberately does not
implement a generic coding loop, a planner or a tool executor: the parent
agent keeps that responsibility.  Latency expectations are part of the tool
description so that a calling agent can decide whether a slow model is
appropriate for the task.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .jobs import JobController
from .server import ModelController

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "stream_agent_submit_review",
        "description": (
            "Submit a bounded review task to the locally served model and return a "
            "job id. The model may be slow: the weight cache is bounded, so the "
            "first call can take minutes. Poll with stream_agent_job_status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The bounded task for the model."},
                "max_tokens": {"type": "integer", "description": "Upper bound on generated tokens."},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "stream_agent_job_status",
        "description": "Return the state, queue delay and result of a submitted job.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "stream_agent_cancel",
        "description": "Cancel a running job and release the reserved cache capacity.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
]


class MCPAdapter:
    """Speaks the MCP JSON-RPC dialect over stdio."""

    def __init__(self, controller: ModelController, jobs: JobController) -> None:
        self.controller = controller
        self.jobs = jobs
        self.initialized = False

    # -- tool implementations ---------------------------------------------
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "stream_agent_submit_review":
            prompt = str(arguments.get("prompt", ""))
            if not prompt:
                return {"isError": True, "content": [{"type": "text", "text": "prompt is required"}]}
            max_tokens = int(arguments.get("max_tokens", 512))

            async def runner(job: Any) -> str:
                result = await self.controller.generate(prompt, max_tokens=max_tokens)
                job.weight_wait_ms = result["weight_wait_ms"]
                job.generated_tokens = result["generated_tokens"]
                return result["text"]

            job = self.jobs.submit(model=self.controller.model_alias, prompt=prompt, runner=runner)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"job_id": job.job_id, "state": job.state, "model": job.model}
                        ),
                    }
                ]
            }
        if name == "stream_agent_job_status":
            job = self.jobs.get(str(arguments.get("job_id", "")))
            if job is None:
                return {"isError": True, "content": [{"type": "text", "text": "unknown job id"}]}
            return {"content": [{"type": "text", "text": json.dumps(job.to_dict())}]}
        if name == "stream_agent_cancel":
            ok = await self.jobs.cancel(str(arguments.get("job_id", "")))
            return {
                "content": [
                    {"type": "text", "text": "cancelled" if ok else "unknown job id"}
                ]
            }
        return {"isError": True, "content": [{"type": "text", "text": f"unknown tool {name!r}"}]}

    # -- protocol ----------------------------------------------------------
    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            self.initialized = True
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stream-agent", "version": "0.1.0"},
                },
            }
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        if method == "tools/call":
            name = str(params.get("name", ""))
            arguments = params.get("arguments") or {}
            result = await self.call_tool(name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method {method!r}"},
        }

    async def serve_stdio(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            response = await self.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


__all__ = ["MCPAdapter", "TOOLS", "PROTOCOL_VERSION"]
