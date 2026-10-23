"""Implementations of the ``run``, ``serve``, ``bench``, ``report`` and ``mcp``
commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config
from .sizes import format_size, parse_size
from .telemetry import load_report, save_report


async def cmd_run(args: argparse.Namespace, config: Config) -> int:
    from .cli_actions import print_json, with_runtime

    async def _run(runtime) -> int:
        telemetry = runtime.telemetry
        session = await runtime.open_session(
            args.model,
            backend_id=args.backend,
            materialize=args.materialize,
            device=args.device,
        )
        started = time.perf_counter()
        result = await session.generate(args.prompt, max_tokens=args.max_tokens)
        elapsed = (time.perf_counter() - started) * 1000.0
        telemetry.model_alias = args.model
        telemetry.repo = session.repo
        telemetry.revision = session.revision
        telemetry.artifact_format = session.layout.artifact_format
        telemetry.backend_id = getattr(session.backend.capabilities, "backend_id", "")
        telemetry.backend_version = getattr(session.backend.capabilities, "version", "")
        telemetry.inference.prompt_tokens = result.prompt_tokens
        telemetry.inference.generated_tokens = result.generated_tokens
        telemetry.inference.ttft_ms = result.ttft_ms
        telemetry.inference.generation_ms = result.generation_ms
        telemetry.inference.weight_wait_ms = result.weight_wait_ms
        telemetry.inference.load_weight_wait_ms = result.load_weight_wait_ms
        telemetry.inference.total_inference_ms = elapsed
        telemetry.inference.status = "ok"
        status = runtime.cache.status()
        telemetry.storage.weight_cache_limit_bytes = config.storage.weight_cache_limit_bytes
        telemetry.storage.managed_store_limit_bytes = config.storage.managed_store_limit_bytes
        telemetry.storage.current_weight_bytes = status.used_weight_bytes
        telemetry.storage.peak_weight_bytes = max(
            telemetry.storage.peak_weight_bytes, status.used_weight_bytes
        )
        telemetry.storage.peak_managed_store_bytes = max(
            telemetry.storage.peak_managed_store_bytes, status.managed_store_bytes
        )
        telemetry.set_unique_verified_bytes(status.used_weight_bytes)
        telemetry.finalize()
        report_path = save_report(telemetry, config.storage.log_root)
        if args.json:
            print_json(telemetry.to_report())
        else:
            print(result.text)
            print()
            print(
                f"[tokens {result.generated_tokens} | weight wait "
                f"{result.weight_wait_ms:.0f} ms | downloaded "
                f"{format_size(telemetry.network.downloaded_bytes)} | cache hits "
                f"{telemetry.cache.hits} / misses {telemetry.cache.misses}]"
            )
            print(f"report: {report_path}")
        await session.close()
        return 0

    return await with_runtime(config, _run)


def cmd_serve(args: argparse.Namespace, config: Config) -> int:
    from .runtime import StreamAgentRuntime
    from .server import ModelController, build_app, check_bind_policy

    async def _run() -> int:
        try:
            import uvicorn
        except ModuleNotFoundError:
            print(
                "the local endpoint requires uvicorn; install stream-agent[server]",
                file=sys.stderr,
            )
            return 2
        check_bind_policy(args.host or config.server.host, config.server.allow_remote_bind)
        runtime = StreamAgentRuntime(config)
        await runtime.start()
        controller = ModelController(
            runtime,
            model_alias=args.model,
            max_concurrent_requests=config.server.max_concurrent_requests,
            request_timeout_seconds=config.server.request_timeout_seconds,
        )
        app = build_app(runtime, controller)
        host = args.host or config.server.host
        port = args.port or config.server.port
        print(f"stream-agent serving {args.model} on http://{host}:{port}")
        try:
            server = uvicorn.Server(
                uvicorn.Config(app, host=host, port=port, log_level="warning")
            )
            await server.serve()
        finally:
            await controller.close()
            await runtime.close()
        return 0

    return asyncio.run(_run())


async def cmd_bench(args: argparse.Namespace, config: Config) -> int:
    from .bench import format_bench_table, run_sweep, write_benchmark_json
    from .cli_actions import print_json, with_runtime

    async def _run(runtime) -> int:
        budgets = []
        for raw in args.cache_sweep.split(","):
            raw = raw.strip()
            if not raw:
                continue
            budgets.append(parse_size(raw, field_name="cache budget"))
        if not budgets:
            budgets = [config.storage.weight_cache_limit_bytes]
        if args.block_size:
            runtime.storage = runtime.storage.__class__(
                **{**runtime.storage.__dict__, "block_size_bytes": int(args.block_size)}
            )
        results = await run_sweep(
            runtime,
            model=args.model,
            budgets=budgets,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            runs=args.runs,
            cold=not args.warm,
            backend=args.backend,
        )
        target = config.storage.log_root / "benchmarks" / f"bench-{int(time.time())}.json"
        write_benchmark_json(results, target)
        if args.json:
            print_json([r.summary() for r in results])
        else:
            print(format_bench_table(results))
            print(f"\nreport written to {target}")
        return 0

    return await with_runtime(config, _run)


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    path = Path(args.run).expanduser()
    if not path.is_absolute():
        candidate = config.storage.log_root / "runs" / path
        if candidate.is_file():
            path = candidate
        elif not path.suffix:
            candidate = config.storage.log_root / "runs" / f"{path}.json"
            if candidate.is_file():
                path = candidate
    if not path.is_file():
        print(f"no report found at {path}", file=sys.stderr)
        return 1
    payload = load_report(path)
    if args.format == "markdown":
        print(report_markdown(payload))
    else:
        print(json.dumps(payload, indent=2, default=str))
    return 0


def report_markdown(payload: dict[str, Any]) -> str:
    model = payload.get("model", {})
    budget = payload.get("budget", {})
    network = payload.get("network", {})
    cache = payload.get("cache", {})
    inference = payload.get("inference", {})
    derived = payload.get("derived", {})
    lines = [
        f"# Stream Agent run {payload.get('run_id', '')}",
        "",
        "| field | value |",
        "|---|---|",
        f"| repo | {model.get('repo', '')} |",
        f"| revision | {model.get('revision', '')} |",
        f"| format | {model.get('artifact_format', '')} |",
        f"| backend | {model.get('backend_id', '')} {model.get('backend_version', '')} |",
        f"| weight cache limit | {format_size(int(budget.get('weight_cache_limit_bytes') or 0))} |",
        f"| peak weight cache | {format_size(int(budget.get('peak_weight_cache_bytes') or 0))} |",
        f"| downloaded bytes | {format_size(int(network.get('downloaded_bytes_total') or 0))} |",
        f"| range requests | {network.get('range_requests')} |",
        f"| cache hits/misses | {cache.get('hits')}/{cache.get('misses')} |",
        f"| evictions | {cache.get('evictions')} |",
        f"| generated tokens | {inference.get('generated_tokens')} |",
        f"| bytes per output token | {derived.get('bytes_per_output_token')} |",
        f"| ttft ms | {inference.get('ttft_ms')} |",
        f"| tokens/s | {inference.get('tokens_per_second')} |",
        f"| weight wait fraction | {derived.get('weight_wait_fraction')} |",
        f"| status | {inference.get('status')} |",
    ]
    errors = payload.get("errors") or []
    if errors:
        lines += ["", "## Errors", ""]
        for error in errors:
            lines.append(f"- `{error.get('code')}` {error.get('message')}")
    return "\n".join(lines)


def cmd_mcp(args: argparse.Namespace, config: Config) -> int:
    from .jobs import JobController
    from .mcp import MCPAdapter
    from .runtime import StreamAgentRuntime
    from .server import ModelController

    async def _run() -> int:
        runtime = StreamAgentRuntime(config)
        await runtime.start()
        controller = ModelController(runtime, model_alias=args.model)
        jobs = JobController(result_dir=config.storage.log_root / "jobs")
        adapter = MCPAdapter(controller, jobs)
        try:
            await adapter.serve_stdio()
        finally:
            await jobs.aclose()
            await controller.close()
            await runtime.close()
        return 0

    return asyncio.run(_run())


__all__ = [
    "cmd_bench",
    "cmd_mcp",
    "cmd_report",
    "cmd_run",
    "cmd_serve",
    "report_markdown",
]
