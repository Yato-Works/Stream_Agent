"""Benchmark runner and report writer (Plan.md section 16 / Appendix B).

The runner measures what the design asks for: bytes transferred per generated
token, cache hit ratio, weight wait time and the effect of the cache budget.
Cold and warm runs are reported separately and the actual starting state is
always recorded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .runtime import StreamAgentRuntime
from .telemetry import RunTelemetry, save_report


@dataclass
class BenchCase:
    model: str
    cache_budget_bytes: int
    prompt: str
    max_tokens: int = 32
    runs: int = 1
    cold: bool = True


@dataclass
class BenchResult:
    model: str
    cache_budget_bytes: int
    cold_or_warm: str
    runs: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if not self.runs:
            return {}

        def values(path: Sequence[str]) -> list[float]:
            collected: list[float] = []
            for run in self.runs:
                node: Any = run
                for key in path:
                    node = (node or {}).get(key)
                if node is not None:
                    collected.append(float(node))
            return collected

        def avg(path: Sequence[str]) -> float | None:
            data = values(path)
            if not data:
                return None
            return round(sum(data) / len(data), 4)

        network_bytes = sum(
            float((run.get("network") or {}).get("downloaded_bytes_total") or 0)
            for run in self.runs
        )
        evictions = sum(
            int((run.get("cache") or {}).get("evictions") or 0) for run in self.runs
        )
        return {
            "runs": len(self.runs),
            "downloaded_bytes_total": int(network_bytes),
            "bytes_per_output_token": avg(("derived", "bytes_per_output_token")),
            "cache_hit_ratio": avg(("derived", "cache_hit_ratio")),
            "weight_wait_fraction": avg(("derived", "weight_wait_fraction")),
            "ttft_ms": avg(("inference", "ttft_ms")),
            "tokens_per_second": avg(("inference", "tokens_per_second")),
            "generated_tokens": sum(
                int((run.get("inference") or {}).get("generated_tokens") or 0)
                for run in self.runs
            ),
            "evictions": evictions,
        }


async def clear_cache(runtime: StreamAgentRuntime) -> None:
    """Drop every cached block so the next run starts cold.

    The reset itself is not part of the measured run: its evictions are not
    counted as churn and the refetch history is cleared so that a deliberate
    cold start is not misreported as cache thrashing.
    """
    if runtime.cache is None:
        return
    telemetry = runtime.cache.telemetry
    runtime.cache.telemetry = None
    try:
        for entry in list(runtime.cache._entries.values()):
            await runtime.cache.evict_block(entry.block_key, reason="bench_reset")
    finally:
        runtime.cache.telemetry = telemetry
    runtime.cache._used_bytes = runtime.cache._compute_unique_bytes()
    runtime.cache._evicted_content_hashes.clear()


async def run_case(
    runtime: StreamAgentRuntime,
    case: BenchCase,
    *,
    backend: str | None = None,
    materialize: str = "lazy",
    device: str = "cpu",
) -> BenchResult:
    """Run one benchmark case with the given cache budget."""
    assert runtime.cache is not None
    result = BenchResult(
        model=case.model,
        cache_budget_bytes=case.cache_budget_bytes,
        cold_or_warm="cold" if case.cold else "warm",
    )
    original_limit = runtime.cache.storage.weight_cache_limit_bytes
    original_block_size = runtime.storage.block_size_bytes
    for index in range(case.runs):
        telemetry = RunTelemetry(run_id=f"bench-{int(time.time()*1000)}-{index}")
        runtime.telemetry = telemetry
        runtime.cache.telemetry = telemetry
        runtime.cache.run_id = telemetry.run_id
        runtime.cache.set_weight_limit(case.cache_budget_bytes)
        runtime.storage = runtime.storage.__class__(
            **{**runtime.storage.__dict__, "weight_cache_limit_bytes": case.cache_budget_bytes}
        )
        if case.cold:
            await clear_cache(runtime)
        session = await runtime.open_session(
            case.model, backend_id=backend, materialize=materialize, device=device
        )
        generation = await session.generate(case.prompt, max_tokens=case.max_tokens)
        telemetry.inference.prompt_tokens = generation.prompt_tokens
        telemetry.inference.generated_tokens = generation.generated_tokens
        telemetry.inference.ttft_ms = generation.ttft_ms
        telemetry.inference.generation_ms = generation.generation_ms
        telemetry.inference.weight_wait_ms = generation.weight_wait_ms
        telemetry.inference.load_weight_wait_ms = generation.load_weight_wait_ms
        telemetry.inference.total_inference_ms = generation.generation_ms
        telemetry.inference.status = "ok"
        telemetry.model_alias = case.model
        telemetry.repo = session.repo
        telemetry.revision = session.revision
        telemetry.artifact_format = session.layout.artifact_format
        telemetry.backend_id = getattr(session.backend.capabilities, "backend_id", "")
        telemetry.backend_version = getattr(session.backend.capabilities, "version", "")
        telemetry.cold_or_warm = result.cold_or_warm
        telemetry.storage.weight_cache_limit_bytes = case.cache_budget_bytes
        telemetry.storage.managed_store_limit_bytes = runtime.storage.managed_store_limit_bytes
        status = runtime.cache.status()
        telemetry.storage.current_weight_bytes = status.used_weight_bytes
        telemetry.storage.peak_weight_bytes = max(
            telemetry.storage.peak_weight_bytes, status.used_weight_bytes
        )
        telemetry.storage.peak_managed_store_bytes = max(
            telemetry.storage.peak_managed_store_bytes, status.managed_store_bytes
        )
        telemetry.set_unique_verified_bytes(status.used_weight_bytes)
        telemetry.quota_policy = runtime.storage.quota_policy
        telemetry.eviction_policy = runtime.storage.eviction_policy
        telemetry.finalize()
        report = telemetry.to_report()
        result.runs.append(report)
        save_report(telemetry, runtime.storage.log_root)
        await session.close()
    runtime.storage = runtime.storage.__class__(
        **{**runtime.storage.__dict__, "weight_cache_limit_bytes": original_limit}
    )
    runtime.cache.set_weight_limit(original_limit)
    runtime.storage = runtime.storage.__class__(
        **{**runtime.storage.__dict__, "block_size_bytes": original_block_size}
    )
    return result


async def run_sweep(
    runtime: StreamAgentRuntime,
    *,
    model: str,
    budgets: Sequence[int],
    prompt: str,
    max_tokens: int = 32,
    runs: int = 1,
    cold: bool = True,
    backend: str | None = None,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for budget in budgets:
        results.append(
            await run_case(
                runtime,
                BenchCase(
                    model=model,
                    cache_budget_bytes=budget,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    runs=runs,
                    cold=cold,
                ),
                backend=backend,
            )
        )
    return results


def format_bench_table(results: list[BenchResult]) -> str:
    header = (
        f"{'budget':>10} {'mode':>6} {'dl bytes':>12} {'bytes/token':>12} "
        f"{'hit ratio':>10} {'ttft ms':>10} {'tok/s':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        summary = result.summary()
        lines.append(
            f"{result.cache_budget_bytes:>10} {result.cold_or_warm:>6} "
            f"{summary.get('downloaded_bytes_total', 0):>12} "
            f"{str(summary.get('bytes_per_output_token', '-')):>12} "
            f"{str(summary.get('cache_hit_ratio', '-')):>10} "
            f"{str(summary.get('ttft_ms', '-')):>10} "
            f"{str(summary.get('tokens_per_second', '-')):>8}"
        )
    return "\n".join(lines)


def write_benchmark_json(results: list[BenchResult], path: Path) -> Path:
    payload = {
        "schema_version": "0.1",
        "results": [
            {
                "model": r.model,
                "cache_budget_bytes": r.cache_budget_bytes,
                "cold_or_warm": r.cold_or_warm,
                "summary": r.summary(),
                "runs": r.runs,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "BenchCase",
    "BenchResult",
    "clear_cache",
    "format_bench_table",
    "run_case",
    "run_sweep",
    "write_benchmark_json",
]
