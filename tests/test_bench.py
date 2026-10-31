"""Tests for the benchmark runner.

``bench.py`` is the source of every number we publish, so the accounting has to
be tested as carefully as the cache itself.  These tests use the in-memory
fixture backend and never touch the network.
"""

from __future__ import annotations

import json

import pytest

from stream_agent.bench import (
    BenchCase,
    BenchResult,
    clear_cache,
    format_bench_table,
    run_case,
    run_sweep,
    write_benchmark_json,
)


@pytest.fixture()
def registered(runtime):
    """Register the fixture model so the runtime can open a session."""
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
    runtime.cache.set_model_quota("tiny", 128 * 1024, "fixed")
    return runtime


def _sample_run(**overrides):
    run = {
        "network": {"downloaded_bytes_total": 1000},
        "cache": {"evictions": 2},
        "derived": {
            "bytes_per_output_token": 25.0,
            "cache_hit_ratio": 0.5,
            "weight_wait_fraction": 0.25,
        },
        "inference": {
            "ttft_ms": 40.0,
            "tokens_per_second": 10.0,
            "generated_tokens": 4,
        },
    }
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        run[section][field] = value
    return run


def test_summary_is_empty_without_runs():
    result = BenchResult(model="tiny", cache_budget_bytes=1, cold_or_warm="cold")
    assert result.summary() == {}


def test_summary_averages_and_sums_runs():
    result = BenchResult(
        model="tiny",
        cache_budget_bytes=1,
        cold_or_warm="cold",
        runs=[
            _sample_run(),
            _sample_run(
                network__downloaded_bytes_total=500,
                inference__tokens_per_second=20.0,
                cache__evictions=1,
            ),
        ],
    )
    summary = result.summary()
    assert summary["runs"] == 2
    assert summary["downloaded_bytes_total"] == 1500
    assert summary["bytes_per_output_token"] == 25.0
    assert summary["tokens_per_second"] == 15.0
    assert summary["evictions"] == 3
    assert summary["generated_tokens"] == 8


def test_summary_ignores_null_ttft_instead_of_treating_it_as_zero():
    result = BenchResult(
        model="tiny",
        cache_budget_bytes=1,
        cold_or_warm="cold",
        runs=[
            _sample_run(inference__ttft_ms=None),
            _sample_run(inference__ttft_ms=60.0),
        ],
    )
    # A backend that cannot observe the first token reports null.  Averaging it
    # as 0 would understate TTFT by 2x for this pair.
    assert result.summary()["ttft_ms"] == 60.0


def test_summary_all_null_ttft_reports_none():
    result = BenchResult(
        model="tiny",
        cache_budget_bytes=1,
        cold_or_warm="cold",
        runs=[_sample_run(inference__ttft_ms=None)],
    )
    assert result.summary()["ttft_ms"] is None


@pytest.mark.asyncio
async def test_clear_cache_evicts_without_counting_reset_churn(registered):
    session = await registered.open_session("tiny")
    await session.generate("hello", max_tokens=2)
    await session.close()

    cache = registered.cache
    assert cache is not None
    assert cache.used_weight_bytes > 0
    cache._evicted_content_hashes["deadbeef"] = 3
    telemetry_evictions_before = cache.telemetry.cache.evictions

    await clear_cache(registered)

    assert cache.used_weight_bytes == 0
    assert list(cache._entries) == []
    assert cache._evicted_content_hashes == {}
    # The reset must not be recorded as cache thrashing.
    assert cache.telemetry.cache.evictions == telemetry_evictions_before


@pytest.mark.asyncio
async def test_clear_cache_is_a_noop_without_a_cache():
    class _Runtime:
        cache = None

    await clear_cache(_Runtime())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_case_reports_measured_run_and_restores_limits(registered, tmp_path):
    original_limit = registered.storage.weight_cache_limit_bytes
    original_cache_limit = registered.cache.storage.weight_cache_limit_bytes
    budget = 64 * 1024

    result = await run_case(
        registered,
        BenchCase(model="tiny", cache_budget_bytes=budget, prompt="hello", max_tokens=3),
    )

    assert result.cold_or_warm == "cold"
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run["inference"]["status"] == "ok"
    assert run["inference"]["generated_tokens"] > 0
    assert run["budget"]["weight_cache_limit_bytes"] == budget
    assert run["network"]["downloaded_bytes_total"] > 0
    assert run["model"]["backend_id"] == "fixture"

    # A sweep must not leak its budget into the next case or the caller.
    assert registered.storage.weight_cache_limit_bytes == original_limit
    assert registered.cache.storage.weight_cache_limit_bytes == original_cache_limit


@pytest.mark.asyncio
async def test_run_case_restores_limits_when_generation_fails(registered, monkeypatch):
    original_storage = registered.storage
    original_cache_storage = registered.cache.storage

    async def boom(self, prompt, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("stream_agent.runtime.ModelSession.generate", boom)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        await run_case(
            registered,
            BenchCase(model="tiny", cache_budget_bytes=1024, prompt="hello"),
        )

    assert registered.storage is original_storage
    assert registered.cache.storage is original_cache_storage


@pytest.mark.asyncio
async def test_run_sweep_visits_each_budget(registered):
    results = await run_sweep(
        registered,
        model="tiny",
        budgets=[64 * 1024, 96 * 1024],
        prompt="hello",
        max_tokens=2,
    )
    assert [r.cache_budget_bytes for r in results] == [64 * 1024, 96 * 1024]
    assert all(r.summary()["runs"] == 1 for r in results)


def test_format_table_marks_unknown_ttft_with_a_dash():
    result = BenchResult(
        model="tiny",
        cache_budget_bytes=1024,
        cold_or_warm="cold",
        runs=[_sample_run(inference__ttft_ms=None)],
    )
    table = format_bench_table([result])
    lines = table.splitlines()
    assert lines[0].strip().startswith("budget")
    assert "ttft ms" in lines[0]
    # The TTFT column must not read as a measured 0.
    assert lines[2].split()[5] == "-"


def test_write_benchmark_json_round_trips(tmp_path):
    result = BenchResult(
        model="tiny",
        cache_budget_bytes=1024,
        cold_or_warm="warm",
        runs=[_sample_run()],
    )
    path = write_benchmark_json([result], tmp_path / "nested" / "bench.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1"
    assert payload["results"][0]["model"] == "tiny"
    assert payload["results"][0]["cold_or_warm"] == "warm"
    assert payload["results"][0]["summary"]["downloaded_bytes_total"] == 1000
    assert payload["results"][0]["runs"] == result.runs

