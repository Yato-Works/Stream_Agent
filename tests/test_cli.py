"""CLI tests (Plan.md section 13)."""

from __future__ import annotations

import json
from pathlib import Path


from stream_agent.cli import main


def _run(capsys, *args: str) -> tuple[int, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_doctor_runs_without_a_configuration(tmp_path: Path, capsys):
    code, output = _run(capsys, "--root", str(tmp_path / "store"), "doctor")
    assert code == 0
    assert "Stream Agent doctor" in output
    assert "weight-cache limit" in output
    assert "No performance result is shown" in output


def test_cache_status_reports_the_accounting(tmp_path: Path, capsys):
    code, output = _run(
        capsys, "--root", str(tmp_path / "store"), "--json", "cache", "status"
    )
    assert code == 0
    payload = json.loads(output)
    assert payload["weight_cache_limit_bytes"] > 0
    assert payload["used_weight_bytes"] == 0


def test_model_list_is_empty_for_a_fresh_store(tmp_path: Path, capsys):
    code, output = _run(capsys, "--root", str(tmp_path / "store"), "model", "list")
    assert code == 0
    assert "alias" in output


def test_report_of_a_missing_run_fails_cleanly(tmp_path: Path, capsys):
    code, output = _run(capsys, "--root", str(tmp_path / "store"), "report", "nope")
    assert code == 1
    assert "no report found" in output


def test_config_show_and_init(tmp_path: Path, capsys):
    target = tmp_path / "stream-agent.toml"
    code, output = _run(capsys, "--root", str(tmp_path / "store"), "config", "init", str(target))
    assert code == 0
    assert target.is_file()
    assert "storage" in target.read_text(encoding="utf-8")

    code, output = _run(capsys, "--config", str(target), "--json", "config", "show")
    assert code == 0
    payload = json.loads(output)
    assert payload["storage"]["weight_cache_limit_bytes"] > 0
