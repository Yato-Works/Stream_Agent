"""Standalone, offline-runnable cache sweep.

Runs the cache limit / multi-model / cache-core test suites and prints a compact
JSON summary.  Does not require network access, a GPU, or ``torch``; it exercises
the cache control invariants (W/M budgets, reserves, eviction, pinned survival,
per-model quotas, deadlock avoidance) against the in-memory test doubles.

Usage::

    python scripts/run_cache_sweep.py
    python scripts/run_cache_sweep.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES = [
    "tests/test_cache_core.py",
    "tests/test_cache_limits.py",
    "tests/test_multi_model.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="", help="optional path to write the JSON report")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest", *TEST_FILES, "-q", "-p", "no:cacheprovider"]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    output = (proc.stdout or "") + (proc.stderr or "")

    # Parse the trailing summary line, e.g. "30 passed in 1.23s".
    passed = failed = errors = 0
    for line in output.splitlines():
        if " passed" in line or " failed" in line or " error" in line:
            # last matching line wins
            import re

            m_passed = re.search(r"(\d+)\s+passed", line)
            m_failed = re.search(r"(\d+)\s+failed", line)
            m_errors = re.search(r"(\d+)\s+error", line)
            if m_passed:
                passed = int(m_passed.group(1))
            if m_failed:
                failed = int(m_failed.group(1))
            if m_errors:
                errors = int(m_errors.group(1))

    report = {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and failed == 0 and errors == 0,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
