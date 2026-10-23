"""Cache-size benchmark: compare real-model E2E across cache limits.

For each cache limit (default 512 KiB, 1, 2, 4, 8 MiB) this runs
``scripts/real_model_e2e.py`` twice against an isolated storage root:

* ``cold`` - fresh storage root (first fetch, full Range retrieval).
* ``warm`` - same storage root, no clearing (re-run, expect cache reuse).

Records bytes transferred, range/full-object counts, cache hits/misses, load and
generation time, bytes/token, and peak cache usage vs. budget.  All values are
measured, never estimated.

Usage::

    python scripts/cache_size_benchmark.py
    python scripts/cache_size_benchmark.py --limits 524288 1048576 --max-tokens 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "real_model_e2e.py"
PYTHON = sys.executable


def run_one(root: Path, cache_limit: int, max_tokens: int, device: str, dtype: str) -> dict:
    out_file = ROOT / "docs" / "benchmarks" / "_bench_tmp.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()
    cmd = [
        PYTHON,
        str(SCRIPT),
        "--root",
        str(root),
        "--device",
        device,
        "--dtype",
        dtype,
        "--materialize",
        "eager",
        "--max-tokens",
        str(max_tokens),
        "--cache-limit",
        str(cache_limit),
        "--store-limit",
        str(2 * 1024 ** 3),
        "--out",
        str(out_file),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    wall = round(time.perf_counter() - started, 3)
    if not out_file.exists():
        return {
            "ok": False,
            "returncode": proc.returncode,
            "wall_seconds": wall,
            "stderr_tail": (proc.stderr or "")[-500:],
        }
    data = json.loads(out_file.read_text(encoding="utf-8"))
    data["_wall_seconds"] = wall
    data["_returncode"] = proc.returncode
    gen_tokens = data.get("generated_tokens") or 0
    downloaded = data.get("downloaded_bytes") or 0
    data["_bytes_per_token"] = round(downloaded / gen_tokens, 2) if gen_tokens else None
    data["_within_budget"] = (
        data.get("cache_used_bytes", 0) <= data.get("cache_limit_bytes", 0)
        if data.get("cache_limit_bytes")
        else None
    )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=[512 * 1024, 1 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024],
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out", default="docs/benchmarks/cache-size-benchmark.json")
    args = parser.parse_args()

    base_root = Path.home() / ".stream-agent-bench"
    if base_root.exists():
        shutil.rmtree(base_root)
    base_root.mkdir(parents=True)

    report = {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "device": args.device,
        "dtype": args.dtype,
        "materialize": "eager",
        "max_tokens": args.max_tokens,
        "limits": [],
    }

    for limit in args.limits:
        limit_root = base_root / ("limit_" + str(limit))
        limit_root.mkdir(parents=True, exist_ok=True)
        entry = {"cache_limit_bytes": limit, "cold": None, "warm": None}

        cold_root = limit_root / "cold"
        cold_root.mkdir(parents=True, exist_ok=True)
        entry["cold"] = run_one(cold_root, limit, args.max_tokens, args.device, args.dtype)

        entry["warm"] = run_one(cold_root, limit, args.max_tokens, args.device, args.dtype)

        report["limits"].append(entry)
        print("limit=" + str(limit) + " done", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
