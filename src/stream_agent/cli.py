"""Command line interface (Plan.md C0 / section 13).

The command set follows the design document:

``doctor``            environment and capability diagnostics
``model add/list``    register a pinned revision without downloading weights
``cache status``      storage accounting
``cache evict``       remove eligible blocks safely
``cache verify``      re-check metadata, hashes and orphans
``run``               one bounded local generation
``serve``             the local inference endpoint
``bench``             a cache sweep with recorded metrics
``report``            re-emit a stored run report
``mcp``               the agent-framework adapter
``config show/init``  inspect or write the configuration
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from . import cli_actions, cli_runtime
from .config_file import load_config
from .errors import StreamAgentError


def build_parser() -> argparse.ArgumentParser:
    # ``--json`` is accepted both before and after the sub-command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit JSON instead of text",
    )

    parser = argparse.ArgumentParser(
        prog="stream-agent",
        description="Storage-budgeted on-demand model runtime.",
    )
    parser.add_argument("--config", help="path to a stream-agent TOML configuration file")
    parser.add_argument("--root", help="override the managed storage root")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", parents=[common], help="check the environment and the registered models")

    model_parser = sub.add_parser("model", parents=[common], help="manage registered models")
    model_sub = model_parser.add_subparsers(dest="model_command", required=True)
    add = model_sub.add_parser(
        "add", parents=[common], help="register a pinned revision (no weight download)"
    )
    add.add_argument("repo", help="repository id, for example Qwen/Qwen3.8-27B")
    add.add_argument("--alias", default=None, help="local alias for the model")
    add.add_argument("--revision", default=None, help="commit sha or branch to pin")
    add.add_argument("--format", default="safetensors", choices=["safetensors", "gguf"])
    add.add_argument("--backend", default="", help="backend id to use for this model")
    add.add_argument("--quota-gib", type=float, default=0.0, help="per-model cache quota")
    add.add_argument("--quota-mode", default=None, choices=["fixed", "elastic"])
    add.add_argument(
        "--license-review",
        default="pending",
        choices=["pending", "required", "accepted", "rejected"],
    )
    add.add_argument("--allow-remote-code", action="store_true")
    add.add_argument("--no-probe", action="store_true", help="skip the range capability probe")
    model_sub.add_parser("list", parents=[common], help="list the registered models")

    cache_parser = sub.add_parser("cache", parents=[common], help="inspect and maintain the weight cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("status", parents=[common], help="show the storage accounting")
    evict = cache_sub.add_parser("evict", parents=[common], help="remove eligible cached blocks")
    evict.add_argument("target", help="a model alias or a block key")
    cache_sub.add_parser("verify", parents=[common], help="re-check hashes, metadata and orphan files")

    run = sub.add_parser("run", parents=[common], help="run one bounded local generation")
    run.add_argument("model", help="registered model alias")
    run.add_argument("--prompt", default="Summarize the supplied test fixture.")
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--backend", default=None)
    run.add_argument("--materialize", default="lazy", choices=["lazy", "eager"])
    run.add_argument("--device", default="cpu")

    serve = sub.add_parser("serve", parents=[common], help="start the local inference endpoint")
    serve.add_argument("--model", required=True, help="registered model alias")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    bench = sub.add_parser("bench", parents=[common], help="run a cache sweep with recorded metrics")
    bench.add_argument("--model", required=True)
    bench.add_argument("--cache-sweep", default="", help="comma separated budgets, e.g. 1GiB,3GiB")
    bench.add_argument("--prompt", default="Summarize the supplied test fixture.")
    bench.add_argument("--max-tokens", type=int, default=32)
    bench.add_argument("--runs", type=int, default=1)
    bench.add_argument("--warm", action="store_true", help="keep the cache between runs")
    bench.add_argument("--backend", default=None)
    bench.add_argument("--block-size", type=int, default=0, help="cache block size in bytes")

    report = sub.add_parser("report", parents=[common], help="re-emit a stored run report")
    report.add_argument("run", help="run id or path to a report file")
    report.add_argument("--format", default="json", choices=["json", "markdown"])

    mcp = sub.add_parser("mcp", parents=[common], help="run the MCP adapter over stdio")
    mcp.add_argument("--model", required=True)

    config_parser = sub.add_parser("config", parents=[common], help="inspect the configuration")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", parents=[common], help="show the effective configuration")
    init = config_sub.add_parser("init", parents=[common], help="write a configuration file")
    init.add_argument("path", nargs="?", default="stream-agent.toml")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Generated text may contain characters the console codec cannot encode.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - older streams
            pass
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, root_override=Path(args.root) if args.root else None)
    except StreamAgentError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2

    try:
        if args.command == "doctor":
            return asyncio.run(cli_actions.cmd_doctor(args, config))
        if args.command == "model":
            if args.model_command == "add":
                if not args.alias:
                    args.alias = args.repo.replace("/", "_")
                return asyncio.run(cli_actions.cmd_model_add(args, config))
            return asyncio.run(cli_actions.cmd_model_list(args, config))
        if args.command == "cache":
            if args.cache_command == "status":
                return asyncio.run(cli_actions.cmd_cache_status(args, config))
            if args.cache_command == "evict":
                return asyncio.run(cli_actions.cmd_cache_evict(args, config))
            return asyncio.run(cli_actions.cmd_cache_verify(args, config))
        if args.command == "run":
            return asyncio.run(cli_runtime.cmd_run(args, config))
        if args.command == "serve":
            return cli_runtime.cmd_serve(args, config)
        if args.command == "bench":
            return asyncio.run(cli_runtime.cmd_bench(args, config))
        if args.command == "report":
            return cli_runtime.cmd_report(args, config)
        if args.command == "mcp":
            return cli_runtime.cmd_mcp(args, config)
        if args.command == "config":
            if args.config_command == "show":
                return cli_actions.cmd_config_show(args, config)
            return cli_actions.cmd_config_init(args, config)
    except StreamAgentError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
