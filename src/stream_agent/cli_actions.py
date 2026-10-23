"""Implementations of the ``doctor``, ``model``, ``cache`` and ``config`` commands."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .config import Config
from .errors import StreamAgentError
from .sizes import format_size
from .telemetry import describe_environment


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def table(rows: list[list[Any]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


async def with_runtime(config: Config, coro_factory):
    """Open a runtime, run the action and close it again."""
    from .runtime import StreamAgentRuntime

    runtime = StreamAgentRuntime(config)
    await runtime.start()
    try:
        return await coro_factory(runtime)
    finally:
        await runtime.close()


# ---------------------------------------------------------------- doctor ---
async def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    from .backend.registry import available_backends
    from .runtime import StreamAgentRuntime

    runtime = StreamAgentRuntime(config)
    recovery = await runtime.start()
    try:
        storage = config.storage
        status = runtime.cache.status() if runtime.cache else None
        token_configured = bool(config.source.resolve_token())
        source_state = "configured" if token_configured else "not configured"
        environment = describe_environment()

        lines = [
            "Stream Agent doctor",
            "",
            "Storage",
            f"  managed-store limit:       {format_size(storage.managed_store_limit_bytes or 0)}"
            + ("" if storage.managed_store_limit_bytes else " (unset)"),
            f"  weight-cache limit:        {format_size(storage.weight_cache_limit_bytes)}",
            "  metadata/staging reserve: "
            f"{format_size(storage.metadata_reserve_bytes + storage.staging_reserve_bytes)}",
            f"  current weight cache:      {format_size(status.used_weight_bytes if status else 0)}",
            f"  free headroom:             {format_size(status.free_weight_bytes if status else 0)}",
            f"  managed root:              {storage.root}",
            f"  cache block size:          {format_size(storage.block_size_bytes)}",
            "",
            "Memory (working set, not the cache budget)",
            f"  ram available:             {format_size(environment.get('ram_bytes') or 0)}",
            f"  accelerator:               {environment.get('gpu') or 'cpu only'} "
            f"({format_size(environment.get('vram_bytes') or 0)})",
            "",
            "Network",
            "  source adapter:            Hugging Face Hub/Xet",
            f"  credentials:               {source_state} (env {config.source.token_env})",
            "  remote inference:          disabled",
            "",
            "Models",
        ]
        models = runtime.db.list_models() if runtime.db else []
        if not models:
            lines.append("  (no model registered yet)")
        for record in models:
            lines.append(f"  {record['alias']}")
            lines.append(f"    repo:                    {record['repo']}")
            lines.append(f"    revision:                {record['revision']}")
            lines.append(f"    format:                  {record['artifact_format']}")
            lines.append(f"    backend:                 {record['backend'] or '(unset)'}")
            lines.append(f"    quota:                   {format_size(int(record['quota_bytes']))}")
            lines.append(f"    license review:          {record['license_review']}")
            lines.append(f"    source access:           {await probe_source(runtime, record)}")
        lines += [
            "",
            "Backends",
            f"  available:                 {', '.join(available_backends()) or '(none)'}",
            "",
            "Readiness",
            f"  metadata lookup:           {'pass' if models else 'not run'}",
            "  range fixture:             see `python -m pytest -m network`",
            "  backend compatibility:     not established",
            "  end-to-end generation:     not run",
            "",
            "No performance result is shown until an actual run completes.",
        ]
        if recovery.get("orphan_blocks_removed") or recovery.get("removed_entries"):
            lines.append(
                f"cache recovery: {recovery['removed_entries']} stale entries removed, "
                f"{recovery['orphan_blocks_removed']} orphan blocks removed"
            )
        print("\n".join(lines))
        return 0
    finally:
        await runtime.close()


async def probe_source(runtime, record: dict) -> str:
    """Check that the source really serves ranges for this revision."""
    try:
        revision = await runtime.source.list_files(
            str(record["repo"]), str(record["revision"])
        )
    except StreamAgentError as exc:
        return f"fail ({exc.code})"
    except Exception as exc:  # noqa: BLE001
        return f"fail ({type(exc).__name__})"
    weight_files = [
        name for name in sorted(revision.files) if name.endswith(".safetensors")
    ]
    if not weight_files:
        return "no weight file in the revision"
    support = await runtime.source.probe_range_support(
        revision.repo,
        revision.commit,
        weight_files[0],
        file_size=revision.files[weight_files[0]].size,
    )
    return "pass (206 partial content)" if support.supported else f"fail ({support.reason})"


# ----------------------------------------------------------------- model ---
async def cmd_model_add(args: argparse.Namespace, config: Config) -> int:
    async def _run(runtime) -> int:
        registration = await runtime.register_model(
            alias=args.alias,
            repo=args.repo,
            ref=args.revision or "main",
            artifact_format=args.format,
            backend=args.backend or "",
            quota_bytes=int(args.quota_gib * (1024**3)),
            quota_mode=args.quota_mode,
            license_review=args.license_review,
            allow_remote_code=args.allow_remote_code,
            probe=not args.no_probe,
        )
        if args.json:
            print_json(registration.to_dict())
            return 0
        print(f"registered model alias: {registration.alias}")
        print(f"  repo:            {registration.repo}")
        print(f"  pinned revision: {registration.revision}")
        print(f"  format:          {registration.artifact_format}")
        print(f"  license id:      {registration.license_id or '(unknown)'}")
        print(f"  license review:  {registration.license_review}")
        print(f"  gated:           {registration.gated}")
        print(f"  files:           {registration.file_count}")
        print(f"  total size:      {format_size(registration.total_weight_bytes)}")
        support = registration.range_support or {}
        print(
            f"  range access:    {'verified' if support.get('supported') else 'not verified'}"
            f" ({support.get('reason', '')})"
        )
        if registration.auxiliary_files:
            total = sum(registration.auxiliary_files.values())
            print(
                f"  auxiliary files: {len(registration.auxiliary_files)} "
                f"({format_size(total)}) -- these are budgeted separately from the shards"
            )
            for name, size in sorted(
                registration.auxiliary_files.items(), key=lambda kv: -kv[1]
            )[:10]:
                print(f"    {name:<50} {format_size(size)}")
        metadata_total = sum(registration.metadata_files.values())
        print(
            f"  metadata files:  {len(registration.metadata_files)} "
            f"({format_size(metadata_total)})"
        )
        if registration.requires_custom_code:
            print("  custom code:     required (inference stays blocked until reviewed)")
        print("  no weight bytes were downloaded")
        return 0

    return await with_runtime(config, _run)


async def cmd_model_list(args: argparse.Namespace, config: Config) -> int:
    async def _run(runtime) -> int:
        records = runtime.db.list_models() if runtime.db else []
        if args.json:
            print_json(records)
            return 0
        rows = [
            [
                record["alias"],
                record["repo"],
                str(record["revision"])[:12],
                record["artifact_format"],
                record["backend"] or "-",
                format_size(int(record["quota_bytes"])),
                record["quota_mode"],
                record["license_review"],
            ]
            for record in records
        ]
        print(
            table(
                rows,
                ["alias", "repo", "revision", "format", "backend", "quota", "mode", "license"],
            )
        )
        return 0

    return await with_runtime(config, _run)


# ----------------------------------------------------------------- cache ---
async def cmd_cache_status(args: argparse.Namespace, config: Config) -> int:
    async def _run(runtime) -> int:
        status = runtime.cache.status()
        if args.json:
            print_json(status.to_dict())
            return 0
        print("Stream Agent cache status")
        print(f"  weight cache limit:  {format_size(status.weight_cache_limit_bytes)}")
        managed = (
            format_size(status.managed_store_limit_bytes)
            if status.managed_store_limit_bytes
            else "(unset)"
        )
        print(f"  managed store limit: {managed}")
        print(f"  used (payload):      {format_size(status.used_weight_bytes)}")
        print(f"  reserved (in flight):{format_size(status.reserved_weight_bytes):>12}")
        print(f"  free headroom:       {format_size(status.free_weight_bytes)}")
        print(f"  managed store bytes: {format_size(status.managed_store_bytes)}")
        print(f"  metadata footprint:  {format_size(status.metadata_bytes)}")
        print(f"  log footprint:       {format_size(status.log_bytes)}")
        print(f"  temporary bytes:     {format_size(status.temporary_bytes)}")
        print(f"  entries:             {status.entry_count}")
        print(f"  pinned entries:      {status.pinned_count}")
        if status.per_model:
            print()
            rows = [
                [
                    alias,
                    format_size(info["quota_bytes"]),
                    info["quota_mode"],
                    format_size(info["used_bytes"]),
                    str(info["entry_count"]),
                ]
                for alias, info in sorted(status.per_model.items())
            ]
            print(table(rows, ["model", "quota", "mode", "used", "entries"]))
        return 0

    return await with_runtime(config, _run)


async def cmd_cache_evict(args: argparse.Namespace, config: Config) -> int:
    async def _run(runtime) -> int:
        target = args.target
        if target in runtime.cache.status().per_model:
            report = await runtime.cache.evict_model(target, reason="manual")
        else:
            removed = await runtime.cache.evict_block(target, reason="manual")
            report = {"entries": 1 if removed else 0, "bytes": 0}
        if args.json:
            print_json(report)
        else:
            print(
                f"evicted {report['entries']} entries ({format_size(report['bytes'])}) "
                f"from {target!r}"
            )
        return 0

    return await with_runtime(config, _run)


async def cmd_cache_verify(args: argparse.Namespace, config: Config) -> int:
    async def _run(runtime) -> int:
        report = await runtime.cache.verify_all()
        if args.json:
            print_json(report)
        else:
            print(
                f"checked {report['checked']} blocks: {report['corrupted']} corrupted, "
                f"{report['missing']} missing"
            )
        return 0

    return await with_runtime(config, _run)


# ---------------------------------------------------------------- config ---
def cmd_config_show(args: argparse.Namespace, config: Config) -> int:
    payload = {
        "path": str(config.path) if config.path else None,
        "storage": {
            "root": str(config.storage.root),
            "weight_cache_limit_bytes": config.storage.weight_cache_limit_bytes,
            "managed_store_limit_bytes": config.storage.managed_store_limit_bytes,
            "metadata_reserve_bytes": config.storage.metadata_reserve_bytes,
            "staging_reserve_bytes": config.storage.staging_reserve_bytes,
            "quota_policy": config.storage.quota_policy,
            "eviction_policy": config.storage.eviction_policy,
            "block_size_bytes": config.storage.block_size_bytes,
        },
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "allow_remote_bind": config.server.allow_remote_bind,
        },
        "models": {alias: m.repo for alias, m in config.models.items()},
    }
    if args.json:
        print_json(payload)
    else:
        print(f"config file: {payload['path'] or '(defaults)'}")
        print(f"storage root: {payload['storage']['root']}")
        print(
            f"weight cache limit: {format_size(config.storage.weight_cache_limit_bytes)} | "
            f"managed store limit: "
            f"{format_size(config.storage.managed_store_limit_bytes or 0)}"
        )
        for alias, repo in payload["models"].items():
            print(f"model {alias}: {repo}")
    return 0


def cmd_config_init(args: argparse.Namespace, config: Config) -> int:
    from .config_file import write_config

    target = write_config(config, args.path)
    print(f"wrote {target}")
    return 0


__all__ = [
    "cmd_cache_evict",
    "cmd_cache_status",
    "cmd_cache_verify",
    "cmd_config_init",
    "cmd_config_show",
    "cmd_doctor",
    "cmd_model_add",
    "cmd_model_list",
    "print_json",
    "table",
    "with_runtime",
]
