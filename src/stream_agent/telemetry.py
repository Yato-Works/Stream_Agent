"""Local telemetry and reporting primitives (Plan.md C9 / section 16.3).

All measurement is local by default.  A report must distinguish measured fields
from estimates and unavailable data, therefore every metric in this module is
either an observed value or ``None`` -- never a guessed number.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

GIB = 1024**3
MIB = 1024**2


@dataclass
class NetworkCounters:
    range_requests: int = 0
    full_object_requests: int = 0
    #: Validated Range responses actually received (each carries HTTP 206).
    range_request_samples: list[dict[str, Any]] = field(default_factory=list)
    downloaded_bytes: int = 0
    downloaded_bytes_failed: int = 0
    #: Bytes received for the weight payload (the blocks the cache stores).
    weight_downloaded_bytes: int = 0
    #: Bytes received for small metadata files (config, tokenizer, index).
    metadata_downloaded_bytes: int = 0
    retries: int = 0
    source_errors: int = 0
    rate_limited: int = 0
    auth_failures: int = 0
    destinations: list[str] = field(default_factory=list)


@dataclass
class CacheCounters:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    prefetched_bytes: int = 0
    useful_prefetched_bytes: int = 0
    wasted_prefetch_bytes: int = 0
    promotions: int = 0
    integrity_failures: int = 0
    budget_rejections: int = 0
    coalesced_requests: int = 0
    single_flight_hits: int = 0
    #: Bytes written into the persistent cache (promotions).
    cache_write_bytes: int = 0
    #: Bytes served from the cache without a network transfer.
    reused_bytes: int = 0
    #: Bytes that had to be fetched again after an eviction (thrashing).
    refetched_bytes: int = 0
    #: Bytes that were read but were not part of the requested tensor.
    wasted_prefetch_bytes: int = 0


@dataclass
class InferenceCounters:
    prompt_tokens: int | None = None
    generated_tokens: int = 0
    ttft_ms: float | None = None
    generation_ms: float | None = None
    tokens_per_second: float | None = None
    weight_wait_ms: float = 0.0
    load_weight_wait_ms: float = 0.0
    total_inference_ms: float = 0.0
    status: str = "not_run"


@dataclass
class StorageSnapshot:
    weight_cache_limit_bytes: int = 0
    managed_store_limit_bytes: int | None = None
    current_weight_bytes: int = 0
    reserved_weight_bytes: int = 0
    peak_weight_bytes: int = 0
    peak_managed_store_bytes: int = 0
    metadata_bytes: int = 0
    log_bytes: int = 0
    temporary_bytes: int = 0


@dataclass
class RunTelemetry:
    """Aggregate counters for one run (one model session or benchmark)."""

    run_id: str = ""
    model_alias: str = ""
    repo: str = ""
    revision: str = ""
    artifact_format: str = ""
    backend_id: str = ""
    backend_version: str = ""
    quantization: str | None = None
    network: NetworkCounters = field(default_factory=NetworkCounters)
    cache: CacheCounters = field(default_factory=CacheCounters)
    inference: InferenceCounters = field(default_factory=InferenceCounters)
    storage: StorageSnapshot = field(default_factory=StorageSnapshot)
    quota_policy: str = ""
    eviction_policy: str = ""
    workload: str = ""
    cold_or_warm: str = "cold"
    network_profile: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    # -- recording helpers -------------------------------------------------
    def record_download(
        self, num_bytes: int, *, failed: bool = False, kind: str = "weight"
    ) -> None:
        """Record bytes received from the source.

        ``kind`` is ``"weight"`` for weight payload blocks and ``"metadata"``
        for the small metadata files.  The two are reported separately because
        only the weight payload is bound by the cache budget.
        """
        self.network.downloaded_bytes += int(num_bytes)
        if kind == "metadata":
            self.network.metadata_downloaded_bytes += int(num_bytes)
        else:
            self.network.weight_downloaded_bytes += int(num_bytes)
        if failed:
            self.network.downloaded_bytes_failed += int(num_bytes)

    def record_error(self, code: str, message: str, **details: Any) -> None:
        self.errors.append({"code": code, "message": message, "details": details})
        self.network.source_errors += 1

    def finalize(self) -> "RunTelemetry":
        self.finished_at = time.time()
        inf = self.inference
        if inf.generation_ms and inf.generation_ms > 0 and inf.generated_tokens:
            inf.tokens_per_second = round(inf.generated_tokens / (inf.generation_ms / 1000.0), 4)
        return self

    # -- derived metrics ---------------------------------------------------
    @property
    def bytes_per_output_token(self) -> float | None:
        """Weight payload bytes received per generated output token.

        Metadata bytes are excluded: the tokenizer is downloaded once and is
        not part of the streaming cost of the model.
        """
        if not self.inference.generated_tokens:
            return None
        return round(
            self.network.weight_downloaded_bytes / self.inference.generated_tokens, 2
        )

    @property
    def weight_wait_fraction(self) -> float | None:
        total = self.inference.total_inference_ms
        if not total:
            return None
        return round(self.inference.weight_wait_ms / total, 4)

    @property
    def useful_prefetch_ratio(self) -> float | None:
        if not self.cache.prefetched_bytes:
            return None
        return round(self.cache.useful_prefetched_bytes / self.cache.prefetched_bytes, 4)

    @property
    def cache_hit_ratio(self) -> float | None:
        total = self.cache.hits + self.cache.misses
        if not total:
            return None
        return round(self.cache.hits / total, 4)

    @property
    def unique_verified_bytes(self) -> int:
        # Verified-and-retained bytes are reported by the cache manager; the
        # telemetry copy is refreshed at the end of a run.
        return self._unique_verified_bytes

    _unique_verified_bytes: int = 0

    def set_unique_verified_bytes(self, value: int) -> None:
        self._unique_verified_bytes = int(value)

    @property
    def network_bytes_per_cache_write_byte(self) -> float | None:
        """How many network bytes were needed per byte kept in the cache."""
        if not self.cache.cache_write_bytes:
            return None
        return round(self.network.downloaded_bytes / self.cache.cache_write_bytes, 4)

    @property
    def useful_reuse_ratio(self) -> float | None:
        """Share of the served bytes that came from the cache, not the network."""
        total = self.cache.reused_bytes + self.network.downloaded_bytes
        if not total:
            return None
        return round(self.cache.reused_bytes / total, 4)

    @property
    def failure_breakdown(self) -> dict[str, int]:
        """Count of failures per error category (Plan.md section 16.3)."""
        counts: dict[str, int] = {}
        for error in self.errors:
            code = str(error.get("code", "UNKNOWN"))
            counts[code] = counts.get(code, 0) + 1
        return counts

    # -- serialization -----------------------------------------------------
    def to_report(self) -> dict[str, Any]:
        """Report shape modelled on Appendix B of the design document."""
        return {
            "schema_version": "0.1",
            "run_id": self.run_id,
            "model": {
                "repo": self.repo,
                "revision": self.revision,
                "artifact_format": self.artifact_format,
                "quantization": self.quantization,
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
            },
            "budget": {
                "weight_cache_limit_bytes": self.storage.weight_cache_limit_bytes,
                "managed_store_limit_bytes": self.storage.managed_store_limit_bytes,
                "peak_weight_cache_bytes": self.storage.peak_weight_bytes,
                "peak_managed_store_bytes": self.storage.peak_managed_store_bytes,
            },
            "network": {
                "downloaded_bytes_total": self.network.downloaded_bytes,
                "weight_downloaded_bytes": self.network.weight_downloaded_bytes,
                "metadata_downloaded_bytes": self.network.metadata_downloaded_bytes,
                "downloaded_bytes_failed": self.network.downloaded_bytes_failed,
                "range_requests": self.network.range_requests,
                "full_object_requests": self.network.full_object_requests,
                "range_request_samples": list(self.network.range_request_samples),
                "retries": self.network.retries,
                "source_errors": self.network.source_errors,
                "destinations": list(self.network.destinations),
                "profile": self.network_profile,
            },
            "cache": {
                "hits": self.cache.hits,
                "misses": self.cache.misses,
                "evictions": self.cache.evictions,
                "evicted_bytes": self.cache.evicted_bytes,
                "prefetched_bytes": self.cache.prefetched_bytes,
                "useful_prefetched_bytes": self.cache.useful_prefetched_bytes,
                "cache_write_bytes": self.cache.cache_write_bytes,
                "reused_bytes": self.cache.reused_bytes,
                "refetched_bytes": self.cache.refetched_bytes,
                "cold_or_warm": self.cold_or_warm,
            },
            "inference": {
                "prompt_tokens": self.inference.prompt_tokens,
                "generated_tokens": self.inference.generated_tokens,
                "ttft_ms": self.inference.ttft_ms,
                "generation_ms": self.inference.generation_ms,
                "tokens_per_second": self.inference.tokens_per_second,
                "weight_wait_ms": round(self.inference.weight_wait_ms, 3),
                "load_weight_wait_ms": round(self.inference.load_weight_wait_ms, 3),
                "status": self.inference.status,
            },
            "derived": {
                "bytes_per_output_token": self.bytes_per_output_token,
                "network_bytes_per_cache_write_byte": self.network_bytes_per_cache_write_byte,
                "useful_reuse_ratio": self.useful_reuse_ratio,
                "cache_hit_ratio": self.cache_hit_ratio,
                "weight_wait_fraction": self.weight_wait_fraction,
                "useful_prefetch_ratio": self.useful_prefetch_ratio,
                "unique_verified_bytes": self.unique_verified_bytes,
                "cache_churn_bytes": self.cache.evicted_bytes,
                "failure_breakdown": self.failure_breakdown,
            },
            "environment": describe_environment(),
            "errors": self.errors,
            "notes": self.notes,
        }


def describe_environment() -> dict[str, Any]:
    """Best-effort, local-only environment description."""
    env: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": None,
        "gpu": None,
        "vram_bytes": None,
    }
    try:
        import psutil  # type: ignore

        env["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:  # pragma: no cover - depends on local hardware
        import torch  # type: ignore

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_bytes"] = int(
                torch.cuda.get_device_properties(0).total_memory
            )
    except Exception:
        pass
    return env


class TelemetryBus:
    """Fan-out point for runtime events.

    The cache manager and the provider push events here; the CLI, the server
    and the benchmark runner subscribe.  Keeping this tiny avoids a dependency
    on an external metrics stack, which the design explicitly does not want.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[str, dict[str, Any]], None]] = []
        self.telemetry = RunTelemetry()

    def subscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        for callback in list(self._subscribers):
            try:
                callback(event, payload)
            except Exception:  # pragma: no cover - subscriber errors must not break the runtime
                pass


def save_report(telemetry: RunTelemetry, directory: str | os.PathLike[str]) -> Path:
    """Persist a run report as JSON inside the managed log directory."""
    target = Path(directory).expanduser() / "runs"
    target.mkdir(parents=True, exist_ok=True)
    run_id = telemetry.run_id or f"run-{int(time.time()*1000)}"
    path = target / f"{run_id}.json"
    path.write_text(json.dumps(telemetry.to_report(), indent=2, sort_keys=False), encoding="utf-8")
    return path


def load_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


__all__ = [
    "RunTelemetry",
    "NetworkCounters",
    "CacheCounters",
    "InferenceCounters",
    "StorageSnapshot",
    "TelemetryBus",
    "describe_environment",
    "save_report",
    "load_report",
    "asdict",
]
