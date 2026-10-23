"""Runtime wiring: one place that assembles config, source, cache and backend.

The runtime owns the lifetime of everything that lives between CLI commands
(the metadata database, the block store) and of everything that lives inside
one command (a model session with its backend and provider).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend.base import GenerationResult, InferenceBackend, LoadContext
from .backend.registry import create_backend, register_builtin_backends
from .cache.db import MetadataDB
from .cache.manager import CacheManager
from .cache.store import BlockStore
from .catalog import ModelCatalog
from .config import Config, StorageConfig
from .errors import BackendCapabilityMissing, ModelNotRegistered
from .layout.model_map import ModelLayout, build_layout
from .prefetch import PrefetchScheduler
from .provider import RemoteWeightProvider
from .sources.base import SourceAdapter
from .sources.hf_hub import HuggingFaceSource
from .telemetry import RunTelemetry


@dataclass
class ModelSession:
    """An open model: pinned revision, layout, provider and backend."""

    alias: str
    repo: str
    revision: str
    layout: ModelLayout
    provider: RemoteWeightProvider
    backend: InferenceBackend
    handle: Any = None
    prefetch: PrefetchScheduler | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    opened_at: float = field(default_factory=time.time)

    async def close(self) -> None:
        if self.prefetch is not None:
            await self.prefetch.cancel_all()
        if self.backend is not None:
            await self.backend.unload()

    async def generate(self, prompt: str, **params: Any) -> GenerationResult:
        return await self.backend.generate(self.handle, prompt, **params)

    def describe(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "repo": self.repo,
            "revision": self.revision,
            "session_id": self.session_id,
            "backend": getattr(self.backend, "capabilities", None).backend_id
            if getattr(self.backend, "capabilities", None)
            else "",
            "tensors": len(self.layout.tensors),
        }


class StreamAgentRuntime:
    """Owns the managed store, the metadata database and the model sessions."""

    def __init__(
        self,
        config: Config,
        *,
        source: SourceAdapter | None = None,
        telemetry: RunTelemetry | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.storage: StorageConfig = config.storage
        self.run_id = run_id or f"run-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        self.telemetry = telemetry or RunTelemetry(run_id=self.run_id)
        self.telemetry.run_id = self.run_id
        self._source = source
        self._owns_source = source is None
        self.db: MetadataDB | None = None
        self.store: BlockStore | None = None
        self.cache: CacheManager | None = None
        self.catalog: ModelCatalog | None = None
        self._sessions: list[ModelSession] = []
        register_builtin_backends()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> dict[str, int]:
        for directory in (
            self.storage.root,
            self.storage.blocks_root,
            self.storage.tmp_root,
            self.storage.metadata_root,
            self.storage.log_root,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.db = MetadataDB(self.storage.db_path)
        self.store = BlockStore(self.storage.blocks_root)
        self.cache = CacheManager(
            self.storage,
            self.db,
            self.store,
            telemetry=self.telemetry,
            run_id=self.run_id,
            quota_policy=self.storage.quota_policy,
        )
        if self._source is None:
            self._source = HuggingFaceSource(self.config.source, telemetry=self._on_source_event)
            self._owns_source = True
            endpoint = getattr(self._source, "endpoint", "")
            if endpoint:
                # Only the host is recorded; credentials and signed query
                # parameters are never part of the report.
                self.telemetry.network.destinations.append(
                    endpoint.split("://", 1)[-1].split("/", 1)[0]
                )
        self.catalog = ModelCatalog(self._source, self.db, telemetry=self.telemetry)
        recovery = await self.cache.recover()
        for record in self.db.list_models():
            self.cache.set_model_quota(
                str(record["alias"]),
                int(record.get("quota_bytes") or 0),
                str(record.get("quota_mode") or self.storage.quota_policy),
            )
        return recovery

    async def close(self) -> None:
        for session in list(self._sessions):
            await session.close()
        self._sessions.clear()
        if self._source is not None and self._owns_source:
            await self._source.aclose()
        if self.db is not None:
            self.db.close()

    # -- accessors ---------------------------------------------------------
    def _on_source_event(self, event: str, payload: dict[str, Any]) -> None:
        """Fold source level events into the run telemetry."""
        telemetry = self.telemetry
        if event == "full_object":
            telemetry.network.full_object_requests += 1
            telemetry.record_error(
                "RANGE_UNSUPPORTED",
                "the source answered a range request with a full object; the body was discarded",
                **payload,
            )
        elif event == "retry":
            telemetry.network.retries += 1

    @property
    def source(self) -> SourceAdapter:
        if self._source is None:
            raise RuntimeError("the runtime has not been started")
        return self._source

    # -- sessions ----------------------------------------------------------
    async def open_session(
        self,
        alias: str,
        *,
        backend_id: str | None = None,
        materialize: str = "lazy",
        device: str = "cpu",
        dtype: str = "float32",
        allow_remote_code: bool = False,
        extra_options: dict[str, Any] | None = None,
    ) -> ModelSession:
        if self.cache is None or self.catalog is None:
            raise RuntimeError("the runtime has not been started")
        record = self.catalog.registered(alias)
        ModelCatalog.assert_inference_allowed(record, allow_remote_code=allow_remote_code)

        repo = str(record["repo"])
        revision = str(record["revision"])
        artifact_format = str(record.get("artifact_format") or "safetensors")

        backend_name = backend_id or str(record.get("backend") or "") or "fixture"
        backend = create_backend(backend_name)
        capabilities = backend.capabilities
        if artifact_format not in capabilities.formats:
            raise BackendCapabilityMissing(
                f"backend {backend_name!r} does not support the {artifact_format!r} format",
                details={"backend": backend_name, "format": artifact_format},
            )

        revision_info = await self.source.list_files(repo, revision)
        layout = await build_layout(self.source, revision_info, artifact_format=artifact_format)

        # Gate 2: never start a long remote stream when the source cannot
        # satisfy partial access for this artifact.
        if layout.shards:
            first = sorted(layout.shards)[0]
            support = await self.source.probe_range_support(
                repo,
                revision,
                first,
                file_size=layout.shards[first].size,
            )
            if not support.supported:
                from .errors import RangeUnsupportedError

                raise RangeUnsupportedError(
                    f"the source cannot serve partial access for {first}: {support.reason}",
                    details={"repo": repo, "commit": revision, "path": first},
                )

        provider = RemoteWeightProvider(
            source=self.source,
            layout=layout,
            cache=self.cache,
            model_alias=alias,
            block_size_bytes=self.storage.block_size_bytes,
            telemetry=self.telemetry,
            metadata_root=self.storage.metadata_root / "files",
        )
        for name, packing in (backend.expert_packing() or {}).items():
            layout.register_expert_packing(packing)

        options = {"materialize": materialize}
        options.update(extra_options or {})
        context = LoadContext(
            alias=alias,
            repo=repo,
            revision=revision,
            layout=layout,
            provider=provider,
            telemetry=self.telemetry,
            device=device,
            dtype=dtype,
            options=options,
        )
        handle = await backend.load(context)
        session = ModelSession(
            alias=alias,
            repo=repo,
            revision=revision,
            layout=layout,
            provider=provider,
            backend=backend,
            handle=handle,
            prefetch=PrefetchScheduler(provider, telemetry=self.telemetry),
        )
        self._sessions.append(session)
        return session

    # -- model registration ------------------------------------------------
    async def register_model(
        self,
        *,
        alias: str,
        repo: str,
        ref: str = "main",
        artifact_format: str = "safetensors",
        backend: str = "",
        quota_bytes: int = 0,
        quota_mode: str | None = None,
        license_review: str = "pending",
        allow_remote_code: bool = False,
        probe: bool = True,
    ):
        if self.catalog is None or self.cache is None:
            raise RuntimeError("the runtime has not been started")
        registration = await self.catalog.register(
            alias=alias,
            repo=repo,
            ref=ref,
            artifact_format=artifact_format,
            backend=backend,
            quota_bytes=quota_bytes,
            quota_mode=quota_mode or self.storage.quota_policy,
            license_review=license_review,
            allow_remote_code=allow_remote_code,
            probe=probe,
        )
        self.cache.set_model_quota(alias, quota_bytes, quota_mode)
        return registration

    # -- reporting ---------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "storage": {
                "root": str(self.storage.root),
                "weight_cache_limit_bytes": self.storage.weight_cache_limit_bytes,
                "managed_store_limit_bytes": self.storage.managed_store_limit_bytes,
            },
            "models": [m["alias"] for m in (self.db.list_models() if self.db else [])],
        }


async def open_runtime(
    config: Config,
    *,
    source: SourceAdapter | None = None,
    telemetry: RunTelemetry | None = None,
) -> StreamAgentRuntime:
    runtime = StreamAgentRuntime(config, source=source, telemetry=telemetry)
    await runtime.start()
    return runtime


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ModelSession",
    "ModelNotRegistered",
    "StreamAgentRuntime",
    "open_runtime",
    "load_json",
    "asyncio",
]
