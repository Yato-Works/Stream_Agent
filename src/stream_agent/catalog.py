"""Model catalog and revision resolver (Plan.md C1).

The catalog turns a user friendly repository id plus a ref into an immutable
commit identity, records the metadata that the design requires (license
identifier, gating, file sizes) and never downloads weight payloads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .cache.db import MetadataDB
from .errors import (
    LicenseReviewRequired,
    ModelNotRegistered,
    SourceNotFoundError,
)
from .sources.base import RangeSupport, RevisionInfo, SourceAdapter
from .telemetry import RunTelemetry

#: Weight files are probed for range support; metadata files must be small.
PROBE_TARGETS = ("model.safetensors",)


@dataclass
class ModelRegistration:
    alias: str
    repo: str
    revision: str
    artifact_format: str = "safetensors"
    backend: str = ""
    quota_bytes: int = 0
    quota_mode: str = "fixed"
    license_id: str | None = None
    license_review: str = "pending"
    gated: bool | None = None
    requires_custom_code: bool = False
    total_weight_bytes: int = 0
    file_count: int = 0
    range_support: dict[str, Any] = field(default_factory=dict)
    auxiliary_files: dict[str, int] = field(default_factory=dict)
    metadata_files: dict[str, int] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "repo": self.repo,
            "revision": self.revision,
            "artifact_format": self.artifact_format,
            "backend": self.backend,
            "quota_bytes": self.quota_bytes,
            "quota_mode": self.quota_mode,
            "license_id": self.license_id,
            "license_review": self.license_review,
            "gated": self.gated,
            "requires_custom_code": self.requires_custom_code,
            "total_weight_bytes": self.total_weight_bytes,
            "file_count": self.file_count,
            "range_support": self.range_support,
            "auxiliary_files": self.auxiliary_files,
            "metadata_files": self.metadata_files,
        }


class ModelCatalog:
    """Resolves, validates and registers model revisions."""

    def __init__(
        self,
        source: SourceAdapter,
        db: MetadataDB,
        *,
        telemetry: RunTelemetry | None = None,
    ) -> None:
        self.source = source
        self.db = db
        self.telemetry = telemetry

    # -- resolution --------------------------------------------------------
    async def resolve(self, repo: str, ref: str = "main") -> RevisionInfo:
        return await self.source.resolve_revision(repo, ref)

    async def list_files(self, repo: str, commit: str) -> RevisionInfo:
        return await self.source.list_files(repo, commit)

    # -- capability probing ------------------------------------------------
    async def probe_range_support(
        self,
        revision: RevisionInfo,
        *,
        artifact_format: str = "safetensors",
    ) -> dict[str, Any]:
        """Verify that partial access really works for this revision."""
        targets = [
            name
            for name in sorted(revision.files)
            if name.endswith(".safetensors") or name.endswith(".gguf")
        ]
        if not targets:
            return {"supported": False, "reason": "no weight file in the revision"}
        support: RangeSupport = await self.source.probe_range_support(
            revision.repo,
            revision.commit,
            targets[0],
            file_size=revision.files[targets[0]].size,
        )
        result = support.to_dict()
        result["probe_path"] = targets[0]
        return result

    # -- registration ------------------------------------------------------
    async def register(
        self,
        *,
        alias: str,
        repo: str,
        ref: str = "main",
        artifact_format: str = "safetensors",
        backend: str = "",
        quota_bytes: int = 0,
        quota_mode: str = "fixed",
        license_review: str = "pending",
        allow_remote_code: bool = False,
        probe: bool = True,
    ) -> ModelRegistration:
        revision = await self.resolve(repo, ref)
        if not revision.commit:
            raise SourceNotFoundError(f"{repo}@{ref} could not be pinned to a commit")
        if revision.requires_custom_code and not allow_remote_code:
            # The registration itself stays possible, but inference will be
            # blocked by the remote code policy until the user opts in.
            pass
        range_support: dict[str, Any] = {}
        if probe:
            range_support = await self.probe_range_support(revision, artifact_format=artifact_format)
        auxiliary = {
            path: info.size or 0
            for path, info in revision.files.items()
            if path.endswith((".bin", ".pt", ".pth", ".onnx", ".msgpack", ".mlmodel"))
        }
        metadata = {
            path: info.size or 0
            for path, info in revision.files.items()
            if path.endswith((".json", ".txt", ".jinja", ".model"))
        }
        registration = ModelRegistration(
            alias=alias,
            repo=repo,
            revision=revision.commit,
            artifact_format=artifact_format,
            backend=backend,
            quota_bytes=quota_bytes,
            quota_mode=quota_mode,
            license_id=revision.license_id,
            license_review=license_review,
            gated=revision.gated,
            requires_custom_code=revision.requires_custom_code,
            total_weight_bytes=revision.total_weight_bytes,
            file_count=len(revision.files),
            range_support=range_support,
            auxiliary_files=auxiliary,
            metadata_files=metadata,
        )
        self.db.upsert_model(
            {
                "alias": alias,
                "repo": repo,
                "revision": revision.commit,
                "artifact_format": artifact_format,
                "backend": backend,
                "quota_bytes": quota_bytes,
                "quota_mode": quota_mode,
                "license_review": license_review,
                "allow_remote_code": allow_remote_code,
                "created_at": time.time(),
                "metadata": {
                    "license_id": revision.license_id,
                    "gated": revision.gated,
                    "requires_custom_code": revision.requires_custom_code,
                    "total_weight_bytes": revision.total_weight_bytes,
                    "range_support": range_support,
                },
            }
        )
        return registration

    # -- lookups -----------------------------------------------------------
    def registered(self, alias: str) -> dict[str, Any]:
        record = self.db.get_model(alias)
        if record is None:
            raise ModelNotRegistered(
                f"model alias {alias!r} is not registered",
                details={"known": [m["alias"] for m in self.db.list_models()]},
            )
        return record

    def list_registered(self) -> list[dict[str, Any]]:
        return self.db.list_models()

    # -- policy ------------------------------------------------------------
    @staticmethod
    def assert_inference_allowed(record: dict[str, Any], *, allow_remote_code: bool = False) -> None:
        """Enforce the license and remote-code gates before an inference run."""
        review = str(record.get("license_review", "pending")).lower()
        if review == "required":
            raise LicenseReviewRequired(
                f"model {record.get('alias')!r} needs a human license review before inference",
                details={
                    "alias": record.get("alias"),
                    "license_review": review,
                },
            )
        metadata = record.get("metadata") or record.get("metadata_json")
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata = metadata or {}
        if metadata.get("requires_custom_code") and not allow_remote_code:
            from .errors import RemoteCodeBlocked

            raise RemoteCodeBlocked(
                f"model {record.get('alias')!r} requires repository supplied code "
                "which is disabled by the current security policy",
                details={"alias": record.get("alias")},
            )


__all__ = ["ModelCatalog", "ModelRegistration"]
