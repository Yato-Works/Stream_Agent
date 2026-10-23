"""Hugging Face Hub source adapter, including the Xet-backed path (Plan.md C2 / 8.3).

Design rules that this module implements:

* Range access is *source specific*.  A plain ``Range`` header is attempted but
  the response is validated: a ``200`` full-object body where ``206`` was
  expected is a failure, never a silent full download.
* Errors are typed and distinguish authentication, authorization/gating,
  missing revision, rate limiting and provider outage.
* Signed URLs and credentials are never persisted or logged.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable

import httpx

from ..config import SourceConfig
from ..errors import (
    ErrorCode,
    RangeMismatchError,
    RangeUnsupportedError,
    SourceAuthError,
    SourceForbiddenError,
    SourceNotFoundError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from .base import RangeSupport, RepoFile, RevisionInfo

HF_ENDPOINT = "https://huggingface.co"

#: Status codes that must never be retried aggressively.
_TERMINAL = {401, 403, 404, 410}


class HuggingFaceSource:
    """Adapter for the public Hugging Face Hub."""

    name = "huggingface"

    def __init__(
        self,
        config: SourceConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        endpoint: str = HF_ENDPOINT,
        telemetry: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or SourceConfig()
        self.endpoint = endpoint.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._telemetry = telemetry or (lambda event, payload: None)
        self._sem = asyncio.Semaphore(max(1, self.config.max_parallel_ranges))

    # -- plumbing ----------------------------------------------------------
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Encoding": "identity",
        }
        token = self.config.resolve_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if extra:
            headers.update(extra)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(self.config.request_timeout_seconds, connect=30.0),
                limits=httpx.Limits(max_connections=self.config.max_parallel_ranges * 2),
            )
        return self._client

    def _map_status(self, response: httpx.Response, path: str) -> Exception:
        status = response.status_code
        if status == 401:
            return SourceAuthError(
                f"source rejected the credential for {path}",
                details={"repo": self._repo_from_url(response.request.url), "status": status},
            )
        if status == 403:
            return SourceForbiddenError(
                f"no access to {path} (gated repository or missing approval)",
                details={"repo": self._repo_from_url(response.request.url), "status": status},
            )
        if status in (404, 410):
            return SourceNotFoundError(
                f"{path} was not found at the requested revision",
                details={"status": status},
            )
        if status == 429:
            return SourceRateLimitedError(
                f"source rate limit hit while fetching {path}",
                details={"status": status},
            )
        return SourceUnavailableError(
            f"source returned HTTP {status} while fetching {path}",
            details={"status": status},
        )

    @staticmethod
    def _repo_from_url(url: Any) -> str:
        try:
            parts = url.path.split("/")
            if len(parts) >= 3:
                return f"{parts[1]}/{parts[2]}"
        except Exception:
            pass
        return ""

    # -- metadata ----------------------------------------------------------
    async def resolve_revision(self, repo: str, ref: str = "main") -> RevisionInfo:
        client = await self._get_client()
        url = f"{self.endpoint}/api/models/{repo}/revision/{ref}?blobs=true"
        try:
            response = await client.get(url, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise SourceUnavailableError(f"timed out resolving {repo}@{ref}") from exc
        except httpx.HTTPError as exc:
            raise SourceUnavailableError(f"failed to resolve {repo}@{ref}: {exc}") from exc

        if response.status_code >= 400:
            raise self._map_status(response, f"{repo}@{ref}")

        payload = response.json()
        commit = payload.get("sha") or payload.get("lastModified")
        if not commit:
            raise SourceNotFoundError(f"{repo}@{ref} did not resolve to a commit")
        return self._revision_from_payload(repo, str(commit), payload)

    def _revision_from_payload(self, repo: str, commit: str, payload: dict[str, Any]) -> RevisionInfo:
        files: dict[str, RepoFile] = {}
        for sibling in payload.get("siblings", []) or []:
            name = sibling.get("rfilename")
            if not name:
                continue
            lfs = sibling.get("lfs") or {}
            files[name] = RepoFile(
                path=name,
                size=sibling.get("size") or lfs.get("size"),
                sha256=lfs.get("oid"),
                xet_hash=sibling.get("xetHash"),
                is_lfs=bool(lfs) or bool(sibling.get("xetHash")),
            )
        card = payload.get("cardData") or {}
        license_id = card.get("license")
        license_url = None
        if isinstance(license_id, dict):
            license_url = license_id.get("url")
            license_id = license_id.get("id")
        if isinstance(license_id, list) and license_id:
            license_id = license_id[0]
        gated = payload.get("gated")
        return RevisionInfo(
            repo=repo,
            commit=commit,
            files=files,
            license_id=str(license_id) if license_id else None,
            license_url=license_url,
            gated=gated,
            requires_custom_code=bool(payload.get("custom_code")),
            card_data=card,
            source=self.name,
        )

    async def list_files(self, repo: str, commit: str) -> RevisionInfo:
        client = await self._get_client()
        url = f"{self.endpoint}/api/models/{repo}/revision/{commit}?blobs=true"
        try:
            response = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise SourceUnavailableError(f"failed to list {repo}@{commit}: {exc}") from exc
        if response.status_code >= 400:
            raise self._map_status(response, f"{repo}@{commit}")
        return self._revision_from_payload(repo, commit, response.json())

    # -- range access ------------------------------------------------------
    def _resolve_url(self, repo: str, commit: str, path: str) -> str:
        return f"{self.endpoint}/{repo}/resolve/{commit}/{path}"

    async def _request_range(
        self,
        repo: str,
        commit: str,
        path: str,
        start: int,
        end: int,
        *,
        attempt: int = 0,
    ) -> bytes:
        if end < start:
            raise RangeMismatchError(
                f"invalid range [{start}, {end}] requested for {path}",
                details={"path": path, "start": start, "end": end},
            )
        client = await self._get_client()
        url = self._resolve_url(repo, commit, path)
        headers = self._headers({"Range": f"bytes={start}-{end}"})
        expected = end - start + 1
        try:
            async with self._sem:
                response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            if attempt < self.config.retry_limit:
                self._telemetry("retry", {"reason": "timeout", "path": path})
                await self._backoff(attempt)
                return await self._request_range(
                    repo, commit, path, start, end, attempt=attempt + 1
                )
            raise SourceUnavailableError(f"timed out fetching {path}[{start}:{end}]") from exc
        except httpx.HTTPError as exc:
            if attempt < self.config.retry_limit:
                self._telemetry("retry", {"reason": "transport", "path": path})
                await self._backoff(attempt)
                return await self._request_range(
                    repo, commit, path, start, end, attempt=attempt + 1
                )
            raise SourceUnavailableError(f"transport failure fetching {path}: {exc}") from exc

        status = response.status_code
        if status >= 400:
            error = self._map_status(response, path)
            if isinstance(error, SourceUnavailableError) and attempt < self.config.retry_limit:
                self._telemetry("retry", {"reason": f"http_{status}", "path": path})
                await self._backoff(attempt)
                return await self._request_range(
                    repo, commit, path, start, end, attempt=attempt + 1
                )
            raise error

        if status == 200:
            # The source ignored the Range header and is sending the whole
            # object.  This is exactly the "silent full download" case the
            # design forbids, so the body is discarded and the error raised.
            self._telemetry("full_object", {"repo": repo, "path": path})
            raise RangeUnsupportedError(
                f"source ignored the Range header for {path}; partial access is unavailable",
                details={"path": path, "status": status, "code": ErrorCode.RANGE_UNSUPPORTED},
            )

        if status != 206:
            raise RangeMismatchError(
                f"unexpected HTTP {status} for range request on {path}",
                details={"path": path, "status": status},
            )

        content_range = response.headers.get("content-range")
        if content_range:
            try:
                rng = content_range.split(" ", 1)[1]
                got_start, got_end = rng.split("/", 1)[0].split("-", 1)
                if int(got_start) != start or int(got_end) != end:
                    raise RangeMismatchError(
                        f"source returned the wrong byte range for {path}",
                        details={
                            "requested": [start, end],
                            "returned": [int(got_start), int(got_end)],
                            "content_range": content_range,
                        },
                    )
            except ValueError as exc:
                raise RangeMismatchError(
                    f"unparsable Content-Range for {path}: {content_range!r}",
                    details={"content_range": content_range},
                ) from exc

        body = response.content
        if len(body) != expected:
            raise RangeMismatchError(
                f"source returned {len(body)} bytes for a {expected} byte range of {path}",
                details={"path": path, "expected": expected, "actual": len(body)},
            )
        self._telemetry(
            "range_ok",
            {
                "repo": repo,
                "commit": commit,
                "path": path,
                "start": start,
                "end": end,
                "status": status,
                "content_range": content_range,
                "bytes": len(body),
            },
        )
        return body

    async def _backoff(self, attempt: int) -> None:
        delay = min(30.0, (2**attempt) * 0.5) * (0.5 + random.random() / 2)
        await asyncio.sleep(delay)

    async def fetch_range(
        self,
        repo: str,
        commit: str,
        path: str,
        start: int,
        end: int,
    ) -> bytes:
        self._telemetry("range_request", {"repo": repo, "commit": commit, "path": path,
                                          "start": start, "end": end})
        return await self._request_range(repo, commit, path, start, end)

    async def probe_range_support(
        self,
        repo: str,
        commit: str,
        path: str,
        *,
        file_size: int | None = None,
    ) -> RangeSupport:
        start = 0
        end = 63
        try:
            data = await self._request_range(repo, commit, path, start, end)
        except RangeUnsupportedError as exc:
            return RangeSupport(supported=False, reason=str(exc), probe_start=start, probe_end=end)
        except SourceAuthError:
            raise
        except (SourceForbiddenError, SourceNotFoundError):
            raise
        except Exception as exc:  # network problems are not a capability answer
            return RangeSupport(
                supported=False,
                reason=f"probe failed: {exc}",
                probe_start=start,
                probe_end=end,
            )
        if len(data) != end - start + 1:
            return RangeSupport(
                supported=False,
                reason="probe returned an unexpected length",
                probe_start=start,
                probe_end=end,
            )
        return RangeSupport(supported=True, reason="206 partial content", probe_start=start, probe_end=end)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


__all__ = ["HuggingFaceSource", "HF_ENDPOINT"]
