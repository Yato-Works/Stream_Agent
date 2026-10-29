"""Source adapter tests: range validation and error mapping (Plan.md C2 / 8.2)."""

from __future__ import annotations

import httpx
import pytest

from stream_agent.config import SourceConfig
from stream_agent.errors import (
    ErrorCode,
    RangeMismatchError,
    RangeUnsupportedError,
    SourceAuthError,
    SourceForbiddenError,
    SourceNotFoundError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from stream_agent.sources.hf_hub import HuggingFaceSource
from stream_agent.sources.local import LocalDirectorySource


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        base_url="https://huggingface.co",
    )


async def test_range_request_requires_206():
    async def handler(request: httpx.Request) -> httpx.Response:
        # A source that ignores the Range header answers with the full object.
        return httpx.Response(200, content=b"x" * 4096)

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    with pytest.raises(RangeUnsupportedError) as excinfo:
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)
    assert excinfo.value.code == ErrorCode.RANGE_UNSUPPORTED


async def test_full_object_body_is_never_consumed():
    """A 200 response must be refused before its body is read.

    This is stronger than checking the error type: an implementation that
    calls ``response.content`` has already paid for the complete shard.
    """

    class TrackingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.read = False

        async def __aiter__(self):
            self.read = True
            yield b"x" * 4096

        async def aclose(self) -> None:
            return None

    stream = TrackingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    with pytest.raises(RangeUnsupportedError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)
    assert stream.read is False, "the full-object body was consumed"


async def test_missing_content_range_is_rejected():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, content=b"0" * 11)

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    with pytest.raises(RangeMismatchError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)


async def test_wrong_content_range_is_rejected():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206,
            content=b"y" * 11,
            headers={"Content-Range": "bytes 100-110/1000"},
        )

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    with pytest.raises(RangeMismatchError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)


async def test_short_body_is_rejected():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206, content=b"y" * 5, headers={"Content-Range": "bytes 0-10/1000"}
        )

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    with pytest.raises(RangeMismatchError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, SourceAuthError),
        (403, SourceForbiddenError),
        (404, SourceNotFoundError),
        (429, SourceRateLimitedError),
        (503, SourceUnavailableError),
    ],
)
async def test_status_codes_map_to_typed_errors(status, expected):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"nope")

    source = HuggingFaceSource(SourceConfig(retry_limit=0), client=_client(handler))
    with pytest.raises(expected):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)


async def test_successful_range_read():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=10-19"
        return httpx.Response(
            206, content=b"0123456789", headers={"Content-Range": "bytes 10-19/100"}
        )

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    data = await source.fetch_range("a/b", "commit", "model.safetensors", 10, 19)
    assert data == b"0123456789"


async def test_probe_reports_unsupported_sources():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 100)

    source = HuggingFaceSource(SourceConfig(), client=_client(handler))
    support = await source.probe_range_support("a/b", "commit", "model.safetensors")
    assert support.supported is False
    assert "Range" in support.reason


async def test_token_is_sent_when_configured(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(
            206, content=b"0" * 4, headers={"Content-Range": "bytes 0-3/100"}
        )

    source = HuggingFaceSource(SourceConfig(token_env="HF_TOKEN"), client=_client(handler))
    await source.fetch_range("a/b", "commit", "model.safetensors", 0, 3)
    assert seen["auth"] == "Bearer secret-token"


async def test_connection_reset_is_retried_and_then_reported():
    attempts = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("connection reset", request=request)

    source = HuggingFaceSource(SourceConfig(retry_limit=1), client=_client(handler))
    with pytest.raises(SourceUnavailableError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)
    assert attempts["count"] == 2  # one attempt plus one retry


async def test_expired_signed_url_is_reported_as_forbidden():
    async def handler(request: httpx.Request) -> httpx.Response:
        # A signed URL that expired between the probe and the transfer.
        return httpx.Response(403, content=b"expired")

    source = HuggingFaceSource(SourceConfig(retry_limit=0), client=_client(handler))
    with pytest.raises(SourceForbiddenError):
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)


async def test_a_transient_server_error_is_retried():
    calls = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, content=b"later")
        return httpx.Response(
            206, content=b"0" * 11, headers={"Content-Range": "bytes 0-10/100"}
        )

    source = HuggingFaceSource(SourceConfig(retry_limit=2), client=_client(handler))
    data = await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)
    assert data == b"0" * 11
    assert calls["count"] == 2


async def test_rate_limit_is_not_retried_but_retry_after_is_reported():
    calls = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, content=b"slow down", headers={"Retry-After": "1"})

    source = HuggingFaceSource(SourceConfig(retry_limit=2), client=_client(handler))
    with pytest.raises(SourceRateLimitedError) as excinfo:
        await source.fetch_range("a/b", "commit", "model.safetensors", 0, 10)
    assert calls["count"] == 1
    assert excinfo.value.details.get("retry_after") == "1"


async def test_local_directory_source_reads_only_the_requested_window(tmp_path):
    source = LocalDirectorySource(tmp_path)
    target = tmp_path / "owner" / "repo" / "rev" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"0123456789abcdef")

    data = await source.fetch_range("owner/repo", "rev", "model.safetensors", 4, 9)
    assert data == b"456789"

    with pytest.raises(RangeMismatchError):
        await source.fetch_range("owner/repo", "rev", "model.safetensors", 4, 99)


async def test_local_directory_source_forbids_path_traversal(tmp_path):
    source = LocalDirectorySource(tmp_path)
    with pytest.raises(SourceForbiddenError):
        await source.fetch_range("owner/repo", "rev", "../../outside", 0, 1)
