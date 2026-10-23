"""Typed error contract for Stream Agent (Plan.md section 15).

Every failure surfaced to a caller carries a stable code so that an agent or
script can react to it without parsing prose.  The mapping between code and
meaning is fixed by the design document; do not rename codes casually.
"""

from __future__ import annotations

from typing import Any


class ErrorCode:
    """Stable error identifiers (Plan.md section 15)."""

    SOURCE_AUTH = "SOURCE_AUTH"
    SOURCE_FORBIDDEN = "SOURCE_FORBIDDEN"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_RATE_LIMITED = "SOURCE_RATE_LIMITED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    RANGE_UNSUPPORTED = "RANGE_UNSUPPORTED"
    RANGE_MISMATCH = "RANGE_MISMATCH"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    MODEL_FORMAT_UNSUPPORTED = "MODEL_FORMAT_UNSUPPORTED"
    BACKEND_CAPABILITY_MISSING = "BACKEND_CAPABILITY_MISSING"
    CACHE_BUDGET_EXCEEDED = "CACHE_BUDGET_EXCEEDED"
    WORKING_SET_EXCEEDS_BUDGET = "WORKING_SET_EXCEEDS_BUDGET"
    BACKEND_OOM = "BACKEND_OOM"
    SESSION_CANCELLED = "SESSION_CANCELLED"
    REMOTE_CODE_BLOCKED = "REMOTE_CODE_BLOCKED"
    MODEL_LICENSE_REVIEW_REQUIRED = "MODEL_LICENSE_REVIEW_REQUIRED"
    CONFIG_INVALID = "CONFIG_INVALID"
    MODEL_NOT_REGISTERED = "MODEL_NOT_REGISTERED"
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    CAPABILITY_PROBE_FAILED = "CAPABILITY_PROBE_FAILED"


#: Codes that describe a source-side failure rather than a local runtime fault.
SOURCE_CODES = frozenset(
    {
        ErrorCode.SOURCE_AUTH,
        ErrorCode.SOURCE_FORBIDDEN,
        ErrorCode.SOURCE_NOT_FOUND,
        ErrorCode.SOURCE_RATE_LIMITED,
        ErrorCode.SOURCE_UNAVAILABLE,
    }
)

#: Codes for which the source should not be retried aggressively.
NON_RETRYABLE_CODES = frozenset(
    {
        ErrorCode.SOURCE_AUTH,
        ErrorCode.SOURCE_FORBIDDEN,
        ErrorCode.SOURCE_NOT_FOUND,
        ErrorCode.RANGE_UNSUPPORTED,
        ErrorCode.MODEL_FORMAT_UNSUPPORTED,
        ErrorCode.BACKEND_CAPABILITY_MISSING,
        ErrorCode.REMOTE_CODE_BLOCKED,
        ErrorCode.MODEL_LICENSE_REVIEW_REQUIRED,
        ErrorCode.CONFIG_INVALID,
        ErrorCode.UNKNOWN_MODEL,
    }
)


class StreamAgentError(Exception):
    """Base class for all Stream Agent failures.

    The ``code`` attribute is the contract; ``message`` explains the situation
    without ever containing credentials or signed URLs.
    """

    code: str = "STREAM_AGENT_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    @property
    def retryable(self) -> bool:
        return self.code not in NON_RETRYABLE_CODES

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class SourceError(StreamAgentError):
    code = "SOURCE_UNAVAILABLE"


class SourceAuthError(SourceError):
    code = ErrorCode.SOURCE_AUTH


class SourceForbiddenError(SourceError):
    code = ErrorCode.SOURCE_FORBIDDEN


class SourceNotFoundError(SourceError):
    code = ErrorCode.SOURCE_NOT_FOUND


class SourceRateLimitedError(SourceError):
    code = ErrorCode.SOURCE_RATE_LIMITED


class SourceUnavailableError(SourceError):
    code = ErrorCode.SOURCE_UNAVAILABLE


class RangeUnsupportedError(StreamAgentError):
    code = ErrorCode.RANGE_UNSUPPORTED


class RangeMismatchError(StreamAgentError):
    code = ErrorCode.RANGE_MISMATCH


class IntegrityFailure(StreamAgentError):
    code = ErrorCode.INTEGRITY_FAILURE


class ModelFormatUnsupported(StreamAgentError):
    code = ErrorCode.MODEL_FORMAT_UNSUPPORTED


class BackendCapabilityMissing(StreamAgentError):
    code = ErrorCode.BACKEND_CAPABILITY_MISSING


class CacheBudgetExceeded(StreamAgentError):
    code = ErrorCode.CACHE_BUDGET_EXCEEDED


class WorkingSetExceedsBudget(StreamAgentError):
    code = ErrorCode.WORKING_SET_EXCEEDS_BUDGET


class BackendOutOfMemory(StreamAgentError):
    code = ErrorCode.BACKEND_OOM


class SessionCancelled(StreamAgentError):
    code = ErrorCode.SESSION_CANCELLED


class RemoteCodeBlocked(StreamAgentError):
    code = ErrorCode.REMOTE_CODE_BLOCKED


class LicenseReviewRequired(StreamAgentError):
    code = ErrorCode.MODEL_LICENSE_REVIEW_REQUIRED


class ConfigInvalid(StreamAgentError):
    code = ErrorCode.CONFIG_INVALID


class ModelNotRegistered(StreamAgentError):
    code = ErrorCode.MODEL_NOT_REGISTERED


class UnknownModel(StreamAgentError):
    code = ErrorCode.UNKNOWN_MODEL


class CapabilityProbeFailed(StreamAgentError):
    code = ErrorCode.CAPABILITY_PROBE_FAILED


__all__ = [
    "ErrorCode",
    "StreamAgentError",
    "SourceError",
    "SourceAuthError",
    "SourceForbiddenError",
    "SourceNotFoundError",
    "SourceRateLimitedError",
    "SourceUnavailableError",
    "RangeUnsupportedError",
    "RangeMismatchError",
    "IntegrityFailure",
    "ModelFormatUnsupported",
    "BackendCapabilityMissing",
    "CacheBudgetExceeded",
    "WorkingSetExceedsBudget",
    "BackendOutOfMemory",
    "SessionCancelled",
    "RemoteCodeBlocked",
    "LicenseReviewRequired",
    "ConfigInvalid",
    "ModelNotRegistered",
    "UnknownModel",
    "CapabilityProbeFailed",
    "SOURCE_CODES",
    "NON_RETRYABLE_CODES",
]
