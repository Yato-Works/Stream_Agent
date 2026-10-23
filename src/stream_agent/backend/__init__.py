"""Inference backend adapters (Plan.md C7)."""

from .base import (
    BackendCapabilities,
    GenerationResult,
    GRANULARITIES,
    InferenceBackend,
    LoadContext,
)
from .registry import (
    available_backends,
    create_backend,
    register_backend,
    register_builtin_backends,
)

__all__ = [
    "BackendCapabilities",
    "GenerationResult",
    "GRANULARITIES",
    "InferenceBackend",
    "LoadContext",
    "available_backends",
    "create_backend",
    "register_backend",
    "register_builtin_backends",
]
