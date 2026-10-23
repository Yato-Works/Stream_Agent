"""Backend registry.

Backends are registered by id; the configuration refers to them by that id so
that a model declaration never depends on an import path.
"""

from __future__ import annotations

from typing import Callable

from .base import BackendCapabilities, InferenceBackend

_REGISTRY: dict[str, Callable[[], InferenceBackend]] = {}


def register_backend(backend_id: str, factory: Callable[[], InferenceBackend]) -> None:
    _REGISTRY[backend_id] = factory


def create_backend(backend_id: str) -> InferenceBackend:
    try:
        factory = _REGISTRY[backend_id]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"unknown backend {backend_id!r}; available backends: {available}"
        ) from exc
    return factory()


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def register_builtin_backends() -> None:
    from .fixture import FixtureBackend

    register_backend("fixture", FixtureBackend)
    try:
        from .transformers_backend import TransformersBackend

        register_backend("transformers", TransformersBackend)
    except Exception:  # pragma: no cover - torch/transformers are optional
        pass


__all__ = [
    "BackendCapabilities",
    "InferenceBackend",
    "available_backends",
    "create_backend",
    "register_backend",
    "register_builtin_backends",
]
