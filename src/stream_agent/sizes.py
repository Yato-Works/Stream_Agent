"""Size parsing and formatting.

All sizes are carried as integer bytes.  Display units (GB/GiB) exist only in
the CLI output; every comparison inside the runtime uses bytes.
"""

from __future__ import annotations

import re
from typing import Any

from .errors import ConfigInvalid

GIB = 1024**3
MIB = 1024**2
KIB = 1024

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?)\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "ki": KIB,
    "kib": KIB,
    "m": 1000**2,
    "mb": 1000**2,
    "mi": MIB,
    "mib": MIB,
    "g": 1000**3,
    "gb": 1000**3,
    "gi": GIB,
    "gib": GIB,
    "t": 1000**4,
    "tb": 1000**4,
    "ti": 1024**4,
    "tib": 1024**4,
}


def parse_size(value: Any, *, field_name: str = "size") -> int:
    """Parse a size expressed as an int or a human string into bytes."""
    if isinstance(value, bool):
        raise ConfigInvalid(f"{field_name}: expected a size, got a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ConfigInvalid(f"{field_name}: size must not be negative")
        return value
    if isinstance(value, float):
        if value < 0:
            raise ConfigInvalid(f"{field_name}: size must not be negative")
        return int(value)
    if isinstance(value, str):
        match = _SIZE_RE.match(value)
        if not match:
            raise ConfigInvalid(f"{field_name}: cannot parse size {value!r}")
        number, unit = match.groups()
        unit = unit.lower()
        if unit not in _UNITS:
            raise ConfigInvalid(f"{field_name}: unknown size unit in {value!r}")
        return int(float(number) * _UNITS[unit])
    raise ConfigInvalid(f"{field_name}: cannot parse size {value!r}")


def format_size(num_bytes: int | None) -> str:
    """Human-readable size for CLI output only."""
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"  # pragma: no cover - unreachable


__all__ = ["GIB", "MIB", "KIB", "parse_size", "format_size"]
