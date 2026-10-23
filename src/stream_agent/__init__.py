"""Stream Agent -- storage-budgeted on-demand model runtime.

The public surface of the package is intentionally small:

* :func:`stream_agent.runtime.open_runtime` builds a wired runtime.
* :class:`stream_agent.provider.RemoteWeightProvider` is the boundary a
  backend uses to request weights.
* :mod:`stream_agent.cli` exposes the ``stream-agent`` command line.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .errors import ErrorCode, StreamAgentError

__all__ = ["ErrorCode", "StreamAgentError", "__version__"]
