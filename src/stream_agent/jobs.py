"""Asynchronous job controller (Plan.md section 12.3).

Long running work (a slow model reviewing a patch) must not hold an HTTP
connection open for hours.  A job is submitted, polled and cancelled; the
result stays local by default.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import StreamAgentError

JOB_STATES = ("queued", "running", "completed", "failed", "cancelled")


@dataclass
class Job:
    job_id: str
    model: str
    prompt_hash: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    state: str = "queued"
    queue_delay_ms: float = 0.0
    weight_wait_ms: float = 0.0
    generated_tokens: int = 0
    generation_ms: float = 0.0
    result: str | None = None
    error: dict[str, Any] | None = None
    result_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobController:
    """In-memory job registry with optional on-disk persistence."""

    def __init__(self, *, result_dir: Path | None = None, max_jobs: int = 256) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self.result_dir = result_dir
        self.max_jobs = max_jobs

    # -- submission --------------------------------------------------------
    def submit(
        self,
        *,
        model: str,
        prompt: str,
        runner: Callable[[Job], Any],
        **params: Any,
    ) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:16],
            model=model,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        )
        self._jobs[job.job_id] = job
        self._cancel[job.job_id] = asyncio.Event()
        self._trim()

        async def _run() -> None:
            job.started_at = time.time()
            job.queue_delay_ms = (job.started_at - job.created_at) * 1000.0
            job.state = "running"
            try:
                result = await runner(job)
                job.state = "completed"
                job.result = result if isinstance(result, str) else json.dumps(result)
                if self.result_dir is not None:
                    target = self.result_dir / f"{job.job_id}.txt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(job.result or "", encoding="utf-8")
                    job.result_location = str(target)
            except asyncio.CancelledError:
                job.state = "cancelled"
                raise
            except StreamAgentError as exc:
                job.state = "failed"
                job.error = exc.to_dict()
            except Exception as exc:  # noqa: BLE001 - report unexpected failures
                job.state = "failed"
                job.error = {"code": "INTERNAL_ERROR", "message": str(exc)}
            finally:
                job.finished_at = time.time()
                if job.started_at:
                    job.generation_ms = (job.finished_at - job.started_at) * 1000.0

        self._tasks[job.job_id] = asyncio.create_task(_run())
        return job

    def _trim(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at)
        for job in ordered[: len(self._jobs) - self.max_jobs]:
            if job.state in ("queued", "running"):
                continue
            self._jobs.pop(job.job_id, None)

    # -- inspection --------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def is_cancelled(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        return bool(event and event.is_set())

    # -- cancellation ------------------------------------------------------
    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        event = self._cancel.get(job_id)
        if event is not None:
            event.set()
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        job.state = "cancelled"
        job.finished_at = time.time()
        return True

    async def wait(self, job_id: str, timeout: float = 60.0) -> Job | None:
        task = self._tasks.get(job_id)
        if task is None:
            return self._jobs.get(job_id)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception:
            pass
        return self._jobs.get(job_id)

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except BaseException:  # noqa: BLE001 - cancellation is expected here
                pass


__all__ = ["Job", "JobController", "JOB_STATES"]
