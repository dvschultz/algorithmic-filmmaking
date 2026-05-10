"""In-memory job runtime: ``ThreadPoolExecutor`` + cancellation protocol.

The runtime owns the worker pool, the in-memory ``cancel_event`` and
``Future`` registries, and the bridge between ``JobStore`` (durable state)
and the running closures.

Submission flow (R21 idempotency, R17 per-project mutex):

1. Validate ``idempotency_key`` length (<= 255 chars).
2. If a row already exists for ``(kind, project_path, idempotency_key)``:
   - terminal-success (``completed``) → return its ``task_id`` without
     spawning a worker.
   - terminal-error (``failed`` / ``cancelled`` / ``crashed``) → delete the
     row and continue to step 3.
   - in-flight (``queued`` / ``running`` / ``cancelling``) → return that
     row's ``task_id`` so the caller can poll it.
3. Insert a new row in ``queued``.
4. Register a ``threading.Event`` and ``Future`` in the in-memory dicts.
5. Submit the closure to the pool.

The submitted worker:

1. Acquires the per-project mutex (blocking; while blocked the row stays in
   ``queued`` with ``queue_position`` / ``blocking_job_id`` populated).
2. Transitions the row to ``running``.
3. Calls the spine fn with ``progress_callback`` (debounced 2s writes) and
   ``cancel_event``.
4. On success: writes ``result_json`` and ``completed``.
5. On ``cancel_event.is_set()``: writes ``cancelled``.
6. On exception (non-cancel): writes ``failed`` with sanitized traceback.
7. Releases the mutex; clears the in-memory registry entries.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from scene_ripper_mcp.jobs.lock import ProjectLockRegistry
from scene_ripper_mcp.jobs.store import (
    JobStore,
    STATUS_CANCELLED,
    STATUS_CANCELLING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_ERROR_STATUSES,
    sanitize_traceback,
)

logger = logging.getLogger(__name__)

# Maximum length of an idempotency_key, validated at submit (R21).
IDEMPOTENCY_KEY_MAX_LENGTH = 255

# How often the worker may push progress updates to the DB. Debouncing keeps
# write pressure low under per-clip / per-frame iteration loops.
PROGRESS_DEBOUNCE_SECONDS = 2.0

# Default poll interval the runtime suggests to the caller in submit results.
DEFAULT_POLL_INTERVAL_SECONDS = 5

# Default worker pool size. The plan target is 3-10 concurrent jobs (R-doc).
DEFAULT_MAX_WORKERS = 10


class InvalidIdempotencyKeyError(ValueError):
    """Raised when ``idempotency_key`` exceeds ``IDEMPOTENCY_KEY_MAX_LENGTH``."""


# Type alias for the work callable. Receives ``progress_callback`` and
# ``cancel_event``; returns the spine result dict (which becomes
# ``result_json`` in the row on success).
RunCallable = Callable[
    [Callable[[float, str], None], threading.Event],
    dict,
]


@dataclass
class _JobHandle:
    """In-memory handle for a running job.

    Stored in ``JobRuntime._handles`` keyed by ``task_id`` so ``cancel`` can
    find the ``cancel_event`` and the worker thread can find its closure
    output without re-querying the store.
    """

    task_id: str
    cancel_event: threading.Event
    future: Future
    last_progress_write: float = 0.0
    last_progress: float = 0.0
    last_status_message: str = ""


class JobRuntime:
    """Owns the executor, cancellation registry, and the per-project mutex."""

    def __init__(
        self,
        store: JobStore,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        lock_registry: Optional[ProjectLockRegistry] = None,
    ) -> None:
        self.store = store
        self.lock_registry = lock_registry or ProjectLockRegistry()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._handles: dict[str, _JobHandle] = {}
        self._handles_lock = threading.Lock()

    # --- Lifecycle ---

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the executor; blocks until in-flight jobs settle when
        ``wait=True``."""
        self._executor.shutdown(wait=wait)

    # --- Submission ---

    def submit(
        self,
        *,
        kind: str,
        args: dict,
        run: RunCallable,
        project_path: Optional[str | Path] = None,
        project_mtime_at_start: Optional[float] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Submit a job. Returns a result dict suitable for the MCP
        ``start_*`` tool response.

        Result shape:

            {
                "task_id": "...",
                "status": "queued" | "completed",
                "poll_interval": 5,
            }

        ``status="completed"`` is returned synchronously when an idempotency
        key matches an already-completed row — no worker is spawned.
        """
        if idempotency_key is not None:
            if len(idempotency_key) > IDEMPOTENCY_KEY_MAX_LENGTH:
                raise InvalidIdempotencyKeyError(
                    f"idempotency_key exceeds {IDEMPOTENCY_KEY_MAX_LENGTH} chars"
                )

        canonical_path = ProjectLockRegistry.canonical(project_path)

        # Idempotency check (R21).
        if idempotency_key is not None:
            existing = self.store.find_by_idempotency(
                kind=kind,
                project_path=canonical_path,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                if existing.status == STATUS_COMPLETED:
                    return {
                        "task_id": existing.id,
                        "status": existing.status,
                        "poll_interval": DEFAULT_POLL_INTERVAL_SECONDS,
                    }
                if existing.status in TERMINAL_ERROR_STATUSES:
                    # Replace the terminal-error row so the same key can be
                    # retried (R21).
                    self.store.delete(existing.id)
                else:
                    # In-flight (queued / running / cancelling). Caller can
                    # poll this task_id.
                    return {
                        "task_id": existing.id,
                        "status": existing.status,
                        "poll_interval": DEFAULT_POLL_INTERVAL_SECONDS,
                    }

        # Compute queue state. If another job currently holds this project's
        # lock, this job will queue behind it; surface that to the caller
        # (R17, AE1).
        queue_position: Optional[int] = None
        blocking_job_id: Optional[str] = None
        if canonical_path is not None:
            blocking_job_id = self.lock_registry.current_holder(canonical_path)
            if blocking_job_id is not None:
                # Count rows already queued behind the same project to
                # produce a position estimate.
                queued_rows = self.store.list(
                    status_filter=[STATUS_QUEUED],
                    project_filter=canonical_path,
                )
                queue_position = len(queued_rows) + 1

        row = self.store.insert(
            kind=kind,
            args=args,
            project_path=canonical_path,
            project_mtime_at_start=project_mtime_at_start,
            idempotency_key=idempotency_key,
            status=STATUS_QUEUED,
            queue_position=queue_position,
            blocking_job_id=blocking_job_id,
        )

        cancel_event = threading.Event()
        future = self._executor.submit(
            self._run_job,
            row.id,
            run,
            cancel_event,
            canonical_path,
        )
        with self._handles_lock:
            self._handles[row.id] = _JobHandle(
                task_id=row.id,
                cancel_event=cancel_event,
                future=future,
            )

        return {
            "task_id": row.id,
            "status": STATUS_QUEUED,
            "poll_interval": DEFAULT_POLL_INTERVAL_SECONDS,
        }

    # --- Cancellation ---

    def cancel(self, task_id: str) -> bool:
        """Signal cancellation. Returns True when the in-memory event was
        set; False when no live handle exists for the id (already terminal,
        unknown id, or crashed).
        """
        with self._handles_lock:
            handle = self._handles.get(task_id)
        if handle is None:
            return False

        # Mark the row as cancelling first; the worker will flip to cancelled
        # once it observes the event.
        try:
            self.store.update_status(task_id, STATUS_CANCELLING)
        except Exception:
            logger.exception("Failed to mark job %s as cancelling", task_id)
            # Still set the event — best-effort.
        handle.cancel_event.set()
        return True

    # --- Worker body ---

    def _run_job(
        self,
        task_id: str,
        run: RunCallable,
        cancel_event: threading.Event,
        canonical_path: Optional[str],
    ) -> None:
        """Execute one job inside the worker thread."""
        # Acquire per-project mutex. While blocked the row stays in
        # ``queued`` with the queue_position/blocking_job_id populated by
        # ``submit``.
        lock = (
            self.lock_registry.get_lock(canonical_path)
            if canonical_path
            else None
        )
        if lock is not None:
            lock.acquire()
            try:
                self.lock_registry.set_holder(canonical_path, task_id)
            except Exception:
                logger.exception(
                    "Failed to record lock holder for %s", task_id
                )

        # Transition to running and clear queue state.
        try:
            self.store.update_status(task_id, STATUS_RUNNING)
            self.store.clear_queue_state(task_id)
        except Exception:
            logger.exception("Failed to mark job %s as running", task_id)

        # Build the debounced progress callback.
        with self._handles_lock:
            handle = self._handles[task_id]

        def progress_callback(progress: float, status_message: str) -> None:
            now = time.monotonic()
            # Always store the latest values on the handle so the final
            # commit can flush them.
            handle.last_progress = progress
            handle.last_status_message = status_message
            if now - handle.last_progress_write < PROGRESS_DEBOUNCE_SECONDS:
                return
            handle.last_progress_write = now
            try:
                self.store.update_status(
                    task_id,
                    STATUS_RUNNING,
                    progress=progress,
                    status_message=status_message,
                )
            except Exception:
                logger.exception(
                    "Progress write failed for job %s", task_id
                )

        # Execute.
        try:
            result = run(progress_callback, cancel_event)
            if cancel_event.is_set():
                self.store.update_status(
                    task_id,
                    STATUS_CANCELLED,
                    progress=handle.last_progress,
                    status_message=handle.last_status_message or "cancelled",
                    terminal=True,
                )
            else:
                self.store.update_status(
                    task_id,
                    STATUS_COMPLETED,
                    progress=1.0,
                    status_message=handle.last_status_message or "completed",
                    result=result if isinstance(result, dict) else {"value": result},
                    terminal=True,
                )
        except BaseException as exc:  # noqa: BLE001 — we sanitize and store
            sanitized = sanitize_traceback(exc)
            try:
                self.store.update_status(
                    task_id,
                    STATUS_FAILED,
                    error=sanitized,
                    progress=handle.last_progress,
                    status_message=handle.last_status_message or "failed",
                    terminal=True,
                )
            except Exception:
                logger.exception(
                    "Failed to record failed status for job %s", task_id
                )
            # Don't re-raise — the worker should die quietly so the executor
            # doesn't log the exception twice.
        finally:
            with self._handles_lock:
                self._handles.pop(task_id, None)
            if canonical_path is not None:
                self.lock_registry.clear_holder(canonical_path, task_id)
            if lock is not None:
                lock.release()

    # --- Convenience reads ---

    def is_handle_live(self, task_id: str) -> bool:
        with self._handles_lock:
            return task_id in self._handles
