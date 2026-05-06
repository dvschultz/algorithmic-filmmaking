"""SQLite-backed jobs store.

Schema is in ``migrations/0001_init.sql``. Pragmas applied at connection time:

- ``journal_mode=WAL`` — readers do not block writers; survives concurrent
  process boots.
- ``synchronous=NORMAL`` — durable enough for a job log; faster than FULL.
- ``busy_timeout=5000`` — block up to 5s when another writer holds the file
  before raising.
- ``foreign_keys=ON`` — kept enabled for future schema changes.

The DB file is created with mode ``0o600`` so other local users cannot read
job history (R29).

Status values:
  ``queued`` — submitted, waiting for the per-project mutex.
  ``running`` — worker is executing the spine fn.
  ``cancelling`` — ``cancel_job`` called; ``cancel_event`` set; awaiting the
    worker's next yield-point check.
  ``completed`` — terminal, success.
  ``failed`` — terminal, the spine fn raised.
  ``cancelled`` — terminal, cancellation observed.
  ``crashed`` — terminal, set by the boot sweep when a row was found in
    ``running`` or ``cancelling`` after server restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback as _traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

# Status sentinel values.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_CANCELLING = "cancelling"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_CRASHED = "crashed"

TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_CRASHED}
)
TERMINAL_ERROR_STATUSES = frozenset({STATUS_FAILED, STATUS_CANCELLED, STATUS_CRASHED})

# Hard cap for stored tracebacks: 4 KB, last 10 frames, absolute paths
# stripped. Defends against unbounded payload in the LLM context (R25).
TRACEBACK_FRAME_LIMIT = 10
TRACEBACK_BYTE_CAP = 4096


class JobNotFoundError(LookupError):
    """Raised when a job_id does not match any row in the store."""


@dataclass
class JobRow:
    """In-memory mirror of a row in the ``jobs`` table.

    The ``args_json``, ``result_json``, and ``error`` columns are sensitive
    payload — never include them in agent-facing projections; they live in
    ``get_job_result`` only (R28).
    """

    id: str
    kind: str
    status: str
    args_json: str
    created_at: float
    updated_at: float
    idempotency_key: Optional[str] = None
    project_path: Optional[str] = None
    project_mtime_at_start: Optional[float] = None
    progress: float = 0.0
    status_message: Optional[str] = None
    result_json: Optional[str] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None
    blocking_job_id: Optional[str] = None
    finished_at: Optional[float] = None

    # Convenience: parsed args / result.
    @property
    def args(self) -> dict:
        return json.loads(self.args_json) if self.args_json else {}

    @property
    def result(self) -> Optional[dict]:
        return json.loads(self.result_json) if self.result_json else None

    def to_safe_projection(self) -> dict:
        """Project to the ``list_jobs`` / ``get_job_status`` shape.

        Excludes ``args_json``, ``result_json``, and ``error`` payload (R28).
        """
        return {
            "task_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "status_message": self.status_message,
            "project_path": self.project_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "queue_position": self.queue_position,
            "blocking_job_id": self.blocking_job_id,
        }


def sanitize_traceback(exc: BaseException) -> str:
    """Turn an exception into a sanitized traceback string for storage.

    Returns ``<ExceptionType>: <message>`` followed by up to
    ``TRACEBACK_FRAME_LIMIT`` of the deepest frames, with absolute file paths
    stripped to basenames. Result is byte-capped at ``TRACEBACK_BYTE_CAP``.
    Defends FastMCP SDK leakage of internal paths into agent context (R25).
    """
    frames = _traceback.extract_tb(exc.__traceback__)
    last_frames = frames[-TRACEBACK_FRAME_LIMIT:]
    lines = [f"{type(exc).__name__}: {exc}"]
    for frame in last_frames:
        # Strip absolute prefix; basename is enough for diagnosis without
        # leaking the developer's filesystem layout.
        rel = os.path.basename(frame.filename) if frame.filename else "?"
        lines.append(f'  File "{rel}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            lines.append(f"    {frame.line}")
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > TRACEBACK_BYTE_CAP:
        text = text.encode("utf-8")[:TRACEBACK_BYTE_CAP].decode(
            "utf-8", errors="ignore"
        )
    return text


def _row_to_jobrow(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        args_json=row["args_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        idempotency_key=row["idempotency_key"],
        project_path=row["project_path"],
        project_mtime_at_start=row["project_mtime_at_start"],
        progress=row["progress"] or 0.0,
        status_message=row["status_message"],
        result_json=row["result_json"],
        error=row["error"],
        queue_position=row["queue_position"],
        blocking_job_id=row["blocking_job_id"],
        finished_at=row["finished_at"],
    )


class JobStore:
    """SQLite-backed job persistence layer."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._ensure_db_file()
        self._init_schema()

    def _ensure_db_file(self) -> None:
        """Touch the DB file with mode 0o600 if it does not exist (R29)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            # os.open with O_CREAT respects the mode argument; sqlite3.connect
            # would create the file with default 0o644.
            fd = os.open(
                str(self.db_path),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            os.close(fd)
        else:
            # Tighten permissions if the file was created by an earlier build
            # without the mode argument.
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,  # autocommit; we manage txns explicitly
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        migration_path = (
            Path(__file__).parent / "migrations" / "0001_init.sql"
        )
        sql = migration_path.read_text()
        with self._connect() as conn:
            conn.executescript(sql)

    # --- Mutations ---

    def insert(
        self,
        *,
        kind: str,
        args: dict,
        project_path: Optional[str] = None,
        project_mtime_at_start: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        status: str = STATUS_QUEUED,
        queue_position: Optional[int] = None,
        blocking_job_id: Optional[str] = None,
    ) -> JobRow:
        """Insert a new job row and return it.

        Raises ``sqlite3.IntegrityError`` when the unique
        ``(kind, project_path, idempotency_key)`` constraint is violated.
        """
        now = time.time()
        job_id = str(uuid.uuid4())
        row = JobRow(
            id=job_id,
            kind=kind,
            status=status,
            args_json=json.dumps(args, default=str),
            created_at=now,
            updated_at=now,
            idempotency_key=idempotency_key,
            project_path=project_path,
            project_mtime_at_start=project_mtime_at_start,
            queue_position=queue_position,
            blocking_job_id=blocking_job_id,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, idempotency_key, args_json,
                    project_path, project_mtime_at_start, progress,
                    status_message, queue_position, blocking_job_id,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.id,
                    row.kind,
                    row.status,
                    row.idempotency_key,
                    row.args_json,
                    row.project_path,
                    row.project_mtime_at_start,
                    row.progress,
                    row.status_message,
                    row.queue_position,
                    row.blocking_job_id,
                    row.created_at,
                    row.updated_at,
                ),
            )
        return row

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        progress: Optional[float] = None,
        status_message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
        queue_position: Optional[int] = None,
        blocking_job_id: Optional[str] = None,
        terminal: bool = False,
    ) -> None:
        """Update a job row.

        ``terminal=True`` sets ``finished_at`` to the current timestamp.
        ``progress`` / ``status_message`` are debounced by the caller; the
        store writes whatever it is given.
        """
        now = time.time()
        sets = ["status = ?", "updated_at = ?"]
        params: list = [status, now]

        if progress is not None:
            sets.append("progress = ?")
            params.append(progress)
        if status_message is not None:
            sets.append("status_message = ?")
            params.append(status_message)
        if result is not None:
            sets.append("result_json = ?")
            params.append(json.dumps(result, default=str))
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        # queue_position / blocking_job_id can be cleared by passing None,
        # but our callers only set or leave them. Use sentinel to detect.
        if queue_position is not None:
            sets.append("queue_position = ?")
            params.append(queue_position)
        if blocking_job_id is not None:
            sets.append("blocking_job_id = ?")
            params.append(blocking_job_id)
        if terminal:
            sets.append("finished_at = ?")
            params.append(now)

        params.append(job_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            if cur.rowcount == 0:
                raise JobNotFoundError(job_id)

    def clear_queue_state(self, job_id: str) -> None:
        """Clear ``queue_position`` / ``blocking_job_id`` after a row leaves
        the queued state."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET queue_position = NULL,
                    blocking_job_id = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (time.time(), job_id),
            )

    def delete(self, job_id: str) -> bool:
        """Delete a job row by id. Returns True if a row was removed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    # --- Reads ---

    def get(self, job_id: str) -> JobRow:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return _row_to_jobrow(row)

    def find_by_idempotency(
        self,
        kind: str,
        project_path: Optional[str],
        idempotency_key: str,
    ) -> Optional[JobRow]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE kind = ?
                  AND idempotency_key = ?
                  AND (project_path IS ? OR project_path = ?)
                """,
                (kind, idempotency_key, project_path, project_path),
            ).fetchone()
        return _row_to_jobrow(row) if row else None

    def list(
        self,
        *,
        status_filter: Optional[Sequence[str]] = None,
        kind_filter: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> list[JobRow]:
        sql = "SELECT * FROM jobs"
        clauses: list[str] = []
        params: list = []
        if status_filter:
            placeholders = ",".join("?" for _ in status_filter)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_filter)
        if kind_filter:
            clauses.append("kind = ?")
            params.append(kind_filter)
        if project_filter:
            clauses.append("project_path = ?")
            params.append(project_filter)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_jobrow(r) for r in rows]

    # --- Boot sweep + pruning ---

    def mark_running_jobs_as_crashed(self) -> int:
        """Boot sweep: mark every still-running row as crashed.

        Called unconditionally on lifespan startup. The MCP server is
        single-process by design — any row in ``running`` or ``cancelling``
        after we boot was running under a dead process (R18).

        Returns the number of rows updated.
        """
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'crashed',
                    error = 'server restarted while in flight',
                    finished_at = ?,
                    updated_at = ?,
                    queue_position = NULL,
                    blocking_job_id = NULL
                WHERE status IN ('running', 'cancelling')
                """,
                (now, now),
            )
            return cur.rowcount

    def purge_old_jobs(self, days: int = 30) -> int:
        """Delete terminal-status rows older than ``days``. Running and queued
        rows are never purged.

        Returns the number of rows deleted.
        """
        if days < 0:
            raise ValueError("days must be >= 0")
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed', 'cancelled', 'crashed')
                  AND finished_at IS NOT NULL
                  AND finished_at < ?
                """,
                (cutoff,),
            )
            return cur.rowcount
