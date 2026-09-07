"""SQLite-backed jobs store.

The initial schema is bundled as Python in ``core.jobs.schema``. Pragmas applied at connection time:

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
  ``crashed`` — terminal, set by the boot sweep for abandoned owner jobs,
    or legacy unowned running/cancelling rows after server restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback as _traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence
from core.jobs.spec import OperationSpec

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
    persistence: str = "job_history"
    operation_json: str | None = None

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
        projection = {
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
        if self.persistence == "session_only":
            projection["persistence"] = self.persistence
        if self.operation_json is not None:
            projection["operation"] = OperationSpec.from_json(
                self.operation_json
            ).safe_projection()
        return projection


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


def _row_to_jobrow(row: sqlite3.Row, persistence: str = "job_history") -> JobRow:
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
        persistence=persistence,
        operation_json=row["operation_json"],
    )


class JobStore:
    """SQLite-backed job persistence layer."""

    def __init__(self, db_path: Path | str) -> None:
        self.persistence = "job_history"
        self._memory_connection: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock()
        self.db_path = Path(db_path)
        self._ensure_db_file()
        self._init_schema()

    @classmethod
    def in_memory(cls) -> JobStore:
        """Isolated session-only history, never written to a SQLite file."""
        store = cls.__new__(cls)
        store.persistence = "session_only"
        store.db_path = Path(":memory:")
        store._memory_lock = threading.RLock()
        store._memory_connection = sqlite3.connect(
            ":memory:", isolation_level=None, check_same_thread=False
        )
        store._memory_connection.row_factory = sqlite3.Row
        store._memory_connection.execute("PRAGMA temp_store=MEMORY")
        store._init_schema()
        return store

    def close(self) -> None:
        """Discard session history after its runtime workers have stopped."""
        with self._memory_lock:
            if self._memory_connection is not None:
                self._memory_connection.close()
                self._memory_connection = None

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
        if self.persistence == "session_only":
            with self._memory_lock:
                if self._memory_connection is None:
                    raise RuntimeError("Session job store is closed")
                yield self._memory_connection
            return
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
        from core.jobs.schema import INITIAL_SCHEMA, RESULT_SCHEMA

        sql = INITIAL_SCHEMA
        with self._connect() as conn:
            conn.executescript(sql)
            conn.executescript(RESULT_SCHEMA)
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
                if "owner_id" not in columns:
                    conn.execute("ALTER TABLE jobs ADD COLUMN owner_id TEXT")
                if "operation_json" not in columns:
                    conn.execute("ALTER TABLE jobs ADD COLUMN operation_json TEXT")

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
        owner_id: str | None = None,
        operation: OperationSpec | None = None,
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
            persistence=self.persistence,
            operation_json=operation.to_json() if operation is not None else None,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, idempotency_key, args_json,
                    project_path, project_mtime_at_start, progress,
                    status_message, queue_position, blocking_job_id,
                    created_at, updated_at, owner_id, operation_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    owner_id,
                    row.operation_json,
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
    ) -> bool:
        """Update a job row.

        Returns False for a late write to a terminal row or a running update
        after cancellation was requested. Terminal decisions are immutable.
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
        if terminal or status in TERMINAL_STATUSES:
            sets.append("finished_at = ?")
            params.append(now)

        params.append(job_id)
        terminal_states = sorted(TERMINAL_STATUSES)
        params.extend(terminal_states)
        params.extend([STATUS_CANCELLING, status, STATUS_RUNNING])
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ? "
                f"AND status NOT IN ({','.join('?' for _ in terminal_states)}) "
                "AND NOT (status = ? AND ? = ?)",
                params,
            )
            if cur.rowcount == 0:
                if (
                    conn.execute(
                        "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                    is None
                ):
                    raise JobNotFoundError(job_id)
                return False
            return True

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

    def get_result(self, result_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_results WHERE result_id = ?", (result_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def record_result(
        self, result_id: str, spec_json: str, payload_json: str, digest: str
    ) -> dict:
        """Persist immutable computed output before publishing it to a project."""
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                "INSERT OR IGNORE INTO job_results (result_id,spec_json,payload_json,payload_digest,created_at) VALUES (?,?,?,?,?)",
                (result_id, spec_json, payload_json, digest, time.time()),
            )
        row = self.get_result(result_id)
        assert row is not None
        if (row["spec_json"], row["payload_json"], row["payload_digest"]) != (
            spec_json,
            payload_json,
            digest,
        ):
            raise ValueError("Result identity already contains different data")
        return row

    def checkpoint_result(self, result_id: str, digest: str) -> None:
        """Acknowledge an already-durable project receipt; never apply work here."""
        self.checkpoint_results([(result_id, digest)])

    def checkpoint_results(self, receipts: Sequence[tuple[str, str]]) -> None:
        """Atomically acknowledge a saved group; a missing receipt rolls it back."""
        if not receipts:
            return
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                for result_id, digest in receipts:
                    changed = conn.execute(
                        "UPDATE job_results SET committed=1 WHERE result_id=? AND payload_digest=?",
                        (result_id, digest),
                    ).rowcount
                    if not changed:
                        raise ValueError(
                            "Computed result is missing or has a different digest"
                        )

    # --- Reads ---

    def get(self, job_id: str) -> JobRow:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return _row_to_jobrow(row, self.persistence)

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
        return _row_to_jobrow(row, self.persistence) if row else None

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
        return [_row_to_jobrow(r, self.persistence) for r in rows]

    # --- Boot sweep + pruning ---

    def mark_running_jobs_as_crashed(self) -> int:
        """Recover abandoned owners, preserving live runtimes and terminal rows.

        Legacy unowned rows keep their historical running/cancelling sweep.
        Owned queued jobs are also abandoned when their runtime lease is gone.
        """
        from core.jobs.ownership import acquire_owner
        from core.project_lock import LockUnavailableError

        if self.persistence == "session_only":
            return 0
        now = time.time()
        with self._connect() as conn:
            owners = conn.execute(
                "SELECT DISTINCT owner_id FROM jobs WHERE owner_id IS NOT NULL "
                "AND status IN ('queued', 'running', 'cancelling')"
            ).fetchall()
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'crashed',
                    error = 'server restarted while in flight',
                    finished_at = ?,
                    updated_at = ?,
                    queue_position = NULL,
                    blocking_job_id = NULL
                WHERE owner_id IS NULL AND status IN ('running', 'cancelling')
                """,
                (now, now),
            )
            recovered = cur.rowcount
        for row in owners:
            owner_id = row[0]
            try:
                lease = acquire_owner(owner_id)
            except LockUnavailableError:
                continue
            try:
                with self._connect() as conn:
                    recovered += conn.execute(
                        "UPDATE jobs SET status='crashed', error='job runtime exited', "
                        "finished_at=?, updated_at=?, queue_position=NULL, blocking_job_id=NULL "
                        "WHERE owner_id=? AND status IN ('queued','running','cancelling')",
                        (now, now, owner_id),
                    ).rowcount
            finally:
                lease.close()
        return recovered

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
