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
from typing import Iterator, Optional, Sequence, TYPE_CHECKING
from core.jobs.spec import OperationSpec
from core.jobs.errors import StaleJobResult
from core.jobs.artifact_inputs import referenced_artifacts

if TYPE_CHECKING:
    from core.artifacts import ArtifactStore

# Keep short job-store transactions serialized within this process. Concurrent
# SQLite connection open/close deadlocked in the macOS runtime's native VFS.
# Inference and project-file writes happen outside these connection scopes.
_connection_lock = threading.RLock()

# Bound SQLite receipts without changing their logical recovery representation.
_RESULT_INLINE_BYTES = 16 * 1024

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


def _row_to_jobrow(row: dict, persistence: str = "job_history") -> JobRow:
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
        self._artifact_root: Path | None = None
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
        store._artifact_root = None
        store._memory_connection = sqlite3.connect(
            ":memory:", isolation_level=None, check_same_thread=False
        )
        store._memory_connection.row_factory = sqlite3.Row
        store._memory_connection.execute("PRAGMA temp_store=MEMORY")
        store._init_schema()
        return store

    def close(self) -> None:
        """Discard session history after its runtime workers have stopped."""
        pins: list[str | None] = []
        with self._memory_lock:
            if self._memory_connection is not None:
                for row in self._memory_connection.execute(
                    "SELECT input_artifact_pin,result_artifact_pin FROM jobs"
                ):
                    pins.extend(row)
                pins.extend(row[0] for row in self._memory_connection.execute(
                    "SELECT artifact_pin FROM job_results"
                ))
                self._memory_connection.close()
                self._memory_connection = None
        self._release_job_pins(pins)

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
        with _connection_lock:
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
        from core.jobs.schema import INITIAL_SCHEMA, RESULT_SCHEMA, DOWNLOAD_SCHEMA

        sql = INITIAL_SCHEMA
        with self._connect() as conn:
            conn.executescript(sql)
            conn.executescript(RESULT_SCHEMA)
            conn.executescript(DOWNLOAD_SCHEMA)
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
                if "owner_id" not in columns:
                    conn.execute("ALTER TABLE jobs ADD COLUMN owner_id TEXT")
                if "operation_json" not in columns:
                    conn.execute("ALTER TABLE jobs ADD COLUMN operation_json TEXT")
                for column in ("input_artifact_json", "input_artifact_pin",
                               "result_artifact_json", "result_artifact_pin"):
                    if column not in columns:
                        conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
                result_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(job_results)")
                }
                for column in ("spec_artifact_json", "payload_artifact_json", "artifact_pin"):
                    if column not in result_columns:
                        conn.execute(f"ALTER TABLE job_results ADD COLUMN {column} TEXT")

    # --- Mutations ---

    def _artifact_store(self) -> ArtifactStore:
        from core.artifacts import ArtifactStore

        if self._artifact_root is None:
            store = ArtifactStore(self.db_path.parent / "artifacts"
                                  if self.persistence == "job_history" else None)
            self._artifact_root = store.root
            return store
        return ArtifactStore(self._artifact_root)

    def _stage_job_body(self, fields: dict) -> tuple[dict, str | None, str | None]:
        """Stage a group of logical JSON columns under one durable owner."""
        data = json.dumps(fields, separators=(",", ":")).encode("utf-8")
        refs = referenced_artifacts(fields.values())
        external = self.persistence == "job_history" and len(data) > _RESULT_INLINE_BYTES
        if not external and not refs:
            return fields, None, None
        artifacts = self._artifact_store()
        pin = artifacts.create_pin(refs)
        if not external:
            return fields, None, pin
        try:
            ref = artifacts.put_bytes(data, pin=pin, media_type="application/json")
        except BaseException:
            artifacts.release_pin(pin)
            raise
        return dict.fromkeys(fields, ""), json.dumps(ref.to_dict()), pin

    def _release_job_pins(self, pins: Sequence[str | None]) -> None:
        owners = {pin for pin in pins if pin is not None}
        if owners:
            artifacts = self._artifact_store()
            for owner in owners:
                artifacts.release_pin(owner)

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
        stored, reference, pin = self._stage_job_body(
            {"args_json": row.args_json, "operation_json": row.operation_json}
        )
        try:
            self._insert_job(row, owner_id, stored, reference, pin)
        except sqlite3.IntegrityError:
            # A rejected single INSERT cannot have published this new owner.
            self._release_job_pins([pin])
            raise
        return row

    def _insert_job(self, row: JobRow, owner_id: str | None, stored: dict,
                    reference: str | None, pin: str | None) -> None:
        # Other publication errors are uncertain; retain the staged owner.
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, idempotency_key, args_json,
                    project_path, project_mtime_at_start, progress,
                    status_message, queue_position, blocking_job_id,
                    created_at, updated_at, owner_id, operation_json,
                    input_artifact_json, input_artifact_pin
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.id,
                    row.kind,
                    row.status,
                    row.idempotency_key,
                    stored["args_json"],
                    row.project_path,
                    row.project_mtime_at_start,
                    row.progress,
                    row.status_message,
                    row.queue_position,
                    row.blocking_job_id,
                    row.created_at,
                    row.updated_at,
                    owner_id,
                    stored["operation_json"],
                    reference,
                    pin,
                ),
            )

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
        result_pin = None
        if result is not None:
            stored, result_reference, result_pin = self._stage_job_body(
                {"result_json": json.dumps(result, default=str)}
            )
            sets.append("result_json = ?")
            params.append(stored["result_json"])
            sets.extend(["result_artifact_json = ?", "result_artifact_pin = ?"])
            params.extend([result_reference, result_pin])
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
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                previous = conn.execute(
                    "SELECT result_artifact_pin FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
                changed = conn.execute(
                    f"UPDATE jobs SET {', '.join(sets)} WHERE id = ? "
                    f"AND status NOT IN ({','.join('?' for _ in terminal_states)}) "
                    "AND NOT (status = ? AND ? = ?)",
                    params,
                ).rowcount
        if not changed:
            self._release_job_pins([result_pin])
            if previous is None:
                raise JobNotFoundError(job_id)
            return False
        if result is not None and previous is not None:
            self._release_job_pins([previous[0]])
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
        return bool(self._delete_jobs("id = ?", [job_id]))

    def _delete_jobs(self, predicate: str, params: Sequence) -> int:
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    f"SELECT input_artifact_pin,result_artifact_pin FROM jobs WHERE {predicate}", params
                ).fetchall()
                deleted = conn.execute(f"DELETE FROM jobs WHERE {predicate}", params).rowcount
        # A deletion must be durable before its payload owners can be released.
        self._release_job_pins([pin for row in rows for pin in row])
        return deleted

    def get_result(self, result_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_results WHERE result_id = ?", (result_id,)
            ).fetchone()
        return self._hydrate_result(dict(row)) if row is not None else None

    def _hydrate_result(self, row: dict) -> dict:
        """Resolve payloads outside the job connection to avoid lock inversion."""
        from models.analysis_record import ArtifactRef

        for field in ("spec", "payload"):
            reference = row.pop(f"{field}_artifact_json", None)
            if reference is None:
                continue
            if row[f"{field}_json"] != "" or not row.get("artifact_pin"):
                raise StaleJobResult("Conflicting job result storage metadata")
            try:
                ref = ArtifactRef.from_dict(json.loads(reference))
                artifacts = self._artifact_store()
                row[f"{field}_json"] = artifacts.read_bytes(ref).decode("utf-8")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise StaleJobResult("Job result payload is unavailable or corrupt") from exc
        row.pop("artifact_pin", None)
        return row

    def get_pending_results(self, result_ids: Sequence[str]) -> list[dict]:
        """Read pending receipts in bounded queries using one connection."""
        if not result_ids:
            return []
        rows: list[dict] = []
        with self._connect() as conn:
            for offset in range(0, len(result_ids), 500):
                batch = result_ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    dict(row)
                    for row in conn.execute(
                        f"SELECT * FROM job_results WHERE committed=0 AND result_id IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        return [self._hydrate_result(row) for row in rows]

    def get_download_receipt(self, request_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM download_receipts WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def record_download_receipt(
        self, request_id: str, spec_json: str, payload_json: str, digest: str
    ) -> None:
        """Replace a verified file receipt after downloading a missing output."""
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                "INSERT INTO download_receipts "
                "(request_id,spec_json,payload_json,payload_digest,updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(request_id) DO UPDATE SET spec_json=excluded.spec_json, "
                "payload_json=excluded.payload_json, payload_digest=excluded.payload_digest, "
                "updated_at=excluded.updated_at",
                (request_id, spec_json, payload_json, digest, time.time()),
            )

    def record_result(
        self, result_id: str, spec_json: str, payload_json: str, digest: str
    ) -> dict:
        """Persist immutable computed output before publishing it to a project."""
        values = {"spec": spec_json, "payload": payload_json}
        references: dict[str, str | None] = {"spec": None, "payload": None}
        artifacts = None
        pin = None
        refs = referenced_artifacts(values.values())
        external = self.persistence == "job_history" and any(
            len(value.encode("utf-8")) > _RESULT_INLINE_BYTES for value in values.values()
        )
        if external or refs:
            # The default jobs.db lives in the configured cache directory. Keep
            # custom databases self-contained too; session-only stores stay inline.
            artifacts = self._artifact_store()
            pin = artifacts.create_pin(refs)
            try:
                for field, value in values.items():
                    data = value.encode("utf-8")
                    if external and len(data) > _RESULT_INLINE_BYTES:
                        ref = artifacts.put_bytes(data, pin=pin, media_type="application/json")
                        references[field] = json.dumps(ref.to_dict(), sort_keys=True)
                        values[field] = ""
            except BaseException:
                artifacts.release_pin(pin)
                raise
        # Do not release the pin on uncertain SQLite publication: the row may
        # already be durable. Proven duplicate inserts can release their new pin.
        with self._connect() as conn:
            conn.execute("PRAGMA synchronous=FULL")
            inserted = conn.execute(
                "INSERT OR IGNORE INTO job_results "
                "(result_id,spec_json,payload_json,payload_digest,created_at,"
                "spec_artifact_json,payload_artifact_json,artifact_pin) VALUES (?,?,?,?,?,?,?,?)",
                (result_id, values["spec"], values["payload"], digest, time.time(),
                 references["spec"], references["payload"], pin),
            ).rowcount
        if not inserted and artifacts is not None and pin is not None:
            artifacts.release_pin(pin)
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

    def _read_jobs(self, sql: str, params: Sequence = ()) -> list[JobRow]:
        from models.analysis_record import ArtifactRef

        artifacts = None
        lease = None
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
            if any(row[f"{group}_artifact_json"] is not None
                   for row in rows for group in ("input", "result")):
                # Repeat under the writer lock, then acquire read ownership before
                # a concurrent delete/replacement can retire the persisted pins.
                # Staging releases artifact transactions before opening this DB;
                # deletion releases pins only after closing it (no reverse order).
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
                    try:
                        refs = [ArtifactRef.from_dict(json.loads(row[f"{group}_artifact_json"]))
                                for row in rows for group in ("input", "result")
                                if row[f"{group}_artifact_json"] is not None]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise StaleJobResult("Job history reference is corrupt") from exc
                    if refs:
                        artifacts = self._artifact_store()
                        lease = artifacts.create_pin(refs)
        try:
            for row in rows:
                for group, fields in (("input", ("args_json", "operation_json")),
                                      ("result", ("result_json",))):
                    reference = row[f"{group}_artifact_json"]
                    if reference is None:
                        continue
                    if not row[f"{group}_artifact_pin"] or any(row[field] != "" for field in fields):
                        raise StaleJobResult("Conflicting job history storage metadata")
                    try:
                        assert artifacts is not None
                        ref = ArtifactRef.from_dict(json.loads(reference))
                        payload = json.loads(artifacts.read_bytes(ref))
                        if not isinstance(payload, dict) or set(payload) != set(fields):
                            raise ValueError("Unexpected job history fields")
                        if any(not isinstance(value, str) and not (key == "operation_json" and value is None)
                               for key, value in payload.items()):
                            raise ValueError("Invalid job history JSON column")
                        row.update(payload)
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        raise StaleJobResult("Job history payload is unavailable or corrupt") from exc
            return [_row_to_jobrow(row, self.persistence) for row in rows]
        finally:
            if artifacts is not None and lease is not None:
                artifacts.release_pin(lease)

    def get(self, job_id: str) -> JobRow:
        rows = self._read_jobs("SELECT * FROM jobs WHERE id = ?", [job_id])
        if not rows:
            raise JobNotFoundError(job_id)
        return rows[0]

    def find_by_idempotency(
        self,
        kind: str,
        project_path: Optional[str],
        idempotency_key: str,
    ) -> Optional[JobRow]:
        rows = self._read_jobs(
                """
                SELECT * FROM jobs
                WHERE kind = ?
                  AND idempotency_key = ?
                  AND (project_path IS ? OR project_path = ?)
                """,
                (kind, idempotency_key, project_path, project_path),
            )
        return rows[0] if rows else None

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
        return self._read_jobs(sql, params)

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
        return self._delete_jobs(
            "status IN ('completed', 'failed', 'cancelled', 'crashed') "
            "AND finished_at IS NOT NULL AND finished_at < ?", [cutoff]
        )
