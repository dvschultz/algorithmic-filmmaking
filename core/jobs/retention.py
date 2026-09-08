"""Persist receipt ownership across project saves before enabling reclamation."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Iterator, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from core.jobs.store import JobStore

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipt_manifests (
    project_path TEXT NOT NULL,
    result_id TEXT NOT NULL,
    PRIMARY KEY (project_path, result_id)
);
CREATE TABLE IF NOT EXISTS receipt_project_history (
    project_path TEXT NOT NULL,
    result_id TEXT NOT NULL,
    PRIMARY KEY (project_path, result_id)
);
CREATE TABLE IF NOT EXISTS receipt_pending_saves (
    token TEXT PRIMARY KEY,
    project_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipt_pending_refs (
    token TEXT NOT NULL REFERENCES receipt_pending_saves(token) ON DELETE CASCADE,
    result_id TEXT NOT NULL,
    PRIMARY KEY (token, result_id)
);
CREATE INDEX IF NOT EXISTS receipt_manifests_result ON receipt_manifests(result_id);
CREATE INDEX IF NOT EXISTS receipt_history_result ON receipt_project_history(result_id);
CREATE INDEX IF NOT EXISTS receipt_pending_result ON receipt_pending_refs(result_id);
"""


def document_receipts(document: dict) -> tuple[str, ...]:
    """Malformed receipt maps cannot establish that a project released results."""
    receipts = document.get("job_results", {})
    if not isinstance(receipts, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        or len(key) != 64 or len(value) != 64
        or any(char not in "0123456789abcdef" for char in key + value)
        for key, value in receipts.items()
    ):
        raise ValueError("Invalid project result receipts")
    return tuple(receipts)


@contextmanager
def retain_receipt_manifest(store: "JobStore", path: Path, document: dict) -> Iterator[None]:
    """Protect incoming refs before publication; replace saved refs afterwards.

    The caller holds the project writer and writes this exact document inside
    the context. Failed or uncertain publication deliberately retains its token.
    Historical path associations remain so reclamation can protect live undo
    state by independently acquiring every former owner's project writer.
    """
    canonical = str(path.expanduser().resolve())
    receipts = document_receipts(document)
    token = str(uuid4())
    with store._connect() as db:
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO receipt_pending_saves VALUES (?, ?)", (token, canonical))
            db.executemany("INSERT INTO receipt_pending_refs VALUES (?, ?)", ((token, rid) for rid in receipts))
            db.executemany("INSERT OR IGNORE INTO receipt_project_history VALUES (?, ?)", ((canonical, rid) for rid in receipts))
    yield
    try:
        with store._connect() as db:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM receipt_manifests WHERE project_path=?", (canonical,))
                db.executemany("INSERT INTO receipt_manifests VALUES (?, ?)", ((canonical, rid) for rid in receipts))
                # Only this publication's token is discharged. Older uncertain
                # saves need independent writer-owned reconciliation.
                db.execute("DELETE FROM receipt_pending_saves WHERE token=?", (token,))
    except Exception:
        logger.warning("Project saved; receipt ownership update remains pending", exc_info=True)


@contextmanager
def receipt_manifest_write(path: Path, document: dict) -> Iterator[None]:
    """Use an existing job cache only when this save adds or removes receipts."""
    receipts = document_receipts(document)
    if not receipts:
        # Keep ordinary project saves independent of cache/settings startup.
        previous = json.loads(path.read_text()) if path.is_file() else {}
        if not document_receipts(previous):
            yield
            return
    from core.settings import load_settings
    from core.jobs.store import JobStore

    database = load_settings().cache_dir / "jobs.db"
    if not database.is_file():
        # Imported projects may name receipts from another machine. There is
        # no local payload to reclaim, so do not create an empty cache.
        yield
        return
    store = JobStore(database)
    try:
        with retain_receipt_manifest(store, path, document):
            yield
    finally:
        store.close()


def retain_loaded_receipts(path: Path, receipts: dict[str, str]) -> None:
    """Merge a loaded copy's refs without dropping a concurrent newer save."""
    if not receipts:
        return
    references = document_receipts({"job_results": receipts})
    from core.settings import load_settings
    from core.jobs.store import JobStore

    database = load_settings().cache_dir / "jobs.db"
    if not database.is_file():
        return
    store = JobStore(database)
    canonical = str(path.expanduser().resolve())
    try:
        with store._connect() as db:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("BEGIN IMMEDIATE")
                for table in ("receipt_manifests", "receipt_project_history"):
                    db.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", ((canonical, rid) for rid in references))
    finally:
        store.close()


def reconcile_receipt_manifests(store: "JobStore") -> int:
    """Resolve abandoned saves from valid disk state under independent writers.

    Missing, unsupported, invalid, busy, or concurrently changed documents keep
    their old and pending owners. This does not acknowledge or delete results.
    """
    from core.project import _validate_project_structure
    from core.project_lock import ProjectBusyError, ProjectWriter
    from core.project_migrations import is_future_schema

    with store._connect() as db:
        paths = [row[0] for row in db.execute("SELECT DISTINCT project_path FROM receipt_pending_saves")]
    reconciled = 0
    for value in paths:
        path = Path(value)
        try:
            # Never borrow a caller's active writer, even on the same thread.
            with ProjectWriter(path):
                encoded = path.read_bytes()
                document = json.loads(encoded)
                if (
                    not isinstance(document, dict)
                    or not isinstance(document.get("id"), str)
                    or not document["id"]
                    or _validate_project_structure(document)
                    or is_future_schema(document["version"])
                ):
                    continue
                references = document_receipts(document)
                with store._connect() as db:
                    db.execute("PRAGMA synchronous=FULL")
                    with db:
                        db.execute("BEGIN IMMEDIATE")
                        if path.read_bytes() != encoded:
                            continue
                        db.execute("DELETE FROM receipt_manifests WHERE project_path=?", (value,))
                        db.executemany("INSERT INTO receipt_manifests VALUES (?, ?)", ((value, rid) for rid in references))
                        db.executemany("INSERT OR IGNORE INTO receipt_project_history VALUES (?, ?)", ((value, rid) for rid in references))
                        reconciled += db.execute("DELETE FROM receipt_pending_saves WHERE project_path=?", (value,)).rowcount
        except (ProjectBusyError, OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return reconciled


# An absent legacy owner is uncertainty, not evidence of release. Uncommitted
# receipts remain the recovery journal, irrespective of age or visible output.
_RECLAIMABLE = """
    committed=1 AND retention_managed=1 AND created_at < ?
    AND EXISTS (SELECT 1 FROM receipt_project_history h WHERE h.result_id=job_results.result_id)
    AND NOT EXISTS (SELECT 1 FROM receipt_manifests m WHERE m.result_id=job_results.result_id)
    AND NOT EXISTS (SELECT 1 FROM receipt_pending_refs p WHERE p.result_id=job_results.result_id)
    AND NOT EXISTS (SELECT 1 FROM jobs WHERE status IN ('queued','running','cancelling'))
"""


def purge_receipts(store: "JobStore", *, days: int = 30) -> int:
    """Explicitly reclaim obsolete receipts, preserving saved and live owners.

    Every historical project must be readable, supported, unchanged, and free
    of an independent writer. This protects closed copies and live undo state.
    Active jobs (including unowned legacy jobs) inhibit deletion. SQLite removal
    is durable before artifact pins are released; concurrent body readers retain
    their separate leases. No source files or project documents are deleted.
    """
    from contextlib import ExitStack
    import time
    from core.project import _validate_project_structure
    from core.project_lock import ProjectBusyError, ProjectWriter
    from core.project_migrations import is_future_schema

    if type(days) is not int or days < 0:
        raise ValueError("days must be a nonnegative integer")
    if store.persistence != "job_history":
        return 0
    reconcile_receipt_manifests(store)
    cutoff = time.time() - days * 86400
    last_id = ""
    deleted = 0
    while True:
        with store._connect() as db:
            candidates = [row[0] for row in db.execute(
                f"SELECT result_id FROM job_results WHERE {_RECLAIMABLE} AND result_id > ? ORDER BY result_id LIMIT 100",
                (cutoff, last_id),
            )]
        if not candidates:
            return deleted
        for rid in candidates:
            last_id = rid
            with store._connect() as db:
                paths = tuple(row[0] for row in db.execute(
                    "SELECT project_path FROM receipt_project_history WHERE result_id=? ORDER BY project_path", (rid,),
                ))
            if not paths:
                continue
            pin = None
            removed = False
            try:
                with ExitStack() as writers:
                    snapshots: dict[Path, bytes] = {}
                    for value in paths:
                        path = Path(value)
                        if str(path.resolve()) != value:
                            raise ValueError("Project path was retargeted")
                        writers.enter_context(ProjectWriter(path))
                        encoded = path.read_bytes()
                        document = json.loads(encoded)
                        if (
                            not isinstance(document, dict)
                            or not isinstance(document.get("id"), str)
                            or not document["id"]
                            or _validate_project_structure(document)
                            or is_future_schema(document["version"])
                            or rid in document_receipts(document)
                        ):
                            raise ValueError("Project has not verifiably released the receipt")
                        snapshots[path] = encoded
                    with store._connect() as db:
                        db.execute("PRAGMA synchronous=FULL")
                        with db:
                            db.execute("BEGIN IMMEDIATE")
                            # Save As/load may have registered another owner
                            # while these writers were acquired.
                            current_paths = tuple(row[0] for row in db.execute(
                                "SELECT project_path FROM receipt_project_history WHERE result_id=? ORDER BY project_path", (rid,),
                            ))
                            if current_paths != paths or any(path.read_bytes() != data for path, data in snapshots.items()):
                                continue
                            row = db.execute(
                                f"SELECT artifact_pin FROM job_results WHERE {_RECLAIMABLE} AND result_id=?", (cutoff, rid),
                            ).fetchone()
                            if row is None:
                                continue
                            pin = row[0]
                            db.execute("DELETE FROM job_results WHERE result_id=?", (rid,))
                            db.execute("DELETE FROM receipt_project_history WHERE result_id=?", (rid,))
                            removed = True
            except (ProjectBusyError, OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
            if removed:
                # Never hold the job transaction while retiring artifact owners.
                store._release_job_pins([pin])
                deleted += 1
