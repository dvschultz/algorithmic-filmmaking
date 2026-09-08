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
