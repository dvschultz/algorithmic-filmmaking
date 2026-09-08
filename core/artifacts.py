"""Content-addressed derived files with durable project references and pins.

Writers must hold a pin before publishing. File publication and registration
share the collector's SQLite transaction; a project records its manifest before
the writer releases its pin. Unknown files and replaced inodes are never swept.
"""

from contextlib import contextmanager
from collections import deque
from hashlib import sha256
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import BinaryIO, Callable, Iterable, Iterator, TYPE_CHECKING
from uuid import uuid4
import weakref

from models.analysis_record import ANALYSIS_FIELDS, AnalysisRecord, ArtifactRef

if TYPE_CHECKING:
    from models.sequence import Sequence, SequenceClip

logger = logging.getLogger(__name__)
_retired_pins: deque[tuple[Path, str]] = deque()


def _drain_retired_pins() -> None:
    """Try GC cleanup without waiting for a transaction on this same thread."""
    for _ in range(len(_retired_pins)):
        try:
            database, owner = _retired_pins.popleft()
        except IndexError:
            break  # Another thread drained the remaining entries.
        if not database.is_file():
            continue
        try:
            db = sqlite3.connect(database, timeout=0, isolation_level=None)
            try:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("DELETE FROM owners WHERE id=? AND kind='pin'", (owner,))
            finally:
                db.close()
        except sqlite3.Error:
            # Keep the durable pin, then retry after the next store transaction.
            _retired_pins.append((database, owner))


def document_references(document: object) -> tuple[ArtifactRef, ...]:
    """Retain recognizable references, including those in unknown record versions."""
    refs: dict[str, ArtifactRef] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in ("artifact", "prerender_artifact"):
                artifact = value.get(key)
                if isinstance(artifact, dict):
                    try:
                        ref = ArtifactRef.from_dict(artifact)
                        refs[ref.digest] = ref
                    except (ValueError, TypeError, KeyError):
                        pass
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(document)
    return tuple(refs.values())


def analysis_references(targets: Iterable[object]) -> tuple[ArtifactRef, ...]:
    refs: dict[str, ArtifactRef] = {}
    for target in targets:
        for record in getattr(target, "analysis_records", {}).values():
            if isinstance(record, AnalysisRecord):
                if record.artifact is not None:
                    refs[record.artifact.digest] = record.artifact
            else:
                for ref in document_references(record.to_dict()):
                    refs[ref.digest] = ref
    return tuple(refs.values())


def sequence_references(sequences: Iterable["Sequence | None"]) -> tuple[ArtifactRef, ...]:
    return tuple({entry.prerender_artifact
                  for sequence in sequences if sequence is not None
                  for entry in sequence.get_all_clips()
                  if entry.prerender_artifact is not None})


def restore_prerender_projections(sequences: Iterable["Sequence | None"], store: "ArtifactStore | None" = None) -> None:
    """Restore managed playback paths; keep missing references for recovery."""
    for sequence in sequences:
        if sequence is None:
            continue
        for entry in sequence.get_all_clips():
            if entry.prerender_artifact is not None:
                if store is None:
                    store = ArtifactStore()
                try:
                    entry.prerendered_path = str(store.path_for(entry.prerender_artifact))
                except (ArtifactUnavailable, OSError):
                    entry.prerendered_path = None


def bind_prerender_path(entry: "SequenceClip", path: Path | None, store: "ArtifactStore | None" = None) -> None:
    """Attach registered content to an unpublished sequence draft without hashing on the GUI thread."""
    entry.prerendered_path = str(path) if path is not None else None
    entry.prerender_artifact = None
    entry._unreadable_prerender_artifact = None
    if path is not None:
        store = store or ArtifactStore()
        try:
            entry.prerender_artifact = store.reference_for_path(path, media_type="video/mp4")
        except (ArtifactUnavailable, OSError):
            entry.prerendered_path = None


class ArtifactLease:
    """An idempotent lifetime pin for detached worker inputs."""

    def __init__(self, references: Iterable[ArtifactRef], root: Path | None = None) -> None:
        refs = tuple(references)
        self._release: weakref.finalize | None = None
        if refs:
            store = ArtifactStore(root)
            owner = store.create_pin(refs)
            self._release = weakref.finalize(self, store.retire_pin, owner)

    @classmethod
    def for_snapshot(cls, snapshot: dict) -> "ArtifactLease":
        targets = (*snapshot.get("clips", ()), *snapshot.get("frames", ()), *snapshot.get("audio_sources", ()))
        root = snapshot.get("extra_data", {}).get("_artifact_store_root")
        sequences = snapshot.get("extra_data", {}).get("_all_sequences", [snapshot.get("sequence")])
        return cls(analysis_references(targets) + sequence_references(sequences), Path(root) if root else None)

    def close(self) -> None:
        if self._release is not None:
            self._release()

    def __enter__(self) -> "ArtifactLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _array_records(document: dict) -> Iterator[tuple[dict, str, AnalysisRecord]]:
    for target in document.get("clips", []):
        for operation in ("embeddings", "boundary_embeddings", "face_embeddings"):
            data = target.get("analysis_records", {}).get(operation)
            if data is None:
                fields = ANALYSIS_FIELDS[operation]
                projection = {field: target.get(field) for field in fields}
                if not any(value is not None for field, value in projection.items() if field != "embedding_model"):
                    continue
                # Unsaved legacy consumers may publish arrays without records.
                # Manage their storage without claiming verified provenance.
                data = AnalysisRecord.legacy(projection).to_dict()
                target.setdefault("analysis_records", {})[operation] = data
            try:
                record = AnalysisRecord.from_dict(data)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if record.state in ("succeeded", "missing") or (
                record.state == "failed" and (record.artifact is not None or any(
                    target.get(field) is not None for field in ANALYSIS_FIELDS[operation]
                    if field != "embedding_model"
                ))
            ):
                yield target, operation, record


def _externalize_arrays(document: dict, store: "ArtifactStore", pin: str) -> None:
    from dataclasses import replace

    for target, operation, record in _array_records(document):
        fields = ANALYSIS_FIELDS[operation]
        projection = {field: target.get(field) for field in fields}
        vectors = tuple(field for field in fields if field != "embedding_model")
        has_projection = any(projection[field] is not None for field in vectors)
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        changed = record.artifact is None and record.value != projection
        if has_projection and record.artifact is not None and record.artifact.digest != sha256(encoded).hexdigest():
            try:
                changed = json.loads(store.read_bytes(record.artifact)) != projection
            except (ArtifactUnavailable, OSError, ValueError):
                changed = True
        if has_projection and changed and record.state != "failed":
            # Legacy consumers can still edit projected fields. Preserve those
            # values without attributing their edits to the previous model run.
            record = AnalysisRecord.legacy(projection)
        elif has_projection and changed and record.state == "failed":
            # Old display vectors survive a failed refresh, but the failed
            # attempt must never become reusable through their storage.
            record = replace(record, value_json=None, artifact=None)
        if record.artifact is None:
            if not has_projection:
                continue
            ref = store.put_bytes(encoded, pin=pin, media_type="application/json")
            record = replace(record, value_json=None, artifact=ref)
        elif has_projection and record.artifact.digest == sha256(encoded).hexdigest() and not store.available_fast(record.artifact):
            # The live projection can reconstruct an exact lost payload without
            # attributing any new computation or changing its semantic identity.
            store.put_bytes(encoded, pin=pin, media_type=record.artifact.media_type)
        target["analysis_records"][operation] = record.to_dict()
        for field in vectors:
            target.pop(field, None)


@contextmanager
def manifest_write(
    project_path: Path, document: dict, root: Path | None = None, *,
    portable: bool = False, cancel_check: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Bridge atomic project publication and the independent artifact index.

    On an uncertain publication failure the durable pin remains: the destination
    may already contain the new references. A successful index update transfers
    ownership to the closed-project manifest before releasing this safety pin.
    """
    if portable:
        for collection in ("clips", "frames", "audio_sources"):
            for target in document.get(collection, []):
                for operation, data in tuple(target.get("analysis_records", {}).items()):
                    try:
                        record = AnalysisRecord.from_dict(data)
                    except (ValueError, TypeError, KeyError, AttributeError):
                        continue
                    if record.input_json is not None:
                        # Semantic identity survives relocation; stamps and paths
                        # describe the original files, never the portable copies.
                        target["analysis_records"][operation] = {**data, "input_json": None}
    refs = document_references(document)
    has_arrays = any(_array_records(document))
    if root is None and not refs and not has_arrays:
        # An ordinary project must not initialize cache settings or storage.
        # Replacing an artifact-bearing manifest still releases its old refs.
        previous_refs = document_references(json.loads(project_path.read_text())) if project_path.exists() else ()
        if not previous_refs:
            yield
            return
    if root is None:
        from core.paths import get_artifact_store_dir

        root = get_artifact_store_dir()
    if not refs and not has_arrays and not (root / "index.sqlite3").exists():
        yield
        return
    store = ArtifactStore(root)
    pin = store.create_manifest_pin(project_path, refs)
    _externalize_arrays(document, store, pin)
    refs = document_references(document)
    if portable:
        for ref in refs:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Bundle export cancelled")
            store.copy_to(ref, project_path.parent / "artifacts" / f"{ref.digest}.blob", cancel_check=cancel_check)
    yield
    try:
        store.set_manifest(project_path, refs)
        store.release_manifest_pins(project_path)
    except Exception:
        logger.warning("Project saved; artifact manifest update pending, safety pin retained: %s", pin, exc_info=True)


class ArtifactUnavailable(ValueError):
    """A derived file is missing, corrupt, or no longer managed by this store."""


def _stamp(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from core.paths import get_artifact_store_dir

            root = get_artifact_store_dir()
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        if self.objects.is_symlink():
            raise ArtifactUnavailable("Artifact object directory must not be a symlink")
        self.objects.mkdir(exist_ok=True)
        self.database = self.root / "index.sqlite3"
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    digest TEXT PRIMARY KEY, size INTEGER NOT NULL,
                    stamp TEXT NOT NULL, filename TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS owners (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('project', 'pin'))
                );
                CREATE TABLE IF NOT EXISTS refs (
                    owner TEXT NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
                    digest TEXT NOT NULL, PRIMARY KEY(owner, digest)
                );
                CREATE INDEX IF NOT EXISTS refs_digest ON refs(digest);
                CREATE TABLE IF NOT EXISTS pending_manifests (
                    owner TEXT PRIMARY KEY REFERENCES owners(id) ON DELETE CASCADE,
                    project_path TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        _drain_retired_pins()
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    yield db
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
        finally:
            _drain_retired_pins()

    def _path(self, digest: str, filename: str | None = None) -> Path:
        # ArtifactRef validates names before any digest reaches a filesystem.
        ArtifactRef(digest, 0)
        if self.objects.is_symlink() or self.objects.resolve().parent != self.root:
            raise ArtifactUnavailable("Artifact object directory changed")
        if filename is not None:
            import re

            if not re.fullmatch(digest + r"-[0-9a-f]{32}\.blob", filename):
                raise ArtifactUnavailable("Invalid managed artifact filename")
        return self.objects / (filename or digest)

    def create_pin(self, references: Iterable[ArtifactRef] = ()) -> str:
        owner = "pin:" + uuid4().hex
        with self._transaction() as db:
            db.execute("INSERT INTO owners VALUES (?, 'pin')", (owner,))
            db.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?)", ((owner, ref.digest) for ref in references))
        return owner

    @contextmanager
    def pin(self, references: Iterable[ArtifactRef] = ()) -> Iterator[str]:
        owner = self.create_pin(references)
        try:
            yield owner
        finally:
            self.release_pin(owner)

    def release_pin(self, owner: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM owners WHERE id=? AND kind='pin'", (owner,))

    def retire_pin(self, owner: str) -> None:
        """Release a finalized owner's pin now, or after an active transaction."""
        _retired_pins.append((self.database, owner))
        _drain_retired_pins()

    def replace_pin(self, owner: str, references: Iterable[ArtifactRef]) -> None:
        refs = tuple(references)
        with self._transaction() as db:
            self._require_pin(db, owner)
            db.execute("DELETE FROM refs WHERE owner=?", (owner,))
            db.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?)", ((owner, ref.digest) for ref in refs))

    def create_manifest_pin(self, path: Path, references: Iterable[ArtifactRef]) -> str:
        owner = "pin:" + uuid4().hex
        with self._transaction() as db:
            db.execute("INSERT INTO owners VALUES (?, 'pin')", (owner,))
            db.execute("INSERT INTO pending_manifests VALUES (?, ?)", (owner, str(path.expanduser().resolve())))
            db.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?)", ((owner, ref.digest) for ref in references))
        return owner

    def release_manifest_pins(self, path: Path) -> None:
        """Reconcile interrupted saves while the caller owns the project writer."""
        with self._transaction() as db:
            db.execute("DELETE FROM owners WHERE id IN (SELECT owner FROM pending_manifests WHERE project_path=?)", (str(path.expanduser().resolve()),))

    def set_manifest(self, project_path: Path, references: Iterable[ArtifactRef]) -> None:
        """Retain a known project's references even while its media is offline."""
        owner = "project:" + str(project_path.expanduser().resolve())
        refs = tuple(references)
        with self._transaction() as db:
            db.execute("INSERT OR IGNORE INTO owners VALUES (?, 'project')", (owner,))
            db.execute("DELETE FROM refs WHERE owner=?", (owner,))
            db.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?)", ((owner, ref.digest) for ref in refs))

    def retain_loaded_manifest(self, project_path: Path, references: Iterable[ArtifactRef]) -> None:
        """Loading may race a save; merge read references without dropping newer ones."""
        owner = "project:" + str(project_path.expanduser().resolve())
        with self._transaction() as db:
            db.execute("INSERT OR IGNORE INTO owners VALUES (?, 'project')", (owner,))
            db.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?)", ((owner, ref.digest) for ref in references))

    @staticmethod
    def _require_pin(db: sqlite3.Connection, owner: str) -> None:
        if db.execute("SELECT 1 FROM owners WHERE id=? AND kind='pin'", (owner,)).fetchone() is None:
            raise ValueError("Artifact publication requires a live writer pin")

    def put_bytes(
        self, data: bytes, *, pin: str, media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        return self._put(io.BytesIO(data), pin=pin, media_type=media_type)

    def put_file(
        self, path: Path, *, pin: str, media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        path = Path(path).resolve()
        before = _stamp(path)

        def verify_source() -> None:
            if _stamp(path) != before:
                raise ArtifactUnavailable("Source changed while staging its derived copy")

        with path.open("rb") as stream:
            return self._put(stream, pin=pin, media_type=media_type, verify_source=verify_source)

    def _put(
        self, source: BinaryIO, *, pin: str, media_type: str,
        verify_source: Callable[[], None] | None = None,
    ) -> ArtifactRef:
        with self._transaction() as db:
            self._require_pin(db, pin)
        # Slow copying stays outside the writer lock. The pin is checked again
        # immediately before publishing; cancelled producers cannot publish late.
        self._path("0" * 64)
        fd, name = tempfile.mkstemp(prefix=".stage-", suffix=".tmp", dir=self.objects)
        temporary = Path(name)
        digest, size = sha256(), 0
        try:
            with os.fdopen(fd, "wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if verify_source is not None:
                verify_source()
            ref = ArtifactRef(digest.hexdigest(), size, media_type)
            with self._transaction() as db:
                self._require_pin(db, pin)
                existing = db.execute("SELECT size, stamp, filename FROM objects WHERE digest=?", (ref.digest,)).fetchone()
                if existing is not None:
                    previous = self._path(ref.digest, existing[2])
                    try:
                        unchanged = list(_stamp(previous)) == json.loads(existing[1])
                    except OSError:
                        unchanged = False
                    if unchanged and existing[0] == size:
                        db.execute("INSERT OR IGNORE INTO refs VALUES (?, ?)", (pin, ref.digest))
                        return ref
                # A crash after rename but before the database commit can leave
                # an untracked file. A fresh physical name permits safe retry
                # without deleting or adopting that unknown file.
                filename = f"{ref.digest}-{uuid4().hex}.blob"
                destination = self._path(ref.digest, filename)
                os.replace(temporary, destination)
                if os.name != "nt":
                    directory = os.open(self.objects, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                db.execute("INSERT OR REPLACE INTO objects VALUES (?, ?, ?, ?)", (ref.digest, size, json.dumps(_stamp(destination)), filename))
                db.execute("INSERT OR IGNORE INTO refs VALUES (?, ?)", (pin, ref.digest))
            return ref
        finally:
            temporary.unlink(missing_ok=True)

    def _verify(self, db: sqlite3.Connection, ref: ArtifactRef) -> Path:
        row = db.execute("SELECT size, filename FROM objects WHERE digest=?", (ref.digest,)).fetchone()
        if row is None:
            raise ArtifactUnavailable("Artifact is unregistered")
        path = self._path(ref.digest, row[1])
        try:
            if row[0] != ref.size or not stat.S_ISREG(path.lstat().st_mode):
                raise ArtifactUnavailable("Artifact is missing or unregistered")
            before = _stamp(path)
            digest, size = sha256(), 0
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            if size != ref.size or digest.hexdigest() != ref.digest or _stamp(path) != before:
                raise ArtifactUnavailable("Artifact content is corrupt or changed")
        except OSError as exc:
            raise ArtifactUnavailable("Artifact is unavailable") from exc
        return path

    def path_for(self, ref: ArtifactRef) -> Path:
        """Verify a path; callers must pin it for the entire period of use."""
        with self._transaction() as db:
            return self._verify(db, ref)

    def reference_for_path(self, path: Path, *, media_type: str = "application/octet-stream") -> ArtifactRef | None:
        """Recognize unchanged registered content without rehashing on the owner thread."""
        if path.parent.resolve() != self.objects.resolve():
            return None
        with self._transaction() as db:
            row = db.execute("SELECT digest,size,stamp FROM objects WHERE filename=?", (path.name,)).fetchone()
            if row is None:
                return None
            ref = ArtifactRef(row[0], row[1], media_type)
            if not stat.S_ISREG(path.lstat().st_mode) or list(_stamp(path)) != json.loads(row[2]):
                raise ArtifactUnavailable("Registered media changed before binding")
            return ref

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        with self._transaction() as db:
            data = self._verify(db, ref).read_bytes()
            if len(data) != ref.size or sha256(data).hexdigest() != ref.digest:
                raise ArtifactUnavailable("Artifact changed while reading")
            return data

    def available(self, ref: ArtifactRef) -> bool:
        try:
            self.path_for(ref)
            return True
        except (ArtifactUnavailable, OSError):
            return False

    def available_fast(self, ref: ArtifactRef) -> bool:
        """Owner-thread check: only unchanged files previously hashed by this store."""
        try:
            with self._connection() as db:
                row = db.execute("SELECT size, stamp, filename FROM objects WHERE digest=?", (ref.digest,)).fetchone()
            if row is None or row[0] != ref.size:
                return False
            path = self._path(ref.digest, row[2])
            return stat.S_ISREG(path.lstat().st_mode) and list(_stamp(path)) == json.loads(row[1])
        except (ArtifactUnavailable, OSError, sqlite3.Error):
            return False

    def copy_to(self, ref: ArtifactRef, destination: Path, *, cancel_check: Callable[[], bool] | None = None) -> None:
        """Copy a pinned object to a new bundle file and verify the actual copy."""
        with self.pin((ref,)):
            source = self.path_for(ref)
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("xb")
            try:
                digest, size = sha256(), 0
                with output, source.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        if cancel_check is not None and cancel_check():
                            raise InterruptedError("Artifact copy cancelled")
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if digest.hexdigest() != ref.digest or size != ref.size:
                    raise ArtifactUnavailable("Artifact changed during bundle export")
            except BaseException:
                destination.unlink(missing_ok=True)
                raise

    def restore_from(self, ref: ArtifactRef, source: Path, *, pin: str) -> None:
        """Import a portable copy only if its content matches the manifest."""
        restored = self.put_file(source, pin=pin, media_type=ref.media_type)
        if restored != ref:
            raise ArtifactUnavailable("Bundled artifact does not match its reference")

    def collect(self) -> list[str]:
        """Delete only unreferenced, unchanged regular files owned by this index."""
        removed = []
        with self._transaction() as db:
            candidates = db.execute("SELECT digest, stamp, filename FROM objects WHERE NOT EXISTS (SELECT 1 FROM refs WHERE refs.digest=objects.digest) ORDER BY digest").fetchall()
            for digest, stamp_json, filename in candidates:
                path = self._path(digest, filename)
                try:
                    value = path.lstat()
                    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or list(_stamp(path)) != json.loads(stamp_json):
                        continue
                    path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    removed.append(digest)
                db.execute("DELETE FROM objects WHERE digest=?", (digest,))
        return removed
