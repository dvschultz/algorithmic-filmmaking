"""Registered derived media with durable cache and temporary reader ownership."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterator

from core.artifacts import ArtifactLease, ArtifactStore, ArtifactUnavailable
from models.analysis_record import ArtifactRef


def media_file_stamp(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return None


@dataclass(frozen=True)
class CachedMedia:
    path: Path
    reference: ArtifactRef
    lease: ArtifactLease
    stamp: tuple[int, ...] | None


class MediaCache:
    def __init__(self, root: Path, artifact_root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(artifact_root)
        self.database = root / "media-cache.sqlite3"
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS entries ("
                       "namespace TEXT NOT NULL, key TEXT NOT NULL, reference TEXT NOT NULL, "
                       "pin TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(namespace,key))")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, isolation_level=None, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def get(self, namespace: str, key: str) -> CachedMedia | None:
        lease = None
        try:
            with self._connection() as db, db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT reference FROM entries WHERE namespace=? AND key=?",
                                 (namespace, key)).fetchone()
                if row is None:
                    return None
                ref = ArtifactRef.from_dict(json.loads(row[0]))
                # Acquire the reader before a concurrent prune can drop the cache
                # owner. Publication never holds an artifact transaction here.
                lease = ArtifactLease([ref], self.artifacts.root)
            if not self.artifacts.available(ref):
                lease.close()
                return None
            path = self.artifacts.path_for(ref)
            return CachedMedia(path, ref, lease, media_file_stamp(path))
        except (ArtifactUnavailable, ValueError, KeyError, TypeError, OSError):
            if lease is not None:
                lease.close()
            return None

    def publish(self, namespace: str, key: str, source: Path) -> CachedMedia:
        if source.stat().st_size <= 0:
            raise ArtifactUnavailable("Rendered media is empty")
        pin = self.artifacts.create_pin()
        try:
            ref = self.artifacts.put_file(source, pin=pin, media_type="video/mp4")
        except BaseException:
            self.artifacts.release_pin(pin)
            raise
        # Uncertain cache publication retains the staged pin conservatively.
        with self._connection() as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT pin FROM entries WHERE namespace=? AND key=?",
                                  (namespace, key)).fetchone()
            db.execute("INSERT INTO entries VALUES (?,?,?,?,?) "
                       "ON CONFLICT(namespace,key) DO UPDATE SET reference=excluded.reference, "
                       "pin=excluded.pin,created_at=excluded.created_at",
                       (namespace, key, json.dumps(ref.to_dict()), pin, time.time()))
            lease = ArtifactLease([ref], self.artifacts.root)
        if previous is not None:
            self.artifacts.release_pin(previous[0])
        path = self.artifacts.path_for(ref)
        return CachedMedia(path, ref, lease, media_file_stamp(path))

    def prune(self, keep_latest: int = 5) -> int:
        if type(keep_latest) is not int or keep_latest < 0:
            raise ValueError("keep_latest must be a nonnegative integer")
        with self._connection() as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT namespace,key,pin FROM entries "
                              "ORDER BY namespace,created_at DESC,key").fetchall()
            counts: dict[str, int] = {}
            retired = []
            for namespace, key, pin in rows:
                counts[namespace] = counts.get(namespace, 0) + 1
                if counts[namespace] > keep_latest:
                    db.execute("DELETE FROM entries WHERE namespace=? AND key=?", (namespace, key))
                    retired.append(pin)
        for pin in retired:
            self.artifacts.release_pin(pin)
        self.artifacts.collect()
        return len(retired)
