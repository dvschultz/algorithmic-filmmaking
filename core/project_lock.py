"""OS-backed project writer scopes, independent of project-file replacement."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import errno
from hashlib import sha256
import os
from pathlib import Path
import sys
from threading import get_ident
from typing import BinaryIO, Iterator
import unicodedata


class ProjectBusyError(OSError):
    """Another writer owns this path or an alias of its current file."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Project is already open for writing: {path}")

    def to_dict(self) -> dict[str, str]:
        return {"code": "project_busy", "path": str(self.path), "message": str(self)}


def _lock_directory() -> Path:
    from core.paths import get_app_support_dir

    return get_app_support_dir() / "project-locks"


def _owner() -> tuple[int, int, int | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), id(task) if task is not None else None


def _path_key(path: Path) -> str:
    key = os.path.normcase(str(path))
    if sys.platform == "darwin":
        # Also serialize new-file aliases on the usual case-insensitive APFS.
        # Conservative on case-sensitive volumes: distinct case variants conflict.
        key = unicodedata.normalize("NFD", key).casefold()
    return key


def _lock(stream: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class ProjectWriter:
    """Own a canonical path and its file identity until explicitly closed.

    Lock records are permanent: deleting a record would split its lock identity.
    File identity ownership is transferred before publishing a replacement.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self.owner = _owner()
        self.closed = True
        self._path_lock: BinaryIO | None = None
        self._identity_lock: BinaryIO | None = None

    def _acquire_record(self, key: str) -> BinaryIO:
        directory = _lock_directory()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = directory / (sha256(key.encode()).hexdigest() + ".lock")
        fd = os.open(record, os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(fd, "r+b")
        try:
            _lock(stream)
        except OSError as exc:
            stream.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise ProjectBusyError(self.path) from exc
            raise
        return stream

    def _acquire_identity(self, path: Path) -> BinaryIO:
        stat = path.stat()
        return self._acquire_record(f"inode:{stat.st_dev}:{stat.st_ino}")

    def acquire(self) -> ProjectWriter:
        if not self.closed:
            raise RuntimeError("Writer is already acquired")
        self._path_lock = self._acquire_record(f"path:{_path_key(self.path)}")
        try:
            if self.path.exists():
                self._identity_lock = self._acquire_identity(self.path)
        except BaseException:
            self.close()
            raise
        self.closed = False
        return self

    def replace(self, temporary: Path) -> None:
        """Atomically replace the project while retaining writer ownership."""
        if self.closed or self.owner != _owner():
            raise RuntimeError("Project writer is not active in this execution context")
        replacement_lock = self._acquire_identity(temporary)
        try:
            os.replace(temporary, self.path)
        except BaseException:
            replacement_lock.close()
            raise
        if self._identity_lock is not None:
            self._identity_lock.close()
        self._identity_lock = replacement_lock

    def close(self) -> None:
        if not self.closed and self.owner != _owner():
            raise RuntimeError("Only the acquiring context can close a project writer")
        for stream in (self._identity_lock, self._path_lock):
            if stream is not None:
                stream.close()
        self._identity_lock = self._path_lock = None
        self.closed = True

    def __enter__(self) -> ProjectWriter:
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.close()


_writers: ContextVar[tuple[ProjectWriter, ...]] = ContextVar(
    "project_writers", default=()
)


@contextmanager
def project_writer(path: Path | str) -> Iterator[ProjectWriter]:
    """Reuse only a scope belonging to the current thread and async task."""
    canonical = Path(path).expanduser().resolve()
    for writer in _writers.get():
        if (
            not writer.closed
            and writer.path == canonical
            and writer.owner == _owner()
        ):
            yield writer
            return
    with ProjectWriter(canonical) as writer:
        token = _writers.set((*_writers.get(), writer))
        try:
            yield writer
        finally:
            _writers.reset(token)


def replace_project_file(temporary: Path | str, destination: Path | str) -> None:
    """Publish through the surrounding writer scope."""
    canonical = Path(destination).expanduser().resolve()
    for writer in _writers.get():
        if (
            not writer.closed
            and writer.path == canonical
            and writer.owner == _owner()
        ):
            writer.replace(Path(temporary))
            return
    raise RuntimeError("Project replacement requires writer ownership")
