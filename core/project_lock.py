"""OS-backed project writer scopes, independent of project-file replacement."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import errno
from hashlib import sha256
import os
from pathlib import Path
import sys
from threading import Lock, get_ident
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
        self._use_lock = Lock()

    def _is_active_here(self) -> bool:
        return any(
            borrow.active and borrow.writer is self and borrow.owner == _owner()
            for borrow in _borrowed_writers.get()
        )

    @contextmanager
    def activate(self) -> Iterator[ProjectWriter]:
        """Explicitly lend this lease to one save operation in this process.

        Possession of the writer is the capability; inherited thread/task
        context alone never grants access. The acquiring context retains
        responsibility for closing the lease after the operation returns.
        """
        if self.closed or self.owner[0] != os.getpid():
            raise RuntimeError("Project writer is closed or belongs to another process")
        if self._is_active_here():
            yield self
            return
        if not self._use_lock.acquire(blocking=False):
            raise RuntimeError("Project writer is in use")
        try:
            if self.closed:
                raise RuntimeError("Project writer is closed")
            borrow = _WriterBorrow(self, _owner())
            token = _borrowed_writers.set((*_borrowed_writers.get(), borrow))
            try:
                yield self
            finally:
                borrow.active = False
                _borrowed_writers.reset(token)
        finally:
            self._use_lock.release()

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
        if self.closed or (self.owner != _owner() and not self._is_active_here()):
            raise RuntimeError("Project writer is not active in this execution context")
        with self.activate():
            self._replace(temporary)

    def _replace(self, temporary: Path) -> None:
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
        if not self._use_lock.acquire(blocking=False):
            raise RuntimeError("Project writer is in use")
        try:
            for stream in (self._identity_lock, self._path_lock):
                if stream is not None:
                    stream.close()
            self._identity_lock = self._path_lock = None
            self.closed = True
        finally:
            self._use_lock.release()

    def __enter__(self) -> ProjectWriter:
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.close()


_writers: ContextVar[tuple[ProjectWriter, ...]] = ContextVar(
    "project_writers", default=()
)


@dataclass
class _WriterBorrow:
    writer: ProjectWriter
    owner: tuple[int, int, int | None]
    active: bool = True


_borrowed_writers: ContextVar[tuple[_WriterBorrow, ...]] = ContextVar(
    "borrowed_project_writers", default=()
)


def _current_writer(canonical: Path) -> ProjectWriter | None:
    for borrow in _borrowed_writers.get():
        writer = borrow.writer
        if (
            borrow.active
            and not writer.closed
            and writer.path == canonical
            and borrow.owner == _owner()
        ):
            return writer
    for writer in _writers.get():
        if not writer.closed and writer.path == canonical and writer.owner == _owner():
            return writer
    return None


@contextmanager
def project_writer(path: Path | str) -> Iterator[ProjectWriter]:
    """Reuse only a scope belonging to the current thread and async task."""
    canonical = Path(path).expanduser().resolve()
    existing = _current_writer(canonical)
    if existing is not None:
        yield existing
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
    writer = _current_writer(canonical)
    if writer is not None:
        writer.replace(Path(temporary))
        return
    raise RuntimeError("Project replacement requires writer ownership")
