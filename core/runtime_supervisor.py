"""Host-side supervisor for managed native-inference workers (KTD8).

Launches ``python -m runtime_worker`` under an explicitly chosen interpreter,
performs the protocol handshake, drains both pipes on reader threads, bounds
every message, validates result paths against the task's staging directory,
and tears down the whole process tree on cancellation or shutdown. A worker
crash is contained: the supervisor raises :class:`WorkerCrashed` and the
caller's project stays untouched.

This module is Qt-free and imports no model runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from core.runtime_worker.protocol import (
    MAX_MESSAGE_BYTES, PROTOCOL_VERSION, WORKER_TYPES, ProtocolError, decode, encode,
)

logger = logging.getLogger(__name__)

DEFAULT_HANDSHAKE_TIMEOUT = 60.0
DEFAULT_CANCEL_GRACE = 3.0
DEFAULT_SHUTDOWN_GRACE = 3.0
MAX_STDERR_LINES = 2000


class WorkerError(RuntimeError):
    """Base class for supervisor failures."""


class WorkerCrashed(WorkerError):
    """The worker process exited while a task was outstanding."""

    def __init__(self, message: str, returncode: int | None, stderr_tail: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class WorkerProtocolViolation(WorkerError):
    """The worker sent something outside the bounded protocol."""


class WorkerTaskError(WorkerError):
    """The worker reported a task failure."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class WorkerCancelled(WorkerError):
    """The task was cancelled by the host."""


class WorkerUnavailable(WorkerError):
    """No interpreter or worker package could be resolved."""


@dataclass(frozen=True)
class WorkerLaunch:
    """Everything needed to start a worker: an explicit interpreter, not the app binary."""

    interpreter: Path
    worker_root: Path
    """Directory containing the ``runtime_worker`` package."""
    package_paths: tuple[Path, ...] = ()
    """Managed package directories placed on the worker's ``PYTHONPATH``."""
    env: dict[str, str] = field(default_factory=dict)
    family: str = "default"

    def command(self) -> list[str]:
        return [str(self.interpreter), "-X", "utf8", "-m", "runtime_worker"]

    def environment(self) -> dict[str, str]:
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"))
        }
        env.update(self.env)
        pythonpath = [str(self.worker_root), *(str(p) for p in self.package_paths)]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Never inherit credentials into workers; providers run in the host.
        for key in list(env):
            if key.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
                env.pop(key, None)
        return env


def worker_package_root() -> Path:
    """Directory whose ``runtime_worker`` child is the worker package."""
    from core.paths import get_resource_path, is_frozen

    if is_frozen():
        staged = get_resource_path("runtime_worker_src")
        if (staged / "runtime_worker" / "__main__.py").is_file():
            return staged
    return Path(__file__).resolve().parent


def resolve_worker_interpreter(*, ensure: bool = False) -> Path:
    """The explicit interpreter workers run under.

    Frozen apps use the managed python-build-standalone interpreter, never the
    frozen executable. Source runs use the current interpreter, which is a real
    Python. ``SCENE_RIPPER_WORKER_PYTHON`` overrides both (tests, CI).
    """
    override = os.environ.get("SCENE_RIPPER_WORKER_PYTHON")
    if override:
        path = Path(override)
        if not path.is_file():
            raise WorkerUnavailable(f"SCENE_RIPPER_WORKER_PYTHON does not exist: {path}")
        return path
    from core.paths import get_managed_python_dir, is_frozen

    if is_frozen():
        python_dir = get_managed_python_dir()
        candidate = python_dir / ("python.exe" if sys.platform == "win32" else "bin/python3")
        if candidate.is_file():
            return candidate
        if ensure:
            from core.dependency_manager import ensure_python

            return Path(ensure_python())
        raise WorkerUnavailable("Managed Python runtime is not installed; install a native feature first")
    if getattr(sys, "frozen", False):
        raise WorkerUnavailable("Frozen executable cannot host workers")
    return Path(sys.executable)


def managed_interpreter_path() -> Path | None:
    from core.paths import get_managed_python_dir

    candidate = get_managed_python_dir() / ("python.exe" if sys.platform == "win32" else "bin/python3")
    return candidate if candidate.is_file() else None


def default_launch(family: str = "default", *, ensure_interpreter: bool = False) -> WorkerLaunch:
    """Pair an interpreter with the package directories built for it.

    Managed package directories are compiled against the managed Python; they
    are placed on the worker path only when that interpreter runs the worker.
    A source-mode worker under the developer's interpreter uses that
    environment's packages instead of shadowing them with incompatible builds.
    """
    from core.paths import get_managed_package_search_paths

    interpreter = resolve_worker_interpreter(ensure=ensure_interpreter)
    managed = managed_interpreter_path()
    packages: tuple[Path, ...] = ()
    if managed is not None and interpreter.resolve() == managed.resolve():
        packages = tuple(path for path in get_managed_package_search_paths() if path.is_dir())
    return WorkerLaunch(
        interpreter=interpreter,
        worker_root=worker_package_root(),
        package_paths=packages,
        family=family,
    )


def _terminate_tree(process: subprocess.Popen, grace: float) -> None:
    """Terminate the worker and everything it spawned (POSIX session / Windows tree)."""
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True, timeout=grace + 5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        process.wait(timeout=grace + 5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Worker %s did not exit cleanly: %s", process.pid, exc)
        try:
            process.kill()
        except OSError:
            pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def validate_result_paths(result: Any, staging_dir: Path) -> None:
    """Reject any path-like value that points outside the task's staging directory."""
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, str) and (key.endswith("_path") or key == "path"):
                candidate = Path(value)
                if not candidate.is_absolute() or not _within(candidate, staging_dir):
                    raise WorkerProtocolViolation(f"Result path {value!r} is outside the worker staging directory")
            else:
                validate_result_paths(value, staging_dir)
    elif isinstance(result, list):
        for item in result:
            validate_result_paths(item, staging_dir)


class ManagedWorker:
    """One warm worker process for a runtime family."""

    def __init__(
        self,
        launch: WorkerLaunch,
        *,
        staging_root: Path | None = None,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
        allow_test_tasks: bool = False,
    ) -> None:
        self.launch = launch
        self.worker_id = uuid.uuid4().hex
        self.staging_dir = Path(staging_root or tempfile.mkdtemp(prefix="scene-ripper-worker-")) / self.worker_id
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._owns_staging_root = staging_root is None
        self._handshake_timeout = handshake_timeout
        self._allow_test_tasks = allow_test_tasks
        self._process: subprocess.Popen | None = None
        self._inbox: "queue.Queue[dict[str, Any] | Exception | None]" = queue.Queue()
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self.capabilities: tuple[str, ...] = ()
        self.python: str | None = None
        self.pid: int | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if not (self.launch.worker_root / "runtime_worker" / "__main__.py").is_file():
            raise WorkerUnavailable(f"Worker package missing under {self.launch.worker_root}")
        if not self.launch.interpreter.is_file():
            raise WorkerUnavailable(f"Worker interpreter missing: {self.launch.interpreter}")
        popen_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        else:
            popen_kwargs["start_new_session"] = True  # own process group for tree teardown
        try:
            self._process = subprocess.Popen(
                self.launch.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(self.staging_dir), env=self.launch.environment(), **popen_kwargs,
            )
        except OSError as exc:
            raise WorkerUnavailable(f"Could not launch worker: {exc}") from exc
        threading.Thread(target=self._drain_stdout, daemon=True, name="worker-stdout").start()
        threading.Thread(target=self._drain_stderr, daemon=True, name="worker-stderr").start()
        self._send({
            "type": "hello", "protocol": PROTOCOL_VERSION, "worker_id": self.worker_id,
            "staging_dir": str(self.staging_dir), "allow_test_tasks": self._allow_test_tasks,
        })
        message = self._next(self._handshake_timeout)
        if message.get("type") == "error":
            raise WorkerProtocolViolation(f"Worker refused handshake: {message.get('error')}")
        if message.get("type") != "ready":
            raise WorkerProtocolViolation(f"Expected ready, got {message.get('type')!r}")
        if message.get("protocol") != PROTOCOL_VERSION:
            self.close()
            raise WorkerProtocolViolation(
                f"Worker protocol {message.get('protocol')!r} differs from host {PROTOCOL_VERSION}"
            )
        self.capabilities = tuple(str(c) for c in message.get("capabilities", []))
        self.python = message.get("python")
        self.pid = message.get("pid")

    def close(self, grace: float = DEFAULT_SHUTDOWN_GRACE) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._send({"type": "shutdown"})
                process.wait(timeout=grace)
            except (WorkerError, subprocess.TimeoutExpired, OSError):
                _terminate_tree(process, grace)
        self._process = None
        if self._owns_staging_root:
            shutil.rmtree(self.staging_dir.parent, ignore_errors=True)

    def __enter__(self) -> "ManagedWorker":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- transport -----------------------------------------------------------

    def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise WorkerCrashed("Worker is not running", process.returncode if process else None, self.stderr_tail())
        try:
            with self._lock:
                process.stdin.write(encode(message))
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerCrashed(f"Worker pipe closed: {exc}", process.poll(), self.stderr_tail()) from exc

    def _drain_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        stream = process.stdout
        while True:
            line = stream.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                self._inbox.put(None)
                return
            if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                self._inbox.put(WorkerProtocolViolation(f"Worker sent a line over {MAX_MESSAGE_BYTES} bytes"))
                # Skip the remainder of the oversized line without buffering it.
                while line and not line.endswith(b"\n"):
                    line = stream.readline(MAX_MESSAGE_BYTES + 1)
                continue
            try:
                self._inbox.put(decode(line, expected_types=WORKER_TYPES))
            except ProtocolError as exc:
                self._inbox.put(WorkerProtocolViolation(str(exc)))

    def _drain_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        for raw in process.stderr:
            text = raw.decode("utf-8", "replace").rstrip()
            logger.debug("worker[%s] %s", self.worker_id[:8], text)
            with self._lock:
                self._stderr.append(text)
                if len(self._stderr) > MAX_STDERR_LINES:
                    del self._stderr[: len(self._stderr) - MAX_STDERR_LINES]

    def stderr_tail(self, lines: int = 20) -> str:
        with self._lock:
            return "\n".join(self._stderr[-lines:])

    def _next(self, timeout: float | None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                item = self._inbox.get(timeout=min(remaining, 0.5) if remaining is not None else 0.5)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerError("Timed out waiting for the worker")
                if not self.alive and self._inbox.empty():
                    raise WorkerCrashed(
                        "Worker exited unexpectedly", self._process.returncode if self._process else None, self.stderr_tail(),
                    )
                continue
            if item is None:
                raise WorkerCrashed(
                    "Worker closed its output", self._process.returncode if self._process else None, self.stderr_tail(),
                )
            if isinstance(item, Exception):
                raise item
            return item

    # -- tasks ---------------------------------------------------------------

    def run(
        self,
        kind: str,
        args: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[float, str], None] | None = None,
        timeout: float | None = None,
        cancel_grace: float = DEFAULT_CANCEL_GRACE,
    ) -> dict[str, Any]:
        """Run one task; results are validated against this task's staging directory."""
        if not self.alive:
            raise WorkerCrashed("Worker is not running", self._process.returncode if self._process else None, self.stderr_tail())
        task_id = uuid.uuid4().hex
        task_staging = self.staging_dir / task_id
        self._send({"type": "task", "id": task_id, "kind": kind, "args": json.loads(json.dumps(args))})
        cancel_sent_at: float | None = None
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set() and cancel_sent_at is None:
                    cancel_sent_at = time.monotonic()
                    try:
                        self._send({"type": "cancel", "id": task_id})
                    except WorkerCrashed:
                        raise WorkerCancelled("Task cancelled")
                if cancel_sent_at is not None and time.monotonic() - cancel_sent_at > cancel_grace:
                    process = self._process
                    if process is not None:
                        _terminate_tree(process, cancel_grace)
                    raise WorkerCancelled("Task cancelled; worker terminated after grace period")
                if deadline is not None and time.monotonic() > deadline:
                    process = self._process
                    if process is not None:
                        _terminate_tree(process, cancel_grace)
                    raise WorkerError(f"Task {kind} timed out after {timeout}s; worker terminated")
                try:
                    message = self._next(0.25)
                except WorkerError as exc:
                    if isinstance(exc, WorkerCrashed) or "Timed out" not in str(exc):
                        if cancel_sent_at is not None and isinstance(exc, WorkerCrashed):
                            raise WorkerCancelled("Task cancelled") from exc
                        raise
                    continue
                if message.get("id") != task_id and message.get("type") != "error":
                    continue
                kind_ = message["type"]
                if kind_ == "progress":
                    if progress is not None:
                        progress(float(message.get("fraction", 0.0)), str(message.get("message", "")))
                elif kind_ == "result":
                    if cancel_sent_at is not None:
                        raise WorkerCancelled("Task cancelled before its result was accepted")
                    result = message.get("result")
                    if not isinstance(result, dict):
                        raise WorkerProtocolViolation("Task result must be an object")
                    validate_result_paths(result, task_staging)
                    return result
                elif kind_ == "cancelled":
                    raise WorkerCancelled("Task cancelled")
                elif kind_ == "error":
                    raise WorkerTaskError(str(message.get("error")), str(message.get("kind", "task")))
        except WorkerProtocolViolation:
            # A worker that breaks the protocol is not trusted with further tasks.
            process = self._process
            if process is not None:
                _terminate_tree(process, cancel_grace)
            raise

    def task_staging(self, result: dict[str, Any]) -> Path | None:
        path = result.get("result_path")
        return Path(path).parent if isinstance(path, str) else None


class RuntimeSupervisor:
    """One warm worker per runtime family, serialized per family."""

    def __init__(self, *, staging_root: Path | None = None, allow_test_tasks: bool = False) -> None:
        self._workers: dict[str, ManagedWorker] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._staging_root = staging_root
        self._allow_test_tasks = allow_test_tasks
        self.launch_factory: Callable[[str], WorkerLaunch] = default_launch

    def worker(self, family: str = "default") -> ManagedWorker:
        with self._guard:
            worker = self._workers.get(family)
            if worker is None or not worker.alive:
                worker = ManagedWorker(
                    self.launch_factory(family), staging_root=self._staging_root,
                    allow_test_tasks=self._allow_test_tasks,
                )
                worker.start()
                self._workers[family] = worker
            self._locks.setdefault(family, threading.Lock())
            return worker

    def run(self, family: str, kind: str, args: dict[str, Any], **options: Any) -> dict[str, Any]:
        """Run one task on the family's worker; accelerator access is serialized per family."""
        worker = self.worker(family)
        with self._locks[family]:
            try:
                return worker.run(kind, args, **options)
            except (WorkerCrashed, WorkerProtocolViolation):
                with self._guard:
                    if self._workers.get(family) is worker:
                        worker.close()
                        self._workers.pop(family, None)
                raise

    def shutdown(self) -> None:
        with self._guard:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.close()


_default: RuntimeSupervisor | None = None
_default_lock = threading.Lock()


def default_supervisor() -> RuntimeSupervisor:
    global _default
    with _default_lock:
        if _default is None:
            _default = RuntimeSupervisor()
        return _default


def shutdown_default_supervisor() -> None:
    global _default
    with _default_lock:
        supervisor, _default = _default, None
    if supervisor is not None:
        supervisor.shutdown()
