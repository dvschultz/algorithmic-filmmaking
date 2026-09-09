"""Worker entry point: ``python -m runtime_worker``.

Reads host messages from stdin on a reader thread, runs one task at a time on
the main thread, and writes protocol messages to stdout. Anything that is not
a protocol message goes to stderr.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .protocol import HOST_TYPES, PROTOCOL_VERSION, ProtocolError, decode, encode
from .tasks import HANDLERS, TEST_HANDLERS, AnalysisCallError, ModelLoadError, TaskCancelled, WorkerContext

_out_lock = threading.Lock()
_protocol_out = None  # private handle to the original stdout; fd 1 itself is rerouted to stderr


def _claim_protocol_stdout() -> None:
    """Reserve the real stdout for protocol messages only.

    Engine code and native libraries (progress bars, ``print`` in third-party
    packages) write to file descriptor 1; once it points at stderr nothing
    they emit can corrupt the newline-JSON stream the host is parsing.
    """
    global _protocol_out
    if _protocol_out is not None:
        return
    _protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr  # Python-level prints follow the descriptor


def _send(message: dict[str, Any]) -> None:
    data = encode(message)
    with _out_lock:
        if _protocol_out is not None:
            _protocol_out.write(data)
        else:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()


def _log(text: str) -> None:
    sys.stderr.write(text.rstrip() + "\n")
    sys.stderr.flush()


def _reader(
    inbox: "queue.Queue[dict[str, Any] | None]",
    cancel_flags: dict[str, threading.Event],
    pending_cancels: set[str],
) -> None:
    stream = sys.stdin.buffer
    while True:
        line = stream.readline()
        if not line:
            inbox.put(None)
            return
        try:
            message = decode(line, expected_types=HOST_TYPES)
        except ProtocolError as exc:
            _log(f"protocol error: {exc}")
            inbox.put({"type": "shutdown", "reason": str(exc)})
            return
        if message["type"] == "cancel":
            task_id = str(message.get("id"))
            flag = cancel_flags.get(task_id)
            if flag is not None:
                flag.set()
            else:
                pending_cancels.add(task_id)  # cancel arrived before the task started
            continue
        inbox.put(message)


def main() -> int:
    _claim_protocol_stdout()
    inbox: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    cancel_flags: dict[str, threading.Event] = {}
    pending_cancels: set[str] = set()
    threading.Thread(target=_reader, args=(inbox, cancel_flags, pending_cancels), daemon=True).start()

    hello = inbox.get()
    if hello is None or hello.get("type") != "hello":
        _send({"type": "error", "id": None, "error": "Expected hello", "kind": "protocol"})
        return 2
    if hello.get("protocol") != PROTOCOL_VERSION:
        _send({"type": "error", "id": None, "error": f"Protocol {hello.get('protocol')!r} unsupported; worker speaks {PROTOCOL_VERSION}", "kind": "protocol"})
        return 2
    staging = Path(str(hello.get("staging_dir", "")))
    if not staging.is_dir():
        _send({"type": "error", "id": None, "error": "Staging directory does not exist", "kind": "protocol"})
        return 2
    handlers = dict(HANDLERS)
    allow_test_tasks = bool(hello.get("allow_test_tasks"))
    if allow_test_tasks:
        handlers.update(TEST_HANDLERS)
    _send({
        "type": "ready", "protocol": PROTOCOL_VERSION, "pid": os.getpid(),
        "python": sys.executable, "capabilities": sorted(handlers),
    })

    while True:
        message = inbox.get()
        if message is None or message.get("type") == "shutdown":
            return 0
        if message.get("type") != "task":
            continue
        task_id = str(message.get("id"))
        kind = str(message.get("kind"))
        handler = handlers.get(kind)
        if handler is None:
            _send({"type": "error", "id": task_id, "error": f"Unknown task kind {kind!r}", "kind": "unsupported"})
            continue
        cancel = threading.Event()
        cancel_flags[task_id] = cancel
        if task_id in pending_cancels:
            pending_cancels.discard(task_id)
            cancel.set()

        def progress(fraction: float, text: str, _id: str = task_id) -> None:
            _send({"type": "progress", "id": _id, "fraction": max(0.0, min(1.0, float(fraction))), "message": str(text)[:500]})

        task_staging = staging / task_id
        task_staging.mkdir(parents=True, exist_ok=True)
        context = WorkerContext(task_staging, cancel, progress, allow_test_tasks=allow_test_tasks)
        try:
            result = handler(dict(message.get("args") or {}), context)
            if cancel.is_set():
                _send({"type": "cancelled", "id": task_id})
            else:
                _send({"type": "result", "id": task_id, "result": result})
        except TaskCancelled:
            _send({"type": "cancelled", "id": task_id})
        except ProtocolError as exc:
            _send({"type": "error", "id": task_id, "error": str(exc), "kind": "protocol"})
        except ImportError as exc:
            _send({"type": "error", "id": task_id, "error": f"{type(exc).__name__}: {exc}"[:2000], "kind": "dependency_missing"})
        except ModelLoadError as exc:
            _send({"type": "error", "id": task_id, "error": str(exc)[:2000], "kind": "model"})
        except AnalysisCallError as exc:
            _send({
                "type": "error", "id": task_id, "error": str(exc)[:2000], "kind": "task",
                "executions": exc.executions[:50],
            })
        except MemoryError:
            _send({"type": "error", "id": task_id, "error": "Out of memory", "kind": "resource"})
        except BaseException as exc:  # noqa: BLE001 - report, keep the worker alive
            _log(traceback.format_exc())
            _send({"type": "error", "id": task_id, "error": f"{type(exc).__name__}: {exc}"[:2000], "kind": "task"})
        finally:
            for child in list(context.children):
                try:
                    child.kill()
                except OSError:
                    pass
            cancel_flags.pop(task_id, None)


if __name__ == "__main__":
    sys.exit(main())
