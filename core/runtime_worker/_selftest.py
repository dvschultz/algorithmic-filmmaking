"""Diagnostic calls used to prove the host <-> worker analysis round trip."""

from __future__ import annotations

import os
import sys


def identity(**kwargs):
    return {"kwargs": kwargs, "pid": os.getpid(), "python": sys.executable}


def with_execution(on_execution=None, **kwargs):
    """Report a fake model execution the way analysis functions do."""
    if on_execution is not None:
        on_execution({"model": "selftest", "device": "cpu", "kwargs": kwargs})
    return {"ok": True}


def missing_dependency():
    import a_module_that_does_not_exist_anywhere  # noqa: F401


def model_error():
    from core.errors import ModelDownloadError

    raise ModelDownloadError("could not fetch weights")


def big_value(count: int = 1000, progress_cb=None):
    if progress_cb is not None:
        progress_cb("halfway")
    return list(range(int(count)))


def execution_then_fail(on_execution=None, **kwargs):
    if on_execution is not None:
        on_execution({"model": "selftest", "device": "cpu"})
    raise RuntimeError("engine failed after reporting")


def wait_for_cancel(cancel_event=None, **kwargs):
    import time

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return {"cancelled": True}
        time.sleep(0.05)
    return {"cancelled": False}
