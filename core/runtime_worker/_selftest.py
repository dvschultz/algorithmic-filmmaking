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
