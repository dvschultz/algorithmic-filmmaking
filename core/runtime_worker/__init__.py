"""Managed native-inference worker.

This package is executed by a separate, explicitly chosen interpreter
(``python -m runtime_worker``) with this directory's parent on ``sys.path``.
It must stay self-contained: only the standard library at import time, with
model runtimes imported lazily inside task handlers. The host side lives in
``core/runtime_supervisor.py``.
"""

from .protocol import PROTOCOL_VERSION, MAX_MESSAGE_BYTES

__all__ = ["PROTOCOL_VERSION", "MAX_MESSAGE_BYTES"]
