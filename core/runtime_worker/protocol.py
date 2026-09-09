"""Bounded, versioned newline-delimited JSON protocol between host and worker.

Host -> worker (stdin):
    {"type": "hello", "protocol": 1, "worker_id": str, "staging_dir": str, "allow_test_tasks": bool}
    {"type": "task", "id": str, "kind": str, "args": {...}}
    {"type": "cancel", "id": str}
    {"type": "shutdown"}

Worker -> host (stdout):
    {"type": "ready", "protocol": 1, "pid": int, "python": str, "capabilities": [str]}
    {"type": "progress", "id": str, "fraction": float, "message": str}
    {"type": "result", "id": str, "result": {...}}          # paths must live under staging_dir
    {"type": "error", "id": str | None, "error": str, "kind": str}
    {"type": "cancelled", "id": str}

Logs never travel on stdout; they go to stderr. Every line is one message and
must be at most MAX_MESSAGE_BYTES; larger lines are a protocol violation.
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576  # 1 MiB per line; results reference files instead of inlining data

HOST_TYPES = ("hello", "task", "cancel", "shutdown")
WORKER_TYPES = ("ready", "progress", "result", "error", "cancelled")


class ProtocolError(ValueError):
    """A message violated the framing, size, or schema rules."""


def encode(message: dict[str, Any]) -> bytes:
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolError("Messages are objects with a string 'type'")
    try:
        line = json.dumps(message, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Message is not plain JSON: {exc}") from exc
    data = line.encode("ascii") + b"\n"
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Message exceeds {MAX_MESSAGE_BYTES} bytes")
    return data


def decode(line: bytes, *, expected_types: tuple[str, ...]) -> dict[str, Any]:
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Line exceeds {MAX_MESSAGE_BYTES} bytes")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"Malformed message: {exc}") from exc
    if not isinstance(message, dict) or message.get("type") not in expected_types:
        raise ProtocolError(f"Unexpected message: {str(message)[:120]}")
    return message
