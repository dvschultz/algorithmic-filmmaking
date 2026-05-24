"""Utilities for removing secrets from text and exported payloads."""

from __future__ import annotations

import re
import logging
from collections.abc import Mapping
from typing import Any

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|bearer|authorization)",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|token|access[_-]?token|secret|password|authorization|signature|sig|x-goog-signature|x-amz-signature)\s*[:=]\s*)([\"']?)([^&\"'\s,}]+)([\"']?)"
)
_URL_SECRET_PARAM_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|access[_-]?token|signature|sig|x-goog-signature|x-amz-signature)=)([^&#\s]+)"
)
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\br8_[A-Za-z0-9_-]{20,}\b"),
)


def redact_text(text: str) -> str:
    """Return text with common API keys and token assignments removed."""
    if not text:
        return text

    redacted = text
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)

    redacted = _URL_SECRET_PARAM_RE.sub(
        lambda match: f"{match.group(1)}{_REDACTED}",
        redacted,
    )

    return _ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}{match.group(4)}",
        redacted,
    )


def redact_secrets(value: Any) -> Any:
    """Recursively redact secrets in JSON-like values."""
    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key):
                redacted[key] = _REDACTED
            else:
                redacted[key] = redact_secrets(item)
        return redacted

    if isinstance(value, list):
        return [redact_secrets(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)

    return value


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets before handlers persist records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(record.getMessage())
            record.args = ()
        except Exception:
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg)
            if record.args:
                record.args = redact_secrets(record.args)
        return True
