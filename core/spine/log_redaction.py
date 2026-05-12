"""Shared redaction helpers for credential-bearing log lines.

Both ``core/spine/chatgpt_oauth_flow`` and ``core/codex_client`` need to
scrub ``Authorization: Bearer <token>`` from exception messages and
HTTP-error bodies before they propagate into log files or bug-report
attachments. This module owns the single canonical regex so both
callers agree on what gets redacted (R-S4 from the doc-review).

Spine-safe: stdlib only (``re``). No PySide6, litellm, or httpx at
top level — ``tests/test_spine_imports.py`` enforces this.

Add new patterns here as new credential formats arrive (refresh
tokens with distinguishable prefixes, raw API keys in error bodies,
etc.). Callers stay one-line.
"""

from __future__ import annotations

import re


# Authorization: Bearer <token>   or   authorization=Bearer <token>
# Replaces the value half with [REDACTED]; case-insensitive.
_BEARER_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"
)


def redact_bearer_tokens(text: str) -> str:
    """Replace bearer-token values in ``text`` with ``[REDACTED]``.

    Returns ``text`` unchanged when there's nothing to redact, so
    callers can pass this through unconditionally without paying
    for a no-op rewrite.
    """
    if not text:
        return text
    return _BEARER_PATTERN.sub(r"\1[REDACTED]", text)


__all__ = ["redact_bearer_tokens"]
