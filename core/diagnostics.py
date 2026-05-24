"""Diagnostic report helpers for user-copyable bug reports."""

from __future__ import annotations

import platform
import sys

from core.app_version import get_build_identity
from core.redaction import redact_text


def build_diagnostics_report(log_lines: list[str] | None = None) -> str:
    """Return a redacted diagnostic report with environment and recent logs."""
    lines = [
        "Scene Ripper Diagnostics",
        f"Build: {get_build_identity()}",
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}",
    ]
    if log_lines:
        lines.extend(["", "Recent logs:"])
        lines.extend(log_lines[-80:])
    return redact_text("\n".join(lines))
