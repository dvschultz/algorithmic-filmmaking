"""Shared models for desktop update state and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class UpdateChannel(StrEnum):
    """Supported release channels."""

    STABLE = "stable"
    BETA = "beta"


class UpdateAvailability(StrEnum):
    """High-level result of an update lookup."""

    UNKNOWN = "unknown"
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    SKIPPED = "skipped"
    ERROR = "error"


class UpdateCapability(StrEnum):
    """What the current build can do when an update is found."""

    FALLBACK_BROWSER = "fallback_browser"
    NATIVE_CHECK = "native_check"
    NATIVE_INSTALL = "native_install"


@dataclass(frozen=True)
class UpdateInfo:
    """Normalized release metadata used by app update flows."""

    version: str
    release_url: str
    tag_name: str
    channel: UpdateChannel = UpdateChannel.STABLE
    published_at: str | None = None
    notes_url: str | None = None
