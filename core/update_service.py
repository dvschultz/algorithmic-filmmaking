"""Shared update service and fallback GitHub release provider."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

from core.update_models import UpdateAvailability, UpdateCapability, UpdateChannel, UpdateInfo

logger = logging.getLogger(__name__)

GITHUB_OWNER = "dvschultz"
GITHUB_REPO = "algorithmic-filmmaking"
GITHUB_LATEST_RELEASE_API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)


class UpdateProvider(Protocol):
    """Contract for update metadata providers."""

    def fetch_latest_release(self, channel: UpdateChannel = UpdateChannel.STABLE) -> Optional[UpdateInfo]:
        """Return the latest available update for a channel, or None on failure."""


class GitHubReleaseProvider:
    """Fallback update provider based on the GitHub Releases API."""

    def fetch_latest_release(self, channel: UpdateChannel = UpdateChannel.STABLE) -> Optional[UpdateInfo]:
        if channel is not UpdateChannel.STABLE:
            logger.debug("GitHub fallback provider only supports the stable channel today")

        req = urllib.request.Request(
            GITHUB_LATEST_RELEASE_API_URL,
            headers={
                "User-Agent": "Scene-Ripper-UpdateCheck/1.0",
                "Accept": "application/vnd.github.v3+json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode())
        except Exception as exc:
            logger.debug("GitHub API request failed: %s", exc)
            return None

        tag_name = str(payload.get("tag_name", "")).strip()
        release_url = str(payload.get("html_url", "")).strip()
        if not tag_name or not release_url:
            return None

        return UpdateInfo(
            version=tag_name.lstrip("v"),
            release_url=release_url,
            tag_name=tag_name,
            channel=channel,
            published_at=payload.get("published_at"),
        )


@dataclass(frozen=True)
class UpdateCheckResult:
    """Result returned by the shared update service."""

    availability: UpdateAvailability
    update: UpdateInfo | None = None
    error_message: str | None = None


class UpdateService:
    """Shared update behavior used by UI, settings, and worker threads."""

    def __init__(self, current_version: str, settings=None, provider: UpdateProvider | None = None):
        self._current_version = current_version
        self._settings = settings
        self._provider = provider or GitHubReleaseProvider()

    def get_capability(
        self,
        *,
        native_backend_available: bool = False,
        native_install_available: bool = False,
    ) -> UpdateCapability:
        """Report the update path supported by the current build."""
        if native_install_available:
            return UpdateCapability.NATIVE_INSTALL
        if native_backend_available:
            return UpdateCapability.NATIVE_CHECK
        return UpdateCapability.FALLBACK_BROWSER

    def should_check_automatically(self, check_interval_seconds: int) -> bool:
        """Return True when an automatic check should run."""
        if self._settings is None:
            return True
        if not getattr(self._settings, "check_for_updates", True):
            return False
        last_check = getattr(self._settings, "last_update_check", 0) or 0
        return (time.time() - last_check) >= check_interval_seconds

    def get_latest_release(self, *, interactive: bool) -> UpdateCheckResult:
        """Fetch and evaluate update state for the configured channel."""
        channel_value = getattr(self._settings, "update_channel", UpdateChannel.STABLE.value)
        try:
            channel = UpdateChannel(channel_value)
        except ValueError:
            channel = UpdateChannel.STABLE

        latest = self._provider.fetch_latest_release(channel=channel)
        if latest is None:
            return UpdateCheckResult(
                availability=UpdateAvailability.ERROR,
                error_message="Could not reach GitHub Releases. Check your network connection and try again.",
            )

        if self.is_skipped_version(latest.version) and not interactive:
            return UpdateCheckResult(UpdateAvailability.SKIPPED, update=latest)

        if self.is_newer(latest.version, self._current_version):
            return UpdateCheckResult(UpdateAvailability.UPDATE_AVAILABLE, update=latest)

        return UpdateCheckResult(UpdateAvailability.UP_TO_DATE, update=latest)

    def record_check_completed(self) -> None:
        """Persist the timestamp of a completed update check."""
        if self._settings is not None:
            self._settings.last_update_check = int(time.time())

    def record_result(self, result: UpdateCheckResult) -> None:
        """Persist the latest update check outcome for diagnostics."""
        if self._settings is None:
            return

        self._settings.last_update_check = int(time.time())
        self._settings.last_update_status = result.availability.value
        self._settings.last_update_version = result.update.version if result.update is not None else ""
        self._settings.last_update_error = result.error_message or ""

    def record_native_check_started(self) -> None:
        """Persist that a native updater UI was launched."""
        if self._settings is None:
            return

        self._settings.last_update_check = int(time.time())
        self._settings.last_update_status = "native_check_started"
        self._settings.last_update_version = ""
        self._settings.last_update_error = ""

    def is_skipped_version(self, version: str) -> bool:
        """Return True when the provided version is currently skipped."""
        if self._settings is None:
            return False
        skipped_version = getattr(self._settings, "skipped_update_version", "") or ""
        return skipped_version.lstrip("v") == version.lstrip("v")

    def skip_version(self, version: str) -> None:
        """Persist the version the user chose to skip."""
        if self._settings is not None:
            normalized = version.lstrip("v")
            self._settings.skipped_update_version = normalized
            self._settings.last_prompted_update_version = normalized

    def clear_skipped_version(self) -> None:
        """Clear any previously skipped update."""
        if self._settings is not None:
            self._settings.skipped_update_version = ""

    def mark_prompted(self, version: str) -> None:
        """Record that the user was shown an update prompt."""
        if self._settings is not None:
            self._settings.last_prompted_update_version = version.lstrip("v")

    @staticmethod
    def is_newer(latest_version: str, current_version: str) -> bool:
        """Compare semantic-ish version strings and ignore leading `v`."""
        latest = latest_version.lstrip("v")
        current = current_version.lstrip("v")

        try:
            latest_parts = [int(x) for x in latest.split(".")]
            current_parts = [int(x) for x in current.split(".")]
        except (ValueError, AttributeError):
            return latest > current

        while len(latest_parts) < len(current_parts):
            latest_parts.append(0)
        while len(current_parts) < len(latest_parts):
            current_parts.append(0)

        return latest_parts > current_parts
