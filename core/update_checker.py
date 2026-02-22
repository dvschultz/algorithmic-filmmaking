"""Background version update checker using GitHub Releases API.

Queries the latest release once per 24 hours and signals when a newer
version is available. Designed to run in a QThread from the main window.
"""

import json
import logging
import time
import urllib.request
from typing import Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

# GitHub repository coordinates
_GITHUB_OWNER = "dvschultz"
_GITHUB_REPO = "algorithmic-filmmaking"
_API_URL = f"https://api.github.com/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/releases/latest"

# Throttle: check at most once per 24 hours
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60


class UpdateCheckWorker(QThread):
    """Background worker that checks for app updates via GitHub Releases.

    Emits `update_available` with (version, release_url) if a newer version
    is found. Throttled to one check per 24-hour period.
    """

    update_available = Signal(str, str)  # version, release_url

    def __init__(self, current_version: str, settings=None):
        super().__init__()
        self._current_version = current_version
        self._settings = settings

    def run(self):
        try:
            # Check throttle
            if self._settings and not self._should_check():
                logger.debug("Update check throttled — skipping")
                return

            latest = self._fetch_latest_release()
            if latest is None:
                return

            version, url = latest
            if self._is_newer(version, self._current_version):
                logger.info(f"Update available: {version} (current: {self._current_version})")
                self.update_available.emit(version, url)
            else:
                logger.debug(f"Up to date (latest: {version}, current: {self._current_version})")

            # Record check timestamp
            if self._settings:
                self._settings.last_update_check = int(time.time())

        except Exception as e:
            # Never crash from a failed update check
            logger.debug(f"Update check failed: {e}")

    def _should_check(self) -> bool:
        """Return True if enough time has passed since the last check."""
        if not hasattr(self._settings, "check_for_updates"):
            return True
        if not self._settings.check_for_updates:
            return False

        last_check = getattr(self._settings, "last_update_check", 0) or 0
        return (time.time() - last_check) >= _CHECK_INTERVAL_SECONDS

    @staticmethod
    def _fetch_latest_release() -> Optional[tuple[str, str]]:
        """Fetch the latest release from GitHub.

        Returns:
            Tuple of (tag_name, html_url), or None on failure.
        """
        req = urllib.request.Request(
            _API_URL,
            headers={
                "User-Agent": "Scene-Ripper-UpdateCheck/1.0",
                "Accept": "application/vnd.github.v3+json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                tag = data.get("tag_name", "")
                url = data.get("html_url", "")
                if tag and url:
                    return tag, url
        except Exception as e:
            logger.debug(f"GitHub API request failed: {e}")

        return None

    @staticmethod
    def _is_newer(latest_tag: str, current_version: str) -> bool:
        """Compare version tags. Handles 'v' prefix."""
        latest = latest_tag.lstrip("v")
        current = current_version.lstrip("v")

        try:
            latest_parts = [int(x) for x in latest.split(".")]
            current_parts = [int(x) for x in current.split(".")]

            # Pad shorter list with zeros
            while len(latest_parts) < len(current_parts):
                latest_parts.append(0)
            while len(current_parts) < len(latest_parts):
                current_parts.append(0)

            return latest_parts > current_parts
        except (ValueError, AttributeError):
            # Non-numeric version — fall back to string comparison
            return latest > current
