"""Background version update checker using the shared update service.

Queries the latest release once per 24 hours and signals when a newer
version is available. Designed to run in a QThread from the main window.
"""

import logging

from PySide6.QtCore import QThread, Signal

from core.update_models import UpdateAvailability
from core.update_service import (
    GITHUB_OWNER as _GITHUB_OWNER,
    GITHUB_REPO as _GITHUB_REPO,
    UpdateService,
)

logger = logging.getLogger(__name__)

# Throttle: check at most once per 24 hours
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60


class UpdateCheckWorker(QThread):
    """Background worker that checks for app updates via GitHub Releases.

    Emits `update_available` with (version, release_url) if a newer version
    is found. Throttled to one check per 24-hour period.
    """

    update_available = Signal(str, str)  # version, release_url
    up_to_date = Signal(str, str)  # latest_version, release_url
    check_failed = Signal(str)

    def __init__(self, current_version: str, settings=None, interactive: bool = False):
        super().__init__()
        self._current_version = current_version
        self._settings = settings
        self._interactive = interactive
        self._service = UpdateService(current_version, settings)

    def run(self):
        try:
            # Check throttle
            if not self._interactive and not self._service.should_check_automatically(_CHECK_INTERVAL_SECONDS):
                logger.debug("Update check throttled — skipping")
                return

            result = self._service.get_latest_release(interactive=self._interactive)
            if result.availability is UpdateAvailability.ERROR:
                if self._interactive:
                    self.check_failed.emit(result.error_message or "Update check failed.")
                return

            if result.update is None:
                return

            if result.availability is UpdateAvailability.UPDATE_AVAILABLE:
                logger.info(
                    "Update available: %s (current: %s)",
                    result.update.version,
                    self._current_version,
                )
                self._service.mark_prompted(result.update.version)
                self.update_available.emit(result.update.version, result.update.release_url)
            elif result.availability is UpdateAvailability.UP_TO_DATE:
                logger.debug(
                    "Up to date (latest: %s, current: %s)",
                    result.update.version,
                    self._current_version,
                )
                if self._interactive:
                    self.up_to_date.emit(result.update.version, result.update.release_url)
            elif result.availability is UpdateAvailability.SKIPPED:
                logger.debug("Skipping suppressed update banner for version %s", result.update.version)

            self._service.record_check_completed()

        except Exception as e:
            # Never crash from a failed update check
            logger.debug(f"Update check failed: {e}")
            if self._interactive:
                self.check_failed.emit(str(e))
