"""Tests for the shared update service."""

from core.settings import Settings
from core.update_models import UpdateAvailability, UpdateChannel, UpdateInfo
from core.update_service import UpdateCheckResult, UpdateService


class StubProvider:
    def __init__(self, update: UpdateInfo | None):
        self.update = update

    def fetch_latest_release(self, channel: UpdateChannel = UpdateChannel.STABLE) -> UpdateInfo | None:
        return self.update


def test_update_service_reports_available_update():
    """Newer releases should be surfaced as update available."""
    service = UpdateService(
        "0.1.0",
        Settings(),
        provider=StubProvider(UpdateInfo("0.1.1", "https://example.com", "v0.1.1")),
    )

    result = service.get_latest_release(interactive=False)

    assert result.availability is UpdateAvailability.UPDATE_AVAILABLE
    assert result.update is not None
    assert result.update.version == "0.1.1"


def test_update_service_respects_skipped_version_on_automatic_checks():
    """Automatic checks should suppress a version the user skipped."""
    service = UpdateService(
        "0.1.0",
        Settings(skipped_update_version="0.1.1"),
        provider=StubProvider(UpdateInfo("0.1.1", "https://example.com", "v0.1.1")),
    )

    result = service.get_latest_release(interactive=False)

    assert result.availability is UpdateAvailability.SKIPPED


def test_update_service_ignores_skipped_version_for_manual_checks():
    """Manual checks should still surface skipped versions."""
    service = UpdateService(
        "0.1.0",
        Settings(skipped_update_version="0.1.1"),
        provider=StubProvider(UpdateInfo("0.1.1", "https://example.com", "v0.1.1")),
    )

    result = service.get_latest_release(interactive=True)

    assert result.availability is UpdateAvailability.UPDATE_AVAILABLE


def test_update_service_skip_version_persists_normalized_value():
    """Skipped versions should be stored without a leading v-prefix."""
    settings = Settings()
    service = UpdateService("0.1.0", settings, provider=StubProvider(None))

    service.skip_version("v0.1.2")

    assert settings.skipped_update_version == "0.1.2"
    assert settings.last_prompted_update_version == "0.1.2"


def test_update_service_records_last_result_metadata():
    """Completed checks should persist status, version, and error details."""
    settings = Settings()
    service = UpdateService("0.1.0", settings, provider=StubProvider(None))

    service.record_result(
        UpdateCheckResult(
            availability=UpdateAvailability.UPDATE_AVAILABLE,
            update=UpdateInfo("0.1.1", "https://example.com", "v0.1.1"),
        )
    )

    assert settings.last_update_status == UpdateAvailability.UPDATE_AVAILABLE.value
    assert settings.last_update_version == "0.1.1"
    assert settings.last_update_error == ""


def test_update_service_records_native_check_started():
    """Native updater launches should be visible in diagnostics."""
    settings = Settings()
    service = UpdateService("0.1.0", settings, provider=StubProvider(None))

    service.record_native_check_started()

    assert settings.last_update_status == "native_check_started"
    assert settings.last_update_version == ""
    assert settings.last_update_error == ""
