"""Tests for GitHub release update checking."""

from core.settings import Settings
from core.update_models import UpdateInfo
from core.update_checker import UpdateCheckWorker


def _set_update(worker, version: str | None):
    if version is None:
        worker._service._provider.fetch_latest_release = lambda channel=None: None
        return

    worker._service._provider.fetch_latest_release = lambda channel=None: UpdateInfo(
        version=version,
        release_url="https://example.com/release",
        tag_name=f"v{version}",
    )


def test_interactive_update_check_emits_up_to_date():
    """Manual update checks should report when the app is already current."""
    worker = UpdateCheckWorker("0.1.0", Settings(check_for_updates=False), interactive=True)
    received = []

    worker.up_to_date.connect(lambda version, url: received.append((version, url)))
    worker.update_available.connect(lambda *_: received.append(("unexpected", "")))
    worker.check_failed.connect(lambda message: received.append(("failed", message)))
    _set_update(worker, "0.1.0")

    worker.run()

    assert received == [("0.1.0", "https://example.com/release")]


def test_interactive_update_check_emits_failure():
    """Manual checks should report network/release fetch failures."""
    worker = UpdateCheckWorker("0.1.0", Settings(), interactive=True)
    failures = []

    worker.check_failed.connect(failures.append)
    _set_update(worker, None)

    worker.run()

    assert failures == ["Could not reach GitHub Releases. Check your network connection and try again."]


def test_interactive_update_check_bypasses_disabled_launch_setting():
    """Manual checks should still run even if launch-time update checks are disabled."""
    worker = UpdateCheckWorker("0.1.0", Settings(check_for_updates=False), interactive=True)
    received = []

    worker.update_available.connect(lambda version, url: received.append((version, url)))
    _set_update(worker, "0.1.1")

    worker.run()

    assert received == [("0.1.1", "https://example.com/release")]


def test_automatic_update_check_suppresses_skipped_version():
    """Automatic checks should not re-announce a version the user skipped."""
    settings = Settings(skipped_update_version="0.1.1")
    worker = UpdateCheckWorker("0.1.0", settings, interactive=False)
    received = []

    worker.update_available.connect(lambda version, url: received.append((version, url)))
    _set_update(worker, "0.1.1")

    worker.run()

    assert received == []
