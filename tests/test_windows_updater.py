"""Tests for the Windows native updater adapter."""

from pathlib import Path
from unittest.mock import patch

from core.update_models import UpdateCapability
from core.windows_updater import (
    WindowsUpdaterStatus,
    _configure_library,
    get_status,
    shutdown,
    start_interactive_update_check,
)


class FakeWinSparkleLibrary:
    """Minimal fake of the WinSparkle C API surface used by the adapter."""

    def __init__(self):
        self.calls = []

    def win_sparkle_set_appcast_url(self, value):
        self.calls.append(("feed", value))

    def win_sparkle_set_eddsa_public_key(self, value):
        self.calls.append(("public_key", value))
        return 1

    def win_sparkle_set_app_details(self, company, app_name, app_version):
        self.calls.append(("details", company, app_name, app_version))

    def win_sparkle_set_app_build_version(self, build_version):
        self.calls.append(("build_version", build_version))

    def win_sparkle_set_registry_path(self, registry_path):
        self.calls.append(("registry_path", registry_path))

    def win_sparkle_set_automatic_check_for_updates(self, state):
        self.calls.append(("automatic", state))

    def win_sparkle_set_update_check_interval(self, interval):
        self.calls.append(("interval", interval))

    def win_sparkle_init(self):
        self.calls.append(("init",))

    def win_sparkle_cleanup(self):
        self.calls.append(("cleanup",))

    def win_sparkle_check_update_with_ui(self):
        self.calls.append(("check_with_ui",))


def teardown_function():
    """Reset module globals between tests."""
    shutdown()


def test_get_status_requires_metadata_for_native_updates():
    """Frozen Windows builds without metadata should fall back to browser updates."""
    with patch("core.windows_updater.sys.platform", "win32"), \
         patch("core.windows_updater.is_frozen", return_value=True), \
         patch("core.windows_updater.get_feed_url", return_value=""), \
         patch("core.windows_updater.get_public_ed_key", return_value=""), \
         patch("core.windows_updater.find_winsparkle_dll", return_value=None):
        status = get_status()

    assert status.available is False
    assert status.capability is UpdateCapability.FALLBACK_BROWSER
    assert "metadata" in status.reason.lower()


def test_get_status_beta_channel_falls_back_to_stable_feed(tmp_path):
    """Beta checks should use the stable feed when no beta-specific feed is configured."""
    dll_path = tmp_path / "WinSparkle.dll"
    dll_path.write_text("binary", encoding="utf-8")

    with patch("core.windows_updater.sys.platform", "win32"), \
         patch("core.windows_updater.is_frozen", return_value=True), \
         patch("core.windows_updater.get_feed_url", return_value="https://example.com/appcast.xml"), \
         patch("core.windows_updater.get_public_ed_key", return_value="pubkey"), \
         patch("core.windows_updater.find_winsparkle_dll", return_value=dll_path):
        status = get_status(update_channel="beta")

    assert status.available is True
    assert status.capability is UpdateCapability.NATIVE_CHECK
    assert status.feed_url == "https://example.com/appcast.xml"


def test_get_status_reports_native_check_when_dll_and_metadata_exist(tmp_path):
    """Installed builds with WinSparkle metadata should expose native checks."""
    dll_path = tmp_path / "WinSparkle.dll"
    dll_path.write_text("binary", encoding="utf-8")

    with patch("core.windows_updater.sys.platform", "win32"), \
         patch("core.windows_updater.is_frozen", return_value=True), \
         patch("core.windows_updater.get_feed_url", return_value="https://example.com/appcast.xml"), \
         patch("core.windows_updater.get_public_ed_key", return_value="pubkey"), \
         patch("core.windows_updater.find_winsparkle_dll", return_value=dll_path):
        status = get_status()

    assert status.available is True
    assert status.capability is UpdateCapability.NATIVE_CHECK
    assert status.feed_url == "https://example.com/appcast.xml"
    assert status.dll_path == dll_path


def test_configure_library_sets_app_metadata():
    """WinSparkle should be configured with feed, key, and build metadata."""
    library = FakeWinSparkleLibrary()

    with patch("core.windows_updater.get_display_version", return_value="1.2.3"), \
         patch("core.windows_updater.get_machine_version", return_value="1.2.3+45"):
        _configure_library(
            library,
            feed_url="https://example.com/appcast.xml",
            public_key="pubkey",
            automatically_check=False,
        )

    assert ("feed", b"https://example.com/appcast.xml") in library.calls
    assert ("public_key", b"pubkey") in library.calls
    assert ("details", "Algorithmic Filmmaking", "Scene Ripper", "1.2.3") in library.calls
    assert ("build_version", "1.2.3+45") in library.calls
    assert ("automatic", 0) in library.calls


def test_start_interactive_update_check_initializes_and_launches_ui(tmp_path):
    """Interactive checks should configure WinSparkle and show its UI."""
    dll_path = tmp_path / "WinSparkle.dll"
    dll_path.write_text("binary", encoding="utf-8")
    library = FakeWinSparkleLibrary()

    with patch("core.windows_updater.sys.platform", "win32"), \
         patch("core.windows_updater.is_frozen", return_value=True), \
         patch("core.windows_updater._load_library", return_value=library), \
         patch("core.windows_updater.get_display_version", return_value="1.2.3"), \
         patch("core.windows_updater.get_machine_version", return_value="1.2.3+45"), \
         patch("core.windows_updater.get_feed_url", return_value="https://example.com/appcast.xml"), \
         patch("core.windows_updater.get_public_ed_key", return_value="pubkey"), \
         patch("core.windows_updater.find_winsparkle_dll", return_value=dll_path):
        status = start_interactive_update_check(update_channel="stable", automatically_check=True)

    assert status == WindowsUpdaterStatus(
        available=True,
        capability=UpdateCapability.NATIVE_CHECK,
        dll_path=dll_path,
        feed_url="https://example.com/appcast.xml",
        public_key="pubkey",
        initialized=True,
        launched=True,
    )
    assert ("init",) in library.calls
    assert ("check_with_ui",) in library.calls
