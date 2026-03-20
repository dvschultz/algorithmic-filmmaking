"""Tests for the macOS native updater adapter."""

from pathlib import Path
from unittest.mock import patch

from core.update_models import UpdateCapability
from core.macos_updater import (
    MacOSUpdaterStatus,
    get_bundle_path,
    get_status,
    is_running_from_disk_image,
    is_translocated_path,
    start_interactive_update_check,
)


def test_get_bundle_path_finds_enclosing_app():
    """Executable paths inside a bundle should resolve to the enclosing .app."""
    bundle = get_bundle_path("/Applications/Scene Ripper.app/Contents/MacOS/Scene Ripper")
    assert bundle == Path("/Applications/Scene Ripper.app")


def test_detects_translocated_bundle_path():
    """App Translocation paths should be treated as unsupported for native updates."""
    assert is_translocated_path(
        Path("/private/var/folders/xx/AppTranslocation/yy/d/Scene Ripper.app")
    )


def test_detects_disk_image_bundle_path():
    """Apps running from /Volumes should be treated as DMG-mounted."""
    assert is_running_from_disk_image(Path("/Volumes/Scene Ripper/Scene Ripper.app"))


def test_get_status_requires_bundle_metadata_for_native_updates(tmp_path):
    """Frozen macOS builds without feed metadata should fall back to browser updates."""
    bundle = tmp_path / "Scene Ripper.app"
    info_plist = bundle / "Contents" / "Info.plist"
    info_plist.parent.mkdir(parents=True)
    info_plist.write_bytes(
        (
            b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            b"<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
            b"\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">"
            b"<plist version=\"1.0\"><dict></dict></plist>"
        )
    )

    with patch("core.macos_updater.sys.platform", "darwin"), \
         patch("core.macos_updater.is_frozen", return_value=True), \
         patch("core.macos_updater.get_bundle_path", return_value=bundle), \
         patch("core.macos_updater.is_in_applications", return_value=True):
        status = get_status()

    assert status.available is False
    assert status.capability is UpdateCapability.FALLBACK_BROWSER
    assert "metadata" in status.reason.lower()


def test_get_status_reports_native_check_when_cli_and_metadata_exist(tmp_path):
    """Installed bundles with Sparkle metadata and CLI should expose native checks."""
    bundle = tmp_path / "Scene Ripper.app"
    info_plist = bundle / "Contents" / "Info.plist"
    info_plist.parent.mkdir(parents=True)
    info_plist.write_bytes(
        (
            b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            b"<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
            b"\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">"
            b"<plist version=\"1.0\"><dict>"
            b"<key>SUFeedURL</key><string>https://example.com/appcast.xml</string>"
            b"<key>SUPublicEDKey</key><string>pubkey</string>"
            b"</dict></plist>"
        )
    )
    sparkle_cli = tmp_path / "sparkle"
    sparkle_cli.write_text("binary", encoding="utf-8")

    with patch("core.macos_updater.sys.platform", "darwin"), \
         patch("core.macos_updater.is_frozen", return_value=True), \
         patch("core.macos_updater.get_bundle_path", return_value=bundle), \
         patch("core.macos_updater.is_in_applications", return_value=True), \
         patch("core.macos_updater.find_sparkle_cli", return_value=sparkle_cli):
        status = get_status()

    assert status.available is True
    assert status.capability is UpdateCapability.NATIVE_CHECK
    assert status.feed_url == "https://example.com/appcast.xml"
    assert status.sparkle_cli_path == sparkle_cli


def test_start_interactive_update_check_launches_sparkle_with_channel(tmp_path):
    """Interactive checks should shell out to sparkle with the requested channel."""
    bundle = tmp_path / "Scene Ripper.app"
    sparkle_cli = tmp_path / "sparkle"
    sparkle_cli.write_text("binary", encoding="utf-8")

    with patch("core.macos_updater.get_status") as get_status_mock, \
         patch("core.macos_updater.subprocess.Popen") as popen_mock:
        get_status_mock.return_value = MacOSUpdaterStatus(
            available=True,
            capability=UpdateCapability.NATIVE_CHECK,
            bundle_path=bundle,
            sparkle_cli_path=sparkle_cli,
            feed_url="https://example.com/appcast.xml",
            public_key="pubkey",
        )

        status = start_interactive_update_check(update_channel="beta")

    assert popen_mock.called
    command = popen_mock.call_args.args[0]
    assert "--channels" in command
    assert "beta" in command
    assert status.launched is True
