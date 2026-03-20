"""Tests for application version resolution."""

from pathlib import Path
from unittest.mock import patch

from core.app_version import get_app_version, get_machine_version, get_release_channel


def test_get_app_version_prefers_env_var():
    """Environment variable should win when explicitly provided."""
    with patch.dict("os.environ", {"APP_VERSION": "9.9.9"}, clear=False):
        assert get_app_version() == "9.9.9"


def test_get_app_version_reads_bundled_resource(tmp_path):
    """Frozen/source resource version file should be used when present."""
    version_file = tmp_path / "app_version.txt"
    version_file.write_text("1.2.3", encoding="utf-8")

    with patch.dict("os.environ", {}, clear=True), \
         patch("core.app_version.get_resource_path", return_value=version_file), \
         patch("core.app_version._version_from_git", return_value=""):
        assert get_app_version() == "1.2.3"


def test_get_app_version_falls_back_to_git_tag():
    """Source checkouts should use the nearest git tag when no bundled version exists."""
    with patch.dict("os.environ", {}, clear=True), \
         patch("core.app_version.get_resource_path", return_value=Path("/missing/version.txt")), \
         patch("core.app_version._version_from_git", return_value="v0.1.0"):
        assert get_app_version() == "v0.1.0"


def test_get_machine_version_reads_bundled_build_version(tmp_path):
    """Bundled build metadata should drive updater version comparisons."""
    build_version_file = tmp_path / "app_build_version.txt"
    build_version_file.write_text("1.2.3+45", encoding="utf-8")

    with patch.dict("os.environ", {}, clear=True), \
         patch("core.app_version.get_resource_path", return_value=build_version_file), \
         patch("core.app_version.get_display_version", return_value="1.2.3"):
        assert get_machine_version() == "1.2.3+45"


def test_get_release_channel_reads_bundled_channel(tmp_path):
    """Bundled release metadata should define the default update channel."""
    channel_file = tmp_path / "app_update_channel.txt"
    channel_file.write_text("beta", encoding="utf-8")

    with patch.dict("os.environ", {}, clear=True), \
         patch("core.app_version.get_resource_path", return_value=channel_file):
        assert get_release_channel() == "beta"
