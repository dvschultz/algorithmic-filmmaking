"""Tests for application version resolution."""

from pathlib import Path
from unittest.mock import patch

from core.app_version import (
    get_app_version,
    get_build_channel,
    get_build_identity,
    get_git_commit,
    get_machine_version,
    get_release_channel,
)


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


def test_get_build_channel_marks_source_checkout():
    """Source checkouts should make the running channel visible."""
    with patch.dict("os.environ", {}, clear=True), \
         patch("core.app_version.is_frozen", return_value=False), \
         patch("core.app_version.get_release_channel", return_value="stable"):
        assert get_build_channel() == "source/stable"


def test_get_git_commit_prefers_env_var():
    """Commit metadata should be available without shelling out in packaged builds."""
    with patch.dict("os.environ", {"GITHUB_SHA": "abcdef1234567890"}, clear=False):
        assert get_git_commit() == "abcdef123456"


def test_get_build_identity_includes_version_channel_and_commit():
    """The UI/log identity should distinguish similar-looking builds."""
    with patch("core.app_version.get_display_version", return_value="1.2.3"), \
         patch("core.app_version.get_build_channel", return_value="source/beta"), \
         patch("core.app_version.get_git_commit", return_value="abcdef1"):
        assert get_build_identity() == "v1.2.3 source/beta abcdef1"
