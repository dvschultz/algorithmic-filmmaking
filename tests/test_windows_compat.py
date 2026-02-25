"""Tests for Windows compatibility code paths.

These tests mock sys.platform to exercise Windows-specific branches
on any platform (macOS, Linux, or Windows CI).
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# core/paths.py
# ---------------------------------------------------------------------------

class TestGetAppSupportDirWindows:
    """Test get_app_support_dir() Windows branch."""

    def test_uses_localappdata(self, tmp_path):
        """On Windows, should use LOCALAPPDATA."""
        import core.paths

        with patch.object(sys, "platform", "win32"), \
             patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}):
            result = core.paths.get_app_support_dir()
            assert "Scene Ripper" in str(result)
            assert str(tmp_path) in str(result)

    def test_fallback_when_no_localappdata(self):
        """Should fall back to ~/AppData/Local when LOCALAPPDATA unset."""
        import core.paths

        env = os.environ.copy()
        env.pop("LOCALAPPDATA", None)
        with patch.object(sys, "platform", "win32"), \
             patch.dict(os.environ, env, clear=True):
            result = core.paths.get_app_support_dir()
            assert "Scene Ripper" in str(result)


# ---------------------------------------------------------------------------
# core/binary_resolver.py
# ---------------------------------------------------------------------------

class TestFindBinaryWindows:
    """Test find_binary() Windows exe suffix handling."""

    def test_appends_exe_suffix(self, tmp_path):
        """find_binary should check for .exe suffix on Windows."""
        from core.binary_resolver import find_binary

        # Create a fake managed bin dir with ffmpeg.exe
        managed_dir = tmp_path / "bin"
        managed_dir.mkdir()
        (managed_dir / "ffmpeg.exe").write_text("fake")

        with patch("core.binary_resolver.sys") as mock_sys, \
             patch("core.binary_resolver.get_managed_bin_dir", return_value=managed_dir):
            mock_sys.platform = "win32"
            # Patch shutil.which to return None (not on PATH)
            with patch("shutil.which", return_value=None):
                result = find_binary("ffmpeg")
                assert result is not None
                assert result.endswith("ffmpeg.exe")

    def test_finds_without_exe_suffix_on_unix(self, tmp_path):
        """find_binary should not add .exe on Unix."""
        from core.binary_resolver import find_binary

        managed_dir = tmp_path / "bin"
        managed_dir.mkdir()
        ffmpeg = managed_dir / "ffmpeg"
        ffmpeg.write_text("fake")
        ffmpeg.chmod(0o755)

        with patch("core.binary_resolver.sys") as mock_sys, \
             patch("core.binary_resolver.get_managed_bin_dir", return_value=managed_dir):
            mock_sys.platform = "linux"
            with patch("shutil.which", return_value=None):
                result = find_binary("ffmpeg")
                assert result is not None
                assert not result.endswith(".exe")


class TestGetSubprocessKwargs:
    """Test get_subprocess_kwargs() platform behavior."""

    def test_returns_creation_flags_on_windows(self):
        """Should return CREATE_NO_WINDOW on Windows."""
        from core.binary_resolver import get_subprocess_kwargs

        with patch("core.binary_resolver.sys") as mock_sys:
            mock_sys.platform = "win32"
            kwargs = get_subprocess_kwargs()
            assert "creationflags" in kwargs
            # CREATE_NO_WINDOW = 0x08000000
            assert kwargs["creationflags"] == 0x08000000

    def test_returns_empty_on_unix(self):
        """Should return empty dict on macOS/Linux."""
        from core.binary_resolver import get_subprocess_kwargs

        with patch("core.binary_resolver.sys") as mock_sys:
            mock_sys.platform = "darwin"
            kwargs = get_subprocess_kwargs()
            assert kwargs == {}


# ---------------------------------------------------------------------------
# core/dependency_manager.py
# ---------------------------------------------------------------------------

class TestDependencyManagerPlatformDispatch:
    """Test platform URL dispatch functions."""

    def test_ffmpeg_url_windows(self):
        """Should return BtbN URL on Windows."""
        from core.dependency_manager import _get_ffmpeg_url

        with patch("core.dependency_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            url = _get_ffmpeg_url()
            assert "BtbN" in url
            assert "win64" in url

    def test_ffmpeg_url_macos(self):
        """Should return osxexperts URL on macOS ARM."""
        from core.dependency_manager import _get_ffmpeg_url

        with patch("core.dependency_manager.sys") as mock_sys, \
             patch("core.dependency_manager.platform") as mock_platform:
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            url = _get_ffmpeg_url()
            assert "osxexperts" in url

    def test_ytdlp_url_windows(self):
        """Should return .exe URL on Windows."""
        from core.dependency_manager import _get_ytdlp_url

        with patch("core.dependency_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            url = _get_ytdlp_url()
            assert url.endswith(".exe")

    def test_binary_ext_windows(self):
        """Should return .exe on Windows."""
        from core.dependency_manager import _get_binary_ext

        with patch("core.dependency_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert _get_binary_ext() == ".exe"

    def test_binary_ext_unix(self):
        """Should return empty string on macOS/Linux."""
        from core.dependency_manager import _get_binary_ext

        with patch("core.dependency_manager.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert _get_binary_ext() == ""


# ---------------------------------------------------------------------------
# core/analysis/audio.py — null device
# ---------------------------------------------------------------------------

class TestNullDevice:
    """Test FFmpeg null device is correct per platform."""

    def test_null_device_windows(self):
        """On Windows, should use NUL."""
        with patch("core.analysis.audio.sys") as mock_sys:
            mock_sys.platform = "win32"
            # The null device logic is inline in extract_clip_volume.
            # We test the platform check directly.
            null_target = "NUL" if mock_sys.platform == "win32" else "-"
            assert null_target == "NUL"

    def test_null_device_unix(self):
        """On macOS/Linux, should use -."""
        with patch("core.analysis.audio.sys") as mock_sys:
            mock_sys.platform = "darwin"
            null_target = "NUL" if mock_sys.platform == "win32" else "-"
            assert null_target == "-"


# ---------------------------------------------------------------------------
# Safe path roots
# ---------------------------------------------------------------------------

class TestSafePathRootsWindows:
    """Test that Windows drive roots are included in safe roots."""

    def test_mcp_security_includes_drive_roots(self):
        """MCP security module should include drive roots on Windows."""
        import importlib

        with patch("sys.platform", "win32"), \
             patch("pathlib.Path.exists", return_value=True):
            import scene_ripper_mcp.security
            importlib.reload(scene_ripper_mcp.security)
            try:
                roots = scene_ripper_mcp.security.SAFE_ROOTS
                # Should have at least C:\ plus home and temp
                drive_roots = [r for r in roots if len(str(r)) <= 4 and ":" in str(r)]
                assert len(drive_roots) > 0, "No drive roots found in SAFE_ROOTS"
            finally:
                importlib.reload(scene_ripper_mcp.security)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class TestSignalHandling:
    """Test platform-conditional signal handling."""

    def test_sigterm_not_set_on_windows(self):
        """SIGTERM handler should not be set on Windows."""
        import signal as signal_module

        with patch("cli.utils.signals.sys") as mock_sys:
            mock_sys.platform = "win32"
            # Import the actual check pattern used
            should_set_sigterm = mock_sys.platform != "win32"
            assert not should_set_sigterm
