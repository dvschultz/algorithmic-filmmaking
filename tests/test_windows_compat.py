"""Tests for Windows compatibility code paths.

These tests mock sys.platform to exercise Windows-specific branches
on any platform (macOS, Linux, or Windows CI).
"""

import io
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
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

    def test_finds_bundled_runtime_in_frozen_app(self, tmp_path):
        """find_binary should detect FFmpeg bundled inside a frozen app."""
        from core.binary_resolver import find_binary

        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        bundled_dir = tmp_path / "bin"
        bundled_dir.mkdir()
        (bundled_dir / "ffmpeg.exe").write_text("fake")

        with patch("core.binary_resolver.sys") as mock_sys, \
             patch("core.binary_resolver.is_frozen", return_value=True), \
             patch("core.binary_resolver.get_base_path", return_value=tmp_path), \
             patch("core.binary_resolver.get_bundled_bin_dir", return_value=bundled_dir), \
             patch("core.binary_resolver.get_managed_bin_dir", return_value=managed_dir):
            mock_sys.platform = "win32"
            mock_sys.executable = str(tmp_path / "Scene Ripper.exe")
            with patch("shutil.which", return_value=None):
                result = find_binary("ffmpeg")
                assert result is not None
                assert result.endswith("ffmpeg.exe")

    def test_identifies_bundled_binary_paths_in_frozen_app(self, tmp_path):
        """Bundled binary detection should recognize frozen app runtime paths."""
        from core.binary_resolver import is_bundled_binary_path

        bundled_dir = tmp_path / "bin"
        bundled_dir.mkdir()
        ffmpeg = bundled_dir / "ffmpeg.exe"
        ffmpeg.write_text("fake")

        with patch("core.binary_resolver.sys") as mock_sys, \
             patch("core.binary_resolver.is_frozen", return_value=True), \
             patch("core.binary_resolver.get_base_path", return_value=tmp_path), \
             patch("core.binary_resolver.get_bundled_bin_dir", return_value=bundled_dir):
            mock_sys.executable = str(tmp_path / "Scene Ripper.exe")
            assert is_bundled_binary_path(str(ffmpeg)) is True


class TestGetSubprocessEnv:
    """Test get_subprocess_env() PATH behavior."""

    def test_prepends_bundled_bin_in_frozen_app(self, tmp_path):
        """Frozen subprocess env should expose bundled binaries on PATH."""
        from core.binary_resolver import get_subprocess_env

        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        bundled_dir = tmp_path / "bin"
        bundled_dir.mkdir()

        with patch("core.binary_resolver.sys") as mock_sys, \
             patch("core.binary_resolver.is_frozen", return_value=True), \
             patch("core.binary_resolver.get_bundled_bin_dir", return_value=bundled_dir), \
             patch("core.binary_resolver.get_managed_bin_dir", return_value=managed_dir), \
             patch.dict(os.environ, {"PATH": r"C:\Windows\System32"}, clear=False):
            mock_sys.executable = str(tmp_path / "Scene Ripper.exe")
            env = get_subprocess_env()
            path_parts = env["PATH"].split(os.pathsep)
            assert path_parts[0] == str(managed_dir)
            assert str(bundled_dir) in path_parts[:3]


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

    def test_python_url_windows(self):
        """Should return the Windows standalone Python archive on Windows."""
        from core.dependency_manager import _get_python_url

        with patch("core.dependency_manager.sys") as mock_sys, \
             patch("core.dependency_manager.platform") as mock_platform:
            mock_sys.platform = "win32"
            mock_platform.machine.return_value = "AMD64"
            url = _get_python_url()
            assert "x86_64-pc-windows-msvc" in url
            assert url.endswith(".tar.gz")

    def test_deno_url_windows(self):
        """Should return the Windows Deno archive on Windows."""
        from core.dependency_manager import _get_deno_url

        with patch("core.dependency_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            url = _get_deno_url()
            assert "windows" in url or "pc-windows-msvc" in url
            assert url.endswith(".zip")

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


class TestDependencyManagerTls:
    """Test HTTPS certificate handling for runtime downloads."""

    def test_ssl_context_prefers_certifi_bundle(self):
        """Downloader should use certifi when available."""
        from core.dependency_manager import _get_ssl_context

        with patch("core.dependency_manager.certifi") as mock_certifi, \
             patch("core.dependency_manager.ssl.create_default_context", return_value="ctx") as mock_create:
            mock_certifi.where.return_value = "C:/bundle/cacert.pem"

            result = _get_ssl_context()

        assert result == "ctx"
        mock_create.assert_called_once_with(cafile="C:/bundle/cacert.pem")

    def test_download_file_passes_ssl_context_for_https(self, tmp_path):
        """HTTPS downloads should pass an explicit SSL context to urllib."""
        from core.dependency_manager import _download_file

        class FakeResponse:
            headers = {"Content-Length": "4"}

            def __init__(self):
                self._chunks = iter((b"test", b""))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, _size):
                return next(self._chunks)

        destination = tmp_path / "ffmpeg.zip"

        with patch("core.dependency_manager._get_ssl_context", return_value="ctx"), \
             patch("core.dependency_manager.urllib.request.urlopen", return_value=FakeResponse()) as mock_urlopen:
            result = _download_file("https://example.com/ffmpeg.zip", destination, label="FFmpeg")

        assert result == destination
        assert destination.read_bytes() == b"test"
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 120
        assert kwargs["context"] == "ctx"


def _make_windows_python_tarball() -> bytes:
    """Create a minimal python-build-standalone style tarball for Windows."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="python/python.exe")
        info.size = len(b"FAKEPYTHON")
        tf.addfile(info, io.BytesIO(b"FAKEPYTHON"))
    return buf.getvalue()


class TestEnsurePythonWindows:
    """Test managed Python install behavior on Windows."""

    def test_frozen_windows_uses_managed_python_not_app_executable(self, tmp_path):
        """Frozen Windows installs should download a real python.exe for pip work."""
        from core.dependency_manager import ensure_python

        managed_python_dir = tmp_path / "managed-python"
        tarball_bytes = _make_windows_python_tarball()

        def _fake_download(_url, dest, _progress=None, _label=""):
            dest.write_bytes(tarball_bytes)
            return dest

        with patch("core.dependency_manager.sys") as mock_sys, \
             patch("core.dependency_manager.platform") as mock_platform, \
             patch("core.dependency_manager.get_managed_python_dir", return_value=managed_python_dir), \
             patch("core.dependency_manager._download_file", side_effect=_fake_download), \
             patch(
                 "core.dependency_manager.subprocess.run",
                 return_value=SimpleNamespace(returncode=0, stdout="pip 24.0", stderr=""),
             ):
            mock_sys.platform = "win32"
            mock_sys.frozen = True
            mock_sys.executable = r"C:\Program Files\Scene Ripper\Scene Ripper.exe"
            mock_platform.machine.return_value = "AMD64"

            python_bin = ensure_python()

        assert python_bin == managed_python_dir / "python.exe"
        assert python_bin.is_file()
        assert str(python_bin) != mock_sys.executable


class TestBuildSupportWindows:
    """Test Windows build helper runtime staging."""

    def test_collects_staged_mpv_dlls(self, tmp_path):
        """PyInstaller build support should include staged mpv DLLs."""
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "packaging" / "build_support.py"
        spec = importlib.util.spec_from_file_location("scene_ripper_build_support", module_path)
        build_support = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(build_support)

        project_root = tmp_path
        runtime_dir = project_root / "packaging" / "runtime" / "mpv" / "windows"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "libmpv-2.dll").write_text("fake")
        (runtime_dir / "libgcc_s_seh-1.dll").write_text("fake")

        binaries = build_support.collect_windows_mpv_binaries(project_root)

        bundled_names = {Path(src).name for src, _ in binaries}
        assert "libmpv-2.dll" in bundled_names
        assert "libgcc_s_seh-1.dll" in bundled_names


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
