"""Regression tests for user-facing yt-dlp error handling."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.downloader import (
    VideoDownloader,
    DOWNLOAD_ERROR_COOKIES_REQUIRED,
    DOWNLOAD_ERROR_GENERIC,
    DOWNLOAD_ERROR_JS_RUNTIME_REQUIRED,
    YTDLP_COOKIE_HELP_URL,
    classify_download_error_message,
    _format_yt_dlp_diagnostics,
)


def test_get_video_info_reports_missing_js_runtime_on_windows(tmp_path):
    """Metadata fetch should surface a Deno hint instead of raw yt-dlp stderr."""
    stderr = (
        "WARNING: [youtube] No supported JavaScript runtime could be found. "
        "Only deno is enabled by default.\n"
        "ERROR: [youtube] abc123: Sign in to confirm you're not a bot."
    )

    with patch("core.downloader.find_binary", return_value="C:/bin/yt-dlp.exe"), \
         patch("core.downloader.get_subprocess_env", return_value={}), \
         patch("core.downloader.get_subprocess_kwargs", return_value={}), \
         patch("core.downloader.sys.platform", "win32"), \
         patch(
             "core.downloader.subprocess.run",
             return_value=SimpleNamespace(returncode=1, stdout="", stderr=stderr),
         ):
        downloader = VideoDownloader(download_dir=tmp_path)

        with pytest.raises(RuntimeError, match="JavaScript runtime"):
            downloader.get_video_info("https://www.youtube.com/watch?v=abc123")


def test_get_video_info_reports_cookie_requirement(tmp_path):
    """Metadata fetch should explain when YouTube requires browser cookies."""
    stderr = (
        "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )

    with patch("core.downloader.find_binary", return_value="yt-dlp"), \
         patch("core.downloader.get_subprocess_env", return_value={}), \
         patch("core.downloader.get_subprocess_kwargs", return_value={}), \
         patch(
             "core.downloader.subprocess.run",
             return_value=SimpleNamespace(returncode=1, stdout="", stderr=stderr),
         ):
        downloader = VideoDownloader(download_dir=tmp_path)

        with pytest.raises(RuntimeError, match="browser authentication cookies"):
            downloader.get_video_info("https://www.youtube.com/watch?v=abc123")


def test_get_video_info_cookie_requirement_includes_help_url(tmp_path):
    """Cookie-required errors should point users to the official docs."""
    stderr = (
        "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )

    with patch("core.downloader.find_binary", return_value="yt-dlp"), \
         patch("core.downloader.get_subprocess_env", return_value={}), \
         patch("core.downloader.get_subprocess_kwargs", return_value={}), \
         patch(
             "core.downloader.subprocess.run",
             return_value=SimpleNamespace(returncode=1, stdout="", stderr=stderr),
         ):
        downloader = VideoDownloader(download_dir=tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            downloader.get_video_info("https://www.youtube.com/watch?v=abc123")

    assert YTDLP_COOKIE_HELP_URL in str(exc_info.value)
    assert "yt-dlp exit code: 1" in str(exc_info.value)


def test_download_reports_startup_failure_with_diagnostics(tmp_path):
    """If yt-dlp cannot exec, the returned error should be actionable."""
    with patch("core.downloader.find_binary", return_value="yt-dlp"), \
         patch("core.downloader.get_subprocess_env", return_value={}), \
         patch("core.downloader.get_subprocess_kwargs", return_value={}), \
         patch.object(VideoDownloader, "get_video_info", return_value={"title": "Example", "duration": 1}), \
         patch("core.downloader.subprocess.Popen", side_effect=OSError("no executable")):
        downloader = VideoDownloader(download_dir=tmp_path)
        result = downloader.download("https://www.youtube.com/watch?v=abc123")

    assert result.success is False
    assert "could not be started" in result.error
    assert result.diagnostics is not None
    assert "no executable" in result.diagnostics


def test_yt_dlp_diagnostics_redacts_command_and_output_secrets():
    diagnostics = _format_yt_dlp_diagnostics(
        exit_code=1,
        command=["yt-dlp", "https://example.com/video.mp4?token=abc123&keep=ok"],
        output_lines=["ERROR: signed URL x-goog-signature=deadbeef failed"],
    )

    assert "abc123" not in diagnostics
    assert "deadbeef" not in diagnostics
    assert "keep=ok" in diagnostics


def test_classify_download_error_message_detects_cookie_requirement():
    """Cookie-auth messages should be classified for dedicated UI handling."""
    error = (
        "YouTube is asking for browser authentication cookies for this video. "
        f"Export cookies from a signed-in browser and retry. See {YTDLP_COOKIE_HELP_URL}"
    )
    assert classify_download_error_message(error) == DOWNLOAD_ERROR_COOKIES_REQUIRED


def test_classify_download_error_message_detects_js_runtime_requirement():
    """Missing runtime messages should be classified separately."""
    error = "yt-dlp requires a JavaScript runtime for YouTube downloads. Install Deno from Settings > Dependencies"
    assert classify_download_error_message(error) == DOWNLOAD_ERROR_JS_RUNTIME_REQUIRED


def test_classify_download_error_message_defaults_to_generic():
    """Other download failures should fall back to the generic path."""
    assert classify_download_error_message("Download timed out") == DOWNLOAD_ERROR_GENERIC
