"""Tests for the video-download spine.

The actual yt-dlp subprocess is heavily stubbed — these tests focus on
spine orchestration: URL validation, per-URL failure aggregation,
cancellation between URLs, and progress callback behaviour.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.spine.downloads import download_videos


def _ok_result(file_path: Path, title: str = "Demo") -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        file_path=file_path,
        title=title,
        duration=42.0,
        error=None,
    )


def _fail_result(message: str) -> SimpleNamespace:
    return SimpleNamespace(
        success=False, file_path=None, title=None, duration=None, error=message
    )


def test_download_videos_happy_path(tmp_path):
    fake_file = tmp_path / "video.mp4"
    fake_file.write_bytes(b"fake")

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download",
        return_value=_ok_result(fake_file),
    ):
        result = download_videos(
            ["https://www.youtube.com/watch?v=abc"],
            tmp_path,
        )

    assert result["success"] is True
    payload = result["result"]
    assert len(payload["succeeded"]) == 1
    entry = payload["succeeded"][0]
    assert entry["url"] == "https://www.youtube.com/watch?v=abc"
    assert entry["file_path"] == str(fake_file)
    assert payload["failed"] == []
    assert payload["cancelled"] == []


def test_download_videos_invalid_url_fails_fast(tmp_path):
    """URL that fails scheme/host validation never reaches yt-dlp."""
    download_calls = []

    def fake_download(self, url):
        download_calls.append(url)
        return _ok_result(tmp_path / "x.mp4")

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download", new=fake_download
    ):
        result = download_videos(
            ["javascript://example.com/x", "https://www.youtube.com/watch?v=abc"],
            tmp_path,
        )

    payload = result["result"]
    # Bad URL surfaces as an invalid_url failure without ever calling
    # downloader.download.
    assert any(
        f["url"] == "javascript://example.com/x"
        and f["error_code"] == "invalid_url"
        for f in payload["failed"]
    )
    assert len(download_calls) == 1


def test_download_videos_per_url_failure_aggregated(tmp_path):
    fake_file = tmp_path / "video.mp4"
    fake_file.write_bytes(b"fake")

    call_count = [0]

    def fake_download(self, url):
        call_count[0] += 1
        if "geo" in url:
            return _fail_result("geo-blocked")
        return _ok_result(fake_file, title=f"v-{call_count[0]}")

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download", new=fake_download
    ):
        result = download_videos(
            [
                "https://www.youtube.com/watch?v=ok-1",
                "https://www.youtube.com/watch?v=geo-blocked",
                "https://www.youtube.com/watch?v=ok-2",
            ],
            tmp_path,
        )

    payload = result["result"]
    assert len(payload["succeeded"]) == 2
    assert len(payload["failed"]) == 1
    assert payload["failed"][0]["error_code"] == "download_failed"
    assert "geo-blocked" in payload["failed"][0]["error_message"]


def test_download_videos_cancellation(tmp_path):
    fake_file = tmp_path / "v.mp4"
    fake_file.write_bytes(b"fake")
    cancel = threading.Event()
    cancel.set()

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download",
        return_value=_ok_result(fake_file),
    ):
        result = download_videos(
            ["https://youtube.com/x", "https://youtube.com/y"],
            tmp_path,
            cancel_event=cancel,
        )

    payload = result["result"]
    # Both URLs land in cancelled because cancel was set before the loop.
    assert "https://youtube.com/x" in payload["cancelled"]
    assert "https://youtube.com/y" in payload["cancelled"]


def test_download_videos_creates_target_dir(tmp_path):
    target = tmp_path / "downloads"
    assert not target.exists()
    fake_file = target / "video.mp4"

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download",
        return_value=_ok_result(fake_file),
    ):
        download_videos(
            ["https://www.youtube.com/watch?v=abc"],
            target,
        )

    assert target.exists()


def test_download_videos_downloader_unavailable(tmp_path):
    def raise_runtime(self, **kwargs):
        raise RuntimeError("yt-dlp not found")

    with patch(
        "core.downloader.VideoDownloader.__init__", new=raise_runtime
    ):
        result = download_videos(
            ["https://www.youtube.com/watch?v=abc"], tmp_path
        )

    assert result["success"] is False
    assert result["error"]["code"] == "downloader_unavailable"


def test_download_videos_progress_callback(tmp_path):
    fake_file = tmp_path / "v.mp4"
    fake_file.write_bytes(b"fake")
    calls = []

    def cb(p, msg):
        calls.append((p, msg))

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download",
        return_value=_ok_result(fake_file),
    ):
        download_videos(
            ["https://www.youtube.com/watch?v=a"],
            tmp_path,
            progress_callback=cb,
        )

    assert calls
    assert any(p == 1.0 for p, _ in calls)


def test_download_videos_per_url_exception_is_aggregated(tmp_path):
    """If yt-dlp raises mid-download, we capture it without poisoning the batch."""
    fake_file = tmp_path / "v.mp4"
    fake_file.write_bytes(b"fake")

    call_count = [0]

    def fake_download(self, url):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("network error")
        return _ok_result(fake_file)

    with patch(
        "core.downloader.VideoDownloader.__init__", return_value=None
    ), patch(
        "core.downloader.VideoDownloader.download", new=fake_download
    ):
        result = download_videos(
            [
                "https://www.youtube.com/watch?v=a",
                "https://www.youtube.com/watch?v=b",
            ],
            tmp_path,
        )

    payload = result["result"]
    assert len(payload["failed"]) == 1
    assert payload["failed"][0]["error_code"] == "download_exception"
    assert len(payload["succeeded"]) == 1
