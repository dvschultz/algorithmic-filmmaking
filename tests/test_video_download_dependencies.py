"""Regression tests for yt-dlp dependency prompts in download flows."""

from types import SimpleNamespace
from unittest.mock import patch

from core.downloader import VideoDownloader
from ui.main_window import MainWindow


def test_ytdlp_error_in_frozen_app_points_to_managed_install():
    """Frozen apps should not tell users to run pip for yt-dlp."""
    with patch("core.downloader.find_binary", return_value=None), \
         patch("core.downloader.is_frozen", return_value=True), \
         patch("core.downloader.get_managed_bin_dir", return_value="C:/Users/test/AppData/Local/Scene Ripper/bin"):
        try:
            VideoDownloader()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("VideoDownloader() should raise when yt-dlp is missing")

    assert "Settings > Dependencies" in message
    assert "pip install yt-dlp" not in message


def test_download_video_aborts_when_video_download_dependency_missing():
    """Single URL downloads should prompt for yt-dlp before starting a worker."""
    class Harness:
        def __init__(self):
            self.collect_tab = SimpleNamespace(
                set_downloading=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set downloading"))
            )
            self.progress_bar = SimpleNamespace(
                setVisible=lambda *_args: (_ for _ in ()).throw(AssertionError("should not show progress")),
                setRange=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set range")),
            )
            self._gui_state = SimpleNamespace(
                set_processing=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set processing"))
            )
            self.download_worker = None

        def _ensure_video_download_available(self):
            return False

    harness = Harness()

    MainWindow._download_video(harness, "https://youtube.com/watch?v=abc")

    assert harness.download_worker is None


def test_bulk_download_aborts_when_video_download_dependency_missing():
    """Bulk downloads should not start a worker when yt-dlp is unavailable."""
    class Harness:
        def __init__(self):
            self.collect_tab = SimpleNamespace(
                youtube_search_panel=SimpleNamespace(
                    set_downloading=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set downloading"))
                )
            )
            self.progress_bar = SimpleNamespace(
                setVisible=lambda *_args: (_ for _ in ()).throw(AssertionError("should not show progress")),
                setRange=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set range")),
                setValue=lambda *_args: (_ for _ in ()).throw(AssertionError("should not set progress value")),
            )
            self.bulk_download_worker = None
            self.settings = SimpleNamespace(
                download_dir="/tmp/downloads",
                youtube_parallel_downloads=2,
            )

        def _ensure_video_download_available(self):
            return False

    harness = Harness()
    videos = [SimpleNamespace(video_id="vid-1", title="Video 1")]

    MainWindow._on_bulk_download(harness, videos)

    assert harness.bulk_download_worker is None
