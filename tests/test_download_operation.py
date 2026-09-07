"""Shared downloads freeze requests and honor cancellation between stages."""

from threading import Event
from unittest.mock import Mock
import pytest
from core.operations.downloads import DownloadRequest, DownloadCancelled, run_download


def test_request_options_and_cancellation_reach_backend(tmp_path):
    cancel = Event()
    backend = Mock()
    backend.get_video_info.return_value = {"duration": 600, "height": 2160}
    request = DownloadRequest(
        "https://youtube.com/watch?v=test",
        tmp_path,
        resolution="720p",
        adaptive_timeout=True,
    )
    assert (
        run_download(request, downloader=backend, cancel_event=cancel)
        is backend.download.return_value
    )
    kwargs = backend.download.call_args.kwargs
    assert kwargs["resolution"] == "720p"
    assert kwargs["max_download_seconds"] == 1200
    assert not kwargs["cancel_check"]()
    cancel.set()
    assert kwargs["cancel_check"]()


@pytest.mark.parametrize("when", ["before", "metadata", "download"])
def test_cancelled_download_does_not_publish(when):
    cancel = Event()
    backend = Mock()
    backend.get_video_info.return_value = {}
    if when == "before":
        cancel.set()
    elif when == "metadata":
        backend.get_video_info.side_effect = lambda *a, **k: (cancel.set() or {})
    else:
        backend.download.side_effect = lambda *a, **k: (
            cancel.set() or Mock(success=True)
        )
    with pytest.raises(DownloadCancelled):
        run_download(
            DownloadRequest("https://youtube.com/test", adaptive_timeout=True),
            downloader=backend,
            cancel_event=cancel,
        )
    assert backend.download.call_count == (when == "download")


def test_invalid_url_never_fetches_metadata():
    backend = Mock()
    result = run_download(
        DownloadRequest("file:///etc/passwd", adaptive_timeout=True), downloader=backend
    )
    assert not result.success
    backend.get_video_info.assert_not_called()
    backend.download.assert_not_called()


def test_native_error_after_cancel_is_cancellation():
    cancel = Event()
    backend = Mock()

    def fail(*args, **kwargs):
        cancel.set()
        raise RuntimeError("terminated subprocess")

    backend.download.side_effect = fail
    with pytest.raises(DownloadCancelled):
        run_download(
            DownloadRequest("https://youtube.com/test"),
            downloader=backend,
            cancel_event=cancel,
        )
