"""Thin Qt adapters for shared single and batch download execution."""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Signal
from core.operations.downloads import (
    DownloadRequest,
    DownloadCancelled,
    DownloadOutcome,
)
from ui.workers.base import CancellableWorker
from core.jobs.downloads import open_download_store, run_recoverable_batch


def _run_batch(
    requests,
    target,
    *,
    max_workers=1,
    cancel_event=None,
    item_callback=None,
    native_progress=None,
):
    """Translate storage failures into per-item UI outcomes, retaining successes."""
    received = {}

    def deliver(outcome):
        received[outcome.index] = outcome
        if item_callback is not None:
            item_callback(outcome)

    store = None
    try:
        store = open_download_store()
        return run_recoverable_batch(
            store,
            requests,
            target,
            max_workers=max_workers,
            cancel_event=cancel_event,
            item_callback=deliver,
            native_progress=native_progress,
        )
    except Exception as exc:
        for index, request in enumerate(requests):
            if index not in received:
                deliver(
                    DownloadOutcome(
                        index,
                        request,
                        "failed",
                        error_code="download_batch_failed",
                        error_message=str(exc),
                    )
                )
        return tuple(received[index] for index in range(len(requests)))
    finally:
        if store is not None:
            store.close()


class DownloadWorker(CancellableWorker):
    """Background worker for video downloads."""

    progress = Signal(float, str)  # progress (0-100), status message
    download_completed = Signal(
        object
    )  # DownloadResult (renamed from 'finished' to avoid shadowing QThread.finished)

    def __init__(self, url: str, resolution: Optional[str] = None):
        super().__init__()
        self.url = url
        self.resolution = resolution
        from core.settings import get_default_download_dir

        self.request = DownloadRequest(
            url, get_default_download_dir(), resolution=resolution
        )

    def run(self):
        try:
            self.progress.emit(0, "Preparing download")
            outcome = _run_batch(
                (self.request,),
                self.request.download_dir,
                cancel_event=self._cancel_event,
                native_progress=lambda index, value, message: self.progress.emit(
                    min(value, 99), message
                ),
            )[0]
            if outcome.status == "succeeded":
                self.progress.emit(100, "Download complete")
                self.download_completed.emit(outcome.result)
            elif outcome.status == "failed":
                self.error.emit(outcome.error_message or "Download failed")
        except DownloadCancelled:
            return
        except Exception as e:
            self.error.emit(str(e))


class URLBulkDownloadWorker(CancellableWorker):
    """Download URL batches with bounded parallelism and per-item delivery."""

    progress = Signal(int, int, str)
    video_finished = Signal(str, object)
    all_finished = Signal(list)
    MAX_WORKERS = 3

    def __init__(self, urls: list[str], download_dir: Path):
        super().__init__()
        self.download_dir = Path(download_dir).expanduser().resolve()
        self.requests = tuple(
            DownloadRequest(url, download_dir, adaptive_timeout=True) for url in urls
        )

    def run(self) -> None:
        total = len(self.requests)
        completed = 0
        self.progress.emit(0, total, f"Starting {total} downloads...")

        def on_item(outcome: DownloadOutcome) -> None:
            nonlocal completed
            completed += 1
            if outcome.status == "succeeded":
                self.video_finished.emit(outcome.request.url, outcome.result)
            self.progress.emit(
                completed, total, f"Processed {completed}/{total} downloads"
            )

        outcomes = _run_batch(
            self.requests,
            self.download_dir,
            max_workers=self.MAX_WORKERS,
            cancel_event=self._cancel_event,
            item_callback=on_item,
        )
        results = []
        for outcome in outcomes:
            result = outcome.result
            if outcome.status == "succeeded" and result is not None:
                results.append(
                    {
                        "url": outcome.request.url,
                        "success": True,
                        "file_path": str(result.file_path)
                        if result.file_path
                        else None,
                        "title": result.title,
                        "duration": result.duration,
                    }
                )
            else:
                item = {
                    "url": outcome.request.url,
                    "success": False,
                    "error": outcome.error_message or "Download cancelled",
                }
                if outcome.status == "cancelled":
                    item["cancelled"] = True
                results.append(item)
        self.all_finished.emit(results)


class BulkDownloadWorker(CancellableWorker):
    """Adapt search-result identity to the shared scheduler."""

    progress = Signal(int, int, str)
    video_finished = Signal(object)
    video_error = Signal(str, str)
    all_finished = Signal()

    def __init__(self, videos: list, download_dir: Path, max_parallel: int = 2):
        super().__init__()
        # Snapshot provider objects before crossing the thread boundary.
        self.items = tuple(
            (
                video.video_id,
                DownloadRequest(
                    getattr(video, "youtube_url", None)
                    or getattr(video, "download_url", "")
                    or "",
                    download_dir,
                ),
            )
            for video in videos
        )
        self.max_parallel = max_parallel
        self.download_dir = Path(download_dir).expanduser().resolve()

    def run(self) -> None:
        total = len(self.items)
        completed = 0
        self.progress.emit(0, total, f"Starting {total} downloads...")

        def on_item(outcome: DownloadOutcome) -> None:
            nonlocal completed
            completed += 1
            if outcome.status == "succeeded":
                self.video_finished.emit(outcome.result)
            elif outcome.status == "failed":
                self.video_error.emit(
                    self.items[outcome.index][0],
                    outcome.error_message or "Download failed",
                )
            self.progress.emit(
                completed, total, f"Processed {completed}/{total} downloads"
            )

        _run_batch(
            tuple(request for _, request in self.items),
            self.download_dir,
            max_workers=self.max_parallel,
            cancel_event=self._cancel_event,
            item_callback=on_item,
        )
        self.all_finished.emit()
