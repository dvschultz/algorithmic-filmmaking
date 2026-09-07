"""Video-download spine.

Bulk URL → file downloads with per-URL failure aggregation, optional
``progress_callback``, and ``cancel_event`` support. Used by both the GUI
agent's ``download_video`` chat tool and the MCP ``start_download_videos``
job.

URL validation goes through ``core.spine.url_security.validate_url``;
``core.downloader.VideoDownloader`` is the underlying yt-dlp wrapper.

Cancellation reaches the downloader's cooperative cancellation check and is
checked between stages by the shared operation. Native metadata requests may
still run until their existing timeout.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from core.operations.downloads import (
    DownloadRequest,
    DownloadOutcome,
    format_download_results,
)

logger = logging.getLogger(__name__)


def download_videos(
    urls: list[str],
    target_dir: Path | str,
    *,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Download every URL in ``urls`` to ``target_dir``.

    Per-URL failures (geo-block, DRM, deleted, invalid scheme/host) are
    aggregated into ``failed`` and never raised mid-batch. Returns

        {
            "success": True,
            "result": {
                "succeeded": [{"url", "file_path", "title", "duration"}],
                "failed":    [{"url", "error_code", "error_message"}],
                "cancelled": [<urls not started before cancel>],
                "target_dir": str(target_dir),
            },
        }
    """
    from core.downloader import VideoDownloader

    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    try:
        downloader = VideoDownloader(download_dir=target)
    except RuntimeError as exc:
        return {
            "success": False,
            "error": {"code": "downloader_unavailable", "message": str(exc)},
        }

    requests = tuple(DownloadRequest(url, target) for url in urls)
    completed = 0
    if progress_callback is not None:
        progress_callback(0.0, f"Starting {len(requests)} downloads")

    def on_item(outcome: DownloadOutcome) -> None:
        nonlocal completed
        completed += 1
        if progress_callback is not None:
            progress_callback(
                completed / max(len(requests), 1),
                f"Processed {completed}/{len(requests)} downloads",
            )

    from core.jobs.downloads import open_download_store, run_recoverable_batch

    store = open_download_store()
    try:
        outcomes = run_recoverable_batch(
            store,
            requests,
            target,
            downloader=downloader,
            cancel_event=cancel_event,
            item_callback=on_item,
        )
    finally:
        store.close()
    output = format_download_results(outcomes, target)
    if progress_callback is not None:
        payload = output["result"]
        progress_callback(
            1.0,
            f"Downloads complete: {len(payload['succeeded'])} ok, {len(payload['failed'])} failed, "
            f"{len(payload['cancelled'])} cancelled",
        )
    return output
