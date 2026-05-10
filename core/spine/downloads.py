"""Video-download spine.

Bulk URL → file downloads with per-URL failure aggregation, optional
``progress_callback``, and ``cancel_event`` support. Used by both the GUI
agent's ``download_video`` chat tool and the MCP ``start_download_videos``
job.

URL validation goes through ``core.spine.url_security.validate_url``;
``core.downloader.VideoDownloader`` is the underlying yt-dlp wrapper.

Cancellation: per-URL granularity. The yt-dlp subprocess inside the
``VideoDownloader.download`` call cannot be interrupted mid-download from
this layer; if cancellation lands during a download, it is observed
between URLs. This is acceptable for v1 (typical clip downloads complete
in 10-60s), but a follow-up could plumb ``proc.terminate()`` into the
downloader for true mid-download cancellation.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from core.spine.url_security import validate_url

logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


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

    succeeded: list[dict] = []
    failed: list[dict] = []
    cancelled: list[str] = []

    try:
        downloader = VideoDownloader(download_dir=target)
    except RuntimeError as exc:
        return {
            "success": False,
            "error": {"code": "downloader_unavailable", "message": str(exc)},
        }

    total = max(len(urls), 1)
    for i, url in enumerate(urls):
        if _check_cancel(cancel_event):
            cancelled.extend(urls[i:])
            break

        if progress_callback is not None:
            progress_callback(
                i / total, f"Downloading ({i + 1}/{len(urls)}): {url}"
            )

        # Scheme + host whitelist check before yt-dlp gets the URL.
        url_ok, url_err = validate_url(url)
        if not url_ok:
            failed.append(
                {
                    "url": url,
                    "error_code": "invalid_url",
                    "error_message": url_err,
                }
            )
            continue

        try:
            result = downloader.download(url)
        except Exception as exc:  # noqa: BLE001 — per-URL resilience
            failed.append(
                {
                    "url": url,
                    "error_code": "download_exception",
                    "error_message": str(exc),
                }
            )
            continue

        if result.success:
            succeeded.append(
                {
                    "url": url,
                    "file_path": str(result.file_path)
                    if result.file_path
                    else None,
                    "title": result.title,
                    "duration": result.duration,
                }
            )
        else:
            failed.append(
                {
                    "url": url,
                    "error_code": "download_failed",
                    "error_message": result.error or "unknown",
                }
            )

    if progress_callback is not None:
        progress_callback(
            1.0,
            f"Downloads complete: {len(succeeded)} ok, {len(failed)} failed, "
            f"{len(cancelled)} cancelled",
        )

    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "cancelled": cancelled,
            "target_dir": str(target),
        },
    }
