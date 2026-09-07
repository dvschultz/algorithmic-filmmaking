"""Shared, GUI-independent download execution and frozen request settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable

from core.spine.url_security import validate_url

if TYPE_CHECKING:
    from core.downloader import DownloadResult, VideoDownloader


class DownloadCancelled(RuntimeError):
    """Cancellation prevents later download stages or result publication."""


@dataclass(frozen=True)
class DownloadRequest:
    url: str
    download_dir: Path | None = None
    resolution: str | None = None
    adaptive_timeout: bool = False

    def __post_init__(self) -> None:
        if self.download_dir is not None:
            object.__setattr__(
                self, "download_dir", Path(self.download_dir).expanduser().absolute()
            )


def calculate_download_timeout(duration_seconds: float, height: int | None) -> int:
    """Calculate download timeout based on video duration and resolution.

    Args:
        duration_seconds: Video duration in seconds
        height: Video height in pixels (e.g., 1080 for 1080p), or None if unknown

    Returns:
        Timeout in seconds
    """
    # Seconds of timeout per minute of video, by resolution
    # Higher resolutions = larger files = more download time needed
    TIMEOUT_MULTIPLIERS = {
        4320: 180,  # 8K: 3 min timeout per video minute
        2160: 120,  # 4K: 2 min timeout per video minute
        1440: 90,  # 1440p: 1.5 min timeout per video minute
        1080: 60,  # 1080p: 1 min timeout per video minute
        720: 45,  # 720p: 45 sec timeout per video minute
        480: 30,  # 480p: 30 sec timeout per video minute
        360: 20,  # 360p: 20 sec timeout per video minute
    }
    MIN_TIMEOUT = 120  # 2 minutes minimum
    MAX_TIMEOUT = 3600  # 1 hour cap
    DEFAULT_MULTIPLIER = 60  # Default to 1080p assumption

    # Find the appropriate multiplier based on resolution
    if height is None:
        multiplier = DEFAULT_MULTIPLIER
    else:
        # Find closest resolution tier (round down to nearest tier)
        multiplier = DEFAULT_MULTIPLIER
        for tier_height, tier_multiplier in sorted(TIMEOUT_MULTIPLIERS.items()):
            if height >= tier_height:
                multiplier = tier_multiplier

    duration_minutes = duration_seconds / 60
    timeout = int(duration_minutes * multiplier)
    return max(MIN_TIMEOUT, min(timeout, MAX_TIMEOUT))


def run_download(
    request: DownloadRequest,
    *,
    downloader: VideoDownloader | None = None,
    cancel_event: Event | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> DownloadResult:
    """Run one download, checking cancellation before and after native stages."""
    from core.downloader import DownloadResult, VideoDownloader

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled("Download cancelled")

    check_cancelled()
    valid, error = validate_url(request.url)
    if not valid:
        return DownloadResult(success=False, error=error)
    if downloader is None:
        downloader = VideoDownloader(download_dir=request.download_dir)
    timeout = 3600
    if request.adaptive_timeout:
        try:
            info = downloader.get_video_info(request.url, include_format_details=True)
            timeout = calculate_download_timeout(
                info.get("duration", 0) or 0, info.get("height")
            )
        except Exception:
            timeout = 600
    check_cancelled()
    try:
        result = downloader.download(
            request.url,
            progress_callback=progress_callback,
            cancel_check=cancel_event.is_set if cancel_event is not None else None,
            max_download_seconds=timeout,
            resolution=request.resolution,
        )
    except Exception:
        check_cancelled()
        raise
    check_cancelled()
    return result
