"""Shared, GUI-independent download execution and frozen request settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable, Literal

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


@dataclass(frozen=True)
class DownloadOutcome:
    index: int
    request: DownloadRequest
    status: Literal["succeeded", "failed", "cancelled"]
    result: DownloadResult | None = None
    error_code: str | None = None
    error_message: str | None = None


def run_download_batch(
    requests: list[DownloadRequest] | tuple[DownloadRequest, ...],
    *,
    max_workers: int = 1,
    cancel_event: Event | None = None,
    item_callback: Callable[[DownloadOutcome], None] | None = None,
    downloader: VideoDownloader | None = None,
    cached_outcomes: dict[int, DownloadOutcome] | None = None,
) -> tuple[DownloadOutcome, ...]:
    """Run a bounded batch and return one outcome per input in input order.

    Item callbacks run on the calling thread as results arrive. A callback error
    stops dispatch and propagates after active downloads observe cancellation.
    Completed successes remain successes even when a later item is cancelled.
    A supplied downloader is supported only for serial execution.
    """
    from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if downloader is not None and max_workers != 1:
        raise ValueError("A shared downloader requires serial execution")
    pending_requests = tuple(requests)
    cached = dict(cached_outcomes or {})
    for index, outcome in cached.items():
        if (
            not 0 <= index < len(pending_requests)
            or outcome.index != index
            or outcome.request != pending_requests[index]
        ):
            raise ValueError("Cached download outcome does not match its request")
    cancel = cancel_event if cancel_event is not None else Event()
    results: dict[int, DownloadOutcome] = {}

    def execute(index: int) -> DownloadOutcome:
        request = pending_requests[index]
        if cancel.is_set():
            return DownloadOutcome(index, request, "cancelled")
        valid, error = validate_url(request.url)
        if not valid:
            return DownloadOutcome(
                index, request, "failed", error_code="invalid_url", error_message=error
            )
        if index in cached:
            return cached[index]
        try:
            result = run_download(request, downloader=downloader, cancel_event=cancel)
            if result.success:
                return DownloadOutcome(index, request, "succeeded", result=result)
            return DownloadOutcome(
                index,
                request,
                "failed",
                result=result,
                error_code="download_failed",
                error_message=result.error or "unknown",
            )
        except DownloadCancelled:
            return DownloadOutcome(index, request, "cancelled")
        except Exception as exc:
            return DownloadOutcome(
                index,
                request,
                "failed",
                error_code="download_exception",
                error_message=str(exc),
            )

    def publish(outcome: DownloadOutcome) -> None:
        results[outcome.index] = outcome
        if item_callback is not None:
            item_callback(outcome)

    next_index = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active: dict[Future[DownloadOutcome], int] = {}
        try:
            while active or next_index < len(pending_requests):
                while (
                    not cancel.is_set()
                    and len(active) < max_workers
                    and next_index < len(pending_requests)
                ):
                    active[executor.submit(execute, next_index)] = next_index
                    next_index += 1
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: active[item]):
                    del active[future]
                    publish(future.result())
            for index in range(next_index, len(pending_requests)):
                publish(DownloadOutcome(index, pending_requests[index], "cancelled"))
        except BaseException:
            cancel.set()
            raise
    return tuple(results[index] for index in range(len(pending_requests)))
