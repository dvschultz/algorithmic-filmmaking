"""Shared, GUI-independent download execution and frozen request settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable, Literal

from core.spine.url_security import validate_url

if TYPE_CHECKING:
    from core.downloader import DownloadResult, VideoDownloader
    from core.project import Project
    from models.clip import Source


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


@dataclass(frozen=True)
class DownloadItem:
    """Immutable per-item delivery captured when the file download completes."""

    index: int
    url: str
    status: Literal["succeeded", "failed", "cancelled"]
    path: Path | None
    stamp: tuple[int, ...] | None
    title: str | None = None
    duration: float | None = None
    error: str | None = None

    @classmethod
    def capture(cls, outcome: DownloadOutcome) -> DownloadItem:
        from core.jobs.media import media_stamp

        result = outcome.result
        path = Path(result.file_path) if result and result.file_path else None
        status = outcome.status
        if status == "succeeded" and (
            result is None or not result.success or path is None
        ):
            status = "failed"
        return cls(
            outcome.index,
            outcome.request.url,
            status,
            path,
            media_stamp(path) if path is not None else None,
            result.title if result else None,
            result.duration if result else None,
            outcome.error_message or (result.error if result else None),
        )

    def as_result(self) -> DownloadResult:
        from core.downloader import DownloadResult

        return DownloadResult(
            success=self.status == "succeeded",
            file_path=self.path,
            title=self.title,
            duration=self.duration,
            error=self.error
            or (None if self.status == "succeeded" else "Download did not complete"),
        )


class DownloadApplication:
    """Admit completed files once without overwriting newer source identity."""

    def __init__(self, project: Project) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path
        self.original_sources = tuple(
            (source, Path(source.file_path)) for source in project.sources
        )

    def apply(
        self, item: DownloadItem, *, still_current: Callable[[], bool]
    ) -> tuple[Source, bool] | None:
        from core.jobs.media import media_stamp
        from core.spine.sources import find_source_by_path, same_source_path
        from models.clip import Source

        project = self.project

        def publish() -> tuple[Source, bool] | None:
            if (
                not still_current()
                or project.session.session_id != self.session_id
                or project.path != self.path
            ):
                return None
            if (
                item.status != "succeeded"
                or item.path is None
                or item.stamp is None
                or media_stamp(item.path) != item.stamp
                or not item.path.is_file()
            ):
                raise ValueError("Downloaded media changed or is unavailable")
            original = next(
                (
                    source
                    for source, path in self.original_sources
                    if same_source_path(path, item.path)
                ),
                None,
            )
            current = find_source_by_path(project, item.path)
            added = current is None
            if original is not None and current is not original:
                raise ValueError("Download source was removed or replaced")
            if current is None:
                current = Source(
                    file_path=item.path,
                    duration_seconds=item.duration or 0,
                    fps=30,
                    width=1920,
                    height=1080,
                )
                project.add_source(current)
            if not still_current():
                return None
            if project.sources_by_id.get(current.id) is not current:
                raise ValueError("Download source was replaced during publication")
            return current, added

        return project.session.apply_external(publish)


def run_download_batch(
    requests: list[DownloadRequest] | tuple[DownloadRequest, ...],
    *,
    max_workers: int = 1,
    cancel_event: Event | None = None,
    item_callback: Callable[[DownloadOutcome], None] | None = None,
    downloader: VideoDownloader | None = None,
    cached_outcomes: dict[int, DownloadOutcome] | None = None,
    native_progress: Callable[[int, float, str], None] | None = None,
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
            result = run_download(
                request,
                downloader=downloader,
                cancel_event=cancel,
                progress_callback=(
                    lambda value, message: native_progress(index, value, message)
                )
                if native_progress is not None
                else None,
            )
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


def format_download_results(
    outcomes: tuple[DownloadOutcome, ...], target: Path
) -> dict:
    """Keep the established headless envelope across volatile and durable runs."""
    succeeded: list[dict] = []
    failed: list[dict] = []
    cancelled: list[str] = []
    for outcome in outcomes:
        url = outcome.request.url
        result = outcome.result
        if outcome.status == "succeeded" and result is not None:
            succeeded.append(
                {
                    "url": url,
                    "file_path": str(result.file_path) if result.file_path else None,
                    "title": result.title,
                    "duration": result.duration,
                }
            )
        elif outcome.status == "cancelled":
            cancelled.append(url)
        else:
            failed.append(
                {
                    "url": url,
                    "error_code": outcome.error_code,
                    "error_message": outcome.error_message,
                }
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
