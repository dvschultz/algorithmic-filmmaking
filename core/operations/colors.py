"""Resolve, compute and apply color analysis without GUI dependencies.

Workers compute immutable snapshots. The owner of the project applies the
result through a one-use application object guarded by the project session.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable, Iterable
from uuid import uuid4

from core.operations.contracts import ColorOutcome, ColorResult, Palette

if TYPE_CHECKING:
    from core.analysis_target import AnalysisTarget
    from core.project import Project


class ColorDependencyError(RuntimeError):
    """The color backend could not load; the batch must not be applied."""


def _file_stamp(path: Path | None) -> tuple[int, int] | None:
    try:
        stat = path.stat() if path is not None else None
        return (stat.st_mtime_ns, stat.st_size) if stat is not None else None
    except OSError:
        return None


@dataclass(frozen=True)
class ColorTarget:
    target_id: str
    target_type: str = "clip"
    video_path: Path | None = None
    image_path: Path | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    existing_colors: Palette | None = None
    file_stamp: tuple[int, int] | None = None
    missing: bool = False


@dataclass(frozen=True)
class ColorRequest:
    targets: tuple[ColorTarget, ...]
    num_colors: int = 5
    skip_existing: bool = True
    skip_empty: bool = False
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if type(self.num_colors) is not int or self.num_colors < 1:
            raise ValueError("num_colors must be a positive integer")
        if len({t.target_id for t in self.targets}) != len(self.targets):
            raise ValueError("Color targets must have unique IDs")


def _snapshot(target: AnalysisTarget) -> ColorTarget:
    video = target.video_path
    if target.start_frame is None or target.end_frame is None:
        video = None
    image = None if video is not None else target.image_path
    colors = target.dominant_colors
    return ColorTarget(
        target_id=target.id,
        target_type=target.target_type,
        video_path=video,
        image_path=image,
        start_frame=target.start_frame if video is not None else None,
        end_frame=target.end_frame if video is not None else None,
        existing_colors=tuple((int(r), int(g), int(b)) for r, g, b in colors)
        if colors is not None
        else None,
        file_stamp=_file_stamp(video or image),
    )


def request_from_targets(
    targets: Iterable[AnalysisTarget],
    *,
    num_colors: int = 5,
    skip_existing: bool = True,
    skip_empty: bool = True,
) -> ColorRequest:
    """Snapshot clip/frame targets on the model-owning thread."""
    return ColorRequest(
        tuple(_snapshot(t) for t in targets),
        num_colors,
        skip_existing,
        skip_empty,
    )


def _project_target(
    project: Project,
    target_id: str,
    target_type: str = "clip",
    *,
    image_fallback: bool = False,
) -> ColorTarget:
    from core.analysis_target import AnalysisTarget

    if target_type == "frame":
        frame = project.frames_by_id.get(target_id)
        if frame is not None:
            return _snapshot(AnalysisTarget.from_frame(frame))
    else:
        clip = project.clips_by_id.get(target_id)
        if clip is not None:
            source = project.sources_by_id.get(clip.source_id)
            target = AnalysisTarget.from_clip(clip, source)
            # Project clip operations require the source; thumbnails are only
            # a fallback when an adapter explicitly supplies an image target.
            if not image_fallback:
                target.image_path = None
            return _snapshot(target)
    return ColorTarget(target_id, target_type, missing=True)


def color_request(
    project: Project,
    clip_ids: list[str] | None = None,
    num_colors: int = 5,
    *,
    skip_existing: bool = True,
    skip_empty: bool = False,
) -> ColorRequest:
    """Resolve exact IDs without silently discarding missing targets."""
    ids = (
        [c.id for c in project.clips]
        if clip_ids is None
        else list(dict.fromkeys(clip_ids))
    )
    return ColorRequest(
        tuple(_project_target(project, target_id) for target_id in ids),
        num_colors,
        skip_existing,
        skip_empty,
    )


def _compute_target(
    target: ColorTarget, request: ColorRequest, cancel_event: Event | None
) -> ColorOutcome:
    if cancel_event is not None and cancel_event.is_set():
        return ColorOutcome(target.target_id, "unprocessed", code="cancelled")
    if target.missing:
        return ColorOutcome(target.target_id, "failed", code="target_not_found")
    existing = target.existing_colors
    if request.skip_existing and (
        bool(existing) or (request.skip_empty and existing is not None)
    ):
        return ColorOutcome(target.target_id, "skipped", code="already_populated")
    path = target.video_path or target.image_path
    try:
        if path is None or not path.is_file():
            return ColorOutcome(target.target_id, "failed", code="source_file_missing")
        try:
            from core.analysis.color import extract_dominant_colors
        except ImportError as exc:
            raise ColorDependencyError(str(exc)) from exc

        colors = extract_dominant_colors(
            video_path=target.video_path or Path(),
            start_frame=target.start_frame or 0,
            end_frame=target.end_frame or 0,
            n_colors=request.num_colors,
            image_path=target.image_path,
        )
        if not colors:
            return ColorOutcome(target.target_id, "failed", code="no_colors_extracted")
        return ColorOutcome(
            target.target_id,
            "succeeded",
            tuple((int(r), int(g), int(b)) for r, g, b in colors),
        )
    except ColorDependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 — one failed clip must not discard the batch
        return ColorOutcome(
            target.target_id, "failed", code="extraction_failed", message=str(exc)
        )


def compute_colors(
    request: ColorRequest,
    *,
    parallelism: int = 1,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, ColorOutcome], None] | None = None,
) -> ColorResult:
    """Compute with bounded dispatch; callbacks run on the calling thread.

    Cancellation stops dispatch. Already running extractions finish and their
    successful results remain usable. Output order always matches input order.
    """
    workers = min(max(1, parallelism), 8)
    outcomes: dict[str, ColorOutcome] = {}
    pending_targets = iter(request.targets)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[ColorOutcome]] = set()

        def dispatch() -> None:
            while len(pending) < workers and not cancelled():
                target = next(pending_targets, None)
                if target is None:
                    break
                pending.add(pool.submit(_compute_target, target, request, cancel_event))

        dispatch()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                outcome = future.result()
                outcomes[outcome.target_id] = outcome
                if progress_callback is not None:
                    progress_callback(len(outcomes), len(request.targets), outcome)
            dispatch()

    return ColorResult(
        request.request_id,
        tuple(
            outcomes.get(
                t.target_id, ColorOutcome(t.target_id, "unprocessed", code="cancelled")
            )
            for t in request.targets
        ),
    )


class ColorApplication:
    """One-use application of a request's result to its originating project."""

    def __init__(self, project: Project, request: ColorRequest) -> None:
        project.session.assert_owner()
        self.project = project
        self.request = request
        self._applied = False
        self._session_id = project.session.session_id

    def apply(self, result: ColorResult) -> ColorResult:
        return self.project.session.apply_external(lambda: self._apply(result))

    def _apply(self, result: ColorResult) -> ColorResult:
        if self._session_id != self.project.session.session_id:
            raise ValueError("Color result belongs to an expired project session")
        if self._applied:
            raise ValueError("Color result already applied")
        if result.request_id != self.request.request_id:
            raise ValueError("Color result belongs to a different request")
        if tuple(o.target_id for o in result.outcomes) != tuple(
            t.target_id for t in self.request.targets
        ):
            raise ValueError("Color result targets do not match the request")
        self._applied = True
        updated = []
        outcomes = []
        for target, outcome in zip(self.request.targets, result.outcomes):
            if outcome.status == "succeeded":
                current = _project_target(
                    self.project,
                    target.target_id,
                    target.target_type,
                    image_fallback=target.image_path is not None,
                )
                if current != target:
                    outcome = replace(
                        outcome, status="failed", colors=(), code="stale_input"
                    )
                elif target.target_type == "frame":
                    self.project.update_frame(
                        target.target_id, dominant_colors=list(outcome.colors)
                    )
                else:
                    clip = self.project.clips_by_id[target.target_id]
                    clip.dominant_colors = list(outcome.colors)
                    updated.append(clip)
            outcomes.append(outcome)
        if updated:
            self.project.update_clips(updated)
        return ColorResult(result.request_id, tuple(outcomes))
