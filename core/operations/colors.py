"""Resolve, compute and apply color analysis without GUI dependencies.

Workers compute immutable snapshots. The owner of the project applies the
result through a one-use application object guarded by the project session.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable, Iterable
from uuid import uuid4

from core.operations.contracts import ColorOutcome, ColorResult, Palette
from core.analysis_records import AnalysisFingerprints, AnalysisInput, model_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

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
    inputs: AnalysisInput | None = None
    record_json: str | None = None


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
    record = target.analysis_records.get("colors")
    path = video or image
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
        inputs=AnalysisInput.capture(
            {"video" if video is not None else "image": path} if path is not None else {},
            {"start_frame": target.start_frame, "end_frame": target.end_frame} if video is not None else {"frame_number": target.frame_number},
            binding={"target_id": target.id, "target_type": target.target_type, "source_id": target.source_id},
        ),
        record_json=json.dumps(record.to_dict(), sort_keys=True) if isinstance(record, AnalysisRecord) else None,
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


def color_identity(
    target: ColorTarget, num_colors: int, fingerprints: AnalysisFingerprints,
    runtime: dict | None = None,
) -> AnalysisIdentity:
    """Identify the color algorithm, its exact input range, and sampling policy."""
    if target.inputs is None:
        raise ValueError("Color analysis input snapshot is missing")
    return fingerprints.identity(
        target.inputs, operation="colors", operation_version=2,
        model=runtime if runtime is not None else model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python")),
        parameters={"num_colors": num_colors, "random_state": 42, "n_init": 1, "max_iter": 100},
        sampling={"policy": "inner-15-50-85/v1" if target.video_path else "single-image/v1", "resize": [50, 50]},
    )


def reusable_colors(target: ColorTarget, identity: AnalysisIdentity, *, skip_empty: bool = False) -> bool:
    if target.record_json is None or not (target.existing_colors or (skip_empty and target.existing_colors is not None)):
        return False
    try:
        record = AnalysisRecord.from_dict(json.loads(target.record_json))
        raw = record.value["dominant_colors"]
        palette = tuple((color["r"], color["g"], color["b"]) if isinstance(color, dict) else tuple(color) for color in raw)
        return record.reusable(identity) and palette == target.existing_colors
    except (ValueError, TypeError, KeyError):
        return False


def _compute_target(
    target: ColorTarget, request: ColorRequest, cancel_event: Event | None,
    fingerprints: AnalysisFingerprints, runtime: dict,
) -> ColorOutcome:
    if cancel_event is not None and cancel_event.is_set():
        return ColorOutcome(target.target_id, "unprocessed", code="cancelled")
    if target.missing:
        return ColorOutcome(target.target_id, "failed", code="target_not_found")
    path = target.video_path or target.image_path
    identity = None

    def failure(code: str, message: str | None = None) -> ColorOutcome:
        record_json = None
        if identity is not None and target.inputs is not None and target.inputs.unchanged():
            record = replace(
                AnalysisRecord.failure(identity, message or code),
                input_json=json.dumps(target.inputs.to_dict(), sort_keys=True),
            )
            record_json = json.dumps(record.to_dict(), sort_keys=True)
        return ColorOutcome(target.target_id, "failed", code=code, message=message, record_json=record_json)

    try:
        if path is None or not path.is_file():
            return ColorOutcome(target.target_id, "failed", code="source_file_missing")
        identity = color_identity(target, request.num_colors, fingerprints, runtime)
        if request.skip_existing and reusable_colors(target, identity, skip_empty=request.skip_empty):
            record = AnalysisRecord.from_dict(json.loads(target.record_json or "null"))
            assert target.inputs is not None
            record = replace(record, input_json=json.dumps(target.inputs.to_dict(), sort_keys=True))
            return ColorOutcome(
                target.target_id, "skipped", target.existing_colors or (), code="valid_analysis",
                record_json=json.dumps(record.to_dict(), sort_keys=True),
            )
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
            return failure("no_colors_extracted")
        if target.inputs is None or not target.inputs.unchanged():
            return ColorOutcome(target.target_id, "failed", code="stale_input")
        record = AnalysisRecord.success(
            identity, {"dominant_colors": colors}, input_snapshot=target.inputs.to_dict(),
        )
        return ColorOutcome(
            target.target_id,
            "succeeded",
            tuple((int(r), int(g), int(b)) for r, g, b in colors),
            record_json=json.dumps(record.to_dict(), sort_keys=True),
        )
    except ColorDependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 — one failed clip must not discard the batch
        from core.jobs.media import FingerprintCancelled

        if isinstance(exc, FingerprintCancelled):
            return ColorOutcome(target.target_id, "unprocessed", code="cancelled")
        return failure("extraction_failed", str(exc))


def compute_colors(
    request: ColorRequest,
    *,
    parallelism: int = 1,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, ColorOutcome], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> ColorResult:
    """Compute with bounded dispatch; callbacks run on the calling thread.

    Cancellation stops dispatch. Already running extractions finish and their
    successful results remain usable. Output order always matches input order.
    """
    workers = min(max(1, parallelism), 8)
    outcomes: dict[str, ColorOutcome] = {}
    pending_targets = iter(request.targets)
    fingerprints = fingerprints if fingerprints is not None else AnalysisFingerprints(cancel_event)
    runtime = runtime if runtime is not None else model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python"))

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[ColorOutcome]] = set()

        def dispatch() -> None:
            while len(pending) < workers and not cancelled():
                target = next(pending_targets, None)
                if target is None:
                    break
                pending.add(pool.submit(_compute_target, target, request, cancel_event, fingerprints, runtime))

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
            if outcome.status == "succeeded" or (outcome.status in ("skipped", "failed") and outcome.record_json is not None):
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
                else:
                    if outcome.status == "skipped" and outcome.record_json == target.record_json:
                        outcomes.append(outcome)
                        continue
                    record = (
                        AnalysisRecord.from_dict(json.loads(outcome.record_json))
                        if outcome.record_json else AnalysisRecord.legacy({"dominant_colors": outcome.colors})
                    )
                    if outcome.record_json and (
                        record.identity is None or record.identity.operation != "colors"
                        or record.state != ("failed" if outcome.status == "failed" else "succeeded")
                        or (outcome.status != "failed" and record.value != {"dominant_colors": [list(c) for c in outcome.colors]})
                        or target.inputs is None
                        or json.loads(record.input_json or "null") != target.inputs.to_dict()
                    ):
                        outcome = replace(outcome, status="failed", code="invalid_analysis_record")
                    else:
                        self.project.record_analysis(target.target_type, target.target_id, "colors", record)
                        if outcome.status == "failed":
                            outcomes.append(outcome)
                            continue
                        if target.target_type == "frame":
                            self.project.update_frame(target.target_id, dominant_colors=list(outcome.colors))
                        else:
                            clip = self.project.clips_by_id[target.target_id]
                            clip.dominant_colors = list(outcome.colors)
                            updated.append(clip)
            outcomes.append(outcome)
        if updated:
            self.project.update_clips(updated)
        return ColorResult(result.request_id, tuple(outcomes))
