"""Detached shot classification shared by desktop and headless callers."""

from dataclasses import asdict, dataclass, field, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Literal, TYPE_CHECKING, cast

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import (
    SHOT_CLOUD_PROMPT,
    SHOT_DEFAULT_CLOUD_MODEL,
    SHOT_TYPE_PROMPTS,
    shot_runtime,
)
from models.analysis_record import AnalysisIdentity, AnalysisRecord

from core.jobs.media import media_stamp
from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project

# The local model is a process-wide singleton, including cloud fallback.
_inference_lock = Lock()


@dataclass(frozen=True)
class ShotTypeTask:
    clip_id: str
    thumbnail_path: Path | None
    source_path: Path | None = None
    start_frame: int = 0
    end_frame: int = 0
    fps: float | None = None
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"
    analysis_json: str | None = None
    image_stamp: tuple | None = field(init=False)
    source_stamp: tuple | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_stamp",
            media_stamp(self.thumbnail_path) if self.thumbnail_path else None,
        )
        object.__setattr__(
            self,
            "source_stamp",
            media_stamp(self.source_path) if self.source_path else None,
        )

    def media_current(self) -> bool:
        return (
            media_stamp(self.thumbnail_path) if self.thumbnail_path else None
        ) == self.image_stamp and (
            media_stamp(self.source_path) if self.source_path else None
        ) == self.source_stamp


@dataclass(frozen=True)
class ShotTypeOptions:
    tier: Literal["local", "cloud"] = "local"
    cloud_model: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in ("local", "cloud"):
            raise ValueError("Unsupported shot classification tier")
        object.__setattr__(
            self,
            "cloud_model",
            (self.cloud_model or SHOT_DEFAULT_CLOUD_MODEL)
            if self.tier == "cloud"
            else None,
        )

    @classmethod
    def from_settings(cls) -> "ShotTypeOptions":
        from core.settings import load_settings

        settings = load_settings()
        return cls(
            "cloud" if settings.shot_classifier_tier == "cloud" else "local",
            settings.shot_classifier_cloud_model,
        )


def shot_task(
    target: Any,
    source: Any = None,
    *,
    image_path: Path | None = None,
    skip_existing: bool = True,
) -> ShotTypeTask:
    kind = getattr(
        target, "target_type", "clip" if hasattr(target, "start_frame") else "frame"
    )
    if kind not in ("clip", "frame"):
        raise ValueError("Invalid shot target type")
    image = (
        image_path
        or getattr(target, "image_path", None)
        or (
            getattr(target, "file_path", None)
            if kind == "frame"
            else getattr(target, "thumbnail_path", None)
        )
    )
    video = (
        (source.file_path if source else getattr(target, "video_path", None))
        if kind == "clip"
        else None
    )
    image = Path(image) if image is not None else None
    video = Path(video) if video is not None else None
    files = {"image": image} if image else {}
    if video is not None:
        files["video"] = video
    source_range = (
        {"start_frame": target.start_frame, "end_frame": target.end_frame}
        if kind == "clip"
        else {"frame_number": getattr(target, "frame_number", None)}
    )
    snapshot = AnalysisSnapshot.capture(
        target, "shots", files, source_range, {"shot_type": target.shot_type}
    )
    return ShotTypeTask(
        target.id,
        image,
        video,
        getattr(target, "start_frame", None) or 0,
        getattr(target, "end_frame", None) or 0,
        source.fps if source else getattr(target, "fps", None),
        skip_existing,
        cast(Literal["clip", "frame"], kind),
        snapshot.to_json(),
    )


def shot_identity(
    snapshot: AnalysisSnapshot,
    options: ShotTypeOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
    *,
    backend: str | None = None,
) -> AnalysisIdentity:
    backend = backend or options.tier
    return fingerprints.identity(
        snapshot.inputs,
        operation="shots",
        operation_version=2,
        model={
            "runtime": runtime,
            "backend": backend,
            "cloud_model": options.cloud_model,
        },
        parameters=asdict(options),
        sampling={"policy": "single-image/v1", "local_ensemble": True},
        prompt=SHOT_CLOUD_PROMPT
        if backend == "cloud"
        else json.dumps(SHOT_TYPE_PROMPTS, sort_keys=True),
    )


@dataclass(frozen=True)
class ShotTypeOutcome:
    clip_id: str
    status: OutcomeStatus
    shot_type: str | None = None
    confidence: float = 0.0
    code: str | None = None
    message: str | None = None
    target_type: Literal["clip", "frame"] = "clip"
    record_json: str | None = None

    @property
    def has_result(self) -> bool:
        return self.status == "succeeded" or (
            self.status == "skipped" and self.record_json is not None
        )

    @property
    def can_apply(self) -> bool:
        return self.has_result or (
            self.status == "failed" and self.record_json is not None
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ShotTypeOutcome":
        fields = set(cls.__dataclass_fields__)
        if set(data) not in (fields, fields - {"record_json"}):
            raise ValueError("Invalid recorded shot outcome fields")
        outcome = cls(**data)
        if (
            not isinstance(outcome.clip_id, str)
            or not outcome.clip_id
            or outcome.target_type not in ("clip", "frame")
            or outcome.status not in ("succeeded", "failed", "skipped", "unprocessed")
            or (outcome.status == "succeeded" and not outcome.valid_result())
        ):
            raise ValueError("Invalid recorded shot outcome")
        return outcome

    def valid_result(self) -> bool:
        return (
            isinstance(self.shot_type, str)
            and bool(self.shot_type.strip())
            and self.shot_type != "unknown"
            and isinstance(self.confidence, (int, float))
            and not isinstance(self.confidence, bool)
            and isfinite(self.confidence)
            and 0 <= self.confidence <= 1
        )


def compute_shot_type(
    task: ShotTypeTask,
    options: ShotTypeOptions,
    cancel: Event,
    *,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> ShotTypeOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ShotTypeOutcome:
        return ShotTypeOutcome(
            task.clip_id, status, target_type=task.target_type, **kwargs
        )

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip and task.analysis_json is None:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.is_file():
        return outcome("failed", code="thumbnail_missing")
    snapshot = None
    identity = None
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else shot_runtime()
    backend = options.tier

    def capture_backend(actual: str) -> None:
        nonlocal backend
        backend = cast(Literal["local", "cloud"], actual)

    def failed(code: str, message: str | None = None) -> ShotTypeOutcome:
        record = None
        if (
            identity is not None
            and snapshot is not None
            and snapshot.inputs.unchanged()
            and not cancel.is_set()
        ):
            try:
                attempted = shot_identity(
                    snapshot, options, fingerprints, runtime, backend=backend
                )
            except Exception as exc:
                if cancel.is_set():
                    return outcome("unprocessed", code="cancelled")
                return outcome("failed", code="stale_input", message=str(exc))
            record = replace(
                AnalysisRecord.failure(attempted, message or code),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
        return outcome(
            "failed",
            code=code,
            message=message,
            record_json=json.dumps(record.to_dict(), sort_keys=True)
            if record
            else None,
        )

    try:
        if task.analysis_json is not None:
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            identity = shot_identity(snapshot, options, fingerprints, runtime)
            reused = snapshot.reusable_record(identity) if task.skip else None
            if reused is not None:
                return outcome(
                    "skipped",
                    code="valid_analysis",
                    shot_type=reused.value["shot_type"],
                    record_json=json.dumps(reused.to_dict(), sort_keys=True),
                )
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="stale_input", message=str(exc))
    while not _inference_lock.acquire(timeout=0.05):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
    try:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        if not task.media_current():
            return outcome("failed", code="stale_input")
        from core.analysis.shots import classify_shot_type, classify_shot_type_tiered

        if options.tier == "local":
            label, confidence = classify_shot_type(task.thumbnail_path)
        elif options.tier == "cloud":
            backend_kwargs: dict[str, Any] = (
                {"on_backend": capture_backend} if snapshot is not None else {}
            )
            label, confidence = classify_shot_type_tiered(
                image_path=task.thumbnail_path,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
                tier=options.tier,
                cloud_model=options.cloud_model,
                **backend_kwargs,
            )
        else:
            raise ValueError("Unsupported shot classification tier")
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        if not task.media_current():
            return outcome("failed", code="stale_input")
        result = outcome("succeeded", shot_type=label, confidence=confidence)
        if snapshot is not None:
            identity = shot_identity(
                snapshot, options, fingerprints, runtime, backend=backend
            )
        if not result.valid_result():
            return failed("no_classification")
        if identity is not None and snapshot is not None:
            record = AnalysisRecord.success(
                identity, {"shot_type": label}, input_snapshot=snapshot.inputs.to_dict()
            )
            result = replace(
                result, record_json=json.dumps(record.to_dict(), sort_keys=True)
            )
        return result
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return failed("classification_failed", str(exc))
    finally:
        _inference_lock.release()


def run_shot_types(
    tasks: tuple[ShotTypeTask, ...],
    options: ShotTypeOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ShotTypeOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> tuple[ShotTypeOutcome, ...]:
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else shot_runtime()
    outcomes = []
    for task in tasks:
        result = compute_shot_type(
            task, options, cancel, fingerprints=fingerprints, runtime=runtime
        )
        outcomes.append(result)
        if not cancel.is_set():
            if on_outcome:
                on_outcome(result)
            if progress:
                progress(len(outcomes), len(tasks))
    return tuple(outcomes)


class ShotTypeApplication:
    """Publish once to the same project, session, media and editorial target."""

    def __init__(
        self,
        project: "Project",
        tasks: tuple[ShotTypeTask, ...],
        options: ShotTypeOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.options = options
        self.session_id = project.session.session_id
        self.path = project.path
        self.tasks = {(task.target_type, task.clip_id): task for task in tasks}
        self.bindings = {
            key: self._binding(project, task) for key, task in self.tasks.items()
        }
        self.consumed: set[tuple[str, str]] = set()

    @staticmethod
    def _binding(project: "Project", task: ShotTypeTask) -> tuple | None:
        if task.target_type == "frame":
            target = project.frames_by_id.get(task.clip_id)
            if target is None:
                return None
            return (
                target,
                None,
                (
                    target.file_path,
                    target.source_id,
                    target.clip_id,
                    target.frame_number,
                    target.shot_type,
                    target.analysis_records.get("shots"),
                ),
            )
        clip = project.clips_by_id.get(task.clip_id)
        if clip is None:
            return None
        source = project.sources_by_id.get(clip.source_id)
        return (
            clip,
            source,
            (
                clip.thumbnail_path,
                clip.source_id,
                clip.start_frame,
                clip.end_frame,
                clip.shot_type,
                clip.analysis_records.get("shots"),
                source.file_path if source else None,
                source.fps if source else None,
            ),
        )

    def apply(self, project: "Project", outcome: ShotTypeOutcome) -> bool:
        key = outcome.target_type, outcome.clip_id
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or project.path != self.path
            or not outcome.can_apply
            or (outcome.has_result and not outcome.valid_result())
            or key in self.consumed
        ):
            return False

        def publish() -> bool:
            self.consumed.add(key)
            task = self.tasks.get(key)
            expected = self.bindings.get(key)
            current = self._binding(project, task) if task else None
            if (
                task is None
                or expected is None
                or current is None
                or current[0] is not expected[0]
                or current[1] is not expected[1]
                or current[2] != expected[2]
                or not task.media_current()
            ):
                return False
            value = {"shot_type": outcome.shot_type}
            record = (
                AnalysisRecord.from_dict(json.loads(outcome.record_json))
                if outcome.record_json is not None
                else AnalysisRecord.legacy(value)
            )
            if outcome.record_json is not None:
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task.analysis_json
                    else None
                )
                if (
                    snapshot is None
                    or record.identity is None
                    or record.identity.operation != "shots"
                    or (
                        self.options is not None
                        and record.identity.to_dict()["parameters"]
                        != asdict(self.options)
                    )
                    or json.loads(record.input_json or "null")
                    != snapshot.inputs.to_dict()
                    or (
                        outcome.has_result
                        and (record.state != "succeeded" or record.value != value)
                    )
                    or (outcome.status == "failed" and record.state != "failed")
                ):
                    return False
            project.record_analysis(task.target_type, task.clip_id, "shots", record)
            if outcome.status == "failed":
                return True
            if task.target_type == "frame":
                project.update_frame(task.clip_id, shot_type=outcome.shot_type)
            else:
                current[0].shot_type = outcome.shot_type
                project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
