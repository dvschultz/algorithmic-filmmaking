"""Detached shot classification shared by desktop and headless callers."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Literal, TYPE_CHECKING

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

    @classmethod
    def from_settings(cls) -> "ShotTypeOptions":
        from core.settings import load_settings

        settings = load_settings()
        return cls(
            "cloud" if settings.shot_classifier_tier == "cloud" else "local",
            settings.shot_classifier_cloud_model,
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

    @classmethod
    def from_dict(cls, data: dict) -> "ShotTypeOutcome":
        if set(data) != set(cls.__dataclass_fields__):
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
) -> ShotTypeOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ShotTypeOutcome:
        return ShotTypeOutcome(
            task.clip_id, status, target_type=task.target_type, **kwargs
        )

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.is_file():
        return outcome("failed", code="thumbnail_missing")
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
            label, confidence = classify_shot_type_tiered(
                image_path=task.thumbnail_path,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
                tier=options.tier,
                cloud_model=options.cloud_model,
            )
        else:
            raise ValueError("Unsupported shot classification tier")
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        if not task.media_current():
            return outcome("failed", code="stale_input")
        result = outcome("succeeded", shot_type=label, confidence=confidence)
        if not result.valid_result():
            return outcome("failed", code="no_classification")
        return result
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="classification_failed", message=str(exc))
    finally:
        _inference_lock.release()


def run_shot_types(
    tasks: tuple[ShotTypeTask, ...],
    options: ShotTypeOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ShotTypeOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ShotTypeOutcome, ...]:
    cancel = cancel_event or Event()
    outcomes = []
    for task in tasks:
        result = compute_shot_type(task, options, cancel)
        outcomes.append(result)
        if not cancel.is_set():
            if on_outcome:
                on_outcome(result)
            if progress:
                progress(len(outcomes), len(tasks))
    return tuple(outcomes)


class ShotTypeApplication:
    """Publish once to the same project, session, media and editorial target."""

    def __init__(self, project: "Project", tasks: tuple[ShotTypeTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
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
            or outcome.status != "succeeded"
            or not outcome.valid_result()
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
            if task.target_type == "frame":
                project.update_frame(task.clip_id, shot_type=outcome.shot_type)
            else:
                current[0].shot_type = outcome.shot_type
                project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
