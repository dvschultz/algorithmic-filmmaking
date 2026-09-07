"""Detached ImageNet classification shared by GUI and headless workflows."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Literal, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip
    from models.frame import Frame

# MobileNet inference shares a process-wide model. Serialize across jobs, too.
_inference_lock = Lock()


@dataclass(frozen=True)
class ClassificationTask:
    clip_id: str
    thumbnail_path: Path | None
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class ClassificationOptions:
    top_k: int = 5
    threshold: float = 0.1


@dataclass(frozen=True)
class ClassificationOutcome:
    clip_id: str
    status: OutcomeStatus
    labels: tuple[tuple[str, float], ...] = ()
    code: str | None = None
    message: str | None = None


def compute_classification(
    task: ClassificationTask, options: ClassificationOptions, cancel: Event
) -> ClassificationOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ClassificationOutcome:
        return ClassificationOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    while not _inference_lock.acquire(timeout=0.05):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
    try:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        from core.analysis.classification import classify_frame

        raw = classify_frame(
            task.thumbnail_path, top_k=options.top_k, threshold=options.threshold
        )
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        labels = tuple((label, float(confidence)) for label, confidence in raw)
        if any(
            not isinstance(label, str)
            or not label
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            for label, confidence in labels
        ):
            raise ValueError("Invalid classification result")
        return outcome("succeeded", labels=labels)
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="classification_failed", message=str(exc))
    finally:
        _inference_lock.release()


def run_classification(
    tasks: tuple[ClassificationTask, ...],
    options: ClassificationOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ClassificationOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ClassificationOutcome, ...]:
    """Run serially on the caller thread with cancellable model admission."""
    cancel = cancel_event or Event()
    outcomes = []
    for task in tasks:
        result = compute_classification(task, options, cancel)
        outcomes.append(result)
        if not cancel.is_set():
            if on_outcome:
                on_outcome(result)
            if progress:
                progress(len(outcomes), len(tasks))
    return tuple(outcomes)


class ClassificationApplication:
    """Publish once to unchanged model targets, including external CLI thumbnails."""

    def __init__(
        self, project: "Project", tasks: tuple[ClassificationTask, ...]
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: ClassificationTask) -> tuple | None:
        from core.jobs.media import media_stamp

        target: Clip | Frame | None
        source = None
        identity: tuple
        if task.target_type == "frame":
            target = project.frames_by_id.get(task.clip_id)
            if target is None:
                return None
            identity = (
                target.file_path,
                target.source_id,
                target.clip_id,
                target.frame_number,
            )
        else:
            target = project.clips_by_id.get(task.clip_id)
            if target is None:
                return None
            source = project.sources_by_id.get(target.source_id)
            identity = (
                target.thumbnail_path,
                target.source_id,
                target.start_frame,
                target.end_frame,
                source.file_path if source else None,
                source.fps if source else None,
                media_stamp(source.file_path) if source else None,
            )
        image = media_stamp(task.thumbnail_path) if task.thumbnail_path else None
        if image is None:
            return None
        return (
            target,
            source,
            (
                identity,
                image,
                tuple(target.object_labels)
                if target.object_labels is not None
                else None,
            ),
        )

    def apply(self, project: "Project", outcome: ClassificationOutcome) -> bool:
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self, project: "Project", outcomes: tuple[ClassificationOutcome, ...]
    ) -> tuple[bool, ...]:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or not any(outcome.status == "succeeded" for outcome in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self.tasks.get(outcome.clip_id)
                if (
                    outcome.status == "succeeded"
                    and outcome.clip_id not in self.consumed
                ):
                    self.consumed.add(outcome.clip_id)
                    expected = self.bindings.get(outcome.clip_id)
                    current = self._binding(project, task) if task else None
                    if (
                        task is not None
                        and expected is not None
                        and current is not None
                        and current[0] is expected[0]
                        and current[1] is expected[1]
                        and current[2] == expected[2]
                    ):
                        labels = [label for label, _ in outcome.labels]
                        if task.target_type == "frame":
                            project.update_frame(outcome.clip_id, object_labels=labels)
                        else:
                            current[0].object_labels = labels
                            clips.append(current[0])
                        valid = True
                accepted.append(valid)
            if clips:
                project.update_clips(clips)
            return tuple(accepted)

        return project.session.apply_external(publish)
