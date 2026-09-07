"""Detached object detection and people counting shared by all adapters."""

from copy import deepcopy
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

_inference_lock = Lock()


@dataclass(frozen=True)
class ObjectDetectionTask:
    clip_id: str
    thumbnail_path: Path | None
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class ObjectDetectionOptions:
    confidence: float = 0.5
    detect_all: bool = True


@dataclass(frozen=True)
class DetectedObject:
    label: str
    confidence: float
    bbox: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "DetectedObject":
        label = value["label"]
        confidence = float(value["confidence"])
        bbox = tuple(value["bbox"])
        if (
            not isinstance(label, str)
            or not label
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            or len(bbox) != 4
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not isfinite(x)
                for x in bbox
            )
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError("Invalid object detection result")
        return cls(label, confidence, bbox)


@dataclass(frozen=True)
class ObjectDetectionOutcome:
    clip_id: str
    status: OutcomeStatus
    detections: tuple[DetectedObject, ...] = ()
    person_count: int | None = None
    code: str | None = None
    message: str | None = None

    def detection_dicts(self) -> list[dict]:
        """Return fresh containers for legacy model and signal consumers."""
        return [detection.to_dict() for detection in self.detections]

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectDetectionOutcome":
        """Detach recorded JSON detections, including nested bounding boxes."""
        return cls(
            **{
                **data,
                "detections": tuple(
                    DetectedObject.from_dict(value)
                    for value in data.get("detections", ())
                ),
            }
        )


def compute_object_detection(
    task: ObjectDetectionTask,
    options: ObjectDetectionOptions,
    cancel: Event,
) -> ObjectDetectionOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ObjectDetectionOutcome:
        return ObjectDetectionOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    # GUI jobs, CLI, and headless callers share the same mutable YOLO singleton.
    while not _inference_lock.acquire(timeout=0.05):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
    try:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        from core.analysis.detection import detect_objects, count_people

        if options.detect_all:
            raw = detect_objects(
                task.thumbnail_path, confidence_threshold=options.confidence
            )
            detections = tuple(DetectedObject.from_dict(value) for value in raw)
            person_count = sum(value.label == "person" for value in detections)
        else:
            detections = ()
            person_count = count_people(
                task.thumbnail_path, confidence_threshold=options.confidence
            )
            if type(person_count) is not int or person_count < 0:
                raise ValueError("Invalid person count")
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("succeeded", detections=detections, person_count=person_count)
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        from core.errors import ModelDownloadError

        return outcome(
            "failed",
            code="model_load_failed"
            if isinstance(exc, ModelDownloadError)
            else "detection_failed",
            message=str(exc),
        )
    finally:
        _inference_lock.release()


def run_object_detection(
    tasks: tuple[ObjectDetectionTask, ...],
    options: ObjectDetectionOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ObjectDetectionOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ObjectDetectionOutcome, ...]:
    """Run serial inference; cancellation suppresses late provider replies."""
    cancel = cancel_event or Event()
    outcomes = []
    for index, task in enumerate(tasks):
        result = compute_object_detection(task, options, cancel)
        outcomes.append(result)
        if not cancel.is_set():
            if on_outcome:
                on_outcome(result)
            if progress:
                progress(len(outcomes), len(tasks))
        if result.code == "model_load_failed":
            outcomes.extend(
                ObjectDetectionOutcome(
                    pending.clip_id, "unprocessed", code="model_unavailable"
                )
                for pending in tasks[index + 1 :]
            )
            break
    return tuple(outcomes)


class ObjectDetectionApplication:
    """Publish once to unchanged model targets, including external CLI thumbnails."""

    def __init__(
        self,
        project: "Project",
        tasks: tuple[ObjectDetectionTask, ...],
        options: ObjectDetectionOptions = ObjectDetectionOptions(),
    ) -> None:
        project.session.assert_owner()
        self.options = options
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    def _binding(self, project: "Project", task: ObjectDetectionTask) -> tuple | None:
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
                deepcopy(target.detected_objects) if self.options.detect_all else None,
                target.person_count,
            ),
        )

    def apply(self, project: "Project", outcome: ObjectDetectionOutcome) -> bool:
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self, project: "Project", outcomes: tuple[ObjectDetectionOutcome, ...]
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
                        updates: dict = {"person_count": outcome.person_count}
                        if self.options.detect_all:
                            updates["detected_objects"] = outcome.detection_dicts()
                        if task.target_type == "frame":
                            project.update_frame(outcome.clip_id, **updates)
                        else:
                            current[0].person_count = outcome.person_count
                            if self.options.detect_all:
                                current[0].detected_objects = outcome.detection_dicts()
                            clips.append(current[0])
                        valid = True
                accepted.append(valid)
            if clips:
                project.update_clips(clips)
            return tuple(accepted)

        return project.session.apply_external(publish)
