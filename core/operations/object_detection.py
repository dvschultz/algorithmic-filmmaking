"""Detached object detection and people counting shared by all adapters."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Literal

from core.operations.contracts import OutcomeStatus

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
