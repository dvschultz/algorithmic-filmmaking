"""Detached object detection and people counting shared by all adapters."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Literal, TYPE_CHECKING, cast

from core.operations.contracts import OutcomeStatus
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import object_detection_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

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
    analysis_json: str | None = None


def object_detection_task(target: Any, source: Any = None, *, image_path: Path | None = None, skip_existing: bool = True, detect_all: bool = True) -> ObjectDetectionTask:
    target_type = cast(Literal["clip", "frame"], getattr(target, "target_type", None) or ("frame" if hasattr(target, "frame_number") else "clip"))
    image = image_path or getattr(target, "image_path", None) or (target.file_path if target_type == "frame" else target.thumbnail_path)
    files = {"image": image} if image is not None else {}
    source_path = source.file_path if source is not None else getattr(target, "video_path", None)
    if source_path is not None:
        files["video"] = source_path
    source_range = {"frame_number": target.frame_number} if target_type == "frame" else {"start_frame": target.start_frame, "end_frame": target.end_frame}
    value = {"person_count": target.person_count}
    if detect_all:
        value["detected_objects"] = target.detected_objects
    snapshot = AnalysisSnapshot.capture(target, "detect_objects", files, source_range, value)
    return ObjectDetectionTask(target.id, image, skip_existing, target_type, snapshot.to_json())


@dataclass(frozen=True)
class ObjectDetectionOptions:
    confidence: float = 0.5
    detect_all: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be between zero and one")
        if type(self.detect_all) is not bool:
            raise ValueError("detect_all must be a boolean")
        object.__setattr__(self, "confidence", float(self.confidence))


def object_detection_identity(snapshot: AnalysisSnapshot, options: ObjectDetectionOptions, fingerprints: AnalysisFingerprints, runtime: dict) -> AnalysisIdentity:
    return fingerprints.identity(snapshot.inputs, operation="detect_objects", operation_version=2, model=runtime, parameters=asdict(options), sampling={"policy": "single-image/v1"})


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
    record_json: str | None = None

    @property
    def has_result(self) -> bool:
        return self.status == "succeeded" or (self.status == "skipped" and self.record_json is not None)

    @property
    def can_apply(self) -> bool:
        return self.has_result or (self.status == "failed" and self.record_json is not None)

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
    *, fingerprints: AnalysisFingerprints | None = None, runtime: dict | None = None,
) -> ObjectDetectionOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ObjectDetectionOutcome:
        return ObjectDetectionOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip and task.analysis_json is None:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.is_file():
        return outcome("failed", code="thumbnail_missing")
    snapshot = AnalysisSnapshot.from_json(task.analysis_json) if task.analysis_json else None
    identity = None
    if snapshot is not None:
        from core.jobs.media import FingerprintCancelled
        from core.jobs.commits import StaleJobResult

        try:
            identity = object_detection_identity(snapshot, options, fingerprints or AnalysisFingerprints(cancel), runtime if runtime is not None else object_detection_runtime())
        except FingerprintCancelled:
            return outcome("unprocessed", code="cancelled")
        except (ValueError, StaleJobResult) as exc:
            return outcome("failed", code="stale_input", message=str(exc))
        except OSError as exc:
            return outcome("failed", code="input_unavailable", message=str(exc))
        reused = snapshot.reusable_record(identity) if task.skip else None
        if reused is not None:
            value = reused.value
            return outcome("skipped", detections=tuple(DetectedObject.from_dict(v) for v in value.get("detected_objects", [])), person_count=value["person_count"], code="valid_analysis", record_json=json.dumps(reused.to_dict(), sort_keys=True))
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
        record = None
        if snapshot is not None and identity is not None:
            if not snapshot.inputs.unchanged():
                return outcome("failed", code="stale_input")
            value = {"person_count": person_count}
            if options.detect_all:
                value["detected_objects"] = [d.to_dict() for d in detections]
            record = AnalysisRecord.success(identity, value, input_snapshot=snapshot.inputs.to_dict())
        return outcome("succeeded", detections=detections, person_count=person_count, record_json=json.dumps(record.to_dict(), sort_keys=True) if record else None)
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
            record_json=json.dumps(replace(AnalysisRecord.failure(identity, str(exc)), input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True)).to_dict(), sort_keys=True) if identity is not None and snapshot is not None and snapshot.inputs.unchanged() else None,
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
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> tuple[ObjectDetectionOutcome, ...]:
    """Run serial inference; cancellation suppresses late provider replies."""
    cancel = cancel_event or Event()
    outcomes = []
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else object_detection_runtime()
    for index, task in enumerate(tasks):
        result = compute_object_detection(task, options, cancel, fingerprints=fingerprints, runtime=runtime)
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
                target.analysis_records.get("detect_objects"),
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
            or not any(outcome.can_apply for outcome in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self.tasks.get(outcome.clip_id)
                if (
                    outcome.can_apply
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
                        record = AnalysisRecord.from_dict(json.loads(outcome.record_json)) if outcome.record_json else AnalysisRecord.legacy(updates)
                        if outcome.record_json is not None:
                            snapshot = AnalysisSnapshot.from_json(task.analysis_json) if task.analysis_json else None
                            if (
                                snapshot is None or record.identity is None
                                or record.identity.operation != "detect_objects"
                                or record.identity.to_dict()["parameters"] != asdict(self.options)
                                or json.loads(record.input_json or "null") != snapshot.inputs.to_dict()
                                or (outcome.has_result and (record.state != "succeeded" or record.value != updates))
                                or (outcome.status == "failed" and record.state != "failed")
                            ):
                                accepted.append(False)
                                continue
                        if outcome.status == "failed":
                            project.record_analysis(task.target_type, outcome.clip_id, "detect_objects", record)
                            accepted.append(True)
                            continue
                        project.record_analysis(task.target_type, outcome.clip_id, "detect_objects", record)
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
