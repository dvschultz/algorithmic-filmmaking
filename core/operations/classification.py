"""Detached ImageNet classification shared by GUI and headless workflows."""

from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Literal, TYPE_CHECKING, cast

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import classification_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

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
    analysis_json: str | None = None


@dataclass(frozen=True)
class ClassificationOptions:
    top_k: int = 5
    threshold: float = 0.1

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("Classification top_k must be a positive integer")
        object.__setattr__(self, "top_k", min(self.top_k, 1000))
        if (
            isinstance(self.threshold, bool)
            or not isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("Classification threshold must be between 0 and 1")
        object.__setattr__(self, "threshold", float(self.threshold))


def classification_task(
    target: Any,
    source: Any = None,
    *,
    image_path: Path | None = None,
    skip_existing: bool = True,
) -> ClassificationTask:
    kind = getattr(
        target, "target_type", "clip" if hasattr(target, "start_frame") else "frame"
    )
    if kind not in ("clip", "frame"):
        raise ValueError("Invalid classification target type")
    image = (
        image_path
        or getattr(target, "image_path", None)
        or (
            getattr(target, "file_path", None)
            if kind == "frame"
            else getattr(target, "thumbnail_path", None)
        )
    )
    files = {"image": image} if image else {}
    video = source.file_path if source else getattr(target, "video_path", None)
    if kind == "clip" and video is not None:
        files["video"] = video
    source_range = (
        {"start_frame": target.start_frame, "end_frame": target.end_frame}
        if kind == "clip"
        else {"frame_number": getattr(target, "frame_number", None)}
    )
    snapshot = AnalysisSnapshot.capture(
        target,
        "classify",
        files,
        source_range,
        {"object_labels": getattr(target, "object_labels", None)},
    )
    return ClassificationTask(
        target.id,
        image,
        skip_existing,
        cast(Literal["clip", "frame"], kind),
        snapshot.to_json(),
    )


def classification_identity(
    snapshot: AnalysisSnapshot,
    options: ClassificationOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="classify",
        operation_version=2,
        model=runtime,
        parameters=asdict(options),
        sampling={"policy": "single-image/v1"},
    )


@dataclass(frozen=True)
class ClassificationOutcome:
    clip_id: str
    status: OutcomeStatus
    labels: tuple[tuple[str, float], ...] = ()
    code: str | None = None
    message: str | None = None
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

    @property
    def label_names(self) -> list[str]:
        if self.status == "skipped" and self.record_json is not None:
            return list(
                AnalysisRecord.from_dict(json.loads(self.record_json)).value[
                    "object_labels"
                ]
            )
        return [label for label, _ in self.labels]

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationOutcome":
        """Detach JSON label arrays when loading a recorded outcome."""
        return cls(
            **{
                **data,
                "labels": tuple(
                    (label, confidence) for label, confidence in data.get("labels", ())
                ),
            }
        )


def compute_classification(
    task: ClassificationTask,
    options: ClassificationOptions,
    cancel: Event,
    *,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> ClassificationOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ClassificationOutcome:
        return ClassificationOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip and task.analysis_json is None:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.is_file():
        return outcome("failed", code="thumbnail_missing")
    snapshot = None
    identity = None
    try:
        if task.analysis_json is not None:
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            identity = classification_identity(
                snapshot,
                options,
                fingerprints or AnalysisFingerprints(cancel),
                runtime if runtime is not None else classification_runtime(),
            )
            reused = snapshot.reusable_record(identity) if task.skip else None
            if reused is not None:
                return outcome(
                    "skipped",
                    code="valid_analysis",
                    record_json=json.dumps(reused.to_dict(), sort_keys=True),
                )
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="stale_input", message=str(exc))
    while not _inference_lock.acquire(timeout=0.05):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        if snapshot is not None and not snapshot.inputs.unchanged():
            raise ValueError("Classification input media changed")
    try:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        from core.analysis.classification import classify_frame

        raw = tuple(
            classify_frame(
                task.thumbnail_path, top_k=options.top_k, threshold=options.threshold
            )
        )
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        if any(isinstance(confidence, bool) for _, confidence in raw):
            raise ValueError("Invalid classification confidence")
        labels = tuple((label, float(confidence)) for label, confidence in raw)
        if any(
            not isinstance(label, str)
            or not label
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            for label, confidence in labels
        ):
            raise ValueError("Invalid classification result")
        if snapshot is not None and not snapshot.inputs.unchanged():
            raise ValueError("Classification input media changed")
        record = (
            AnalysisRecord.success(
                identity,
                {"object_labels": [label for label, _ in labels]},
                input_snapshot=snapshot.inputs.to_dict(),
            )
            if identity is not None and snapshot is not None
            else None
        )
        return outcome(
            "succeeded",
            labels=labels,
            record_json=json.dumps(record.to_dict(), sort_keys=True)
            if record
            else None,
        )
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        record = (
            replace(
                AnalysisRecord.failure(identity, str(exc)),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
            if identity is not None
            and snapshot is not None
            and snapshot.inputs.unchanged()
            else None
        )
        return outcome(
            "failed",
            code="classification_failed",
            message=str(exc),
            record_json=json.dumps(record.to_dict(), sort_keys=True)
            if record
            else None,
        )
    finally:
        _inference_lock.release()


def run_classification(
    tasks: tuple[ClassificationTask, ...],
    options: ClassificationOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ClassificationOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> tuple[ClassificationOutcome, ...]:
    """Run serially on the caller thread with cancellable model admission."""
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else classification_runtime()
    outcomes = []
    for task in tasks:
        result = compute_classification(
            task, options, cancel, fingerprints=fingerprints, runtime=runtime
        )
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
        self,
        project: "Project",
        tasks: tuple[ClassificationTask, ...],
        options: ClassificationOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.options = options
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
                target.analysis_records.get("classify"),
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
            or not any(outcome.can_apply for outcome in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self.tasks.get(outcome.clip_id)
                if outcome.can_apply and outcome.clip_id not in self.consumed:
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
                        labels = outcome.label_names
                        value = {"object_labels": labels}
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
                                or record.identity.operation != "classify"
                                or (
                                    self.options is not None
                                    and record.identity.to_dict()["parameters"]
                                    != asdict(self.options)
                                )
                                or json.loads(record.input_json or "null")
                                != snapshot.inputs.to_dict()
                                or (
                                    outcome.has_result
                                    and (
                                        record.state != "succeeded"
                                        or record.value != value
                                    )
                                )
                                or (
                                    outcome.status == "failed"
                                    and record.state != "failed"
                                )
                            ):
                                accepted.append(False)
                                continue
                        project.record_analysis(
                            task.target_type, outcome.clip_id, "classify", record
                        )
                        if outcome.status == "failed":
                            accepted.append(True)
                            continue
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
