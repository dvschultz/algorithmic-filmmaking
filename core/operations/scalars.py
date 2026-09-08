"""Detached, verified brightness and volume analysis with owner publication."""

from dataclasses import dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event
from typing import Callable, Literal, TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot, model_runtime
from core.jobs.media import FingerprintCancelled
from models.analysis_record import AnalysisRecord

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

ScalarOperation = Literal["brightness", "volume"]
FIELDS = {"brightness": "average_brightness", "volume": "rms_volume"}


def scalar_runtime(operation: ScalarOperation) -> dict:
    if operation == "brightness":
        return model_runtime("opencv-gray-mean/v1", ("opencv-python", "numpy"))
    if operation == "volume":
        return {
            "name": "ffmpeg-volumedetect/v1",
            "audio_probe": "ffprobe-audio-streams/v1",
        }
    raise ValueError("Unknown scalar operation")


def scalar_snapshot(
    clip: "Clip", source: "Source | None", operation: ScalarOperation
) -> AnalysisSnapshot:
    files = {"video": Path(source.file_path)} if source else {}
    if operation == "volume":
        from core.binary_resolver import find_binary

        for name in ("ffmpeg", "ffprobe"):
            path = find_binary(name)
            if path is not None:
                files[name] = Path(path)
    return AnalysisSnapshot.capture(
        clip,
        operation,
        files,
        {
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "fps": source.fps if source else 0.0,
        },
        {FIELDS[operation]: getattr(clip, FIELDS[operation])},
    )


@dataclass(frozen=True)
class ScalarTask:
    clip_id: str
    operation: ScalarOperation
    snapshot_json: str
    num_samples: int = 5
    skip_existing: bool = True


def scalar_task(
    clip: "Clip",
    source: "Source | None",
    operation: ScalarOperation,
    *,
    num_samples: int = 5,
    skip_existing: bool = True,
) -> ScalarTask:
    if operation not in FIELDS:
        raise ValueError("Unknown scalar operation")
    if source is not None and source.id != clip.source_id:
        raise ValueError("Scalar source does not belong to the clip")
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("Sample count must be a positive integer")
    return ScalarTask(
        clip.id,
        operation,
        scalar_snapshot(clip, source, operation).to_json(),
        num_samples,
        skip_existing,
    )


def scalar_parameters(task: ScalarTask) -> dict:
    return {"num_samples": task.num_samples} if task.operation == "brightness" else {}


def scalar_sampling(operation: ScalarOperation) -> dict:
    return (
        {"policy": "interior-uniform-integer-frames/v1", "short_clip": "all-frames"}
        if operation == "brightness"
        else {"policy": "half-open-clip-audio/v1"}
    )


def scalar_value(operation: ScalarOperation, value: float | None) -> dict:
    if value is None and operation == "volume":
        return {FIELDS[operation]: None}
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError("Scalar analysis must produce a finite number")
    if operation == "brightness" and not 0 <= value <= 1:
        raise ValueError("Brightness must be between zero and one")
    return {FIELDS[operation]: float(value)}


@dataclass(frozen=True)
class ScalarOutcome:
    clip_id: str
    operation: ScalarOperation
    status: Literal["succeeded", "skipped", "failed", "unprocessed"]
    record_json: str | None = None
    message: str | None = None

    @property
    def has_result(self) -> bool:
        return self.record_json is not None and self.status in ("succeeded", "skipped")

    @property
    def can_apply(self) -> bool:
        return self.record_json is not None and self.status != "unprocessed"


def _compute(
    task: ScalarTask, fingerprints: AnalysisFingerprints, cancel: Event
) -> ScalarOutcome:
    snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
    runtime = scalar_runtime(task.operation)
    identity = None
    try:
        if cancel.is_set():
            raise FingerprintCancelled()
        identity = fingerprints.identity(
            snapshot.inputs,
            operation=task.operation,
            model=runtime,
            parameters=scalar_parameters(task),
            sampling=scalar_sampling(task.operation),
        )
        region = json.loads(snapshot.inputs.range_json)
        start, end, fps = region["start_frame"], region["end_frame"], region["fps"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("Invalid scalar analysis media range")
        files = {role: path for role, path, _ in snapshot.inputs.files}
        if task.skip_existing:
            reused = snapshot.reusable_record(identity)
            if reused is not None:
                reused_value = reused.value
                if (
                    scalar_value(task.operation, reused_value[FIELDS[task.operation]])
                    == reused_value
                ):
                    if cancel.is_set():
                        raise FingerprintCancelled()
                    if (
                        not snapshot.inputs.unchanged()
                        or scalar_runtime(task.operation) != runtime
                    ):
                        raise ValueError("Scalar analysis inputs changed during reuse")
                    return ScalarOutcome(
                        task.clip_id,
                        task.operation,
                        "skipped",
                        json.dumps(reused.to_dict(), sort_keys=True),
                    )
        value: float | None
        if task.operation == "brightness":
            from core.analysis.color import get_average_brightness

            value = get_average_brightness(
                files["video"], start, end, fps, num_samples=task.num_samples
            )
        else:
            if not {"ffmpeg", "ffprobe", "video"}.issubset(files):
                raise RuntimeError("Volume analysis requires FFmpeg and FFprobe")
            from core.analysis.audio import extract_clip_volume

            value = extract_clip_volume(
                files["video"],
                start / fps,
                (end - start) / fps,
                _ffmpeg_path=files["ffmpeg"],
                _ffprobe_path=files["ffprobe"],
            )
        if cancel.is_set():
            raise FingerprintCancelled()
        if not snapshot.inputs.unchanged() or scalar_runtime(task.operation) != runtime:
            raise ValueError("Scalar analysis inputs changed during computation")
        record = AnalysisRecord.success(
            identity,
            scalar_value(task.operation, value),
            input_snapshot=snapshot.inputs.to_dict(),
        )
        return ScalarOutcome(
            task.clip_id,
            task.operation,
            "succeeded",
            json.dumps(record.to_dict(), sort_keys=True),
        )
    except FingerprintCancelled:
        return ScalarOutcome(
            task.clip_id, task.operation, "unprocessed", message="Cancelled"
        )
    except Exception as exc:
        record_json = None
        if cancel.is_set():
            return ScalarOutcome(
                task.clip_id, task.operation, "unprocessed", message="Cancelled"
            )
        if (
            identity is not None
            and snapshot.inputs.unchanged()
            and scalar_runtime(task.operation) == runtime
        ):
            record = replace(
                AnalysisRecord.failure(identity, str(exc)),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
            record_json = json.dumps(record.to_dict(), sort_keys=True)
        return ScalarOutcome(
            task.clip_id, task.operation, "failed", record_json, str(exc)
        )


def run_scalars(
    tasks: tuple[ScalarTask, ...],
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ScalarOutcome], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> tuple[ScalarOutcome, ...]:
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    results = []
    for task in tasks:
        outcome = _compute(task, fingerprints, cancel)
        results.append(outcome)
        if on_outcome is not None and not cancel.is_set():
            on_outcome(outcome)
    return tuple(results)


class ScalarApplication:
    def __init__(self, project: "Project", task: ScalarTask) -> None:
        project.session.assert_owner()
        self.project, self.task = project, task
        self.path, self.session_id = project.path, project.session.session_id
        self.clip = project.clips_by_id.get(task.clip_id)
        self.source = (
            project.sources_by_id.get(self.clip.source_id) if self.clip else None
        )
        self.consumed = False

    def apply(self, project: "Project", outcome: ScalarOutcome) -> bool:
        if (
            project is not self.project
            or project.path != self.path
            or project.session.session_id != self.session_id
            or self.consumed
            or outcome.clip_id != self.task.clip_id
            or outcome.operation != self.task.operation
            or outcome.record_json is None
            or outcome.status == "unprocessed"
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            clip = project.clips_by_id.get(self.task.clip_id)
            if (
                clip is None
                or clip is not self.clip
                or project.sources_by_id.get(clip.source_id) is not self.source
            ):
                return False
            snapshot = AnalysisSnapshot.from_json(self.task.snapshot_json)
            if (
                scalar_snapshot(clip, self.source, self.task.operation) != snapshot
                or not snapshot.inputs.unchanged()
            ):
                return False
            try:
                record = AnalysisRecord.from_dict(
                    json.loads(outcome.record_json or "null")
                )
                if record.identity is None:
                    return False
                data = record.identity.to_dict()
                if (
                    record.input_json
                    != json.dumps(
                        snapshot.inputs.to_dict(), sort_keys=True, separators=(",", ":")
                    )
                    or data["operation"] != self.task.operation
                    or data["operation_version"] != 1
                    or data["schema_version"] != 1
                    or data["model"] != scalar_runtime(self.task.operation)
                    or data["parameters"] != scalar_parameters(self.task)
                    or data["sampling"] != scalar_sampling(self.task.operation)
                    or data["source_range"] != json.loads(snapshot.inputs.range_json)
                    or data["prompt_sha256"] is not None
                    or set(data["sources"])
                    != {role for role, _, _ in snapshot.inputs.files}
                ):
                    return False
                if outcome.has_result:
                    if (
                        record.state != "succeeded"
                        or scalar_value(
                            self.task.operation,
                            record.value[FIELDS[self.task.operation]],
                        )
                        != record.value
                    ):
                        return False
                    if outcome.status == "skipped" and record.value != json.loads(
                        snapshot.value_json
                    ):
                        return False
                elif record.state != "failed":
                    return False
            except (ValueError, TypeError, KeyError, AttributeError):
                return False
            project.record_analysis("clip", clip.id, self.task.operation, record)
            if outcome.status == "succeeded":
                setattr(
                    clip,
                    FIELDS[self.task.operation],
                    record.value[FIELDS[self.task.operation]],
                )
                project.update_clips([clip])
            return True

        return project.session.apply_external(publish)


class ScalarBatchApplication:
    """Route detached outcomes to applications captured on the project owner."""

    def __init__(self, project: "Project", tasks: tuple[ScalarTask, ...]) -> None:
        self.applications = {task.clip_id: ScalarApplication(project, task) for task in tasks}
        if len(self.applications) != len(tasks):
            raise ValueError("Scalar analysis requires unique clip IDs")

    def apply(self, project: "Project", outcome: ScalarOutcome) -> bool:
        application = self.applications.get(outcome.clip_id)
        return application.apply(project, outcome) if application is not None else False
