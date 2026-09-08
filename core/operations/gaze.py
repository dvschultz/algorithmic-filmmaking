"""Detached gaze extraction and guarded project publication."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Iterator, TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import gaze_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project

_inference_lock = Lock()


@dataclass(frozen=True)
class GazeTask:
    clip_id: str
    source_id: str
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float
    skip: bool = False
    analysis_json: str | None = None


@dataclass(frozen=True)
class GazeOptions:
    sample_interval: float = 1.0


def gaze_values(target: Any) -> dict:
    """Match the existing project serialization precision for gaze angles."""
    return {
        "gaze_yaw": round(target.gaze_yaw, 2) if target.gaze_yaw is not None else None,
        "gaze_pitch": round(target.gaze_pitch, 2)
        if target.gaze_pitch is not None
        else None,
        "gaze_category": target.gaze_category,
    }


def gaze_task(
    target: Any, source: Any = None, *, skip_existing: bool = True
) -> GazeTask:
    path = Path(source.file_path) if source is not None else None
    fps = source.fps if source is not None else 0.0
    snapshot = AnalysisSnapshot.capture(
        target,
        "gaze",
        {"video": path} if path is not None else {},
        {"start_frame": target.start_frame, "end_frame": target.end_frame, "fps": fps},
        gaze_values(target),
    )
    return GazeTask(
        target.id,
        target.source_id,
        path,
        target.start_frame,
        target.end_frame,
        fps,
        skip_existing,
        snapshot.to_json(),
    )


def gaze_identity(
    snapshot: AnalysisSnapshot,
    options: GazeOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    if (
        isinstance(options.sample_interval, bool)
        or not isfinite(options.sample_interval)
        or options.sample_interval <= 0
    ):
        raise ValueError("Invalid gaze sampling interval")
    return fingerprints.identity(
        snapshot.inputs,
        operation="gaze",
        operation_version=2,
        model=runtime,
        parameters={"sample_interval": float(options.sample_interval)},
        sampling={
            "policy": "half-open-uniform-frames/v1",
            "short_clip": "midpoint",
            "angle_precision": 2,
        },
    )


@dataclass(frozen=True)
class GazeOutcome:
    clip_id: str
    status: OutcomeStatus
    yaw: float | None = None
    pitch: float | None = None
    category: str | None = None
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

    @classmethod
    def from_dict(cls, data: dict) -> "GazeOutcome":
        outcome = cls(**data)
        if outcome.status == "succeeded":
            if outcome.category is None and outcome.code == "no_gaze_detected":
                if outcome.yaw is not None or outcome.pitch is not None:
                    raise ValueError("Empty gaze observation has angles")
            else:
                cls.from_result(
                    outcome.clip_id,
                    {
                        "gaze_yaw": outcome.yaw,
                        "gaze_pitch": outcome.pitch,
                        "gaze_category": outcome.category,
                    },
                )
        return outcome

    @classmethod
    def from_result(cls, clip_id: str, data: dict) -> "GazeOutcome":
        from core.analysis.gaze import GAZE_CATEGORIES

        yaw, pitch = data["gaze_yaw"], data["gaze_pitch"]
        category = data["gaze_category"]
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not isfinite(v)
                for v in (yaw, pitch)
            )
            or category not in GAZE_CATEGORIES
        ):
            raise ValueError("Invalid gaze result")
        return cls(clip_id, "succeeded", float(yaw), float(pitch), category)


@dataclass
class _GazeModelSession:
    acquired: bool = False
    loaded: bool = False
    model_error: str | None = None

    def close(self) -> None:
        if self.acquired:
            try:
                from core.analysis.gaze import unload_model

                unload_model()
            finally:
                self.acquired = False
                _inference_lock.release()


@contextmanager
def gaze_model_session() -> Iterator[_GazeModelSession]:
    """Retain one model across same-thread result commits; acquire only for inference."""
    session = _GazeModelSession()
    try:
        yield session
    finally:
        session.close()


def _compute_gaze(
    task: GazeTask,
    options: GazeOptions,
    cancel: Event,
    session: _GazeModelSession,
) -> GazeOutcome:
    """Compute one observation under the caller's shared model lease."""
    if cancel.is_set():
        return GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
    if task.skip:
        return GazeOutcome(task.clip_id, "skipped", code="already_populated")
    if task.source_path is None or not task.source_path.is_file():
        return GazeOutcome(task.clip_id, "failed", code="source_file_missing")
    if session.model_error is not None:
        return GazeOutcome(task.clip_id, "unprocessed", code="model_unavailable")
    try:
        if (
            not isfinite(task.fps)
            or task.fps <= 0
            or task.start_frame < 0
            or task.end_frame <= task.start_frame
            or not isfinite(options.sample_interval)
            or options.sample_interval <= 0
        ):
            raise ValueError("Invalid gaze sampling range or interval")
        while not session.acquired and not cancel.is_set():
            session.acquired = _inference_lock.acquire(timeout=0.05)
        if cancel.is_set():
            return GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
        from core.analysis.gaze import load_face_mesh, extract_gaze_from_clip

        if not session.loaded:
            try:
                load_face_mesh()
                session.loaded = True
            except Exception as exc:
                session.model_error = str(exc)
                raise
        raw = extract_gaze_from_clip(
            source_path=str(task.source_path),
            start_frame=task.start_frame,
            end_frame=task.end_frame,
            fps=task.fps,
            sample_interval=options.sample_interval,
        )
        return (
            GazeOutcome(task.clip_id, "succeeded", code="no_gaze_detected")
            if raw is None
            else GazeOutcome.from_result(task.clip_id, raw)
        )
    except Exception as exc:
        return GazeOutcome(
            task.clip_id,
            "failed",
            message=str(exc),
            code="model_load_failed"
            if session.model_error is not None
            else "gaze_failed",
        )


def run_gaze(
    tasks: tuple[GazeTask, ...],
    options: GazeOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[GazeOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_session: _GazeModelSession | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> tuple[GazeOutcome, ...]:
    """Verify reuse before loading the model; retain failed attempts separately."""
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else gaze_runtime()
    session = model_session or _GazeModelSession()
    outcomes = []
    try:
        for task in tasks:
            snapshot = None
            identity = None
            outcome = None
            if cancel.is_set():
                outcome = GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
            elif (
                task.analysis_json is not None
                and task.source_path is not None
                and task.source_path.is_file()
            ):
                try:
                    snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                    identity = gaze_identity(snapshot, options, fingerprints, runtime)
                    reused = snapshot.reusable_record(identity) if task.skip else None
                    if reused is not None:
                        value = reused.value
                        outcome = GazeOutcome(
                            task.clip_id,
                            "skipped",
                            value["gaze_yaw"],
                            value["gaze_pitch"],
                            value["gaze_category"],
                            code="valid_analysis",
                            record_json=json.dumps(reused.to_dict(), sort_keys=True),
                        )
                except Exception as exc:
                    outcome = GazeOutcome(
                        task.clip_id, "failed", code="stale_input", message=str(exc)
                    )
            if outcome is None:
                inference_task = (
                    replace(task, skip=False)
                    if task.analysis_json is not None
                    else task
                )
                outcome = _compute_gaze(inference_task, options, cancel, session)
                if snapshot is not None and identity is not None:
                    if not snapshot.inputs.unchanged():
                        outcome = GazeOutcome(
                            task.clip_id, "failed", code="stale_input"
                        )
                    elif (
                        outcome.status in ("succeeded", "failed")
                        and not cancel.is_set()
                    ):
                        value = {
                            "gaze_yaw": round(outcome.yaw, 2)
                            if outcome.yaw is not None
                            else None,
                            "gaze_pitch": round(outcome.pitch, 2)
                            if outcome.pitch is not None
                            else None,
                            "gaze_category": outcome.category,
                        }
                        record = (
                            AnalysisRecord.success(
                                identity,
                                value,
                                input_snapshot=snapshot.inputs.to_dict(),
                            )
                            if outcome.status == "succeeded"
                            else replace(
                                AnalysisRecord.failure(
                                    identity,
                                    outcome.message or outcome.code or "Gaze failed",
                                ),
                                input_json=json.dumps(
                                    snapshot.inputs.to_dict(), sort_keys=True
                                ),
                            )
                        )
                        outcome = replace(
                            outcome,
                            record_json=json.dumps(record.to_dict(), sort_keys=True),
                        )
            if cancel.is_set():
                outcome = GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
            outcomes.append(outcome)
            if not cancel.is_set():
                if on_outcome:
                    on_outcome(outcome)
                if progress:
                    progress(len(outcomes), len(tasks))
    finally:
        if model_session is None:
            session.close()
    return tuple(outcomes)


class GazeApplication:
    """Publish each result once, only while its source and target remain current."""

    def __init__(
        self,
        project: "Project",
        tasks: tuple[GazeTask, ...],
        options: GazeOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.path = project.path
        self.options = options
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: GazeTask) -> tuple | None:
        from core.jobs.media import media_stamp

        clip = project.clips_by_id.get(task.clip_id)
        source = project.sources_by_id.get(task.source_id)
        if clip is None or source is None or task.source_path is None:
            return None
        if (
            clip.source_id,
            source.file_path,
            clip.start_frame,
            clip.end_frame,
            source.fps,
        ) != (
            task.source_id,
            task.source_path,
            task.start_frame,
            task.end_frame,
            task.fps,
        ):
            return None
        stamp = media_stamp(task.source_path)
        if stamp is None:
            return None
        return (
            clip,
            source,
            (
                clip.source_id,
                source.file_path,
                clip.start_frame,
                clip.end_frame,
                source.fps,
                stamp,
                (clip.gaze_yaw, clip.gaze_pitch, clip.gaze_category),
                clip.analysis_records.get("gaze"),
            ),
        )

    def apply(self, project: "Project", outcome: GazeOutcome) -> bool:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or project.path != self.path
            or not outcome.can_apply
        ):
            return False

        def publish() -> bool:
            if outcome.clip_id in self.consumed:
                return False
            self.consumed.add(outcome.clip_id)
            task = self.tasks.get(outcome.clip_id)
            expected = self.bindings.get(outcome.clip_id)
            current = self._binding(project, task) if task else None
            if (
                expected is None
                or current is None
                or current[0] is not expected[0]
                or current[1] is not expected[1]
                or current[2] != expected[2]
            ):
                return False
            value = {
                "gaze_yaw": round(outcome.yaw, 2) if outcome.yaw is not None else None,
                "gaze_pitch": round(outcome.pitch, 2)
                if outcome.pitch is not None
                else None,
                "gaze_category": outcome.category,
            }
            record = (
                AnalysisRecord.from_dict(json.loads(outcome.record_json))
                if outcome.record_json is not None
                else AnalysisRecord.legacy(value)
            )
            if outcome.record_json is not None:
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task is not None and task.analysis_json
                    else None
                )
                if (
                    snapshot is None
                    or record.identity is None
                    or record.identity.operation != "gaze"
                    or (
                        self.options is not None
                        and record.identity.to_dict()["parameters"]
                        != {"sample_interval": float(self.options.sample_interval)}
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
            project.record_analysis("clip", outcome.clip_id, "gaze", record)
            if outcome.status == "failed":
                return True
            current[0].gaze_yaw = outcome.yaw
            current[0].gaze_pitch = outcome.pitch
            current[0].gaze_category = outcome.category
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
