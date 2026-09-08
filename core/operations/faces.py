"""Detached face extraction and guarded project publication."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Iterator, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus
from core.analysis_records import AnalysisFingerprints, AnalysisInput, AnalysisSnapshot
from models.analysis_record import AnalysisRecord
from core.operations.face_records import (
    FACE_SAMPLING,
    face_snapshot,
    face_value,
    face_result_value,
    face_parameters,
    face_environment,
    face_packages,
    face_runtime,
    execution_inputs,
    saved_execution,
    validate_face_frames,
)

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

_inference_lock = Lock()


@dataclass(frozen=True)
class FaceTask:
    clip_id: str
    source_id: str
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float
    skip: bool = False
    analysis_json: str | None = None


@dataclass(frozen=True)
class FaceOptions:
    sample_interval: float = 1.0


def face_task(
    clip: "Clip", source: "Source | None" = None, *, skip_existing: bool = True
) -> FaceTask:
    """Capture verified face inputs without loading models or hashing files."""
    snapshot = face_snapshot(clip, source)
    return FaceTask(
        clip.id,
        clip.source_id,
        Path(source.file_path) if source else None,
        clip.start_frame,
        clip.end_frame,
        source.fps if source else 0.0,
        skip_existing,
        snapshot.to_json(),
    )


@dataclass(frozen=True)
class Face:
    bbox: tuple[float, ...]
    embedding: tuple[float, ...]
    confidence: float
    frame_number: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Face":
        bbox = tuple(data["bbox"])
        embedding = tuple(data["embedding"])
        confidence = data["confidence"]
        frame = data.get("frame_number")
        if (
            len(bbox) != 4
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or len(embedding) != 512
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not isfinite(v)
                for v in (*bbox, *embedding)
            )
            or bbox[2] < 0
            or bbox[3] < 0
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            or (frame is not None and (type(frame) is not int or frame < 0))
        ):
            raise ValueError("Invalid face result")
        return cls(bbox, embedding, confidence, frame)

    def to_dict(self) -> dict:
        data = {
            "bbox": list(self.bbox),
            "embedding": list(self.embedding),
            "confidence": self.confidence,
        }
        if self.frame_number is not None:
            data["frame_number"] = self.frame_number
        return data


@dataclass(frozen=True)
class FaceOutcome:
    clip_id: str
    status: OutcomeStatus
    faces: tuple[Face, ...] = ()
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
    def from_dict(cls, data: dict) -> "FaceOutcome":
        return cls(
            **{**data, "faces": tuple(Face.from_dict(f) for f in data.get("faces", ()))}
        )

    def face_dicts(self) -> list[dict]:
        return [face.to_dict() for face in self.faces]


@dataclass
class _FaceModelSession:
    acquired: bool = False
    loaded: bool = False
    model_error: str | None = None

    def close(self) -> None:
        if self.acquired:
            try:
                from core.analysis.faces import unload_model

                unload_model()
            finally:
                self.acquired = False
                _inference_lock.release()


@contextmanager
def face_model_session() -> Iterator[_FaceModelSession]:
    """Retain one model across same-thread result commits; acquire only for inference."""
    session = _FaceModelSession()
    try:
        yield session
    finally:
        session.close()


def _run_raw_faces(
    tasks: tuple[FaceTask, ...],
    options: FaceOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[FaceOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_session: _FaceModelSession | None = None,
    on_execution: Callable[[dict], None] | None = None,
) -> tuple[FaceOutcome, ...]:
    """Serialize shared model use and unloading; never mutate project models."""
    cancel = cancel_event or Event()
    outcomes: list[FaceOutcome] = []
    owns_session = model_session is None
    session = model_session or _FaceModelSession()
    try:
        for task in tasks:
            if cancel.is_set():
                outcome = FaceOutcome(task.clip_id, "unprocessed", code="cancelled")
            elif task.skip:
                outcome = FaceOutcome(task.clip_id, "skipped", code="already_populated")
            elif task.source_path is None or not task.source_path.is_file():
                outcome = FaceOutcome(
                    task.clip_id, "failed", code="source_file_missing"
                )
            elif session.model_error is not None:
                outcome = FaceOutcome(
                    task.clip_id, "unprocessed", code="model_unavailable"
                )
            else:
                try:
                    if (
                        not isfinite(task.fps)
                        or task.fps <= 0
                        or task.start_frame < 0
                        or task.end_frame <= task.start_frame
                        or not isfinite(options.sample_interval)
                        or options.sample_interval <= 0
                    ):
                        raise ValueError("Invalid face sampling range or interval")
                    while not session.acquired and not cancel.is_set():
                        session.acquired = _inference_lock.acquire(timeout=0.05)
                    if cancel.is_set():
                        outcome = FaceOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                    else:
                        from core.analysis.faces import (
                            _load_insightface,
                            extract_faces_from_clip,
                        )

                        if not session.loaded:
                            try:
                                _load_insightface()
                                session.loaded = True
                            except Exception as exc:
                                session.model_error = str(exc)
                                raise
                        raw = extract_faces_from_clip(
                            source_path=task.source_path,
                            start_frame=task.start_frame,
                            end_frame=task.end_frame,
                            fps=task.fps,
                            sample_interval=options.sample_interval,
                            **({"on_execution": on_execution} if on_execution else {}),
                        )
                        faces = tuple(Face.from_dict(face) for face in raw)
                        outcome = FaceOutcome(task.clip_id, "succeeded", faces)
                except Exception as exc:
                    outcome = FaceOutcome(
                        task.clip_id,
                        "failed",
                        code="model_load_failed"
                        if session.model_error is not None
                        else "face_detection_failed",
                        message=str(exc),
                    )
                if cancel.is_set():
                    outcome = FaceOutcome(task.clip_id, "unprocessed", code="cancelled")
            outcomes.append(outcome)
            if not cancel.is_set():
                if on_outcome:
                    on_outcome(outcome)
                if progress:
                    progress(len(outcomes), len(tasks))
    finally:
        if owns_session:
            session.close()
    return tuple(outcomes)


def run_faces(
    tasks: tuple[FaceTask, ...],
    options: FaceOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[FaceOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_session: _FaceModelSession | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> tuple[FaceOutcome, ...]:
    """Verify semantic reuse and retain owned failures before model publication."""
    cancel = cancel_event or Event()
    session = model_session or _FaceModelSession()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    outcomes = []
    try:
        for task in tasks:
            if task.analysis_json is None:
                outcome = _run_raw_faces(
                    (task,), options, cancel_event=cancel, model_session=session
                )[0]
            else:
                outcome = _verified_face(task, options, cancel, session, fingerprints)
            if cancel.is_set():
                outcome = FaceOutcome(task.clip_id, "unprocessed", code="cancelled")
            outcomes.append(outcome)
            if not cancel.is_set():
                if on_outcome:
                    on_outcome(deepcopy(outcome))
                if progress:
                    progress(len(outcomes), len(tasks))
    finally:
        if model_session is None:
            session.close()
    return tuple(outcomes)


def _verified_face(
    task: FaceTask,
    options: FaceOptions,
    cancel: Event,
    session: _FaceModelSession,
    fingerprints: AnalysisFingerprints,
) -> FaceOutcome:
    snapshot = None
    identity = None
    inputs = None
    try:
        if cancel.is_set():
            return FaceOutcome(task.clip_id, "unprocessed", code="cancelled")
        snapshot = AnalysisSnapshot.from_json(task.analysis_json or "")
        parameters = face_parameters(options.sample_interval)

        def identify(bound, runtime):
            return fingerprints.identity(
                bound,
                operation="face_embeddings",
                operation_version=2,
                model=runtime,
                parameters=parameters,
                sampling=FACE_SAMPLING,
            )

        inputs = snapshot.inputs
        identity = identify(
            inputs,
            {
                "packages": face_packages(),
                "available_providers": [],
                "model": "buffalo_l",
                "components": [],
            },
        )
        environment = face_environment()
        identity = identify(
            inputs, {**environment, "model": "buffalo_l", "components": []}
        )
        if (
            task.skip
            and snapshot.record is not None
            and snapshot.record.identity is not None
        ):
            try:
                from core.jobs.media import media_stamp
                from core.analysis.faces import _get_model_cache_dir

                execution = saved_execution(snapshot.record)
                directory = (
                    _get_model_cache_dir() / "insightface" / "models" / "buffalo_l"
                ).resolve()
                for item in execution["weight_files"]:
                    if Path(item["path"]).parent != directory:
                        raise ValueError("Face model cache path changed")
                    item["stamp"] = list(media_stamp(Path(item["path"])) or ())
                by_path = {item["path"]: item for item in execution["weight_files"]}
                for component in execution["components"]:
                    component["weights"]["stamp"] = by_path[component["path"]]["stamp"]
                bound = execution_inputs(snapshot, execution)
                runtime = face_runtime(execution, environment)
                candidate = identify(bound, runtime)
                reused = replace(snapshot, inputs=bound).reusable_record(candidate)
                if (
                    reused is not None
                    and bound.unchanged()
                    and face_environment() == environment
                ):
                    value = face_result_value(reused.value["face_embeddings"])
                    if value != reused.value:
                        raise ValueError("Invalid saved face projection")
                    return FaceOutcome(
                        task.clip_id,
                        "skipped",
                        tuple(Face.from_dict(v) for v in value["face_embeddings"]),
                        code="valid_analysis",
                        record_json=json.dumps(reused.to_dict(), sort_keys=True),
                    )
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                pass
        events: list[dict] = []
        outcome = _run_raw_faces(
            (replace(task, skip=False),),
            options,
            cancel_event=cancel,
            model_session=session,
            on_execution=events.append,
        )[0]
        if cancel.is_set() or outcome.status == "unprocessed":
            return outcome
        if not snapshot.inputs.unchanged() or face_environment() != environment:
            return FaceOutcome(task.clip_id, "failed", code="stale_input")
        if events:
            if len(events) != 1:
                raise ValueError("Ambiguous face model execution")
            inputs = execution_inputs(snapshot, events[0])
            runtime = face_runtime(events[0], environment)
            identity = identify(inputs, runtime)
            actual_sources = identity.to_dict()["sources"]
            for item in events[0]["weight_files"]:
                if actual_sources[f"model:{Path(item['path']).name}"] != item["sha256"]:
                    raise ValueError("Face weights differ from loaded model")
        elif outcome.status == "succeeded":
            raise ValueError("Face provider did not report weight execution")
        if outcome.status == "succeeded":
            value = face_result_value(outcome.face_dicts())
            validate_face_frames(
                value,
                task.start_frame,
                task.end_frame,
                task.fps,
                options.sample_interval,
            )
            record = AnalysisRecord.success(
                identity, value, input_snapshot=inputs.to_dict()
            )
            return replace(
                outcome,
                faces=tuple(Face.from_dict(v) for v in value["face_embeddings"]),
                record_json=json.dumps(record.to_dict(), sort_keys=True),
            )
        error = outcome.message or outcome.code or "Face analysis failed"
    except Exception as exc:
        error = str(exc)
    if cancel.is_set():
        return FaceOutcome(task.clip_id, "unprocessed", code="cancelled")
    if snapshot is None or identity is None or inputs is None or not inputs.unchanged():
        return FaceOutcome(task.clip_id, "failed", code="stale_input", message=error)
    record = replace(
        AnalysisRecord.failure(identity, error),
        input_json=json.dumps(inputs.to_dict(), sort_keys=True),
    )
    return FaceOutcome(
        task.clip_id,
        "failed",
        code="face_detection_failed",
        message=error,
        record_json=json.dumps(record.to_dict(), sort_keys=True),
    )


class FaceApplication:
    """Publish each result once, only while its source and target remain current."""

    def __init__(
        self,
        project: "Project",
        tasks: tuple[FaceTask, ...],
        options: FaceOptions | None = None,
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
    def _binding(project: "Project", task: FaceTask) -> tuple | None:
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
                deepcopy(clip.face_embeddings),
                deepcopy(clip.analysis_records.get("face_embeddings")),
            ),
        )

    def apply(self, project: "Project", outcome: FaceOutcome) -> bool:
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
            try:
                value = face_result_value(outcome.face_dicts())
                record = AnalysisRecord.legacy(value)
                if task is not None and task.analysis_json is not None:
                    if outcome.record_json is None:
                        return False
                    snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                    if face_snapshot(current[0], current[1]) != snapshot:
                        return False
                    record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
                    if record.identity is None:
                        return False
                    identity = record.identity.to_dict()
                    inputs = AnalysisInput.from_dict(
                        json.loads(record.input_json or "null")
                    )
                    media = replace(
                        inputs,
                        files=tuple(
                            f for f in inputs.files if not f[0].startswith("model:")
                        ),
                    )
                    if (
                        media != snapshot.inputs
                        or not inputs.unchanged()
                        or identity["operation"] != "face_embeddings"
                        or identity["operation_version"] != 2
                        or identity["schema_version"] != 1
                        or identity["source_range"] != json.loads(inputs.range_json)
                        or identity["sampling"] != FACE_SAMPLING
                        or identity["prompt_sha256"] is not None
                        or set(identity["sources"]) != {f[0] for f in inputs.files}
                        or identity["model"]["packages"] != face_packages()
                        or (
                            self.options is not None
                            and identity["parameters"]
                            != face_parameters(self.options.sample_interval)
                        )
                    ):
                        return False
                    if outcome.has_result:
                        interval = identity["parameters"]["sample_interval"]
                        if identity["parameters"] != face_parameters(interval):
                            return False
                        validate_face_frames(
                            value, task.start_frame, task.end_frame, task.fps, interval
                        )
                        execution = saved_execution(record)
                        environment = {
                            key: identity["model"][key]
                            for key in ("packages", "available_providers")
                        }
                        if (
                            record.state != "succeeded"
                            or record.value != value
                            or face_runtime(execution, environment) != identity["model"]
                            or execution_inputs(snapshot, execution) != inputs
                            or (
                                outcome.status == "skipped"
                                and value != face_value(current[0])
                            )
                        ):
                            return False
                    elif record.state != "failed":
                        return False
                elif outcome.status != "succeeded" or outcome.record_json is not None:
                    return False
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                return False
            project.record_analysis("clip", outcome.clip_id, "face_embeddings", record)
            if outcome.status == "succeeded":
                current[0].face_embeddings = (
                    value["face_embeddings"]
                    if task is not None and task.analysis_json is not None
                    else outcome.face_dicts()
                )
                project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
