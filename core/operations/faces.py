"""Detached face extraction and guarded project publication."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project

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


@dataclass(frozen=True)
class FaceOptions:
    sample_interval: float = 1.0


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
        confidence = float(data["confidence"])
        frame = data.get("frame_number")
        if (
            len(bbox) != 4
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

    def face_dicts(self) -> list[dict]:
        return [face.to_dict() for face in self.faces]


def run_faces(
    tasks: tuple[FaceTask, ...],
    options: FaceOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[FaceOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[FaceOutcome, ...]:
    """Serialize shared model use and unloading; never mutate project models."""
    cancel = cancel_event or Event()
    outcomes: list[FaceOutcome] = []
    acquired = False
    loaded = False
    model_error = None
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
            elif model_error is not None:
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
                    while not acquired and not cancel.is_set():
                        acquired = _inference_lock.acquire(timeout=0.05)
                    if cancel.is_set():
                        outcome = FaceOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                    else:
                        from core.analysis.faces import (
                            _load_insightface,
                            extract_faces_from_clip,
                        )

                        if not loaded:
                            try:
                                _load_insightface()
                                loaded = True
                            except Exception as exc:
                                model_error = str(exc)
                                raise
                        raw = extract_faces_from_clip(
                            source_path=task.source_path,
                            start_frame=task.start_frame,
                            end_frame=task.end_frame,
                            fps=task.fps,
                            sample_interval=options.sample_interval,
                        )
                        faces = tuple(Face.from_dict(face) for face in raw)
                        outcome = FaceOutcome(task.clip_id, "succeeded", faces)
                except Exception as exc:
                    outcome = FaceOutcome(
                        task.clip_id,
                        "failed",
                        code="model_load_failed"
                        if model_error is not None
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
        if acquired:
            try:
                from core.analysis.faces import unload_model

                unload_model()
            finally:
                _inference_lock.release()
    return tuple(outcomes)


class FaceApplication:
    """Publish each result once, only while its source and target remain current."""

    def __init__(self, project: "Project", tasks: tuple[FaceTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
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
            ),
        )

    def apply(self, project: "Project", outcome: FaceOutcome) -> bool:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or outcome.status != "succeeded"
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
            current[0].face_embeddings = outcome.face_dicts()
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
