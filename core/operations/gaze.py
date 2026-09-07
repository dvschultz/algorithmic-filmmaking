"""Detached gaze extraction and guarded project publication."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Iterator, TYPE_CHECKING

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


@dataclass(frozen=True)
class GazeOptions:
    sample_interval: float = 1.0


@dataclass(frozen=True)
class GazeOutcome:
    clip_id: str
    status: OutcomeStatus
    yaw: float | None = None
    pitch: float | None = None
    category: str | None = None
    code: str | None = None
    message: str | None = None

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


def run_gaze(
    tasks: tuple[GazeTask, ...],
    options: GazeOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[GazeOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_session: _GazeModelSession | None = None,
) -> tuple[GazeOutcome, ...]:
    """Serialize shared model use and unloading; never mutate project models."""
    cancel = cancel_event or Event()
    outcomes: list[GazeOutcome] = []
    owns_session = model_session is None
    session = model_session or _GazeModelSession()
    try:
        for task in tasks:
            if cancel.is_set():
                outcome = GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
            elif task.skip:
                outcome = GazeOutcome(task.clip_id, "skipped", code="already_populated")
            elif task.source_path is None or not task.source_path.is_file():
                outcome = GazeOutcome(
                    task.clip_id, "failed", code="source_file_missing"
                )
            elif session.model_error is not None:
                outcome = GazeOutcome(
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
                        raise ValueError("Invalid gaze sampling range or interval")
                    while not session.acquired and not cancel.is_set():
                        session.acquired = _inference_lock.acquire(timeout=0.05)
                    if cancel.is_set():
                        outcome = GazeOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                    else:
                        from core.analysis.gaze import (
                            load_face_mesh,
                            extract_gaze_from_clip,
                        )

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
                        outcome = (
                            GazeOutcome(task.clip_id, "failed", code="no_gaze_detected")
                            if raw is None
                            else GazeOutcome.from_result(task.clip_id, raw)
                        )
                except Exception as exc:
                    outcome = GazeOutcome(
                        task.clip_id,
                        "failed",
                        code="model_load_failed"
                        if session.model_error is not None
                        else "gaze_failed",
                        message=str(exc),
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
        if owns_session:
            session.close()
    return tuple(outcomes)


class GazeApplication:
    """Publish each result once, only while its source and target remain current."""

    def __init__(self, project: "Project", tasks: tuple[GazeTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
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
            ),
        )

    def apply(self, project: "Project", outcome: GazeOutcome) -> bool:
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
            current[0].gaze_yaw = outcome.yaw
            current[0].gaze_pitch = outcome.pitch
            current[0].gaze_category = outcome.category
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
