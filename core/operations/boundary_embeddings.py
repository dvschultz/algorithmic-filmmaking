"""Detached first/last-frame embeddings with shared DINOv2 model ownership."""

from dataclasses import dataclass
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip

from core.operations.contracts import OutcomeStatus
from core.operations.embeddings import (
    EmbeddingOutcome,
    _EmbeddingModelSession,
    embedding_model_session,
)


@dataclass(frozen=True)
class BoundaryEmbeddingTask:
    clip_id: str
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float
    skip: bool = False


@dataclass(frozen=True)
class BoundaryEmbeddingOutcome:
    clip_id: str
    status: OutcomeStatus
    first: tuple[float, ...] = ()
    last: tuple[float, ...] = ()
    model: str | None = None
    code: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "BoundaryEmbeddingOutcome":
        outcome = cls(**payload)
        if outcome.status == "succeeded":
            first = EmbeddingOutcome.from_vector(outcome.clip_id, outcome.first)
            last = EmbeddingOutcome.from_vector(outcome.clip_id, outcome.last)
            if outcome.model != first.model:
                raise ValueError("Boundary embedding model does not match")
            return cls(
                outcome.clip_id, "succeeded", first.vector, last.vector, first.model
            )
        if outcome.status not in ("failed", "skipped", "unprocessed"):
            raise ValueError("Invalid boundary embedding status")
        return outcome


def run_boundary_embeddings(
    tasks: tuple[BoundaryEmbeddingTask, ...],
    *,
    cancel_event: Event | None = None,
    model_session: _EmbeddingModelSession | None = None,
) -> tuple[BoundaryEmbeddingOutcome, ...]:
    """Validate each pair atomically and reject changed media or late cancellation."""
    from core.errors import ModelDownloadError
    from core.jobs.media import media_stamp

    cancel = cancel_event or Event()
    stamps = [
        media_stamp(task.source_path) if task.source_path and not task.skip else None
        for task in tasks
    ]
    outcomes: list[BoundaryEmbeddingOutcome] = []
    with (
        nullcontext(model_session)
        if model_session is not None
        else embedding_model_session()
    ) as session:
        for task, stamp in zip(tasks, stamps):
            if cancel.is_set() or session.failed:
                break
            if task.skip:
                outcomes.append(
                    BoundaryEmbeddingOutcome(
                        task.clip_id, "skipped", code="already_populated"
                    )
                )
                continue
            try:
                if (
                    isinstance(task.fps, bool)
                    or not isfinite(task.fps)
                    or task.fps <= 0
                    or isinstance(task.start_frame, bool)
                    or not isinstance(task.start_frame, int)
                    or isinstance(task.end_frame, bool)
                    or not isinstance(task.end_frame, int)
                    or task.start_frame < 0
                    or task.end_frame <= task.start_frame
                ):
                    raise ValueError(
                        "Boundary embeddings require a valid frame range and positive finite fps"
                    )
                if (
                    task.source_path is None
                    or stamp is None
                    or media_stamp(task.source_path) != stamp
                ):
                    raise ValueError("Boundary embedding source is missing or changed")
                if not session.acquire(cancel):
                    break
                if media_stamp(task.source_path) != stamp:
                    raise ValueError(
                        "Boundary embedding source changed while waiting for model"
                    )
                from core.analysis.embeddings import extract_boundary_embeddings

                session.attempted = True
                first, last = extract_boundary_embeddings(
                    source_path=task.source_path,
                    start_frame=task.start_frame,
                    end_frame=task.end_frame,
                    fps=task.fps,
                    cancel_event=cancel,
                )
                if cancel.is_set():
                    break
                if media_stamp(task.source_path) != stamp:
                    raise ValueError(
                        "Boundary embedding source changed during inference"
                    )
                first_result = EmbeddingOutcome.from_vector(task.clip_id, first)
                last_result = EmbeddingOutcome.from_vector(task.clip_id, last)
                outcome = BoundaryEmbeddingOutcome(
                    task.clip_id,
                    "succeeded",
                    first_result.vector,
                    last_result.vector,
                    first_result.model,
                )
            except Exception as exc:
                if cancel.is_set():
                    break
                if isinstance(exc, ModelDownloadError):
                    session.failed = True
                outcome = BoundaryEmbeddingOutcome(
                    task.clip_id, "failed", code="embedding_failed", message=str(exc)
                )
            outcomes.append(outcome)
        outcomes.extend(
            BoundaryEmbeddingOutcome(
                task.clip_id,
                "unprocessed",
                code="cancelled" if cancel.is_set() else "embedding_failed",
            )
            for task in tasks[len(outcomes) :]
        )
    return tuple(
        BoundaryEmbeddingOutcome(
            task.clip_id,
            "failed",
            code="source_changed",
            message="Boundary embedding source changed before delivery",
        )
        if outcome.status == "succeeded"
        and (task.source_path is None or media_stamp(task.source_path) != stamp)
        else outcome
        for task, stamp, outcome in zip(tasks, stamps, outcomes)
    )


def validate_boundary_model(clip: "Clip", model: str | None) -> None:
    """Do not relabel an existing thumbnail vector through the shared model field."""
    if clip.embedding is not None and clip.embedding_model != model:
        raise ValueError(
            "Existing thumbnail embeddings use a different or unknown model. "
            "Reanalyze or clear thumbnail embeddings before boundary analysis."
        )


class BoundaryEmbeddingApplication:
    """Publish a validated pair once to its original unchanged clip."""

    def __init__(
        self, project: "Project", tasks: tuple[BoundaryEmbeddingTask, ...]
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: BoundaryEmbeddingTask) -> tuple | None:
        from core.jobs.media import media_stamp

        clip = project.clips_by_id.get(task.clip_id)
        source = project.sources_by_id.get(clip.source_id) if clip else None
        if (
            clip is None
            or source is None
            or (source.file_path, clip.start_frame, clip.end_frame, source.fps)
            != (task.source_path, task.start_frame, task.end_frame, task.fps)
        ):
            return None
        stamp = media_stamp(source.file_path)
        if stamp is None:
            return None
        return (
            clip,
            source,
            (
                clip.source_id,
                source.file_path,
                source.fps,
                clip.start_frame,
                clip.end_frame,
                stamp,
                tuple(clip.first_frame_embedding)
                if clip.first_frame_embedding is not None
                else None,
                tuple(clip.last_frame_embedding)
                if clip.last_frame_embedding is not None
                else None,
                clip.embedding_model,
            ),
        )

    def apply(self, project: "Project", outcome: BoundaryEmbeddingOutcome) -> bool:
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
            from dataclasses import asdict

            checked = BoundaryEmbeddingOutcome.from_dict(asdict(outcome))
            validate_boundary_model(current[0], checked.model)
            current[0].first_frame_embedding = list(checked.first)
            current[0].last_frame_embedding = list(checked.last)
            current[0].embedding_model = checked.model
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
