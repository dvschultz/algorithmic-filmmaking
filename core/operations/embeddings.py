"""Detached DINOv2 batches and owner-thread embedding publication."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Sequence, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project

_inference_lock = Lock()


@dataclass(frozen=True)
class EmbeddingTask:
    clip_id: str
    thumbnail_path: Path | None
    skip: bool = False


@dataclass(frozen=True)
class EmbeddingOptions:
    chunk_size: int = 16


@dataclass(frozen=True)
class EmbeddingOutcome:
    clip_id: str
    status: OutcomeStatus
    vector: tuple[float, ...] = ()
    model: str | None = None
    code: str | None = None
    message: str | None = None

    @classmethod
    def from_vector(cls, clip_id: str, values: Sequence[float]) -> "EmbeddingOutcome":
        from core.analysis.embeddings import _EMBEDDING_DIM, _EMBEDDING_MODEL_TAG

        if len(values) != _EMBEDDING_DIM or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in values
        ):
            raise ValueError("Invalid embedding vector dimensions or values")
        if not any(values):
            raise ValueError("Embedding is empty or its image could not be decoded")
        return cls(
            clip_id, "succeeded", tuple(float(v) for v in values), _EMBEDDING_MODEL_TAG
        )


def run_embeddings(
    tasks: tuple[EmbeddingTask, ...],
    options: EmbeddingOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[EmbeddingOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[EmbeddingOutcome, ...]:
    """Compute bounded batches, validate every item, and serialize model ownership."""
    if (
        isinstance(options.chunk_size, bool)
        or not isinstance(options.chunk_size, int)
        or options.chunk_size < 1
    ):
        raise ValueError("Embedding chunk size must be a positive integer")
    cancel = cancel_event or Event()
    outcomes: dict[str, EmbeddingOutcome] = {}
    pending = []
    acquired = attempted = False

    def publish(outcome: EmbeddingOutcome) -> None:
        outcomes[outcome.clip_id] = outcome
        if not cancel.is_set():
            if on_outcome:
                on_outcome(outcome)
            if progress:
                progress(len(outcomes), len(tasks))

    try:
        for task in tasks:
            if cancel.is_set():
                break
            if task.skip:
                publish(
                    EmbeddingOutcome(task.clip_id, "skipped", code="already_populated")
                )
            elif task.thumbnail_path is None or not task.thumbnail_path.is_file():
                publish(
                    EmbeddingOutcome(task.clip_id, "failed", code="thumbnail_missing")
                )
            else:
                pending.append(task)
        if pending and not cancel.is_set():
            while not cancel.is_set():
                if _inference_lock.acquire(timeout=0.05):
                    acquired = True
                    break
            if acquired and not cancel.is_set():
                from core.analysis.embeddings import extract_clip_embeddings_batch

                for start in range(0, len(pending), options.chunk_size):
                    if cancel.is_set():
                        break
                    chunk = pending[start : start + options.chunk_size]
                    try:
                        attempted = True
                        paths: list[Path] = []
                        for task in chunk:
                            assert task.thumbnail_path is not None
                            paths.append(task.thumbnail_path)
                        vectors = extract_clip_embeddings_batch(paths)
                        if cancel.is_set():
                            break
                        if len(vectors) != len(chunk):
                            raise ValueError(
                                "Embedding batch returned the wrong number of vectors"
                            )
                    except Exception as exc:
                        if cancel.is_set():
                            break
                        for task in chunk:
                            publish(
                                EmbeddingOutcome(
                                    task.clip_id,
                                    "failed",
                                    code="embedding_failed",
                                    message=str(exc),
                                )
                            )
                        for task in pending[start + len(chunk) :]:
                            publish(
                                EmbeddingOutcome(
                                    task.clip_id, "unprocessed", code="embedding_failed"
                                )
                            )
                        break
                    for task, vector in zip(chunk, vectors):
                        if cancel.is_set():
                            break
                        try:
                            outcome = EmbeddingOutcome.from_vector(task.clip_id, vector)
                        except (TypeError, ValueError) as exc:
                            outcome = EmbeddingOutcome(
                                task.clip_id,
                                "failed",
                                code="embedding_failed",
                                message=str(exc),
                            )
                        publish(outcome)
    finally:
        if acquired:
            try:
                if attempted:
                    from core.analysis.embeddings import unload_model

                    unload_model()
            finally:
                _inference_lock.release()
    return tuple(
        outcomes.get(
            task.clip_id,
            EmbeddingOutcome(task.clip_id, "unprocessed", code="cancelled"),
        )
        for task in tasks
    )


class EmbeddingApplication:
    """Apply once to the same clip, thumbnail, source, and previous embedding."""

    def __init__(self, project: "Project", tasks: tuple[EmbeddingTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: EmbeddingTask) -> tuple | None:
        from core.jobs.media import media_stamp

        clip = project.clips_by_id.get(task.clip_id)
        if clip is None or task.thumbnail_path is None:
            return None
        stamp = media_stamp(task.thumbnail_path)
        if stamp is None:
            return None
        source = project.sources_by_id.get(clip.source_id)
        return (
            clip,
            source,
            (
                clip.thumbnail_path,
                clip.source_id,
                clip.start_frame,
                clip.end_frame,
                source.file_path if source else None,
                source.fps if source else None,
                media_stamp(source.file_path) if source else None,
                stamp,
                tuple(clip.embedding) if clip.embedding is not None else None,
                clip.embedding_model,
            ),
        )

    def apply(self, project: "Project", outcome: EmbeddingOutcome) -> bool:
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
            checked = EmbeddingOutcome.from_vector(outcome.clip_id, outcome.vector)
            if outcome.model != checked.model:
                raise ValueError("Embedding model identity does not match its vector")
            current[0].embedding = list(checked.vector)
            current[0].embedding_model = checked.model
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
