"""Detached DINOv2 batches and owner-thread embedding publication."""

from dataclasses import dataclass, replace
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
import json
from threading import Event, Lock
from typing import Callable, Iterator, Sequence, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus
from core.analysis_records import AnalysisInput, AnalysisFingerprints
from core.analysis_model_identity import embedding_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

_inference_lock = Lock()


@dataclass(frozen=True)
class EmbeddingTask:
    clip_id: str
    thumbnail_path: Path | None
    skip: bool = False
    inputs: AnalysisInput | None = None
    record_json: str | None = None
    existing_vector: tuple[float, ...] | None = None
    existing_model: str | None = None


def embedding_task(clip: "Clip", source: "Source | None" = None, *, skip_existing: bool = True) -> EmbeddingTask:
    files = {"image": clip.thumbnail_path} if clip.thumbnail_path is not None else {}
    if source is not None:
        files["video"] = source.file_path
    record = clip.analysis_records.get("embeddings")
    return EmbeddingTask(
        clip.id, clip.thumbnail_path, skip_existing,
        AnalysisInput.capture(files, {"start_frame": clip.start_frame, "end_frame": clip.end_frame}, binding={"target_id": clip.id, "source_id": clip.source_id}),
        json.dumps(record.to_dict(), sort_keys=True) if isinstance(record, AnalysisRecord) else None,
        tuple(clip.embedding) if clip.embedding is not None else None, clip.embedding_model,
    )


def embedding_identity(task: EmbeddingTask, fingerprints: AnalysisFingerprints, runtime: dict) -> AnalysisIdentity:
    if task.inputs is None:
        raise ValueError("Embedding input snapshot is missing")
    return fingerprints.identity(
        task.inputs, operation="embeddings", operation_version=2,
        model=runtime, parameters={}, sampling={"policy": "thumbnail/v1", "processor": "dinov2-default"},
    )


def reusable_embedding(task: EmbeddingTask, identity: AnalysisIdentity) -> "EmbeddingOutcome | None":
    if task.record_json is None or task.existing_vector is None or task.inputs is None:
        return None
    from core.artifacts import ArtifactStore, ArtifactUnavailable

    try:
        record = AnalysisRecord.from_dict(json.loads(task.record_json))
        if not record.reusable(identity, artifact_available=lambda _ref: True):
            return None
        payload = json.loads(ArtifactStore().read_bytes(record.artifact)) if record.artifact is not None else record.value
        if payload != {"embedding": list(task.existing_vector), "embedding_model": task.existing_model}:
            return None
        checked = EmbeddingOutcome.from_vector(task.clip_id, task.existing_vector)
        if checked.model != task.existing_model:
            return None
        record = replace(
            record, artifact=None,
            value_json=json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            input_json=json.dumps(task.inputs.to_dict(), sort_keys=True, separators=(",", ":")),
        )
        return replace(checked, status="skipped", code="valid_analysis", record_json=json.dumps(record.to_dict(), sort_keys=True))
    except (ArtifactUnavailable, OSError, ValueError, TypeError, KeyError):
        return None


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
    record_json: str | None = None

    @property
    def can_apply(self) -> bool:
        return self.status == "succeeded" or (self.status == "skipped" and self.record_json is not None)

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingOutcome":
        outcome = cls(**{**data, "vector": tuple(data.get("vector", ()))})
        if outcome.status == "succeeded" or (outcome.status == "skipped" and outcome.record_json is not None):
            checked = cls.from_vector(outcome.clip_id, outcome.vector)
            if outcome.model != checked.model:
                raise ValueError("Embedding model identity does not match its vector")
        return outcome

    @classmethod
    def from_vector(cls, clip_id: str, values: Sequence[float]) -> "EmbeddingOutcome":
        from core.analysis_model_identity import DINOV2_DIMENSIONS, DINOV2_TAG

        if len(values) != DINOV2_DIMENSIONS or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in values
        ):
            raise ValueError("Invalid embedding vector dimensions or values")
        if not any(values):
            raise ValueError("Embedding is empty or its image could not be decoded")
        return cls(
            clip_id, "succeeded", tuple(float(v) for v in values), DINOV2_TAG
        )


class _EmbeddingModelSession:
    def __init__(self) -> None:
        self.acquired = False
        self.attempted = False
        self.failed = False

    def acquire(self, cancel: Event) -> bool:
        if self.failed:
            return False
        while not self.acquired and not cancel.is_set():
            self.acquired = _inference_lock.acquire(timeout=0.05)
        return self.acquired and not cancel.is_set()

    def close(self) -> None:
        if self.acquired:
            try:
                if self.attempted:
                    from core.analysis.embeddings import unload_model

                    unload_model()
            finally:
                self.acquired = False
                _inference_lock.release()


@contextmanager
def embedding_model_session() -> Iterator[_EmbeddingModelSession]:
    """Keep model ownership across durable batches without loading for cache hits."""
    session = _EmbeddingModelSession()
    try:
        yield session
    finally:
        session.close()


def run_embeddings(
    tasks: tuple[EmbeddingTask, ...],
    options: EmbeddingOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[EmbeddingOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    model_session: _EmbeddingModelSession | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
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
    session = model_session or _EmbeddingModelSession()
    fingerprints = fingerprints if fingerprints is not None else AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else embedding_runtime()
    identities: dict[str, AnalysisIdentity] = {}

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
            if task.inputs is not None:
                from core.jobs.media import FingerprintCancelled

                try:
                    identity = embedding_identity(task, fingerprints, runtime)
                    identities[task.clip_id] = identity
                    reused = reusable_embedding(task, identity) if task.skip else None
                    if reused is not None:
                        publish(reused)
                        continue
                except FingerprintCancelled:
                    break
                except (ValueError, OSError) as exc:
                    publish(EmbeddingOutcome(task.clip_id, "failed", code="invalid_analysis_input", message=str(exc)))
                    continue
            if task.skip and task.inputs is None:
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
            if session.acquire(cancel):
                from core.analysis.embeddings import extract_clip_embeddings_batch

                for start in range(0, len(pending), options.chunk_size):
                    if cancel.is_set():
                        break
                    chunk = pending[start : start + options.chunk_size]
                    try:
                        session.attempted = True
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
                        session.failed = True
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
                            if task.inputs is not None:
                                if not task.inputs.unchanged():
                                    raise ValueError("Embedding media changed during computation")
                                record = AnalysisRecord.success(
                                    identities[task.clip_id],
                                    {"embedding": list(outcome.vector), "embedding_model": outcome.model},
                                    input_snapshot=task.inputs.to_dict(),
                                )
                                outcome = replace(outcome, record_json=json.dumps(record.to_dict(), sort_keys=True))
                        except (TypeError, ValueError) as exc:
                            outcome = EmbeddingOutcome(
                                task.clip_id,
                                "failed",
                                code="embedding_failed",
                                message=str(exc),
                            )
                        publish(outcome)
    finally:
        if model_session is None:
            session.close()
    return tuple(
        outcomes.get(
            task.clip_id,
            EmbeddingOutcome(
                task.clip_id,
                "unprocessed",
                code="embedding_failed"
                if session.failed and not cancel.is_set()
                else "cancelled",
            ),
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
                clip.analysis_records.get("embeddings"),
            ),
        )

    def apply(self, project: "Project", outcome: EmbeddingOutcome) -> bool:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or (outcome.status != "succeeded" and not (outcome.status == "skipped" and outcome.record_json is not None))
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
            if outcome.record_json is not None:
                record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
                if (
                    record.identity is None or record.identity.operation != "embeddings"
                    or record.state != "succeeded" or task is None or task.inputs is None
                    or record.input_json != json.dumps(task.inputs.to_dict(), sort_keys=True, separators=(",", ":"))
                    or (record.artifact is None and record.value != {"embedding": list(checked.vector), "embedding_model": checked.model})
                ):
                    raise ValueError("Embedding record does not match its published projection")
            else:
                record = AnalysisRecord.legacy({"embedding": list(checked.vector), "embedding_model": checked.model})
            project.record_analysis("clip", outcome.clip_id, "embeddings", record)
            current[0].embedding = list(checked.vector)
            current[0].embedding_model = checked.model
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
