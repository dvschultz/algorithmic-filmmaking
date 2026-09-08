"""Detached first/last-frame embeddings with shared DINOv2 model ownership."""

from dataclasses import dataclass, replace
import json
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from threading import Event
from typing import Any, TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import boundary_embedding_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

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
    analysis_json: str | None = None


def boundary_embedding_task(
    clip: Any, source: Any = None, *, skip_existing: bool = True
) -> BoundaryEmbeddingTask:
    path = Path(source.file_path) if source is not None else None
    fps = source.fps if source is not None else 0.0
    snapshot = AnalysisSnapshot.capture(
        clip,
        "boundary_embeddings",
        {"video": path} if path is not None else {},
        {"start_frame": clip.start_frame, "end_frame": clip.end_frame, "fps": fps},
        {
            "first_frame_embedding": clip.first_frame_embedding,
            "last_frame_embedding": clip.last_frame_embedding,
            "embedding_model": clip.embedding_model,
        },
    )
    return BoundaryEmbeddingTask(
        clip.id,
        path,
        clip.start_frame,
        clip.end_frame,
        fps,
        skip_existing,
        snapshot.to_json(),
    )


def boundary_embedding_identity(
    snapshot: AnalysisSnapshot, fingerprints: AnalysisFingerprints, runtime: dict
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="boundary_embeddings",
        operation_version=2,
        model=runtime,
        parameters={},
        sampling={
            "policy": "boundary-start/end-minus-one-v1",
            "processor": "dinov2-default",
        },
    )


@dataclass(frozen=True)
class BoundaryEmbeddingOutcome:
    clip_id: str
    status: OutcomeStatus
    first: tuple[float, ...] = ()
    last: tuple[float, ...] = ()
    model: str | None = None
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
    def from_dict(cls, payload: dict) -> "BoundaryEmbeddingOutcome":
        outcome = cls(**{**payload, "first": tuple(payload.get("first", ())), "last": tuple(payload.get("last", ()))})
        if outcome.has_result:
            first = EmbeddingOutcome.from_vector(outcome.clip_id, outcome.first)
            last = EmbeddingOutcome.from_vector(outcome.clip_id, outcome.last)
            if outcome.model != first.model:
                raise ValueError("Boundary embedding model does not match")
            return replace(
                outcome, first=first.vector, last=last.vector, model=first.model
            )
        if outcome.status not in ("failed", "skipped", "unprocessed"):
            raise ValueError("Invalid boundary embedding status")
        return outcome


def run_boundary_embeddings(
    tasks: tuple[BoundaryEmbeddingTask, ...],
    *,
    cancel_event: Event | None = None,
    model_session: _EmbeddingModelSession | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    runtime: dict | None = None,
) -> tuple[BoundaryEmbeddingOutcome, ...]:
    """Validate each pair atomically and reject changed media or late cancellation."""
    from core.errors import ModelDownloadError
    from core.jobs.media import media_stamp

    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    runtime = runtime if runtime is not None else boundary_embedding_runtime()
    stamps = [
        media_stamp(task.source_path) if task.source_path else None for task in tasks
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
            if task.skip and task.analysis_json is None:
                outcomes.append(
                    BoundaryEmbeddingOutcome(
                        task.clip_id, "skipped", code="already_populated"
                    )
                )
                continue
            snapshot = None
            identity = None
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
                if task.analysis_json is not None:
                    snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                    identity = boundary_embedding_identity(
                        snapshot, fingerprints, runtime
                    )
                    reused = snapshot.reusable_record(identity) if task.skip else None
                    if reused is not None:
                        value = reused.value
                        reused_first = EmbeddingOutcome.from_vector(
                            task.clip_id, value["first_frame_embedding"]
                        )
                        reused_last = EmbeddingOutcome.from_vector(
                            task.clip_id, value["last_frame_embedding"]
                        )
                        if value["embedding_model"] != reused_first.model:
                            raise ValueError("Boundary embedding model does not match")
                        outcomes.append(
                            BoundaryEmbeddingOutcome(
                                task.clip_id,
                                "skipped",
                                reused_first.vector,
                                reused_last.vector,
                                reused_first.model,
                                code="valid_analysis",
                                record_json=json.dumps(
                                    reused.to_dict(), sort_keys=True
                                ),
                            )
                        )
                        continue
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
                if snapshot is not None and identity is not None:
                    record = AnalysisRecord.success(
                        identity,
                        {
                            "first_frame_embedding": list(outcome.first),
                            "last_frame_embedding": list(outcome.last),
                            "embedding_model": outcome.model,
                        },
                        input_snapshot=snapshot.inputs.to_dict(),
                    )
                    outcome = replace(
                        outcome,
                        record_json=json.dumps(record.to_dict(), sort_keys=True),
                    )
            except Exception as exc:
                if cancel.is_set():
                    break
                if isinstance(exc, ModelDownloadError):
                    session.failed = True
                outcome = BoundaryEmbeddingOutcome(
                    task.clip_id, "failed", code="embedding_failed", message=str(exc)
                )
                if (
                    snapshot is not None
                    and identity is not None
                    and snapshot.inputs.unchanged()
                ):
                    record = replace(
                        AnalysisRecord.failure(identity, str(exc)),
                        input_json=json.dumps(
                            snapshot.inputs.to_dict(), sort_keys=True
                        ),
                    )
                    outcome = replace(
                        outcome,
                        record_json=json.dumps(record.to_dict(), sort_keys=True),
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
        if outcome.can_apply
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
        self.path = project.path
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
                clip.analysis_records.get("boundary_embeddings"),
            ),
        )

    def apply(self, project: "Project", outcome: BoundaryEmbeddingOutcome) -> bool:
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
            from dataclasses import asdict

            checked = BoundaryEmbeddingOutcome.from_dict(asdict(outcome))
            value = {
                "first_frame_embedding": list(checked.first),
                "last_frame_embedding": list(checked.last),
                "embedding_model": checked.model,
            }
            record = (
                AnalysisRecord.from_dict(json.loads(checked.record_json))
                if checked.record_json is not None
                else AnalysisRecord.legacy(value)
            )
            if checked.record_json is not None:
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task is not None and task.analysis_json
                    else None
                )
                if (
                    snapshot is None
                    or record.identity is None
                    or record.identity.operation != "boundary_embeddings"
                    or record.identity.to_dict()["parameters"] != {}
                    or json.loads(record.input_json or "null")
                    != snapshot.inputs.to_dict()
                    or (
                        checked.has_result
                        and (record.state != "succeeded" or record.value != value)
                    )
                    or (checked.status == "failed" and record.state != "failed")
                ):
                    return False
            if checked.has_result:
                validate_boundary_model(current[0], checked.model)
            project.record_analysis(
                "clip", checked.clip_id, "boundary_embeddings", record
            )
            if checked.status == "failed":
                return True
            current[0].first_frame_embedding = list(checked.first)
            current[0].last_frame_embedding = list(checked.last)
            current[0].embedding_model = checked.model
            project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
