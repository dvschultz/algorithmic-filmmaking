"""Recoverable embedding prerequisites for detached GUI sequence proposals."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Literal

from core.jobs import JobRuntime, JobStore
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingTask,
    BoundaryEmbeddingOutcome,
    run_boundary_embeddings,
)
from core.operations.embeddings import (
    EmbeddingOptions,
    EmbeddingOutcome,
    embedding_task,
    embedding_identity,
    reusable_embedding,
    embedding_model_session,
    run_embeddings,
)
from models.clip import Clip, Source
from models.analysis_record import AnalysisRecord
from core.analysis_records import AnalysisFingerprints

if TYPE_CHECKING:
    from core.project import Project


def _target(clip: Clip, source: Source, mode: str) -> dict:
    return {
        "clip_id": clip.id,
        "clip_source_id": clip.source_id,
        "source_id": source.id,
        "source_path": str(source.file_path),
        "fps": source.fps,
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "thumbnail_path": str(clip.thumbnail_path) if clip.thumbnail_path else None,
        "previous": {"vector": clip.embedding}
        if mode == "thumbnail"
        else {"first": clip.first_frame_embedding, "last": clip.last_frame_embedding},
        "model": clip.embedding_model,
        "record": clip.analysis_records["embeddings"].to_dict() if mode == "thumbnail" and "embeddings" in clip.analysis_records else None,
        "skip": False
        if mode == "thumbnail"
        else clip.first_frame_embedding is not None
        and clip.last_frame_embedding is not None,
    }


class SequenceEmbeddingJob:
    """Capture on the project owner; compute and enrich private copies off-thread."""

    def __init__(
        self,
        clips: list[tuple[Clip, Source]],
        *,
        mode: Literal["thumbnail", "boundary"],
        project: "Project | None" = None,
    ) -> None:
        if project is not None:
            project.session.assert_owner()
        self.mode = mode
        self.thumbnail_tasks = tuple(replace(embedding_task(clip, source), clip_id=str(i)) for i, (clip, source) in enumerate(clips)) if mode == "thumbnail" else ()
        self.path = (
            project.path.expanduser().resolve()
            if project is not None and project.path
            else None
        )
        self.project_id = project.metadata.id if project is not None else None
        targets = []
        paths = set()
        for clip, source in clips:
            skip = (
                False
                if mode == "thumbnail"
                else clip.first_frame_embedding is not None
                and clip.last_frame_embedding is not None
            )
            targets.append(_target(clip, source, mode))
            if not skip:
                paths.add(source.file_path)
                if mode == "thumbnail" and clip.thumbnail_path:
                    paths.add(clip.thumbnail_path)
        self.targets_json = json.dumps(targets, sort_keys=True, allow_nan=False)
        self.stamps = {path: media_stamp(path) for path in paths}
        self.runtime_identity = self._runtime_identity()
        self.operation = OperationSpec.build(
            kind="gui_sequence_embeddings",
            version=1,
            arguments={
                "mode": mode,
                **({"project_path": str(self.path)} if self.path else {}),
            },
            inputs={"targets": targets, "runtime": self.runtime_identity},
            persistence="job_history" if self.path else "session_only",
            publication="owner_thread" if self.path else "worker",
            session_id=project.session.session_id if project is not None else None,
            input_revision=str(project.mutation_generation)
            if project is not None
            else None,
        )
        self.task_id: str | None = None
        self.status: str | None = None

    def _runtime_identity(self) -> dict:
        if self.mode == "boundary":
            from core.jobs.boundary_embeddings import _runtime
        else:
            from core.jobs.embeddings import _runtime
        return _runtime()

    def _check_media(self, paths: set[Path] | None = None) -> None:
        if self._runtime_identity() != self.runtime_identity or any(
            media_stamp(path) != self.stamps[path]
            for path in (self.stamps if paths is None else paths)
        ):
            raise StaleJobResult(
                "Sequencing embedding inputs changed during computation"
            )

    def _compute(self, progress, cancel: Event) -> dict:
        from core.feature_registry import check_feature

        targets = json.loads(self.targets_json)
        outcomes = {}
        requests = {}
        pending = []
        journal = None
        failed = False
        try:
            self._check_media()
            if self.path is not None:
                assert self.project_id is not None
                journal = GuiResultJournal(
                    self.path,
                    self.project_id,
                    {str(i): target["source_id"] for i, target in enumerate(targets)},
                    {},
                    kind="sequence_embedding_prerequisites",
                    arguments={"mode": self.mode},
                    media_stamps=self.stamps,
                )
                journal.start(cancel)
            fingerprints = AnalysisFingerprints(cancel, media_fingerprints=journal.fingerprints if journal is not None else None)
            for index, target in enumerate(targets):
                if cancel.is_set():
                    break
                cid = str(index)
                if self.mode == "thumbnail":
                    task = self.thumbnail_tasks[index]
                    if task.inputs is not None and task.inputs.unchanged():
                        identity = embedding_identity(task, fingerprints, self.runtime_identity)
                        reused = reusable_embedding(task, identity)
                        if reused is not None:
                            outcomes[cid] = asdict(reused)
                            continue
                if target["skip"]:
                    outcomes[cid] = {"clip_id": cid, "status": "skipped"}
                    continue
                media_value = (
                    target["thumbnail_path"]
                    if self.mode == "thumbnail"
                    else target["source_path"]
                )
                if not media_value or not Path(media_value).is_file():
                    outcomes[cid] = {
                        "clip_id": cid,
                        "status": "failed",
                        "code": "thumbnail_missing"
                        if self.mode == "thumbnail"
                        else "source_missing",
                    }
                    continue
                payload = None
                if journal is not None:
                    source = Path(target["source_path"])
                    media = (
                        (
                            Path(target["thumbnail_path"])
                            if target["thumbnail_path"]
                            else None
                        )
                        if self.mode == "thumbnail"
                        else source
                    )
                    requests[cid], payload = journal.prepare(
                        cid,
                        {
                            "target": target,
                            "runtime": self.runtime_identity,
                            "source_media": journal.fingerprints.get(source),
                        },
                        media,
                    )
                if payload is not None:
                    self._validate_payload(payload)
                    outcomes[cid] = payload
                else:
                    pending.append((cid, target))
            if pending and not cancel.is_set():
                available, missing = check_feature("embeddings")
                if not available:
                    raise RuntimeError(
                        f"DINOv2 embeddings require torch and transformers. Missing: {', '.join(missing)}"
                    )
                self._check_media()
                chunk_size = 16 if self.mode == "thumbnail" else 1
                with embedding_model_session() as session:
                    for start in range(0, len(pending), chunk_size):
                        if cancel.is_set() or session.failed:
                            failed = session.failed
                            break
                        chunk = pending[start : start + chunk_size]
                        chunk_paths = {Path(t["source_path"]) for _, t in chunk}
                        if self.mode == "thumbnail":
                            chunk_paths.update(
                                Path(t["thumbnail_path"])
                                for _, t in chunk
                                if t["thumbnail_path"]
                            )
                        self._check_media(chunk_paths)
                        computed: tuple[
                            EmbeddingOutcome | BoundaryEmbeddingOutcome, ...
                        ]
                        if self.mode == "thumbnail":
                            tasks = tuple(self.thumbnail_tasks[int(cid)] for cid, _ in chunk)
                            computed = run_embeddings(
                                tasks,
                                EmbeddingOptions(),
                                cancel_event=cancel,
                                model_session=session,
                                fingerprints=fingerprints,
                                runtime=self.runtime_identity,
                            )
                        else:
                            boundary_tasks = tuple(
                                BoundaryEmbeddingTask(
                                    cid,
                                    Path(t["source_path"]),
                                    t["start_frame"],
                                    t["end_frame"],
                                    t["fps"],
                                )
                                for cid, t in chunk
                            )
                            computed = run_boundary_embeddings(
                                boundary_tasks,
                                cancel_event=cancel,
                                model_session=session,
                            )
                        self._check_media(chunk_paths)
                        # Finish journaling this batch before starting another.
                        for outcome in computed:
                            if journal is not None and outcome.status == "succeeded":
                                journal.record(requests[outcome.clip_id], outcome)
                            outcomes[outcome.clip_id] = asdict(outcome)
            self._check_media()
            progress(1.0, "Sequencing embeddings ready")
            return {
                "outcomes": [
                    outcomes.get(
                        str(i),
                        {
                            "clip_id": str(i),
                            "status": "unprocessed",
                            "code": "embedding_failed" if failed else "cancelled",
                        },
                    )
                    for i in range(len(targets))
                ]
            }
        finally:
            if journal is not None and hasattr(journal, "store"):
                journal.store.close()

    def _validate_payload(self, payload: dict) -> None:
        if payload["status"] != "succeeded" and not (self.mode == "thumbnail" and payload["status"] == "skipped" and payload.get("record_json") is not None):
            return
        if self.mode == "thumbnail":
            EmbeddingOutcome.from_dict(payload)
        else:
            first = EmbeddingOutcome.from_vector(payload["clip_id"], payload["first"])
            EmbeddingOutcome.from_vector(payload["clip_id"], payload["last"])
            if payload["model"] != first.model:
                raise ValueError("Boundary embedding model identity does not match")

    def _check_copies(self, clips: list[tuple[Clip, Source]]) -> None:
        current = json.dumps(
            [_target(c, s, self.mode) for c, s in clips],
            sort_keys=True,
            allow_nan=False,
        )
        if current != self.targets_json:
            raise StaleJobResult("Sequencing embedding targets changed")

    def validate_project(self, project: "Project | None") -> None:
        """Revalidate the live project on its owner before publishing a proposal."""
        if self.project_id is None:
            return
        if (
            project is None
            or project.metadata.id != self.project_id
            or project.session.session_id != self.operation.session_id
        ):
            raise StaleJobResult("Sequencing project session changed")
        project.session.assert_owner()
        current_path = project.path.expanduser().resolve() if project.path else None
        if current_path != self.path:
            raise StaleJobResult("Sequencing project save path changed")
        pairs = []
        for target in json.loads(self.targets_json):
            clip = project.clips_by_id.get(target["clip_id"])
            source = project.sources_by_id.get(target["source_id"])
            if clip is None or source is None:
                raise StaleJobResult("Sequencing target no longer exists")
            pairs.append((clip, source))
        self._check_copies(pairs)
        self._check_media()

    def populate(
        self,
        clips: list[tuple[Clip, Source]],
        cancel: Event,
        *,
        require_all: bool = False,
    ) -> None:
        """Run off the GUI thread; publish only into the caller's private copies."""
        from core.settings import load_settings

        if cancel.is_set():
            return
        self._check_copies(clips)
        if self.path is None:
            runtime = JobRuntime.for_session(max_workers=1)
        else:
            store = JobStore(load_settings().cache_dir / "jobs.db")
            store.mark_running_jobs_as_crashed()
            runtime = JobRuntime(store, max_workers=1)
        try:
            submitted = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=self._compute,
                cancellation_event=cancel,
                project_path=self.path,
            )
            self.task_id = submitted["task_id"]
            runtime.shutdown()
            row = runtime.store.get(self.task_id)
            self.status = row.status
            if cancel.is_set() or row.status == "cancelled":
                return
            if row.status != "completed":
                raise RuntimeError(
                    row.error or "Sequencing embedding prerequisites failed"
                )
            self._check_media()
            self._check_copies(clips)
            payloads = (row.result or {}).get("outcomes", [])
            if len(payloads) != len(clips):
                raise ValueError(
                    "Sequencing embedding result count does not match inputs"
                )
            for index, payload in enumerate(payloads):
                if payload["clip_id"] != str(index):
                    raise ValueError(
                        "Sequencing embedding result target does not match"
                    )
                self._validate_payload(payload)
            for (clip, _), payload in zip(clips, payloads):
                if cancel.is_set():
                    return
                if payload["status"] != "succeeded" and not (self.mode == "thumbnail" and payload["status"] == "skipped" and payload.get("record_json") is not None):
                    if self.mode == "thumbnail":
                        clip.embedding = None
                    continue
                if self.mode == "thumbnail":
                    clip.embedding = list(payload["vector"])
                    if payload.get("record_json") is not None:
                        clip.analysis_records["embeddings"] = AnalysisRecord.from_dict(json.loads(payload["record_json"]))
                else:
                    clip.first_frame_embedding = list(payload["first"])
                    clip.last_frame_embedding = list(payload["last"])
                clip.embedding_model = payload["model"]
            missing = sum(clip.embedding is None for clip, _ in clips)
            if require_all and missing:
                raise RuntimeError(
                    f"Missing DINOv2 embeddings for {missing} clips. Run embedding analysis first or ensure thumbnails exist before generating Staccato."
                )
        finally:
            runtime.shutdown()
            runtime.store.close()
