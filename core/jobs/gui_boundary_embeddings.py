"""Recover boundary pairs before explicit desktop project saves."""

import json
from threading import Event
from typing import Callable

from core.jobs.boundary_embeddings import _runtime, _target, _values
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingTask,
    BoundaryEmbeddingOutcome,
    run_boundary_embeddings,
)
from core.operations.embeddings import embedding_model_session
from core.project import Project


class GuiBoundaryEmbeddingCache(GuiResultJournal):
    def __init__(
        self, project: Project, tasks: tuple[BoundaryEmbeddingTask, ...]
    ) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Boundary recovery requires a saved project")
        super().__init__(
            project.path,
            project.metadata.id,
            {
                task.clip_id: project.clips_by_id[task.clip_id].source_id
                for task in tasks
            },
            project.metadata.job_results,
            kind="gui_boundary_embeddings",
            arguments={},
            media_stamps={
                task.source_path: media_stamp(task.source_path)
                for task in tasks
                if task.source_path and not task.skip
            },
        )
        self.runtime = _runtime()
        self.targets_json = json.dumps(
            {
                task.clip_id: {
                    **_target(project, task.clip_id),
                    "previous": _values(project, task.clip_id),
                    "runtime": self.runtime,
                }
                for task in tasks
            },
            sort_keys=True,
            allow_nan=False,
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        if _runtime() != self.runtime:
            raise StaleJobResult("Boundary embedding runtime changed")

    def run(
        self,
        tasks: tuple[BoundaryEmbeddingTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[BoundaryEmbeddingOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[BoundaryEmbeddingOutcome, ...]:
        outcomes = {}
        targets = json.loads(self.targets_json)
        failed = False
        try:
            self.start(cancel)
            with embedding_model_session() as session:
                for index, task in enumerate(tasks):
                    if cancel.is_set() or session.failed:
                        failed = session.failed
                        break
                    if (
                        task.skip
                        or task.source_path is None
                        or not task.source_path.is_file()
                    ):
                        outcome = run_boundary_embeddings(
                            (task,), cancel_event=cancel, model_session=session
                        )[0]
                    else:
                        request, payload = self.prepare(
                            task.clip_id, targets[task.clip_id], task.source_path
                        )
                        self.validate_media(request)
                        if payload is None:
                            if not prepare():
                                cancel.set()
                                break
                            self.validate_media(request)
                            outcome = run_boundary_embeddings(
                                (task,), cancel_event=cancel, model_session=session
                            )[0]
                        else:
                            outcome = BoundaryEmbeddingOutcome.from_dict(payload)
                        if outcome.status == "succeeded" and not cancel.is_set():
                            self.record(request, outcome)
                    if cancel.is_set():
                        break
                    outcomes[task.clip_id] = outcome
                    deliver(outcome)
                    progress(index + 1, len(tasks))
        except FingerprintCancelled:
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        return tuple(
            outcomes.get(
                task.clip_id,
                BoundaryEmbeddingOutcome(
                    task.clip_id,
                    "unprocessed",
                    code="embedding_failed" if failed else "cancelled",
                ),
            )
            for task in tasks
        )
