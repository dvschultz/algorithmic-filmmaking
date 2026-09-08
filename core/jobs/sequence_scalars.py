"""Recoverable scalar prerequisites for detached GUI sequencing inputs."""

import json
from dataclasses import replace
from threading import Event
from typing import TYPE_CHECKING

from core.analysis_records import AnalysisSnapshot
from core.jobs.commits import StaleJobResult
from core.jobs.gui_scalars import GuiScalarCache
from core.operations.scalars import (
    FIELDS, ScalarOperation, ScalarOutcome, scalar_task, scalar_runtime, run_scalars,
)
from models.analysis_record import AnalysisRecord
from models.clip import Clip, Source

if TYPE_CHECKING:
    from core.project import Project


class SequenceScalarJob:
    """Freeze on the owner, compute on a worker, validate before sequence delivery."""

    def __init__(
        self, clips: list[tuple[Clip, Source]], *, operation: ScalarOperation,
        project: "Project | None" = None,
    ) -> None:
        if project is not None:
            project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id if project else None
        self.path = project.path.expanduser().resolve() if project and project.path else None
        self.operation = operation
        self.runtime = scalar_runtime(operation)
        self.originals = tuple(clips)
        self.tasks = tuple(
            replace(scalar_task(clip, source, operation), clip_id=str(index))
            for index, (clip, source) in enumerate(clips)
        )
        self.cache = None
        self.outcomes: tuple[ScalarOutcome, ...] = ()
        if self.path is not None and project is not None:
            stamps = {
                path: stamp for task in self.tasks
                for _, path, stamp in AnalysisSnapshot.from_json(task.snapshot_json).inputs.files
            }
            self.cache = GuiScalarCache(
                self.path, project.metadata.id,
                {str(index): clip.source_id for index, (clip, _) in enumerate(clips)},
                project.metadata.job_results, operation=operation, media_stamps=stamps,
            )

    def _check_inputs(self, clips: list[tuple[Clip, Source]]) -> None:
        tasks = tuple(
            replace(scalar_task(clip, source, self.operation), clip_id=str(index))
            for index, (clip, source) in enumerate(clips)
        )
        if tasks != self.tasks or scalar_runtime(self.operation) != self.runtime:
            raise StaleJobResult("Sequencing scalar inputs changed")

    def validate_project(self, project: "Project | None") -> None:
        """Only stat inputs here; full content verification runs off the GUI thread."""
        if project is not self.project:
            raise StaleJobResult("Sequencing project changed")
        if project is not None:
            project.session.assert_owner()
            path = project.path.expanduser().resolve() if project.path else None
            if path != self.path or project.session.session_id != self.session_id:
                raise StaleJobResult("Sequencing project session or path changed")
            for clip, source in self.originals:
                if (
                    project.clips_by_id.get(clip.id) is not clip
                    or project.sources_by_id.get(source.id) is not source
                ):
                    raise StaleJobResult("Sequencing targets were replaced")
        self._check_inputs(list(self.originals))

    def populate(self, clips: list[tuple[Clip, Source]], cancel: Event) -> None:
        """Enrich only detached copies; successful computations survive interruption."""
        if cancel.is_set():
            return
        self._check_inputs(clips)
        if self.cache is None:
            outcomes = run_scalars(self.tasks, cancel_event=cancel)
        else:
            outcomes = self.cache.run(self.tasks, cancel, lambda _: None, lambda *_: None)
        self.outcomes = outcomes
        if cancel.is_set():
            return
        self._check_inputs(clips)
        for (clip, _), outcome in zip(clips, outcomes, strict=True):
            if not outcome.has_result or outcome.record_json is None:
                raise RuntimeError(f"{self.operation.capitalize()} analysis failed: {outcome.message}")
            record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
            setattr(clip, FIELDS[self.operation], record.value[FIELDS[self.operation]])
            clip.analysis_records[self.operation] = record
