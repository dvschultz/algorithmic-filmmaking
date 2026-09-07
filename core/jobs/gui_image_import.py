"""Recover imported images before owner delivery and explicit project save."""

import json
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.jobs.image_import import (
    ImageImportRecord,
    image_import_target,
    image_import_runtime,
)
from core.operations.image_import import (
    ImageImportTask,
    ImageImportOutcome,
    image_import_task_inputs,
    run_image_import,
)
from core.project import Project


class GuiImageImportCache(GuiResultJournal):
    def __init__(self, project: Project, task: ImageImportTask, runtime: dict) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Image import recovery requires a saved project")
        self.batch_id = image_import_target(task)
        super().__init__(
            project.path,
            project.metadata.id,
            {self.batch_id: self.batch_id},
            project.metadata.job_results,
            kind="gui_import_images",
            arguments={"copy_files": task.copy_files},
            media_stamps={},
            target_id_field="batch_id",
        )
        self.task = task
        self.inputs_json = canonical_json(image_import_task_inputs(task))
        self.runtime_json = canonical_json(runtime)
        self.media_json: str | None = None
        self.recorded: ImageImportRecord | None = None

    def _media(self) -> dict:
        values: list[dict | None] = []
        for item in self.task.items:
            if item.error is not None or item.path.is_dir():
                values.append(None)
                continue
            if media_stamp(item.path) != item.media_stamp:
                raise StaleJobResult("Image import source changed while queued")
            try:
                values.append(
                    self.fingerprints.get(item.path) if item.path.is_file() else None
                )
            except OSError:
                # Keep unreadable-input errors per item, as the shared import does.
                values.append(None)
        return {"items": values}

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        if canonical_json(self._media()) != self.media_json:
            raise StaleJobResult("Image import media changed")
        if canonical_json(image_import_runtime()) != self.runtime_json:
            raise StaleJobResult("Image import runtime changed")

    def run(
        self, task: ImageImportTask, cancel: Event, progress: Callable[[int, int], None]
    ) -> ImageImportOutcome:
        try:
            if cancel.is_set():
                return ImageImportOutcome(task.request_id, "unprocessed")
            self.start(cancel)
            self.media_json = canonical_json(self._media())
            request, payload = self.prepare(
                self.batch_id,
                {
                    "task": json.loads(self.inputs_json),
                    "runtime": json.loads(self.runtime_json),
                    "media": json.loads(self.media_json),
                },
                None,
            )
            self.validate_media(request)
            if payload is None:
                outcome = run_image_import(task, cancel_event=cancel, progress=progress)
                if outcome.status != "succeeded" or cancel.is_set():
                    return (
                        ImageImportOutcome(task.request_id, "unprocessed")
                        if cancel.is_set()
                        else outcome
                    )
                payload = self.record(request, ImageImportRecord.build(task, outcome))
            recorded = ImageImportRecord.from_dict(payload)
            recovered = ImageImportTask.from_dict(recorded.task)
            if image_import_task_inputs(recovered) != image_import_task_inputs(task):
                raise StaleJobResult(
                    "Recovered image import inputs differ from request"
                )
            if cancel.is_set():
                return ImageImportOutcome(task.request_id, "unprocessed")
            self.recorded = recorded
            return ImageImportOutcome.from_dict(recorded.outcome)
        except FingerprintCancelled:
            cancel.set()
            return ImageImportOutcome(task.request_id, "unprocessed")
        finally:
            if hasattr(self, "store"):
                self.store.close()
