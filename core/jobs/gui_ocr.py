"""Journal clip/frame OCR without saving unrelated desktop edits."""

from dataclasses import asdict
import json
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultReceipt, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.jobs.ocr import _runtime, _task_data
from core.operations.ocr import OcrOptions, OcrOutcome, OcrTask, run_ocr
from core.project import Project


class _OcrJournal(GuiResultJournal):
    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        captured = json.loads(request.spec.identity_json)["inputs"]["task"]["runtime"]
        if _runtime() != captured:
            raise StaleJobResult("OCR runtime changed")


class GuiOcrCache:
    def __init__(
        self, project: Project, tasks: tuple[OcrTask, ...], options: OcrOptions
    ) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Durable OCR requires a saved project")
        self.path = project.path.expanduser().resolve()
        self.options = options
        self.runtime = _runtime()
        self.results: dict[tuple[str, str], GuiResultReceipt] = {}
        self.journals: dict[str, _OcrJournal] = {}
        self.previous: dict[tuple[str, str], str] = {}
        # Separate namespaces preserve clip/frame identity even when IDs collide.
        for kind in dict.fromkeys(task.target_type for task in tasks):
            selected = [task for task in tasks if task.target_type == kind]
            targets = project.frames_by_id if kind == "frame" else project.clips_by_id
            source_ids = {}
            for task in selected:
                target = targets[task.clip_id]
                source_ids[task.clip_id] = target.source_id or ""
                self.previous[task.key] = json.dumps(
                    [text.to_dict() for text in target.extracted_texts]
                    if target.extracted_texts is not None
                    else None,
                    sort_keys=True,
                    allow_nan=False,
                )
            self.journals[kind] = _OcrJournal(
                self.path,
                project.metadata.id,
                source_ids,
                project.metadata.job_results,
                kind=f"gui_ocr_{kind}",
                arguments=asdict(options),
                media_stamps={
                    task.path: media_stamp(task.path) for task in selected if task.path
                },
            )

    def run(
        self,
        tasks: tuple[OcrTask, ...],
        cancel: Event,
        deliver: Callable[[OcrOutcome], None],
        progress: Callable[[int, int, str], None],
    ) -> tuple[OcrOutcome, ...]:
        outcomes: dict[tuple[str, str], OcrOutcome] = {}
        model_failed = False
        try:
            for journal in self.journals.values():
                journal.start(cancel)
            for index, task in enumerate(tasks):
                if cancel.is_set():
                    break
                progress(index, len(tasks), task.clip_id)
                if cancel.is_set():
                    break
                journal = self.journals[task.target_type]
                if model_failed:
                    outcome = OcrOutcome(
                        task.clip_id,
                        "unprocessed",
                        task.target_type,
                        code="model_unavailable",
                    )
                elif task.skip or task.path is None or not task.path.is_file():
                    outcome = run_ocr((task,), self.options, cancel_event=cancel)[0]
                else:
                    data = {
                        **_task_data(task),
                        "runtime": self.runtime,
                        "previous_texts": json.loads(self.previous[task.key]),
                    }
                    request, payload = journal.prepare(task.clip_id, data, task.path)
                    journal.validate_media(request)
                    outcome = (
                        OcrOutcome.from_dict(payload)
                        if payload is not None
                        else run_ocr((task,), self.options, cancel_event=cancel)[0]
                    )
                    if (
                        outcome.clip_id != task.clip_id
                        or outcome.target_type != task.target_type
                    ):
                        raise StaleJobResult("OCR cache target identity changed")
                    if outcome.status == "succeeded" and not cancel.is_set():
                        journal.record(request, outcome)
                        self.results[task.key] = journal.results[task.clip_id]
                if cancel.is_set():
                    break
                model_failed = model_failed or outcome.code == "model_load_failed"
                outcomes[task.key] = outcome
                deliver(outcome)
                progress(index + 1, len(tasks), task.clip_id)
        except FingerprintCancelled:
            pass
        finally:
            for journal in self.journals.values():
                if hasattr(journal, "store"):
                    journal.store.close()
        return tuple(
            outcomes.get(
                task.key,
                OcrOutcome(
                    task.clip_id, "unprocessed", task.target_type, code="cancelled"
                ),
            )
            for task in tasks
        )
