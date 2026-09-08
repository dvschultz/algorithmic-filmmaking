"""Qt adapter for shared OCR computation on detached clip/frame inputs."""

from typing import Optional, TYPE_CHECKING
from dataclasses import asdict
from queue import Empty, Queue

from PySide6.QtCore import Signal

from core.operations.ocr import OcrOptions, OcrOutcome, ocr_task, run_ocr
from core.jobs import JobRuntime
from core.jobs.gui_ocr import GuiOcrCache
from core.jobs.ocr import _runtime, _task_data, resolve_options
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from core.project import Project


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    return summarize_clip_errors(errors, operation_label="Text extraction")


class TextExtractionWorker(CancellableWorker):
    """Preserve legacy signals while publishing typed outcomes to the owner."""

    progress = Signal(int, int, str)
    clip_completed = Signal(str, list)
    extraction_completed = Signal(dict)
    outcome_ready = Signal(object)

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        num_keyframes: int = 3,
        use_vlm_fallback: bool = True,
        vlm_model: Optional[str] = None,
        vlm_only: bool = False,
        use_text_detection: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
        *,
        project: "Project | None" = None,
        options: OcrOptions | None = None,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(parent)
        self.options = resolve_options(
            options
            or OcrOptions(
                min(max(1, num_keyframes), 5),
                use_vlm_fallback,
                vlm_model,
                vlm_only,
                use_text_detection,
            )
        )
        if analysis_targets:
            self.tasks = tuple(
                ocr_task(target, skip_existing=skip_existing)
                for target in analysis_targets
            )
        else:
            self.tasks = tuple(
                ocr_task(
                    clip, sources_by_id.get(clip.source_id), skip_existing=skip_existing
                )
                for clip in clips
            )
        self.result: tuple[OcrOutcome, ...] = ()
        self._media_stamps = {
            task.path: media_stamp(task.path) for task in self.tasks if task.path
        }
        self._runtime_identity = _runtime()
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="ocr",
                version=2,
                arguments={"targets": [list(task.key) for task in self.tasks]},
                inputs={
                    "tasks": [_task_data(task) for task in self.tasks],
                    "options": asdict(self.options),
                    "runtime": self._runtime_identity,
                },
                persistence="session_only",
                session_id=project.session.session_id if project is not None else None,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.cache = (
            GuiOcrCache(project, self.tasks, self.options)
            if project is not None and project.path is not None
            else None
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        self._log_start()
        results = {}
        errors: list[tuple[str, str]] = []

        def emit_outcome(outcome: OcrOutcome) -> None:
            self.outcome_ready.emit(outcome)
            if outcome.has_result:
                results[outcome.clip_id] = outcome.to_models()
                self.clip_completed.emit(outcome.clip_id, outcome.to_models())
            elif outcome.status == "failed":
                errors.append(
                    (outcome.clip_id, outcome.message or outcome.code or "OCR failed")
                )
                results[outcome.clip_id] = []

        events: Queue = Queue()
        runtime = None

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[(outcome.target_type, outcome.clip_id)] = outcome
                events.put(("outcome", outcome))

            def report(current, total, clip_id):
                progress(current / total if total else 1.0, f"OCR ({current}/{total})")
                events.put(("progress", (current, total, clip_id)))

            try:
                if _runtime() != self._runtime_identity or any(
                    media_stamp(path) != stamp
                    for path, stamp in self._media_stamps.items()
                ):
                    raise RuntimeError("OCR media or runtime changed while queued")
                outcomes = (
                    self.cache.run(self.tasks, cancel, deliver, report)
                    if self.cache is not None
                    else run_ocr(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.key not in collected:
                        deliver(
                            OcrOutcome(
                                task.clip_id,
                                "failed",
                                task.target_type,
                                code="ocr_failed",
                                message=str(exc),
                            )
                        )
                return {
                    "success": False,
                    "error": str(exc),
                    "outcomes": [asdict(collected[task.key]) for task in self.tasks],
                }

        def relay(event):
            if self.is_cancelled():
                return
            kind, value = event
            if kind == "progress":
                self.progress.emit(*value)
            else:
                emit_outcome(value)

        try:
            runtime = gui_job_runtime(self.operation)
            self._runtime = runtime
            submission = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
                project_path=self.operation.arguments.get("project_path"),
            )
            self.task_id = submission["task_id"]
            while runtime.is_handle_live(self.task_id):
                try:
                    relay(events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                relay(events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            self.result = tuple(
                OcrOutcome.from_dict(value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    OcrOutcome(
                        task.clip_id, "unprocessed", task.target_type, code="cancelled"
                    )
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "OCR failed")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if runtime is not None:
                close_gui_job_runtime(runtime)
                runtime.store.close()
            self._runtime = None
        if not self.is_cancelled():
            if errors:
                self.error.emit(_summarize_errors(errors))
            self.extraction_completed.emit(results)
            self._log_complete()
        else:
            self._log_cancelled()
