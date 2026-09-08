"""Run-owned clip analysis; publish and advance after native thread exit."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from core.analysis_availability import operation_is_complete_for_clip
from core.operations.analysis_inputs import clip_input
from core.operations.clip_analysis import ClipAnalysisPlan
from core.operations.contracts import OutcomeStatus
from ui.workers.clip_analysis_work import create_clip_analysis_worker


WORKER_ATTRIBUTES = {
    "colors": "color_worker",
    "shots": "shot_type_worker",
    "classify": "classification_worker",
    "detect_objects": "detection_worker_yolo",
    "extract_text": "text_extraction_worker",
    "transcribe": "transcription_worker",
    "face_embeddings": "face_detection_worker",
    "gaze": "_gaze_worker",
    "embeddings": "_embeddings_worker",
    "boundary_embeddings": "_boundary_embeddings_worker",
    "describe": "description_worker",
    "cinematography": "cinematography_worker",
    "custom_query": "custom_query_worker",
}


class ClipAnalysisController(QObject):
    progress = Signal(object, int, str)
    status = Signal(object, str)
    completed = Signal(object, object)

    def __init__(
        self,
        window: Any,
        clips: list,
        operations: list[str],
        *,
        force_rerun: bool = False,
        query: str | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.project = window.project
        self.project.session.assert_owner()
        self.session_id = self.project.session.session_id
        self.path = self.project.path
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.settings = deepcopy(window.settings)
        self.plan = ClipAnalysisPlan((c.id for c in clips), operations)
        self.clips = {c.id: c for c in clips}
        self.sources = tuple(self.project.sources_by_id.get(c.source_id) for c in clips)
        self.inputs = {
            cid: clip_input(self.project, c) for cid, c in self.clips.items()
        }
        self.force_rerun = force_rerun
        self.query = query
        self.workers: dict[str, Any] = {}
        self.applications: dict[str, Any] = {}
        self.outcomes: dict[str, dict[str, OutcomeStatus]] = {}
        self.errors: list[str] = []
        self._transcription_batches: list[list] = []
        self._started = False
        self.finished = False
        self._advance_timer = QTimer(self)
        self._advance_timer.setSingleShot(True)
        self._advance_timer.timeout.connect(self._advance)
        self._transcription_timer = QTimer(self)
        self._transcription_timer.setSingleShot(True)
        self._transcription_timer.timeout.connect(self._next_transcription)

    def is_current(self) -> bool:
        return (
            self.owns_view()
            and self.project.path == self.path
            and (self.reply is None or self.reply.is_current(self.window))
        )

    def owns_view(self) -> bool:
        """The retired reply may still own UI cleanup for this project."""
        return (
            getattr(self.window, "_clip_analysis_controller", None) is self
            and self.window.project is self.project
            and self.project.session.session_id == self.session_id
        )

    def _same_clip(self, cid: str) -> bool:
        clip = self.project.clips_by_id.get(cid)
        return (
            cid in self.clips
            and clip is self.clips[cid]
            and clip_input(self.project, clip) == self.inputs[cid]
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        previous = getattr(self.window, "_clip_analysis_controller", None)
        self.window._clip_analysis_controller = self
        if previous is not None and previous is not self:
            previous.cancel()
        if not hasattr(self.window, "_active_clip_analyses"):
            self.window._active_clip_analyses = set()
        self.window._active_clip_analyses.add(self)
        self._advance()

    def cancel(self) -> None:
        self.plan.cancel()
        self._transcription_batches.clear()
        for operation, worker in tuple(self.workers.items()):
            worker.cancel()
            self._detach(operation, worker)
        # Reserved operations without workers can settle immediately.
        for operation in tuple(self.plan.running - self.workers.keys()):
            self.plan.finish(operation, self.outcomes.get(operation, {}))
        if self._started and not self.workers:
            self._finish()

    def _detach(self, operation: str, worker: Any) -> None:
        attribute = WORKER_ATTRIBUTES[operation]
        if getattr(self.window, attribute, None) is worker:
            setattr(self.window, attribute, None)

    @Slot()
    def _advance(self) -> None:
        if self.finished:
            return
        if not self.is_current():
            self.cancel()
            return
        for operation in self.plan.begin_ready():
            if self.plan.cancelled or not self.is_current():
                self.cancel()
                return
            self.outcomes[operation] = {}
            clips = []
            for cid, clip in self.clips.items():
                if not self._same_clip(cid):
                    self.outcomes[operation][cid] = "failed"
                elif not self.force_rerun and operation_is_complete_for_clip(
                    operation, clip
                ):
                    self.outcomes[operation][cid] = "skipped"
                else:
                    clips.append(clip)
            if operation == "transcribe":
                grouped: dict[str, list] = {}
                for clip in clips:
                    grouped.setdefault(clip.source_id, []).append(clip)
                self._transcription_batches = list(grouped.values())
                self._next_transcription()
            else:
                self._launch(operation, clips)
        if self.plan.finished:
            self._finish()

    def _launch(self, operation: str, clips: list) -> None:
        if not clips:
            self._settle(operation)
            return
        worker = None
        try:
            worker, application = create_clip_analysis_worker(
                self.project,
                self.settings,
                operation,
                clips,
                force_rerun=self.force_rerun,
                query=self.query,
            )
            attribute = WORKER_ATTRIBUTES[operation]
            previous = getattr(self.window, attribute, None)
            if previous is not None and previous.isRunning():
                raise RuntimeError(
                    "Another worker already owns this analysis operation"
                )
            self.workers[operation] = worker
            self.applications[operation] = application
            setattr(self.window, attribute, worker)
            worker.progress.connect(self._progress)
            if hasattr(worker, "status"):
                worker.status.connect(self._status)
            worker.error.connect(
                self._description_error if operation == "describe" else self._error
            )
            worker.finished.connect(self._worker_finished)
            worker.start()
            if self.workers.get(operation) is worker and self.is_current():
                self.progress.emit(self, 0, operation)
        except Exception as exc:
            self.errors.append(f"{operation}: {exc}")
            if worker is not None:
                self._detach(operation, worker)
                worker.deleteLater()
            self.workers.pop(operation, None)
            self.applications.pop(operation, None)
            if operation == "transcribe":
                self._transcription_timer.start(0)
            else:
                self._settle(operation)

    @Slot()
    def _next_transcription(self) -> None:
        if self.finished:
            return
        if self.plan.cancelled or not self.is_current():
            self.cancel()
            return
        if not self._transcription_batches:
            self._settle("transcribe")
            return
        clips = self._transcription_batches.pop(0)
        current = [c for c in clips if self._same_clip(c.id)]
        if not current:
            self._transcription_timer.start(0)
        else:
            self._launch("transcribe", current)

    def _operation_for_sender(self) -> str | None:
        sender = self.sender()
        return next(
            (op for op, worker in self.workers.items() if worker is sender), None
        )

    @Slot(int, int)
    @Slot(int, int, str)
    def _progress(self, current: int, total: int, *_: str) -> None:
        operation = self._operation_for_sender()
        if (
            operation is not None
            and self.is_current()
            and not self.plan.cancelled
            and getattr(self.window, WORKER_ATTRIBUTES[operation], None)
            is self.workers[operation]
        ):
            self.progress.emit(
                self, int(100 * current / total) if total else 0, operation
            )

    @Slot(str)
    def _status(self, message: str) -> None:
        operation = self._operation_for_sender()
        if (
            operation is not None
            and self.is_current()
            and not self.plan.cancelled
            and getattr(self.window, WORKER_ATTRIBUTES[operation], None)
            is self.workers[operation]
        ):
            self.status.emit(self, message)

    @Slot(str)
    def _error(self, message: str) -> None:
        operation = self._operation_for_sender()
        if operation is not None:
            self.errors.append(f"{operation}: {message}")

    @Slot(str, str)
    def _description_error(self, cid: str, message: str) -> None:
        self._error(f"{cid}: {message}")

    def _apply(self, operation: str, worker: Any) -> None:
        application = self.applications[operation]
        outcomes = self.outcomes[operation]
        if operation == "colors":
            if worker.result is None:
                return
            result = replace(
                worker.result,
                outcomes=tuple(
                    outcome
                    if self._same_clip(outcome.target_id)
                    else replace(outcome, status="failed", code="stale_input")
                    for outcome in worker.result.outcomes
                    if outcome.target_id in self.clips
                ),
            )
            for outcome in application.apply(result).outcomes:
                outcomes[outcome.target_id] = outcome.status
            return
        for outcome in worker.result:
            if not self.is_current() or self.plan.cancelled:
                self.cancel()
                return
            cid = outcome.clip_id
            if cid not in self.clips or not self._same_clip(cid):
                continue
            outcomes[cid] = outcome.status
            if outcome.status != "succeeded":
                continue
            try:
                receipt = None
                cache = worker.cache
                if cache is not None:
                    if (
                        self.project.path is None
                        or self.project.path.resolve() != cache.path
                    ):
                        raise ValueError("Analysis save location changed")
                    receipt = (
                        cache.receipt(outcome)
                        if operation == "shots"
                        else cache.results[("clip", cid)]
                        if operation == "extract_text"
                        else cache.results[cid]
                    )
                    if not receipt.matches(outcome):
                        raise ValueError("Analysis differs from its recorded result")
                if not application.apply(self.project, outcome):
                    outcomes[cid] = "failed"
                elif receipt is not None:
                    self.project.record_job_result(receipt.result_id, receipt.digest)
            except Exception as exc:
                outcomes[cid] = "failed"
                self.errors.append(f"{operation}: {exc}")

    @Slot()
    def _worker_finished(self) -> None:
        operation = self._operation_for_sender()
        if operation is None:
            return
        worker = self.workers[operation]
        if worker.isRunning():
            return
        if (
            self.is_current()
            and not self.plan.cancelled
            and not worker.is_cancelled()
            and getattr(self.window, WORKER_ATTRIBUTES[operation], None) is worker
        ):
            try:
                self._apply(operation, worker)
            except Exception as exc:
                self.errors.append(f"{operation}: {exc}")
        else:
            self.cancel()
        self._detach(operation, worker)
        self.workers.pop(operation)
        self.applications.pop(operation)
        worker.deleteLater()
        if (
            operation == "transcribe"
            and self._transcription_batches
            and not self.plan.cancelled
        ):
            self._transcription_timer.start(0)
        else:
            self._settle(operation)

    def _settle(self, operation: str) -> None:
        self.plan.finish(operation, self.outcomes.get(operation, {}))
        self._advance_timer.start(0)

    def _finish(self) -> None:
        if self.finished or self.workers:
            return
        self.finished = True
        self._advance_timer.stop()
        self._transcription_timer.stop()
        analyzed_sources = set()
        for cid, clip in self.clips.items():
            if not self.is_current() or self.plan.cancelled:
                break
            if self._same_clip(cid) and any(
                outcomes.get(cid) in ("succeeded", "skipped")
                for outcomes in self.plan.results.values()
            ):
                source = self.project.sources_by_id.get(clip.source_id)
                if source is not None and source.id not in analyzed_sources:
                    analyzed_sources.add(source.id)
                    if not source.has_analysis:
                        try:
                            self.project.update_source(source.id, has_analysis=True)
                        except Exception as exc:
                            self.errors.append(f"Source analysis status: {exc}")
        if not self.is_current():
            self.plan.cancel()
        accepted = [cid for cid in self.plan.successful_ids() if self._same_clip(cid)]
        result = {
            "succeeded": accepted,
            "failed": [cid for cid in self.plan.clip_ids if cid not in accepted],
            "cancelled": self.plan.cancelled,
            "operations": deepcopy(self.plan.results),
            "errors": list(self.errors),
            "analyzed_sources": sorted(analyzed_sources),
        }
        self.window._active_clip_analyses.discard(self)
        self.completed.emit(self, result)
        if getattr(self.window, "_clip_analysis_controller", None) is self:
            self.window._clip_analysis_controller = None
        self.deleteLater()
