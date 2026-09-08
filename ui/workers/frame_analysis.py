"""Owner-thread coordination of shared frame-analysis workers."""

from dataclasses import asdict, replace
from copy import deepcopy
from typing import Any

from PySide6.QtCore import Signal, Slot, QTimer

from ui.workers.qt_lifetime import RetiringQObject

from core.analysis_target import AnalysisTarget
from core.analysis_availability import VERIFIED_ANALYSIS_OPERATIONS
from core.jobs.media import media_stamp
from core.operations.frame_analysis import FrameAnalysisPlan


def create_frame_analysis_worker(
    project: Any, settings: Any, operation: str, targets: list, *, options: Any = None
) -> tuple[Any, Any]:
    """Build one existing worker and its shared model-publication adapter."""
    worker: Any
    common = {
        "clips": [],
        "analysis_targets": targets,
        "project": project,
        "skip_existing": operation in VERIFIED_ANALYSIS_OPERATIONS,
    }
    if operation == "colors":
        from ui.workers.color_worker import ColorAnalysisWorker

        worker = ColorAnalysisWorker(
            **common, parallelism=settings.color_analysis_parallelism
        )
        return worker, worker.application
    if operation == "shots":
        from ui.workers.shot_type_worker import ShotTypeWorker
        from core.operations.shots import ShotTypeApplication

        worker = ShotTypeWorker(
            **common,
            sources_by_id={},
            parallelism=settings.local_model_parallelism,
            options=options,
        )
        return worker, ShotTypeApplication(project, worker.tasks, worker.options)
    if operation == "classify":
        from ui.workers.classification_worker import ClassificationWorker
        from core.operations.classification import ClassificationApplication

        worker = ClassificationWorker(
            **common, parallelism=settings.local_model_parallelism
        )
        return worker, ClassificationApplication(project, worker.tasks, worker.options)
    if operation == "detect_objects":
        from ui.workers.object_detection_worker import ObjectDetectionWorker
        from core.operations.object_detection import ObjectDetectionApplication

        worker = ObjectDetectionWorker(
            **common, parallelism=settings.local_model_parallelism
        )
        return worker, ObjectDetectionApplication(project, worker.tasks, worker.options)
    if operation == "extract_text":
        from ui.workers.text_extraction_worker import TextExtractionWorker
        from core.operations.ocr import OcrApplication

        method = settings.text_extraction_method
        use_vlm = method in ("vlm", "hybrid")
        worker = TextExtractionWorker(
            **common,
            sources_by_id={},
            use_vlm_fallback=use_vlm,
            vlm_only=method == "vlm",
            vlm_model=settings.text_extraction_vlm_model if use_vlm else None,
            options=options,
        )
        return worker, OcrApplication(project, worker.tasks, worker.options)
    if operation == "describe":
        from ui.workers.description_worker import DescriptionWorker
        from core.operations.description import DescriptionApplication

        worker = DescriptionWorker(
            **common, parallelism=settings.description_parallelism, options=options
        )
        return worker, DescriptionApplication(project, worker.tasks)
    if operation == "cinematography":
        from ui.workers.cinematography_worker import CinematographyWorker
        from core.operations.cinematography import CinematographyApplication

        worker = CinematographyWorker(
            **common,
            sources_by_id={},
            parallelism=min(settings.description_parallelism, 2),
            options=options,
        )
        return worker, CinematographyApplication(project, worker.tasks)
    raise ValueError("Unsupported frame analysis operation")


def _has_result(frame: Any, operation: str) -> bool:
    fields = {
        "colors": "dominant_colors",
        "shots": "shot_type",
        "classify": "object_labels",
        "extract_text": "extracted_texts",
        "describe": "description",
        "cinematography": "cinematography",
    }
    if operation == "detect_objects":
        return frame.person_count is not None and frame.detected_objects is not None
    return getattr(frame, fields[operation]) is not None


class FrameAnalysisController(RetiringQObject):
    progress = Signal(object, int, str)
    completed = Signal(object, object)

    def __init__(
        self, window: Any, frame_ids: list[str], operations: list[str]
    ) -> None:
        super().__init__(window)
        self.window = window
        self.project = window.project
        self.project.session.assert_owner()
        self.session_id = self.project.session.session_id
        self.path = self.project.path
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.plan = FrameAnalysisPlan(frame_ids, operations)
        self.settings = deepcopy(window.settings)
        self.options: dict[str, Any] = {}
        # Resolve provider choices now; construct tasks and publication adapters
        # only when their step starts, after preceding model updates.
        for operation in self.plan.operations:
            if operation == "shots":
                from core.operations.shots import ShotTypeOptions

                self.options[operation] = ShotTypeOptions.from_settings()
            elif operation == "describe":
                from core.operations.description import resolve_options

                self.options[operation] = resolve_options(
                    parallelism=self.settings.description_parallelism
                )
            elif operation == "cinematography":
                from core.operations.cinematography import (
                    resolve_options as resolve_cinematography,
                )

                self.options[operation] = resolve_cinematography(
                    parallelism=min(self.settings.description_parallelism, 2)
                )
            elif operation == "extract_text":
                from core.operations.ocr import OcrOptions
                from core.jobs.ocr import resolve_options as resolve_ocr

                method = self.settings.text_extraction_method
                self.options[operation] = resolve_ocr(
                    OcrOptions(
                        use_vlm_fallback=method in ("vlm", "hybrid"),
                        vlm_only=method == "vlm",
                        vlm_model=self.settings.text_extraction_vlm_model,
                    )
                )
        self.frames = {
            fid: self.project.frames_by_id[fid] for fid in self.plan.frame_ids
        }
        self.identities = {
            fid: self._identity(frame) for fid, frame in self.frames.items()
        }
        self.worker: Any = None
        self.application: Any = None
        self.finished = False
        self._started = False
        self.errors: list[str] = []
        self._outcomes: dict[str, str] = {}

    @staticmethod
    def _identity(frame: Any) -> tuple:
        return (
            frame.file_path,
            frame.source_id,
            frame.clip_id,
            frame.frame_number,
            media_stamp(frame.file_path),
        )

    def _same_frame(self, fid: str) -> bool:
        frame = self.project.frames_by_id.get(fid)
        return (
            frame is self.frames[fid] and self._identity(frame) == self.identities[fid]
        )

    def _current(self) -> bool:
        return (
            getattr(self.window, "_frame_analysis_controller", None) is self
            and self.window.project is self.project
            and self.project.session.session_id == self.session_id
            and self.project.path == self.path
            and (self.reply is None or self.reply.is_current(self.window))
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        previous = getattr(self.window, "_frame_analysis_controller", None)
        self.window._frame_analysis_controller = self
        if previous is not None and previous is not self:
            previous.cancel()
        if not hasattr(self.window, "_active_frame_analyses"):
            self.window._active_frame_analyses = set()
        self.window._active_frame_analyses.add(self)
        self._advance()

    def cancel(self) -> None:
        self.plan.cancel()
        if self.worker is not None:
            self.worker.cancel()
        elif self._started:
            self._finish()

    @Slot()
    def _advance(self) -> None:
        if self.finished:
            return
        if not self._current():
            self.plan.cancel()
        operation = self.plan.begin_next()
        if operation is None:
            self._finish()
            return
        self._outcomes = {}
        targets = []
        for fid in self.plan.frame_ids:
            if not self._same_frame(fid):
                self._outcomes[fid] = "failed"
            elif operation not in VERIFIED_ANALYSIS_OPERATIONS and _has_result(
                self.frames[fid], operation
            ):
                self._outcomes[fid] = "skipped"
            else:
                targets.append(AnalysisTarget.from_frame(self.frames[fid]))
        if not targets:
            self.plan.finish(operation, self._outcomes)
            QTimer.singleShot(0, self._advance)
            return
        try:
            self.worker, self.application = create_frame_analysis_worker(
                self.project,
                self.settings,
                operation,
                targets,
                options=self.options.get(operation),
            )
            self.worker.progress.connect(self._progress)
            self.worker.error.connect(
                self._description_error if operation == "describe" else self._error
            )
            self.worker.finished.connect(self._step_finished)
            self.worker.start()
        except Exception as exc:
            self.errors.append(f"{operation}: {exc}")
            if self.worker is not None:
                self.worker.deleteLater()
            self.worker = None
            self.plan.finish(operation, self._outcomes)
            QTimer.singleShot(0, self._advance)

    @Slot(int, int)
    @Slot(int, int, str)
    def _progress(self, current: int, total: int, *_: str) -> None:
        if self._current() and not self.plan.cancelled:
            value = (
                len(self.plan.results) + min(1.0, current / total if total else 0)
            ) / len(self.plan.operations)
            self.progress.emit(
                self, int(value * 100), f"Analyzing frames: {self.plan.active}"
            )

    @Slot(str)
    def _error(self, message: str) -> None:
        self.errors.append(f"{self.plan.active}: {message}")

    @Slot(str, str)
    def _description_error(self, fid: str, message: str) -> None:
        self._error(f"{fid}: {message}")

    def _apply(self, operation: str, worker: Any) -> None:
        if operation == "colors":
            if worker.result is None:
                return
            result = replace(
                worker.result,
                outcomes=tuple(
                    outcome
                    if self._same_frame(outcome.target_id)
                    else replace(outcome, status="failed", code="stale_input")
                    for outcome in worker.result.outcomes
                ),
            )
            for outcome in self.application.apply(result).outcomes:
                self._outcomes[outcome.target_id] = outcome.status
            return
        for outcome in worker.result:
            # Publication notifies observers, which can cancel or replace this
            # run before the next outcome is applied.
            if not self._current() or self.plan.cancelled:
                self.plan.cancel()
                break
            fid = outcome.clip_id
            if fid not in self.frames or not self._same_frame(fid):
                continue
            self._outcomes[fid] = outcome.status
            if outcome.status != "succeeded" and not getattr(
                outcome, "can_apply", False
            ):
                continue
            try:
                receipt = None
                if worker.cache is not None:
                    cache = worker.cache
                    if (
                        self.project.path is None
                        or self.project.path.resolve() != cache.path
                    ):
                        raise ValueError("Frame analysis save location changed")
                    receipt = (
                        cache.receipt(outcome)
                        if operation == "shots"
                        else cache.results.get(("frame", fid))
                        if operation == "extract_text"
                        else cache.results.get(fid)
                    )
                    key = ("frame", fid) if operation in ("extract_text", "shots") else fid
                    matches = (
                        receipt.matches(outcome)
                        if receipt is not None
                        else getattr(cache, "transient_outcomes", {}).get(key)
                        == asdict(outcome)
                    )
                    if not matches:
                        raise ValueError(
                            "Frame outcome differs from its recorded result"
                        )
                if not self.application.apply(self.project, outcome):
                    self._outcomes[fid] = "failed"
                elif receipt is not None:
                    self.project.record_job_result(receipt.result_id, receipt.digest)
            except Exception as exc:
                self._outcomes[fid] = "failed"
                self.errors.append(f"{operation}: {exc}")

    @Slot()
    def _step_finished(self) -> None:
        worker: Any = self.sender()
        if worker is not self.worker or worker.isRunning():
            return
        operation = self.plan.active
        if operation is None:
            return
        if self._current() and not self.plan.cancelled and not worker.is_cancelled():
            try:
                self._apply(operation, worker)
            except Exception as exc:
                self.errors.append(f"{operation}: {exc}")
        else:
            self.plan.cancel()
        self.plan.finish(operation, self._outcomes)
        self.worker = None
        worker.deleteLater()
        QTimer.singleShot(0, self._advance)

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        accepted = []
        if self._current():
            for fid in self.plan.successful_ids():
                if (
                    self._current()
                    and not self.plan.cancelled
                    and self._same_frame(fid)
                    and all(
                        _has_result(self.frames[fid], op) for op in self.plan.operations
                    )
                ):
                    self.project.update_frame(fid, analyzed=True)
                    accepted.append(fid)
        result = {
            "succeeded": accepted,
            "failed": [fid for fid in self.plan.frame_ids if fid not in accepted],
            "cancelled": self.plan.cancelled,
            "operations": self.plan.results,
            "errors": list(self.errors),
        }
        self.window._active_frame_analyses.discard(self)
        self.completed.emit(self, result)
        if getattr(self.window, "_frame_analysis_controller", None) is self:
            self.window._frame_analysis_controller = None
        self.retire()
