"""Run-owned intention analysis using the existing shared operation workers."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, TYPE_CHECKING

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.intention_workflow import WorkflowState
from core.operations.analysis_inputs import clip_input
from ui.workers.intention_run import IntentionRun

if TYPE_CHECKING:
    from models.clip import Clip


class IntentionAnalysisController(RetiringQObject):
    """Publish and advance only after this plan's native worker has exited."""

    def __init__(
        self, window: Any, run: IntentionRun, clips: list[Clip], settings: Any
    ) -> None:
        super().__init__(window)
        self.window = window
        self.run_identity = run
        self.project = run.project
        self.workflow = run.workflow
        self.settings = settings
        self.clips = {clip.id: clip for clip in clips}
        # Retain source objects as well as their IDs for the whole run.
        self.sources = tuple(self.project.sources_by_id.get(c.source_id) for c in clips)
        self.inputs = {
            cid: clip_input(self.project, clip) for cid, clip in self.clips.items()
        }
        algorithm, _ = self.workflow.get_algorithm_with_direction()
        self.operation = {
            "color": "colors",
            "shot_type": "shots",
            "storyteller": "describe",
        }[algorithm]
        self.attribute = {
            "colors": "color_worker",
            "shots": "shot_type_worker",
            "describe": "description_worker",
        }[self.operation]
        self.worker: Any = None
        self.application: Any = None
        self.cancelled = False
        self.disposed = False
        self.errors: list[str] = []
        window._intention_analysis = self
        if not hasattr(window, "_active_intention_analyses"):
            window._active_intention_analyses = set()
        window._active_intention_analyses.add(self)
        self.workflow.workflow_cancelled.connect(self.cancel)
        self.workflow.workflow_completed.connect(self._workflow_completed)

    def _current(self) -> bool:
        return bool(
            not self.cancelled
            and not self.disposed
            and getattr(self.window, "_intention_analysis", None) is self
            and self.run_identity.is_current(self.window, WorkflowState.ANALYZING)
            and (
                self.worker is None
                or getattr(self.window, self.attribute, None) is self.worker
            )
        )

    def _same_clip(self, cid: str) -> bool:
        clip = self.project.clips_by_id.get(cid)
        return (
            cid in self.clips
            and clip is self.clips[cid]
            and clip_input(self.project, clip) == self.inputs[cid]
        )

    def start(self) -> None:
        if self.worker is not None or self.disposed:
            return
        if not self._current():
            self.cancel()
            return
        existing = getattr(self.window, self.attribute, None)
        if existing is not None:
            self.workflow.on_analysis_failed(
                "Another analysis has not finished cleaning up"
            )
            self._dispose()
            return
        try:
            assert self.run_identity.plan is not None
            missing = self.run_identity.plan.missing_analysis(list(self.clips.values()))
            clips = [clip for cid, clip in self.clips.items() if cid in missing]
            if not all(self._same_clip(cid) for cid in self.clips):
                raise ValueError("Analysis inputs changed; start the workflow again")
            if not clips:
                self.workflow.on_analysis_finished()
                self._dispose()
                return
            settings = self.settings
            if self.operation == "colors":
                from ui.workers.color_worker import ColorAnalysisWorker

                self.worker = ColorAnalysisWorker(
                    clips,
                    parallelism=settings.color_analysis_parallelism,
                    sources_by_id=self.project.sources_by_id,
                    project=self.project,
                )
                self.application = self.worker.application
            elif self.operation == "shots":
                from ui.workers.shot_type_worker import ShotTypeWorker
                from core.operations.shots import ShotTypeApplication

                self.worker = ShotTypeWorker(
                    clips,
                    self.project.sources_by_id,
                    parallelism=settings.local_model_parallelism,
                    project=self.project,
                )
                self.application = ShotTypeApplication(self.project, self.worker.tasks, self.worker.options)
            else:
                from ui.workers.description_worker import DescriptionWorker
                from core.operations.description import DescriptionApplication

                self.worker = DescriptionWorker(
                    clips,
                    tier=settings.description_model_tier,
                    sources=self.project.sources_by_id,
                    parallelism=settings.description_parallelism,
                    project=self.project,
                )
                self.application = DescriptionApplication(
                    self.project, self.worker.tasks
                )
            setattr(self.window, self.attribute, self.worker)
            self.worker.setParent(self)
            self.worker.progress.connect(self._progress)
            if self.operation == "colors":
                self.worker.job_started.connect(self._job_started)
            self.worker.error.connect(
                self._description_error if self.operation == "describe" else self._error
            )
            self.worker.finished.connect(self._native_finished)
            self.worker.start()
        except Exception as exc:
            if self.run_identity.is_current(self.window, WorkflowState.ANALYZING):
                self.workflow.on_analysis_failed(str(exc))
            self.cancel()
            if self.worker is not None and not self.worker.isRunning():
                self._release_worker()
            self._dispose()

    @Slot(int, int)
    def _progress(self, current: int, total: int) -> None:
        if self.sender() is self.worker and self._current():
            self.workflow.on_analysis_progress(current, total)

    @Slot(str, str)
    def _job_started(self, task_id: str, persistence: str) -> None:
        if self.sender() is self.worker and self._current():
            self.window._on_color_job_started(task_id, persistence)

    @Slot(str)
    def _error(self, message: str) -> None:
        if self.sender() is self.worker and self._current():
            self.errors.append(message)

    @Slot(str, str)
    def _description_error(self, cid: str, message: str) -> None:
        self._error(f"{cid}: {message}")

    def _apply(self) -> None:
        worker = self.worker
        if self.operation == "colors":
            if worker.result is None:
                raise ValueError("Color analysis completed without a result")
            result = replace(
                worker.result,
                outcomes=tuple(
                    outcome
                    if self._same_clip(outcome.target_id)
                    else replace(outcome, status="failed", code="stale_input")
                    for outcome in worker.result.outcomes
                ),
            )
            applied = self.application.apply(result)
            if self._current() and any(
                o.status == "succeeded" for o in applied.outcomes
            ):
                self.window._update_window_title()
            return
        for outcome in worker.result:
            if not self._current() or worker.is_cancelled():
                return
            if (outcome.status != "succeeded" and not getattr(outcome, "can_apply", False)) or not self._same_clip(outcome.clip_id):
                continue
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
                    if self.operation == "shots"
                    else cache.results[outcome.clip_id]
                )
                key = (outcome.target_type, outcome.clip_id) if self.operation == "shots" else outcome.clip_id
                matches = receipt.matches(outcome) if receipt is not None else getattr(cache, "transient_outcomes", {}).get(key) == asdict(outcome)
                if not matches:
                    raise ValueError(
                        "Analysis outcome differs from its recorded result"
                    )
            if not self.application.apply(self.project, outcome):
                continue
            if receipt is not None:
                self.project.record_job_result(receipt.result_id, receipt.digest)
            if not self._current():
                return
            if outcome.status != "succeeded":
                continue
            if self.operation == "shots":
                self.window._on_shot_type_ready(
                    outcome.clip_id, outcome.shot_type, outcome.confidence
                )
            else:
                self.window._on_description_ready(
                    outcome.clip_id, outcome.description, outcome.model
                )

    @Slot()
    def _native_finished(self) -> None:
        worker = self.worker
        if worker is None or self.sender() is not worker or worker.isRunning():
            return
        try:
            if not self._current() or worker.is_cancelled():
                self.cancel()
                return
            self._apply()
            if self._current():
                if not all(self._same_clip(cid) for cid in self.clips):
                    self.workflow.on_analysis_failed(
                        "Analysis inputs changed; start the workflow again"
                    )
                elif self.errors:
                    self.workflow.on_analysis_failed("\n".join(self.errors))
                else:
                    self.workflow.on_analysis_finished()
        except Exception as exc:
            if self._current():
                self.workflow.on_analysis_failed(str(exc))
        finally:
            self._release_worker()
            self._dispose()

    def _release_worker(self) -> None:
        if self.worker is not None:
            if getattr(self.window, self.attribute, None) is self.worker:
                setattr(self.window, self.attribute, None)
            self.worker.deleteLater()
            self.worker = None

    @Slot(object)
    def _workflow_completed(self, _result: Any) -> None:
        self.cancel()

    @Slot()
    def cancel(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        if self.worker is not None:
            self.worker.cancel()
            if getattr(self.window, self.attribute, None) is self.worker:
                setattr(self.window, self.attribute, None)
        else:
            self._dispose()
        if (
            self.window.intention_workflow is self.workflow
            and self.workflow.plan is self.run_identity.plan
            and self.workflow.is_running
        ):
            self.workflow.cancel()

    def _dispose(self) -> None:
        if self.disposed or self.worker is not None:
            return
        self.disposed = True
        self.window._active_intention_analyses.discard(self)
        if getattr(self.window, "_intention_analysis", None) is self:
            self.window._intention_analysis = None
        self.retire()
