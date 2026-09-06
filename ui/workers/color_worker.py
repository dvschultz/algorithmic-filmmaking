"""Qt scheduling and signals for the shared color operation."""

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal

from core.analysis_target import AnalysisTarget
from core.operations.colors import (
    ColorApplication,
    compute_colors,
    request_from_targets,
)
from core.operations.contracts import ColorOutcome, ColorResult
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from core.project import Project


class ColorAnalysisWorker(CancellableWorker):
    """Snapshot on construction; compute off-thread; apply on the project thread.

    ``color_ready`` remains available for consumers of individual palettes.
    The main window uses ``result_ready`` to commit a whole batch exactly once.
    """

    progress = Signal(int, int)
    color_ready = Signal(str, list)
    result_ready = Signal(object, object)  # ColorApplication, ColorResult
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list,
        parallelism: int = 4,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        sources_by_id: Optional[dict] = None,
        parent=None,
        *,
        project: Optional["Project"] = None,
    ) -> None:
        super().__init__(parent)
        self._parallelism = min(max(1, parallelism), 8)
        targets = analysis_targets
        if targets is None:
            sources = sources_by_id or {}
            targets = []
            for clip in clips:
                target = AnalysisTarget.from_clip(clip, sources.get(clip.source_id))
                target.image_path = None
                targets.append(target)
        self.request = request_from_targets(
            targets, skip_existing=skip_existing, skip_empty=True
        )
        self.application = (
            ColorApplication(project, self.request) if project is not None else None
        )
        self.result: Optional[ColorResult] = None

    def _on_progress(self, completed: int, total: int, outcome: ColorOutcome) -> None:
        if outcome.status == "succeeded":
            self.color_ready.emit(outcome.target_id, list(outcome.colors))
        self.progress.emit(completed, total)

    def run(self) -> None:
        self._log_start()
        try:
            self.result = compute_colors(
                self.request,
                parallelism=self._parallelism,
                cancel_event=self._cancel_event,
                progress_callback=self._on_progress,
            )
            errors = [
                (o.target_id, o.message or o.code or "Color extraction failed")
                for o in self.result.outcomes
                if o.status == "failed"
            ]
            if errors:
                self.error.emit(
                    summarize_clip_errors(errors, operation_label="Color extraction")
                )
            self.result_ready.emit(self.application, self.result)
        except Exception as exc:
            self._log_error(str(exc))
            self.error.emit(str(exc))
        finally:
            self.analysis_completed.emit()
            self._log_complete()
