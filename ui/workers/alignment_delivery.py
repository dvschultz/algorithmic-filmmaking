"""Owner-thread alignment publication scoped to one tab run and project."""

from __future__ import annotations

from PySide6.QtCore import Slot
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.alignment import AlignmentApplication, AlignmentOutcome

if TYPE_CHECKING:
    from core.project import Project


def apply_alignment_outcome(
    project: Project,
    worker: Any,
    application: AlignmentApplication,
    outcome: AlignmentOutcome,
) -> bool:
    """Authenticate a GUI receipt or transient outcome before owner publication."""
    cache = getattr(worker, "cache", None)
    if cache is not None and (
        project.path is None or project.path.resolve() != cache.path
    ):
        raise ValueError("Project save location changed during alignment")
    receipt = cache.results.get(outcome.clip_id) if cache is not None else None
    if cache is not None and (
        (receipt is not None and not receipt.matches(outcome))
        or (
            receipt is None
            and cache.transient_outcomes.get(outcome.clip_id) != asdict(outcome)
        )
    ):
        raise ValueError("Queued alignment output differs from its recorded result")
    applied = application.apply(project, outcome)
    if applied and receipt is not None:
        project.record_job_result(receipt.result_id, receipt.digest)
    return applied


class AlignmentDelivery(RetiringQObject):
    def __init__(self, tab, worker, project) -> None:
        super().__init__(tab)
        self.tab = tab
        self.worker = worker
        self.project = project
        self.session_id = project.session.session_id
        self.generation = tab._alignment_generation
        self.application = AlignmentApplication(project, worker.tasks)
        self._delivered: set[str] = set()
        worker.progress.connect(self.progress)
        worker.error.connect(self.error)
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.receive)
        else:
            worker.clip_aligned.connect(self.aligned)
        worker.alignment_completed.connect(self.completed)
        worker.finished.connect(self.finished)

    def _current(self) -> bool:
        return (
            self.tab._forced_alignment_worker is self.worker
            and self.tab._alignment_generation == self.generation
            and self.tab._project_provider() is self.project
            and self.project.session.session_id == self.session_id
            and not getattr(self.worker, "is_cancelled", lambda: False)()
        )

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self._current():
            self.tab._on_alignment_progress(current, total)

    @Slot(str)
    def error(self, message: str) -> None:
        if self._current():
            self.tab._on_alignment_error(message)

    @Slot(str, list)
    def aligned(self, clip_id: str, words: list) -> None:
        self.receive(AlignmentOutcome(clip_id, "succeeded", tuple(words)))

    @Slot(object)
    def receive(self, outcome: AlignmentOutcome) -> None:
        if not isinstance(outcome, AlignmentOutcome) or not outcome.can_apply:
            return
        clip_id = outcome.clip_id
        if not self._current() or clip_id in self._delivered:
            return
        self._delivered.add(clip_id)
        try:
            applied = apply_alignment_outcome(
                self.project, self.worker, self.application, outcome
            )
        except Exception as exc:
            self.error(f"Could not apply word alignment: {exc}")
            return
        if applied and outcome.has_result:
            self.tab._on_clip_aligned(clip_id, list(outcome.words))
        elif not applied:
            self.error(
                f"Alignment result discarded for changed clip {clip_id}. Run alignment again."
            )

    @Slot()
    def completed(self) -> None:
        if self._current():
            self.tab._on_alignment_completed()

    @Slot()
    def finished(self) -> None:
        if self.tab._forced_alignment_worker is self.worker:
            self.tab._on_alignment_thread_finished()
        self.worker.deleteLater()
        self.retire()
