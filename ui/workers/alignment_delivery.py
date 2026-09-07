"""Owner-thread alignment publication scoped to one tab run and project."""

from PySide6.QtCore import QObject, Slot

from core.operations.alignment import AlignmentApplication, AlignmentOutcome


class AlignmentDelivery(QObject):
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
        worker.clip_aligned.connect(self.aligned)
        worker.alignment_completed.connect(self.completed)
        worker.finished.connect(self.finished)

    def _current(self) -> bool:
        return (
            self.tab._forced_alignment_worker is self.worker
            and self.tab._alignment_generation == self.generation
            and self.tab._project_provider() is self.project
            and self.project.session.session_id == self.session_id
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
        if not self._current() or clip_id in self._delivered:
            return
        self._delivered.add(clip_id)
        cache = getattr(self.worker, "cache", None)
        if cache is not None and (
            self.project.path is None or self.project.path.resolve() != cache.path
        ):
            self.error(
                "Alignment result discarded because the project save location changed."
            )
            return
        try:
            receipt = cache.results[clip_id] if cache is not None else None
            outcome = AlignmentOutcome(clip_id, "succeeded", tuple(words))
            if receipt is not None and not receipt.matches(outcome):
                raise ValueError(
                    "Queued alignment output differs from its recorded result"
                )
            applied = self.application.apply(
                self.project,
                outcome,
            )
            if applied and receipt is not None:
                self.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.error(f"Could not apply word alignment: {exc}")
            return
        if applied:
            self.tab._on_clip_aligned(clip_id, words)
        else:
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
        self.deleteLater()
