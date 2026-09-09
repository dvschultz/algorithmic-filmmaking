"""Verify detached project data while the owner thread services UI events."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from core.project import ProjectLoadCancelled, ProjectLoadError, hydrate_project_artifacts
from models.audio_source import AudioSource
from models.clip import Clip
from models.frame import Frame
from models.sequence import Sequence
from ui.theme import Spacing
from ui.workers.base import CancellableWorker


class ProjectArtifactWorker(CancellableWorker):
    def __init__(self, path: Path, document: dict, targets: list[Clip | Frame | AudioSource], sequences: list[Sequence], parent=None) -> None:
        super().__init__(parent)
        self.inputs = path, document, targets, sequences
        self.failure: str | None = None

    def run(self) -> None:
        if self.is_cancelled():
            return
        try:
            hydrate_project_artifacts(*self.inputs, cancel_check=self.is_cancelled)
        except Exception as exc:
            self.failure = str(exc)


class ArtifactLoadDialog(QDialog):
    """Cancellation discards the load after native worker completion."""

    def __init__(self, worker: ProjectArtifactWorker, parent=None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.setWindowTitle("Loading project")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        self.label = QLabel("Verifying saved analysis and cached media…", self)
        layout.addWidget(self.label)
        progress = QProgressBar(self)
        progress.setRange(0, 0)
        layout.addWidget(progress)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        worker.finished.connect(self.worker_finished)

    @Slot()
    def reject(self) -> None:
        self.worker.cancel()
        self.label.setText("Cancelling project load…")
        self.cancel_button.setEnabled(False)

    def accept(self) -> None:
        self.reject()

    def done(self, result: int) -> None:
        # Only native completion may dismiss the dialog and release its worker.
        self.reject()

    @Slot()
    def start_worker(self) -> None:
        try:
            self.worker.start()
        except Exception as exc:
            self.worker.failure = str(exc)
            self.worker_finished()

    @Slot()
    def worker_finished(self) -> None:
        QDialog.done(self, 0 if self.worker.is_cancelled() or self.worker.failure else 1)


def hydrate_with_progress(window: Any, path: Path, document: dict, targets: list[Clip | Frame | AudioSource], sequences: list[Sequence]) -> None:
    """Return only after the worker has relinquished all detached load objects."""
    from core.artifacts import document_references

    if not document_references(document):
        return
    worker = ProjectArtifactWorker(path, document, targets, sequences)
    dialog = ArtifactLoadDialog(worker, window)
    if not hasattr(window, "_active_project_loads"):
        window._active_project_loads = set()
    window._active_project_loads.add(worker)
    try:
        QTimer.singleShot(0, dialog.start_worker)
        dialog.exec()
        if worker.is_cancelled():
            raise ProjectLoadCancelled("Project load cancelled")
        if worker.failure:
            raise ProjectLoadError(worker.failure)
    finally:
        window._active_project_loads.discard(worker)
        worker.deleteLater()
        dialog.deleteLater()
