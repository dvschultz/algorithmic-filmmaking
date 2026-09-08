"""Explicit legacy reuse with asynchronous hashing and guarded publication."""

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout

from core.operations.colors import ColorApplication
from ui.theme import Spacing, UISizes
from ui.workers.legacy_reuse_worker import LegacyReuseWorker


class LegacyReuseDialog(QDialog):
    def __init__(self, owner, clip_ids: list[str]) -> None:
        super().__init__(owner)
        self.owner = owner
        self.project = owner.project
        self.clip_ids = list(clip_ids)
        self.worker: LegacyReuseWorker | None = None
        self.closing = False
        self.setWindowTitle("Reuse Legacy Analysis")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)
        explanation = QLabel(
            f"Reuse existing values for {len(clip_ids)} selected {'clip' if len(clip_ids) == 1 else 'clips'}. Their provenance "
            "will remain unknown: this does not verify how they were computed. "
            "The decision applies to the current media and analysis settings. "
            "Changed inputs require a new decision or recomputation."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.operation = QComboBox()
        self.operation.addItem("Colors (default five-color settings)", "colors")
        self.operation.addItem("Thumbnail embeddings (compatible DINO model)", "embeddings")
        self.operation.addItem("Brightness (default five samples)", "brightness")
        self.operation.addItem("Volume", "volume")
        self.operation.addItem("Custom query (exact saved question)", "custom_query")
        self.operation.addItem("Word alignment (saved timings)", "align_words")
        self.operation.addItem("Transcription (current settings)", "transcribe")
        self.operation.addItem("Cinematography (current settings)", "cinematography")
        self.operation.addItem("Description (current settings, default prompt)", "describe")
        self.operation.addItem("OCR text (current model, default sampling)", "extract_text")
        self.operation.addItem("Shot type (current settings)", "shots")
        self.operation.addItem("Gaze (default sampling)", "gaze")
        self.operation.addItem("Boundary embeddings (compatible DINO model)", "boundary_embeddings")
        self.operation.addItem("Classification (default settings)", "classify")
        self.operation.addItem("Object detection (default settings)", "detect_objects")
        self.operation.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.operation.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH_WIDE)
        layout.addWidget(self.operation)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Exact saved question")
        self.query.setAccessibleName("Exact saved question")
        self.query.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.query.setVisible(False)
        self.operation.currentIndexChanged.connect(lambda: self.query.setVisible(self.operation.currentData() == "custom_query"))
        layout.addWidget(self.query)
        self.status = QLabel("Accepted decisions are included the next time you save the project.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.reuse = QPushButton("Reuse selected values")
        self.reuse.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self.reuse.clicked.connect(self.start_reuse)
        layout.addWidget(self.reuse)
        self.close_button = QPushButton("Close")
        self.close_button.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.close_button)

    @Slot()
    def start_reuse(self) -> None:
        if self.worker is not None:
            return
        if self.owner.project is not self.project:
            self.status.setText("Project changed. Close this dialog and select clips again.")
            return
        try:
            worker = LegacyReuseWorker(self.project, self.operation.currentData(), self.clip_ids, self, settings=getattr(self.owner, "settings", None), query=self.query.text())
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self.worker = worker
        self.operation.setEnabled(False)
        self.query.setEnabled(False)
        self.reuse.setEnabled(False)
        self.close_button.setText("Cancel")
        self.status.setText("Checking current media…")
        worker.result_ready.connect(self.apply_result)
        worker.error.connect(self.show_error)
        worker.finished.connect(self.worker_finished)
        try:
            worker.start()
        except Exception as exc:
            self.show_error(str(exc))
            self.worker_finished()

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.status.setText(message)

    @Slot(object)
    def apply_result(self, result) -> None:
        worker = self.worker
        if worker is None or self.closing or worker.is_cancelled():
            return
        if self.owner.project is not self.project:
            self.status.setText("Project changed; no reuse decisions were applied.")
            return
        try:
            if isinstance(worker.application, ColorApplication):
                outcomes = worker.application.apply(result).outcomes
                accepted = sum(outcome.status == "succeeded" for outcome in outcomes)
            else:
                outcomes = result
                accepted = sum(outcome.status in ("succeeded", "skipped") and worker.application.apply(self.project, outcome) for outcome in outcomes)
            failures = [outcome.message or getattr(outcome, "code", "Reuse unavailable") for outcome in outcomes if outcome.status == "failed"]
            self.status.setText(
                f"Accepted {accepted} of {len(self.clip_ids)} clips. Save the project to keep these decisions."
                + ("\n" + "\n".join(str(message) for message in failures[:3]) if failures else "")
            )
        except (ValueError, RuntimeError) as exc:
            self.show_error(str(exc))

    @Slot()
    def worker_finished(self) -> None:
        self.close_button.setText("Close")
        if self.closing:
            super().reject()

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.closing = True
            self.worker.cancel()
            self.status.setText("Cancelling…")
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
