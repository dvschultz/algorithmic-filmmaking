"""Dice Roll dialog — shuffle with optional pre-rendered transforms.

Users configure random H-Flip, V-Flip, and Reverse options.
If any transforms are checked, clips are pre-rendered via FFmpeg
before being placed on the timeline.
"""

import logging

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal, Slot

from core.remix import run_registry_algorithm
from core.remix.prerender import prerender_batch, get_transform_cache_dir
from ui.theme import theme, Spacing, TypeScale
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class DiceRollWorker(CancellableWorker):
    """Background worker for shuffle + optional pre-render."""

    progress_update = Signal(int, int)  # current, total
    progress_message = Signal(str)
    finished_sequence = Signal(list)  # list of (Clip, Source, dict)

    def __init__(
        self,
        clips: list,  # list of (Clip, Source)
        hflip: bool,
        vflip: bool,
        reverse: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._clips = clips
        self._hflip = hflip
        self._vflip = vflip
        self._reverse = reverse
        self.recipe = None  # SequenceRecipe once the shuffle has run

    def run(self):
        """Run shuffle + pre-render pipeline."""
        self._log_start()
        try:
            # Step 1: Shuffle and draw transforms through the registry so the
            # recipe records the realized order and transform flags.
            self.progress_message.emit("Shuffling clips...")
            run = run_registry_algorithm(
                "shuffle", self._clips,
                transform_options={"hflip": self._hflip, "vflip": self._vflip, "reverse": self._reverse},
                cancel_event=self._cancel_event,
            )
            if run is None or self.is_cancelled():
                self._log_cancelled()
                return
            self.recipe = run.recipe
            sorted_clips = run.ordered_clips
            realized = run.recipe.realized

            has_transforms = any(entry.has_transform for entry in realized)

            if not has_transforms:
                result = [
                    (clip, source, {"hflip": False, "vflip": False, "reverse": False, "prerendered_path": None})
                    for clip, source in sorted_clips
                ]
                self.finished_sequence.emit(result)
                return

            # Step 2: Pre-render clips with the realized transforms
            self.progress_message.emit("Pre-rendering transformed clips...")
            clips_with_transforms = [
                (clip, source, {"hflip": entry.hflip, "vflip": entry.vflip, "reverse": entry.reverse})
                for (clip, source), entry in zip(sorted_clips, realized)
            ]

            output_dir = get_transform_cache_dir()
            rendered = prerender_batch(
                clips_with_transforms=clips_with_transforms,
                output_dir=output_dir,
                progress_cb=self._on_progress,
                cancel_event=self._cancel_event,
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Detect total prerender failure: if transforms were requested but
            # every clip came back without a prerendered_path, something is wrong
            # (commonly: project folder is on a disconnected drive).
            expected_prerender_count = sum(1 for entry in realized if entry.has_transform)
            actual_prerender_count = sum(
                1 for _, _, path in rendered if path is not None
            )
            if expected_prerender_count > 0 and actual_prerender_count == 0:
                raise RuntimeError(
                    f"Pre-rendering failed for all {expected_prerender_count} clips "
                    f"with transforms. Check that the project folder is accessible "
                    f"(disconnected external drive?). See logs for details."
                )

            result = []
            for (clip, source, prerendered_path), entry in zip(rendered, realized):
                result.append((clip, source, {
                    "hflip": entry.hflip,
                    "vflip": entry.vflip,
                    "reverse": entry.reverse,
                    "prerendered_path": str(prerendered_path) if prerendered_path else None,
                }))

            self.finished_sequence.emit(result)

        except Exception as e:
            if not self.is_cancelled():
                logger.error("Dice Roll generation failed: %s", e)
                self.error.emit(str(e))
        self._log_complete()

    def _on_progress(self, current: int, total: int):
        """Bridge prerender_batch progress to Qt signal."""
        self.progress_update.emit(current, total)


class DiceRollDialog(QDialog):
    """Dialog for Dice Roll (shuffle) with optional transforms.

    Page 1: Config — checkboxes for transforms + Generate button
    Page 2: Progress — progress bar during pre-rendering
    """

    sequence_ready = Signal(list)  # list of (Clip, Source, dict); see ``recipe``

    def __init__(self, clips: list, parent=None):
        """
        Args:
            clips: List of (Clip, Source) tuples to shuffle.
        """
        super().__init__(parent)
        self._clips = clips
        self._worker = None
        self.recipe = None  # SequenceRecipe of the emitted sequence
        self.setWindowTitle("Hatchet Job")
        self.setMinimumWidth(400)
        self.setMinimumHeight(250)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Page 0: Config
        self._config_page = self._create_config_page()
        self._stack.addWidget(self._config_page)

        # Page 1: Progress
        self._progress_page = self._create_progress_page()
        self._stack.addWidget(self._progress_page)

        self._stack.setCurrentIndex(0)

    def _create_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Hatchet Job")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Randomly shuffle clips into a new order.\n"
            "Optionally apply random transforms to individual clips."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(desc)

        layout.addSpacing(Spacing.SM)

        self._hflip_cb = QCheckBox("Random H-Flip")
        self._hflip_cb.setToolTip("Randomly flip ~50% of clips horizontally")
        layout.addWidget(self._hflip_cb)

        self._vflip_cb = QCheckBox("Random V-Flip")
        self._vflip_cb.setToolTip("Randomly flip ~50% of clips vertically")
        layout.addWidget(self._vflip_cb)

        self._reverse_cb = QCheckBox("Random Reverse")
        self._reverse_cb.setToolTip("Randomly reverse ~50% of clips (max 15s per clip)")
        layout.addWidget(self._reverse_cb)

        layout.addSpacing(Spacing.SM)

        self._clip_count_label = QLabel(f"{len(self._clips)} clips")
        self._clip_count_label.setStyleSheet(f"color: {theme().text_muted};")
        layout.addWidget(self._clip_count_label)

        layout.addStretch()

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-weight: bold;
            }}
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._generate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    def _create_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Hatchet Job")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        self._progress_label = QLabel("Preparing...")
        self._progress_label.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    @Slot()
    def _on_generate(self):
        """Start shuffle + optional pre-render."""
        hflip = self._hflip_cb.isChecked()
        vflip = self._vflip_cb.isChecked()
        reverse = self._reverse_cb.isChecked()

        has_transforms = hflip or vflip or reverse
        if has_transforms:
            # Show progress page
            self._stack.setCurrentIndex(1)
            self._progress_bar.setValue(0)
            self._progress_label.setText("Shuffling clips...")

        self._worker = DiceRollWorker(
            clips=self._clips,
            hflip=hflip,
            vflip=vflip,
            reverse=reverse,
            parent=self,
        )
        self._generate_btn.setEnabled(False)
        worker = self._worker
        worker.progress_update.connect(self._on_progress_update)
        worker.progress_message.connect(self._on_progress_message)
        worker.finished_sequence.connect(lambda data, owner=worker: self._on_finished(data, owner))
        worker.error.connect(self._on_error)
        worker.start()

    @Slot(int, int)
    def _on_progress_update(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)
            self._progress_label.setText(f"Pre-rendering clip {current + 1} of {total}...")

    @Slot(str)
    def _on_progress_message(self, message: str):
        self._progress_label.setText(message)

    def _on_finished(self, sequence_data: list, owner=None):
        """Publish the result with the recipe of the worker that produced it."""
        if owner is not None and owner is not self._worker:
            return  # A superseded worker finished late.
        source = owner if owner is not None else self._worker
        self.recipe = source.recipe if source is not None else None
        self.sequence_ready.emit(sequence_data)
        self.accept()

    @Slot(str)
    def _on_error(self, error_msg: str):
        logger.error("Dice Roll error: %s", error_msg)
        self._stack.setCurrentIndex(0)
        self._generate_btn.setEnabled(True)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Hatchet Job Error", f"Pre-rendering failed:\n{error_msg}")

    @Slot()
    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._worker.wait(3000)
            self._worker = None
        self.reject()

    def closeEvent(self, event):
        if self._worker:
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
