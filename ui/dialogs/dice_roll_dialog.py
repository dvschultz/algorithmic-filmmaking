"""Dice Roll dialog — shuffle with optional pre-rendered transforms.

Users configure random H-Flip, V-Flip, and Reverse options.
If any transforms are checked, clips are pre-rendered via FFmpeg
before being placed on the timeline.
"""

import logging
from pathlib import Path

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
from PySide6.QtCore import Qt, Signal, Slot

from core.remix import generate_sequence, assign_random_transforms
from core.remix.prerender import prerender_batch, get_transform_cache_dir
from models.sequence import SequenceClip
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

    def run(self):
        """Run shuffle + pre-render pipeline."""
        self._log_start()
        try:
            # Step 1: Shuffle clips
            self.progress_message.emit("Shuffling clips...")
            sorted_clips = generate_sequence(
                algorithm="shuffle",
                clips=self._clips,
                clip_count=len(self._clips),
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            has_transforms = self._hflip or self._vflip or self._reverse

            if not has_transforms:
                # No transforms — just emit shuffled clips with empty transform info
                result = [
                    (clip, source, {"hflip": False, "vflip": False, "reverse": False, "prerendered_path": None})
                    for clip, source in sorted_clips
                ]
                self.finished_sequence.emit(result)
                return

            # Step 2: Create ephemeral SequenceClips and assign random transforms
            transform_options = {
                "hflip": self._hflip,
                "vflip": self._vflip,
                "reverse": self._reverse,
            }
            temp_seq_clips = [
                SequenceClip(source_clip_id=clip.id, source_id=source.id)
                for clip, source in sorted_clips
            ]
            assign_random_transforms(temp_seq_clips, transform_options)

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Step 3: Pre-render clips with transforms
            self.progress_message.emit("Pre-rendering transformed clips...")
            clips_with_transforms = [
                (clip, source, {"hflip": sc.hflip, "vflip": sc.vflip, "reverse": sc.reverse})
                for (clip, source), sc in zip(sorted_clips, temp_seq_clips)
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

            # Build result with transform info and prerendered paths
            result = []
            for (clip, source, prerendered_path), sc in zip(rendered, temp_seq_clips):
                result.append((clip, source, {
                    "hflip": sc.hflip,
                    "vflip": sc.vflip,
                    "reverse": sc.reverse,
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

    sequence_ready = Signal(list)  # list of (Clip, Source, dict)

    def __init__(self, clips: list, parent=None):
        """
        Args:
            clips: List of (Clip, Source) tuples to shuffle.
        """
        super().__init__(parent)
        self._clips = clips
        self._worker = None
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
        self._worker.progress_update.connect(self._on_progress_update)
        self._worker.progress_message.connect(self._on_progress_message)
        self._worker.finished_sequence.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(int, int)
    def _on_progress_update(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)
            self._progress_label.setText(f"Pre-rendering clip {current + 1} of {total}...")

    @Slot(str)
    def _on_progress_message(self, message: str):
        self._progress_label.setText(message)

    @Slot(list)
    def _on_finished(self, sequence_data: list):
        self.sequence_ready.emit(sequence_data)
        self.accept()

    @Slot(str)
    def _on_error(self, error_msg: str):
        logger.error("Dice Roll error: %s", error_msg)
        self._stack.setCurrentIndex(0)
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
