"""Eyes Without a Face dialog — gaze-based sequencing.

Three modes:
- Eyeline Match: pair clips with complementary gaze directions (shot/reverse-shot)
- Filter: keep only clips matching a selected gaze category
- Rotation: arrange clips in monotonically progressing angle order
"""

import logging

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from ui.theme import theme, UISizes
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)

# Gaze category display names -> internal keys
_CATEGORY_DISPLAY_TO_KEY = {
    "At Camera": "at_camera",
    "Looking Left": "looking_left",
    "Looking Right": "looking_right",
    "Looking Up": "looking_up",
    "Looking Down": "looking_down",
}


class _EyesWorker(CancellableWorker):
    """Background worker for gaze sequencing."""

    progress_message = Signal(str)
    finished_sequence = Signal(list)  # list of (Clip, Source)

    def __init__(
        self,
        clips,  # list of (Clip, Source)
        mode: str,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._clips = clips
        self._mode = mode
        self._params = params

    def run(self):
        """Run the selected gaze sequencing algorithm."""
        self._log_start()
        try:
            from core.remix.gaze import eyeline_match, gaze_filter, gaze_rotation

            if self.is_cancelled():
                self._log_cancelled()
                return

            if self._mode == "eyeline_match":
                self.progress_message.emit("Pairing eyeline matches...")
                result = eyeline_match(
                    self._clips,
                    tolerance=self._params.get("tolerance", 20.0),
                )
            elif self._mode == "filter":
                category = self._params.get("category", "at_camera")
                self.progress_message.emit(f"Filtering by gaze: {category}...")
                result = gaze_filter(self._clips, category=category)
            elif self._mode == "rotation":
                self.progress_message.emit("Computing gaze rotation sequence...")
                result = gaze_rotation(
                    self._clips,
                    axis=self._params.get("axis", "yaw"),
                    range_start=self._params.get("range_start", -30.0),
                    range_end=self._params.get("range_end", 30.0),
                    ascending=self._params.get("ascending", True),
                )
            else:
                self.error.emit(f"Unknown mode: {self._mode}")
                return

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.finished_sequence.emit(result)
            self._log_complete()

        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Eyes Without a Face generation error: {e}", exc_info=True)
                self.error.emit("Gaze sequencing failed. Check logs for details.")


class EyesWithoutAFaceDialog(QDialog):
    """Gaze-based sequencer dialog.

    Opens as a modal dialog where users select a gaze sequencing mode,
    configure mode-specific parameters, and generate a filtered/sorted sequence.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source) tuples
    """

    sequence_ready = Signal(list)

    PAGE_CONFIG = 0
    PAGE_PROGRESS = 1

    # Mode indices for the mode parameter stack
    MODE_EYELINE = 0
    MODE_FILTER = 1
    MODE_ROTATION = 2

    def __init__(self, clips, parent=None):
        """Initialize the dialog.

        Args:
            clips: List of (Clip, Source) tuples
        """
        super().__init__(parent)
        self._clips = clips
        self.worker: _EyesWorker | None = None

        self.setWindowTitle("Eyes Without a Face")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.resize(560, 440)

        self._setup_ui()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 0: Config
        self.config_page = self._create_config_page()
        self.stack.addWidget(self.config_page)

        # Page 1: Progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.generate_btn)
        layout.addLayout(btn_layout)

    def _create_config_page(self) -> QWidget:
        """Create the configuration page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Title
        title = QLabel("Gaze-based sequencing")
        title_font = QFont()
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        mode_row.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.mode_combo.addItems(["Eyeline Match", "Filter", "Rotation"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Mode-specific parameter pages
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._create_eyeline_page())
        self.mode_stack.addWidget(self._create_filter_page())
        self.mode_stack.addWidget(self._create_rotation_page())
        layout.addWidget(self.mode_stack)

        layout.addStretch()
        return page

    def _create_eyeline_page(self) -> QWidget:
        """Create the Eyeline Match parameter page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        desc = QLabel(
            "Pairs clips with complementary gaze directions to simulate "
            "shot/reverse-shot editing."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Tolerance slider
        tol_row = QHBoxLayout()
        tol_label = QLabel("Tolerance:")
        tol_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        tol_row.addWidget(tol_label)

        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(5)
        self.tolerance_slider.setMaximum(30)
        self.tolerance_slider.setValue(20)
        self.tolerance_slider.setTickInterval(5)
        self.tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.tolerance_slider.valueChanged.connect(self._on_tolerance_changed)
        tol_row.addWidget(self.tolerance_slider)

        self.tolerance_label = QLabel("20\u00b0")
        self.tolerance_label.setFixedWidth(40)
        tol_row.addWidget(self.tolerance_label)

        layout.addLayout(tol_row)
        layout.addStretch()
        return page

    def _create_filter_page(self) -> QWidget:
        """Create the Filter parameter page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        desc = QLabel(
            "Keep only clips where the subject is looking in the selected direction."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Category dropdown
        cat_row = QHBoxLayout()
        cat_label = QLabel("Category:")
        cat_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        cat_row.addWidget(cat_label)

        self.category_combo = QComboBox()
        self.category_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.category_combo.addItems(list(_CATEGORY_DISPLAY_TO_KEY.keys()))
        cat_row.addWidget(self.category_combo)
        cat_row.addStretch()

        layout.addLayout(cat_row)
        layout.addStretch()
        return page

    def _create_rotation_page(self) -> QWidget:
        """Create the Rotation parameter page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        desc = QLabel(
            "Arrange clips so gaze direction smoothly rotates across a range of angles."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Axis selector
        axis_row = QHBoxLayout()
        axis_label = QLabel("Axis:")
        axis_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        axis_row.addWidget(axis_label)

        self.axis_combo = QComboBox()
        self.axis_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.axis_combo.addItems(["Yaw", "Pitch"])
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        axis_row.addWidget(self.axis_combo)
        axis_row.addStretch()
        layout.addLayout(axis_row)

        # Range start slider
        start_row = QHBoxLayout()
        start_label = QLabel("Range start:")
        start_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        start_row.addWidget(start_label)

        self.range_start_slider = QSlider(Qt.Horizontal)
        self.range_start_slider.setMinimum(-30)
        self.range_start_slider.setMaximum(30)
        self.range_start_slider.setValue(-30)
        self.range_start_slider.setTickInterval(10)
        self.range_start_slider.setTickPosition(QSlider.TicksBelow)
        self.range_start_slider.valueChanged.connect(self._on_range_start_changed)
        start_row.addWidget(self.range_start_slider)

        self.range_start_label = QLabel("-30\u00b0")
        self.range_start_label.setFixedWidth(40)
        start_row.addWidget(self.range_start_label)

        layout.addLayout(start_row)

        # Range end slider
        end_row = QHBoxLayout()
        end_label = QLabel("Range end:")
        end_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        end_row.addWidget(end_label)

        self.range_end_slider = QSlider(Qt.Horizontal)
        self.range_end_slider.setMinimum(-30)
        self.range_end_slider.setMaximum(30)
        self.range_end_slider.setValue(30)
        self.range_end_slider.setTickInterval(10)
        self.range_end_slider.setTickPosition(QSlider.TicksBelow)
        self.range_end_slider.valueChanged.connect(self._on_range_end_changed)
        end_row.addWidget(self.range_end_slider)

        self.range_end_label = QLabel("30\u00b0")
        self.range_end_label.setFixedWidth(40)
        end_row.addWidget(self.range_end_label)

        layout.addLayout(end_row)

        # Direction toggle
        dir_row = QHBoxLayout()
        dir_label = QLabel("Direction:")
        dir_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        dir_row.addWidget(dir_label)

        self.direction_combo = QComboBox()
        self.direction_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.direction_combo.addItems(["Ascending", "Descending"])
        dir_row.addWidget(self.direction_combo)
        dir_row.addStretch()
        layout.addLayout(dir_row)

        layout.addStretch()
        return page

    def _create_progress_page(self) -> QWidget:
        """Create the progress page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Generating gaze sequence...")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        layout.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Starting...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        layout.addStretch()
        return page

    # ──────────────────────────────────────────────────────────
    # Mode switching
    # ──────────────────────────────────────────────────────────

    @Slot(int)
    def _on_mode_changed(self, index: int):
        """Switch the mode-specific parameter page."""
        self.mode_stack.setCurrentIndex(index)

    @Slot(int)
    def _on_tolerance_changed(self, value: int):
        """Update the tolerance label."""
        self.tolerance_label.setText(f"{value}\u00b0")

    @Slot(int)
    def _on_axis_changed(self, index: int):
        """Update slider ranges when axis changes."""
        if index == 0:
            # Yaw: -30 to +30
            self.range_start_slider.setMinimum(-30)
            self.range_start_slider.setMaximum(30)
            self.range_start_slider.setValue(-30)
            self.range_end_slider.setMinimum(-30)
            self.range_end_slider.setMaximum(30)
            self.range_end_slider.setValue(30)
        else:
            # Pitch: -20 to +20
            self.range_start_slider.setMinimum(-20)
            self.range_start_slider.setMaximum(20)
            self.range_start_slider.setValue(-20)
            self.range_end_slider.setMinimum(-20)
            self.range_end_slider.setMaximum(20)
            self.range_end_slider.setValue(20)
        self._on_range_start_changed(self.range_start_slider.value())
        self._on_range_end_changed(self.range_end_slider.value())

    @Slot(int)
    def _on_range_start_changed(self, value: int):
        """Update range start label."""
        self.range_start_label.setText(f"{value}\u00b0")

    @Slot(int)
    def _on_range_end_changed(self, value: int):
        """Update range end label."""
        self.range_end_label.setText(f"{value}\u00b0")

    # ──────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────

    def _on_generate(self):
        """Start gaze sequencing."""
        mode_index = self.mode_combo.currentIndex()

        if mode_index == self.MODE_EYELINE:
            mode = "eyeline_match"
            params = {"tolerance": float(self.tolerance_slider.value())}
        elif mode_index == self.MODE_FILTER:
            mode = "filter"
            display_cat = self.category_combo.currentText()
            params = {"category": _CATEGORY_DISPLAY_TO_KEY.get(display_cat, "at_camera")}
        elif mode_index == self.MODE_ROTATION:
            mode = "rotation"
            params = {
                "axis": "yaw" if self.axis_combo.currentIndex() == 0 else "pitch",
                "range_start": float(self.range_start_slider.value()),
                "range_end": float(self.range_end_slider.value()),
                "ascending": self.direction_combo.currentIndex() == 0,
            }
        else:
            return

        # Switch to progress page
        self.stack.setCurrentIndex(self.PAGE_PROGRESS)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")

        # Start worker
        self.worker = _EyesWorker(
            clips=self._clips,
            mode=mode,
            params=params,
            parent=self,
        )
        self.worker.progress_message.connect(self._on_progress, Qt.UniqueConnection)
        self.worker.finished_sequence.connect(self._on_finished, Qt.UniqueConnection)
        self.worker.error.connect(self._on_error, Qt.UniqueConnection)
        self.worker.start()

    @Slot(str)
    def _on_progress(self, message: str):
        """Update progress label."""
        self.progress_label.setText(message)

    @Slot(list)
    def _on_finished(self, sequence: list):
        """Handle generation completion."""
        if not sequence:
            self.progress_label.setText("No clips matched the criteria.")
            self.stack.setCurrentIndex(self.PAGE_CONFIG)
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate")
            QMessageBox.information(
                self,
                "No Matches",
                "No clips matched the gaze criteria.\n\n"
                "Try adjusting the parameters or running gaze analysis on more clips.",
            )
            return

        self.sequence_ready.emit(sequence)
        self.accept()

    @Slot(str)
    def _on_error(self, message: str):
        """Handle generation error."""
        self.stack.setCurrentIndex(self.PAGE_CONFIG)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate")
        QMessageBox.warning(self, "Error", message)

    def _on_cancel(self):
        """Cancel and close."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.reject()

    # ──────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────

    def _apply_theme(self):
        """Apply current theme colors."""
        t = theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t.colors.background_primary};
                color: {t.colors.text_primary};
            }}
            QLabel {{
                color: {t.colors.text_primary};
            }}
            QPushButton {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: {UISizes.BUTTON_MIN_HEIGHT}px;
            }}
            QPushButton:hover {{
                background-color: {t.colors.background_tertiary};
            }}
            QPushButton:disabled {{
                opacity: 0.5;
            }}
            QComboBox {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 4px;
            }}
            QSlider::groove:horizontal {{
                background-color: {t.colors.background_secondary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 3px;
                height: 6px;
            }}
            QSlider::handle:horizontal {{
                background-color: {t.colors.accent_blue};
                border-radius: 6px;
                width: 12px;
                margin: -4px 0;
            }}
            QProgressBar {{
                background-color: {t.colors.background_secondary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {t.colors.accent_blue};
                border-radius: 3px;
            }}
        """)
