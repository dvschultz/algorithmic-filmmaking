"""Signature Style dialog — drawing-based sequencer.

Users draw (or import an image) on a canvas, then the system interprets
the drawing left-to-right to select and arrange clips. Two modes:

- Parametric: pixel-level reading of Y=pacing, color=color match
- VLM: vision-language model interpretation of visual meaning

Both produce the same DrawingSegment[] intermediate, consumed by a shared
clip matching algorithm.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QFont, QScreen

from ui.theme import theme, UISizes
from ui.widgets.drawing_canvas import DrawingCanvas

logger = logging.getLogger(__name__)

# FPS options
_FPS_OPTIONS = ["24", "25", "29.97", "30", "60"]
_DEFAULT_FPS_INDEX = 3  # 30 fps


class SignatureStyleWorker(QThread):
    """Background worker for Signature Style sequence generation."""

    progress = Signal(str)  # Status message
    finished_sequence = Signal(list)  # list of (Clip, Source, in_point, out_point)
    error = Signal(str)

    def __init__(
        self,
        image,  # QImage
        clips,  # list of (Clip, Source)
        mode: str,  # "parametric" or "vlm"
        total_duration_seconds: float,
        fps: float,
        sample_count: int = 64,
        llm_client=None,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._clips = clips
        self._mode = mode
        self._total_duration = total_duration_seconds
        self._fps = fps
        self._sample_count = sample_count
        self._llm_client = llm_client
        self._cancelled = False

    def run(self):
        """Run sequence generation."""
        try:
            if self._mode == "parametric":
                self._run_parametric()
            else:
                self._run_vlm()
        except Exception as e:
            if not self._cancelled:
                logger.error(f"Signature Style generation error: {e}")
                self.error.emit(str(e))

    def _run_parametric(self):
        """Run parametric mode."""
        from core.remix.signature_style import (
            build_sequence_from_matches,
            match_clips_to_segments,
            sample_drawing_parametric,
        )

        self.progress.emit("Sampling drawing...")
        segments = sample_drawing_parametric(
            self._image,
            self._total_duration,
            self._sample_count,
        )

        if not segments:
            self.error.emit("No drawing content found. Draw something on the canvas first.")
            return

        if self._cancelled:
            return

        self.progress.emit(f"Matching {len(segments)} segments to clips...")
        matches = match_clips_to_segments(segments, self._clips, allow_reuse=True)

        if not matches:
            self.error.emit("Could not match any clips to drawing segments.")
            return

        if self._cancelled:
            return

        self.progress.emit("Building sequence...")
        sequence = build_sequence_from_matches(matches, self._fps)

        if not self._cancelled:
            self.finished_sequence.emit(sequence)

    def _run_vlm(self):
        """Run VLM mode."""
        from core.remix.drawing_vlm import interpret_drawing_vlm
        from core.remix.signature_style import (
            build_sequence_from_matches,
            match_clips_to_segments,
        )

        def on_progress(current, total):
            if not self._cancelled:
                self.progress.emit(f"Interpreting slice {current} of {total}...")

        self.progress.emit("Analyzing drawing with VLM...")
        segments = interpret_drawing_vlm(
            self._image,
            self._total_duration,
            self._llm_client,
            progress_callback=on_progress,
        )

        if not segments:
            self.error.emit(
                "VLM could not interpret the drawing. "
                "Try a different drawing or switch to Parametric mode."
            )
            return

        if self._cancelled:
            return

        self.progress.emit(f"Matching {len(segments)} segments to clips...")
        matches = match_clips_to_segments(segments, self._clips, allow_reuse=True)

        if not matches:
            self.error.emit("Could not match any clips to drawing segments.")
            return

        if self._cancelled:
            return

        self.progress.emit("Building sequence...")
        sequence = build_sequence_from_matches(matches, self._fps)

        if not self._cancelled:
            self.finished_sequence.emit(sequence)

    def cancel(self):
        """Cancel the worker."""
        self._cancelled = True


class SignatureStyleDialog(QDialog):
    """Drawing-based sequencer dialog.

    Opens as a large modal dialog with a drawing canvas. Users draw or import
    an image, configure mode and parameters, and generate a sequence.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source, in_point, out_point) tuples
    """

    sequence_ready = Signal(list)

    PAGE_CANVAS = 0
    PAGE_PROGRESS = 1

    def __init__(
        self,
        clips,
        sources_by_id,
        parent=None,
    ):
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.worker: Optional[SignatureStyleWorker] = None

        self.setWindowTitle("Signature Style")
        self.setModal(True)

        # Size to 80% of screen
        screen = QScreen.availableGeometry(self.screen())
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        self.setMinimumSize(800, 600)

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

        # Page 0: Canvas + controls
        self.canvas_page = self._create_canvas_page()
        self.stack.addWidget(self.canvas_page)

        # Page 1: Progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.generate_btn = QPushButton("Generate Sequence")
        self.generate_btn.clicked.connect(self._on_generate)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.generate_btn)
        layout.addLayout(btn_layout)

    def _create_canvas_page(self) -> QWidget:
        """Create the main canvas page with controls."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Top bar: Duration, FPS, Mode, Granularity ──
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        # Duration input
        dur_label = QLabel("Duration:")
        dur_label.setFixedWidth(60)
        top_bar.addWidget(dur_label)


        self.duration_input = QLineEdit("2:30")
        self.duration_input.setFixedWidth(80)
        self.duration_input.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self.duration_input.setPlaceholderText("m:ss")
        self.duration_input.setToolTip("Target output duration (e.g., 2:30 for two and a half minutes)")
        top_bar.addWidget(self.duration_input)

        # FPS dropdown
        fps_label = QLabel("FPS:")
        fps_label.setFixedWidth(30)
        top_bar.addWidget(fps_label)


        self.fps_combo = QComboBox()
        self.fps_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.fps_combo.addItems(_FPS_OPTIONS)
        self.fps_combo.setCurrentIndex(_DEFAULT_FPS_INDEX)
        top_bar.addWidget(self.fps_combo)

        top_bar.addSpacing(20)

        # Mode toggle
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(40)
        top_bar.addWidget(mode_label)


        self.mode_parametric_btn = QPushButton("Parametric")
        self.mode_parametric_btn.setCheckable(True)
        self.mode_parametric_btn.setChecked(True)

        self.mode_vlm_btn = QPushButton("VLM")
        self.mode_vlm_btn.setCheckable(True)

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_parametric_btn, 0)
        self._mode_group.addButton(self.mode_vlm_btn, 1)
        self._mode_group.idClicked.connect(self._on_mode_changed)

        top_bar.addWidget(self.mode_parametric_btn)
        top_bar.addWidget(self.mode_vlm_btn)

        top_bar.addSpacing(20)

        # Granularity slider (parametric only)
        self.granularity_label = QLabel("Granularity:")
        self.granularity_label.setFixedWidth(75)
        top_bar.addWidget(self.granularity_label)

        self.granularity_slider = QSlider(Qt.Horizontal)
        self.granularity_slider.setMinimum(8)
        self.granularity_slider.setMaximum(128)
        self.granularity_slider.setValue(64)
        self.granularity_slider.setFixedWidth(150)
        self.granularity_slider.setToolTip("Number of segments to sample from the drawing")
        top_bar.addWidget(self.granularity_slider)

        self.granularity_value = QLabel("64")
        self.granularity_value.setFixedWidth(30)
        self.granularity_slider.valueChanged.connect(
            lambda v: self.granularity_value.setText(str(v))
        )
        top_bar.addWidget(self.granularity_value)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # ── Y-axis hint ──
        canvas_container = QHBoxLayout()

        y_label = QLabel("Fast cuts\n\u2191\n\n\n\u2193\nSlow holds")
        y_label.setAlignment(Qt.AlignCenter)
        y_label.setFixedWidth(60)
        canvas_container.addWidget(y_label)


        # ── Drawing canvas ──
        self.canvas = DrawingCanvas()
        canvas_container.addWidget(self.canvas, 1)

        layout.addLayout(canvas_container, 1)

        # ── Tool bar ──
        tool_bar = QHBoxLayout()
        tool_bar.setSpacing(8)

        self.pen_btn = QPushButton("Pen")
        self.pen_btn.setCheckable(True)
        self.pen_btn.setChecked(True)
        self.pen_btn.clicked.connect(lambda: self._set_tool("pen"))

        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self._set_tool("eraser"))

        self._tool_group = QButtonGroup(self)
        self._tool_group.addButton(self.pen_btn, 0)
        self._tool_group.addButton(self.eraser_btn, 1)

        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self._on_color_pick)

        # Thickness slider
        thickness_label = QLabel("Size:")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(10, 100)
        self.thickness_slider.setValue(10)
        self.thickness_slider.setFixedWidth(100)
        self.thickness_slider.valueChanged.connect(self._on_thickness_changed)
        self.thickness_value = QLabel("10")
        self.thickness_value.setFixedWidth(24)
        self._thickness_label = thickness_label

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.canvas.undo)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.canvas.redo)

        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self._on_import_image)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)

        tool_bar.addWidget(self.pen_btn)
        tool_bar.addWidget(self.eraser_btn)
        tool_bar.addWidget(self.color_btn)
        tool_bar.addSpacing(12)
        tool_bar.addWidget(self._thickness_label)
        tool_bar.addWidget(self.thickness_slider)
        tool_bar.addWidget(self.thickness_value)
        tool_bar.addSpacing(12)
        tool_bar.addWidget(self.undo_btn)
        tool_bar.addWidget(self.redo_btn)
        tool_bar.addSpacing(12)
        tool_bar.addWidget(self.import_btn)
        tool_bar.addWidget(self.clear_btn)
        tool_bar.addStretch()

        layout.addLayout(tool_bar)

        return page

    def _create_progress_page(self) -> QWidget:
        """Create the progress page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Generating Sequence...")
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
    # Actions
    # ──────────────────────────────────────────────────────────

    def _set_tool(self, tool: str):
        """Switch drawing tool."""
        self.canvas.set_tool(tool)

    def _on_color_pick(self):
        """Open color picker."""
        color = QColorDialog.getColor(self.canvas.color(), self, "Pick pen color")
        if color.isValid():
            self.canvas.set_color(color)

    def _on_thickness_changed(self, value: int):
        """Update pen thickness from slider."""
        self.canvas.set_pen_width(value)
        self.thickness_value.setText(str(value))

    def _on_import_image(self):
        """Import an image onto the canvas."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
        )
        if path:
            if not self.canvas.load_image(path):
                QMessageBox.warning(
                    self,
                    "Import Failed",
                    "Could not load the image. Ensure it is a valid image file "
                    "and dimensions do not exceed 4096x4096.",
                )

    @Slot(int)
    def _on_mode_changed(self, mode_id: int):
        """Toggle between parametric and VLM mode."""
        is_parametric = mode_id == 0
        self.granularity_label.setVisible(is_parametric)
        self.granularity_slider.setVisible(is_parametric)
        self.granularity_value.setVisible(is_parametric)

    def _parse_duration(self) -> Optional[float]:
        """Parse the duration input (m:ss or seconds) to total seconds."""
        text = self.duration_input.text().strip()
        if not text:
            return None

        try:
            if ":" in text:
                parts = text.split(":")
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                return minutes * 60 + seconds
            else:
                return float(text)
        except (ValueError, IndexError):
            return None

    def _get_fps(self) -> float:
        """Get the selected FPS value."""
        try:
            return float(self.fps_combo.currentText())
        except ValueError:
            return 30.0

    def _on_generate(self):
        """Start sequence generation."""
        # Validate duration
        duration = self._parse_duration()
        if not duration or duration <= 0:
            QMessageBox.warning(
                self, "Invalid Duration",
                "Please enter a valid duration (e.g., 2:30 for two and a half minutes)."
            )
            return

        # Validate clips available
        if not self.clips:
            QMessageBox.warning(self, "No Clips", "No clips available for sequencing.")
            return

        # Check for missing analysis
        mode = "parametric" if self.mode_parametric_btn.isChecked() else "vlm"
        from core.remix.signature_style import check_missing_analysis
        missing = check_missing_analysis(self.clips, mode)
        if missing:
            details = ", ".join(f"{count} clips need {op}" for op, count in missing.items())
            reply = QMessageBox.question(
                self,
                "Analysis Required",
                f"Some clips are missing metadata needed for matching:\n{details}\n\n"
                "The algorithm will still work but matching quality may be reduced.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                return

        # Get clip pairs
        clip_pairs = []
        for clip in self.clips:
            source = self.sources_by_id.get(clip.source_id)
            if source:
                clip_pairs.append((clip, source))

        if not clip_pairs:
            QMessageBox.warning(self, "No Sources", "Could not find sources for clips.")
            return

        # Warn if clip pool is very small
        if len(clip_pairs) < 5:
            reply = QMessageBox.question(
                self,
                "Small Clip Pool",
                f"Only {len(clip_pairs)} clips available. Clips will be reused "
                "heavily to fill the sequence. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                return

        # Get mode and parameters
        mode = "parametric" if self.mode_parametric_btn.isChecked() else "vlm"
        fps = self._get_fps()
        sample_count = self.granularity_slider.value()

        # For VLM mode, get LLM client
        llm_client = None
        if mode == "vlm":
            try:
                from core.llm_client import get_llm_client
                llm_client = get_llm_client()
            except Exception as e:
                QMessageBox.warning(
                    self, "VLM Not Available",
                    f"Could not initialize LLM client: {e}\n\n"
                    "Configure a vision-capable model in Settings, or use Parametric mode."
                )
                return

        # Switch to progress page
        self.stack.setCurrentIndex(self.PAGE_PROGRESS)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")

        # Start worker
        self.worker = SignatureStyleWorker(
            image=self.canvas.get_image(),
            clips=clip_pairs,
            mode=mode,
            total_duration_seconds=duration,
            fps=fps,
            sample_count=sample_count,
            llm_client=llm_client,
            parent=self,
        )
        self.worker.progress.connect(self._on_progress, Qt.UniqueConnection)
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
        self.sequence_ready.emit(sequence)
        self.accept()

    @Slot(str)
    def _on_error(self, message: str):
        """Handle generation error."""
        self.stack.setCurrentIndex(self.PAGE_CANVAS)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Sequence")
        QMessageBox.warning(self, "Generation Failed", message)

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
            QPushButton:checked {{
                background-color: {t.colors.accent_blue};
                color: white;
            }}
            QLineEdit {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 4px;
            }}
            QComboBox {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 4px;
            }}
            QSlider::groove:horizontal {{
                background: {t.colors.background_tertiary};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {t.colors.accent_blue};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
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
