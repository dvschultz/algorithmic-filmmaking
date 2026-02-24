"""Configuration dialog for Reference-Guided Remixing.

Allows the artist to select a reference source, configure dimension weights,
and generate a matched sequence from their clip library.
"""

import logging
from typing import Any, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QProgressBar,
    QWidget,
    QGroupBox,
    QMessageBox,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from ui.theme import theme, UISizes

logger = logging.getLogger(__name__)


# Dimension display configuration
DIMENSION_INFO = {
    "color": {
        "label": "Color",
        "tooltip": "Match dominant color hue",
        "analysis_key": "colors",
        "default_weight": 80,
    },
    "brightness": {
        "label": "Brightness",
        "tooltip": "Match average luminance",
        "analysis_key": "brightness",
        "default_weight": 40,
    },
    "shot_scale": {
        "label": "Shot Scale",
        "tooltip": "Match camera distance (wide to close-up)",
        "analysis_key": "shots",
        "default_weight": 60,
    },
    "audio": {
        "label": "Audio Energy",
        "tooltip": "Match audio volume level",
        "analysis_key": "volume",
        "default_weight": 20,
    },
    "embedding": {
        "label": "Visual Match",
        "tooltip": "Match overall visual appearance (DINOv2 embeddings)",
        "analysis_key": "embeddings",
        "default_weight": 100,
    },
    "movement": {
        "label": "Movement",
        "tooltip": "Match camera movement type (requires Rich Analysis)",
        "analysis_key": "cinematography",
        "default_weight": 40,
    },
    "duration": {
        "label": "Duration",
        "tooltip": "Match clip length",
        "analysis_key": None,  # Always available
        "default_weight": 60,
    },
}


class ReferenceMatchWorker(QThread):
    """Background worker for reference-guided matching."""

    progress = Signal(str)
    match_ready = Signal(list)  # List of (Clip, Source) tuples
    error = Signal(str)

    def __init__(
        self,
        reference_clips,
        user_clips,
        weights,
        allow_repeats,
        match_reference_timing,
        parent=None,
    ):
        super().__init__(parent)
        self.reference_clips = reference_clips
        self.user_clips = user_clips
        self.weights = weights
        self.allow_repeats = allow_repeats
        self.match_reference_timing = match_reference_timing
        self._cancelled = False

    def run(self):
        try:
            from core.remix.reference_match import reference_guided_match

            self.progress.emit("Matching clips to reference structure...")

            matched = reference_guided_match(
                reference_clips=self.reference_clips,
                user_clips=self.user_clips,
                weights=self.weights,
                allow_repeats=self.allow_repeats,
                match_reference_timing=self.match_reference_timing,
            )

            if not self._cancelled:
                self.match_ready.emit(matched)

        except Exception as e:
            if not self._cancelled:
                logger.error(f"Reference matching error: {e}")
                self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True


class ReferenceGuideDialog(QDialog):
    """Dialog for configuring reference-guided remixing.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source) tuples when complete
    """

    sequence_ready = Signal(list)

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        project: Any,
        parent=None,
    ):
        """Initialize the dialog.

        Args:
            clips: List of (Clip, Source) tuples — all available clips
            sources_by_id: Dict mapping source_id to Source objects
            project: Project object (for source listing)
            parent: Parent widget
        """
        super().__init__(parent)
        self.all_clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self.worker = None

        # Determine which dimensions have data
        from core.remix.reference_match import get_active_dimensions_for_clips
        all_clip_objects = [clip for clip, _ in clips]
        self._available_dimensions = get_active_dimensions_for_clips(all_clip_objects)

        self.setWindowTitle("Reference Guide")
        self.setMinimumSize(550, 500)
        self.setModal(True)

        self._dimension_widgets: dict[str, dict] = {}
        self._setup_ui()
        self._apply_theme()
        self._update_clip_counts()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("Reference Guide")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._header = header

        desc = QLabel(
            "Match your clips to a reference video's structure. "
            "Select which video is the reference, then adjust dimension weights."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        self._desc = desc

        layout.addSpacing(8)

        # Reference source selector
        source_layout = QHBoxLayout()
        source_label = QLabel("Reference Source:")
        source_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
        source_layout.addWidget(source_label)
        self._source_label = source_label

        self.source_combo = QComboBox()
        self.source_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.source_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self._populate_source_combo()
        self.source_combo.currentIndexChanged.connect(self._update_clip_counts)
        source_layout.addWidget(self.source_combo)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        layout.addSpacing(8)

        # Dimension weights group
        weights_group = QGroupBox("Dimension Weights")
        weights_layout = QVBoxLayout(weights_group)
        weights_layout.setSpacing(6)

        for dim_key, info in DIMENSION_INFO.items():
            row = self._create_dimension_row(dim_key, info)
            weights_layout.addLayout(row)

        layout.addWidget(weights_group)
        self._weights_group = weights_group

        # Options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(4)

        self.allow_repeats_check = QCheckBox("Allow Repeats")
        self.allow_repeats_check.setToolTip(
            "Allow the same clip to match multiple reference positions"
        )
        options_layout.addWidget(self.allow_repeats_check)

        self.match_timing_check = QCheckBox("Match Reference Timing")
        self.match_timing_check.setToolTip(
            "Trim matched clips to match reference clip durations"
        )
        options_layout.addWidget(self.match_timing_check)

        layout.addLayout(options_layout)

        # Clip counts
        self.counts_label = QLabel("")
        layout.addWidget(self.counts_label)
        self._counts_label = self.counts_label

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.generate_btn = QPushButton("Generate Sequence")
        self.generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self.generate_btn)

        layout.addLayout(btn_layout)

    def _populate_source_combo(self):
        """Populate source dropdown with all project sources."""
        self.source_combo.clear()
        for source_id, source in self.sources_by_id.items():
            self.source_combo.addItem(source.filename, source_id)

    def _create_dimension_row(self, dim_key: str, info: dict) -> QHBoxLayout:
        """Create a row with checkbox + slider + value label for a dimension."""
        row = QHBoxLayout()
        row.setSpacing(8)

        checkbox = QCheckBox(info["label"])
        checkbox.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        checkbox.setToolTip(info["tooltip"])

        # Enable if data available, disable with tooltip if not
        is_available = dim_key in self._available_dimensions
        checkbox.setEnabled(is_available)
        checkbox.setChecked(is_available)
        if not is_available:
            checkbox.setToolTip(
                f"{info['tooltip']}\n\n"
                f"Not available — run {info.get('analysis_key', 'analysis')} "
                f"in the Analyze tab first"
            )

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(info["default_weight"] if is_available else 0)
        slider.setEnabled(is_available)
        slider.setTickPosition(QSlider.NoTicks)

        value_label = QLabel(f"{slider.value()}%")
        value_label.setFixedWidth(35)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Wire checkbox to enable/disable slider
        checkbox.toggled.connect(lambda checked, s=slider: s.setEnabled(checked))
        slider.valueChanged.connect(lambda val, lbl=value_label: lbl.setText(f"{val}%"))

        row.addWidget(checkbox)
        row.addWidget(slider)
        row.addWidget(value_label)

        self._dimension_widgets[dim_key] = {
            "checkbox": checkbox,
            "slider": slider,
            "value_label": value_label,
        }

        return row

    def _update_clip_counts(self):
        """Update clip count display based on selected reference source."""
        ref_source_id = self.source_combo.currentData()
        if not ref_source_id:
            self.counts_label.setText("")
            return

        ref_count = sum(1 for clip, _ in self.all_clips if clip.source_id == ref_source_id)
        user_count = sum(1 for clip, _ in self.all_clips if clip.source_id != ref_source_id)

        self.counts_label.setText(
            f"Reference: {ref_count} clips  |  Your footage: {user_count} clips"
        )

        # Disable generate if no clips on either side
        self.generate_btn.setEnabled(ref_count > 0 and user_count > 0)

    def _get_weights(self) -> dict[str, float]:
        """Get current dimension weights from UI."""
        weights = {}
        for dim_key, widgets in self._dimension_widgets.items():
            if widgets["checkbox"].isChecked():
                weights[dim_key] = widgets["slider"].value() / 100.0
            else:
                weights[dim_key] = 0.0
        return weights

    def _on_generate(self):
        """Start the matching process."""
        ref_source_id = self.source_combo.currentData()
        if not ref_source_id:
            QMessageBox.warning(self, "No Reference", "Select a reference source first.")
            return

        weights = self._get_weights()
        active = {k: v for k, v in weights.items() if v > 0}
        if not active:
            QMessageBox.warning(
                self,
                "No Dimensions",
                "Enable at least one dimension with a weight above 0%."
            )
            return

        # Split clips into reference and user pools
        reference_clips = [
            (clip, source) for clip, source in self.all_clips
            if clip.source_id == ref_source_id
        ]
        user_clips = [
            (clip, source) for clip, source in self.all_clips
            if clip.source_id != ref_source_id
        ]

        if not reference_clips:
            QMessageBox.warning(self, "No Reference Clips", "Reference source has no clips.")
            return
        if not user_clips:
            QMessageBox.warning(self, "No User Clips", "No clips available from other sources.")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Matching clips to reference structure...")
        self.generate_btn.setEnabled(False)

        self.worker = ReferenceMatchWorker(
            reference_clips=reference_clips,
            user_clips=user_clips,
            weights=weights,
            allow_repeats=self.allow_repeats_check.isChecked(),
            match_reference_timing=self.match_timing_check.isChecked(),
            parent=self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.match_ready.connect(self._on_match_ready)
        self.worker.error.connect(self._on_match_error)
        self.worker.start()

    def _on_progress(self, message: str):
        self.progress_label.setText(message)

    def _on_match_ready(self, matched_clips: list):
        """Handle matching completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if not matched_clips:
            QMessageBox.information(
                self,
                "No Matches",
                "No clips could be matched. Try adjusting dimension weights "
                "or enabling Allow Repeats."
            )
            self.generate_btn.setEnabled(True)
            return

        ref_source_id = self.source_combo.currentData()
        ref_count = sum(1 for clip, _ in self.all_clips if clip.source_id == ref_source_id)
        unmatched = ref_count - len(matched_clips)

        msg = f"Matched {len(matched_clips)} clips"
        if unmatched > 0:
            msg += f" ({unmatched} reference positions unmatched)"
        logger.info(msg)

        # Store config on the result for sequence metadata
        self._last_weights = self._get_weights()
        self._last_ref_source_id = ref_source_id
        self._last_allow_repeats = self.allow_repeats_check.isChecked()
        self._last_match_timing = self.match_timing_check.isChecked()

        self.sequence_ready.emit(matched_clips)
        self.accept()

    def _on_match_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Matching failed: {error_msg}")

    def _on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        self.reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        event.accept()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme().background_primary};
            }}
            QLabel {{
                color: {theme().text_primary};
            }}
            QGroupBox {{
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QCheckBox {{
                color: {theme().text_primary};
                spacing: 6px;
            }}
            QCheckBox:disabled {{
                color: {theme().text_muted};
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {theme().border_primary};
                height: 6px;
                background: {theme().background_tertiary};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {theme().accent_blue};
                border: 1px solid {theme().border_primary};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:disabled {{
                background: {theme().text_muted};
            }}
            QComboBox {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 28px;
            }}
            QComboBox:hover {{
                background-color: {theme().background_elevated};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QPushButton {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {theme().background_elevated};
            }}
            QPushButton:disabled {{
                background-color: {theme().background_secondary};
                color: {theme().text_muted};
            }}
            QProgressBar {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme().accent_blue};
                border-radius: 3px;
            }}
            QFrame[frameShape="4"] {{
                color: {theme().border_primary};
            }}
        """)

        if hasattr(self, '_desc'):
            self._desc.setStyleSheet(f"color: {theme().text_secondary};")
        if hasattr(self, '_counts_label'):
            self._counts_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: 11px;")
