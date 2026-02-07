"""Frames tab for frame-level extraction, browsing, and analysis."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QStackedWidget,
    QComboBox,
    QSpinBox,
    QSlider,
    QFileDialog,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.frame_browser import FrameBrowser
from ui.widgets import EmptyStateWidget
from ui.theme import theme, UISizes, TypeScale, Spacing
from models.frame import Frame

logger = logging.getLogger(__name__)


class FramesTab(BaseTab):
    """Tab for extracting, browsing, and analyzing individual frames.

    Supports two workflows:
    - Extract frames from video sources at configurable intervals
    - Import standalone images (PNG/JPG) from disk

    Signals:
        extract_frames_requested: Emitted with (source_id, mode, interval)
            when the user clicks Extract Frames.
        import_images_requested: Emitted with list of image file paths.
        analyze_frames_requested: Emitted with list of frame IDs to analyze.
        add_to_sequence_requested: Emitted with list of frame IDs to add.
        frames_selected: Emitted with list of frame IDs on selection change.
    """

    extract_frames_requested = Signal(str, str, int)  # source_id, mode, interval
    import_images_requested = Signal(list)  # list[Path]
    analyze_frames_requested = Signal(list)  # list[str] frame IDs
    add_to_sequence_requested = Signal(list)  # list[str] frame IDs
    frames_selected = Signal(list)  # list[str] frame IDs

    # State constants for stacked widget
    STATE_EMPTY = 0
    STATE_FRAMES = 1

    def __init__(self, parent=None):
        self._project = None
        self._sources: list = []  # Source objects, set by main window
        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Frames tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top controls bar
        controls = self._create_controls()
        layout.addLayout(controls)

        # Stacked widget for empty / content states
        self.state_stack = QStackedWidget()

        # State 0: No frames
        self.empty_widget = EmptyStateWidget(
            "No Frames",
            "Extract frames from a video source or import images from disk."
        )
        self.state_stack.addWidget(self.empty_widget)

        # State 1: Frame browser
        self.frame_browser = FrameBrowser()
        self.frame_browser.frames_selected.connect(self._on_frames_selected)
        self.frame_browser.frame_double_clicked.connect(self._on_frame_double_clicked)
        self.state_stack.addWidget(self.frame_browser)

        layout.addWidget(self.state_stack)

        # Start in empty state
        self.state_stack.setCurrentIndex(self.STATE_EMPTY)

    def _create_controls(self) -> QHBoxLayout:
        """Create the top controls bar."""
        controls = QHBoxLayout()
        controls.setContentsMargins(
            Spacing.MD, Spacing.MD, Spacing.MD, Spacing.SM
        )
        controls.setSpacing(Spacing.SM)

        # Source selector
        source_label = QLabel("Source:")
        controls.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.source_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self.source_combo.setToolTip("Select a video source to extract frames from")
        controls.addWidget(self.source_combo)

        controls.addSpacing(Spacing.SM)

        # Extraction mode
        mode_label = QLabel("Mode:")
        controls.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.mode_combo.addItems(["Every N frames", "Every N seconds"])
        self.mode_combo.setToolTip("Frame extraction mode")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_combo)

        controls.addSpacing(Spacing.SM)

        # Interval spinner
        interval_label = QLabel("Interval:")
        controls.addWidget(interval_label)

        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(9999)
        self.interval_spin.setValue(30)
        self.interval_spin.setSuffix(" frames")
        self.interval_spin.setToolTip("Extract every N frames or seconds")
        controls.addWidget(self.interval_spin)

        controls.addSpacing(Spacing.SM)

        # Extract button
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.setToolTip("Extract frames from the selected source")
        self.extract_btn.clicked.connect(self._on_extract_clicked)
        controls.addWidget(self.extract_btn)

        controls.addSpacing(Spacing.SM)

        # Import images button
        self.import_btn = QPushButton("Import Images...")
        self.import_btn.setToolTip("Import image files (PNG, JPG) as frames")
        self.import_btn.clicked.connect(self._on_import_clicked)
        controls.addWidget(self.import_btn)

        # Separator
        controls.addSpacing(Spacing.LG)
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {theme().text_muted};")
        controls.addWidget(sep)
        controls.addSpacing(Spacing.LG)

        # Analyze selected button
        self.analyze_btn = QPushButton("Analyze Selected")
        self.analyze_btn.setToolTip("Run analysis on selected frames")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        controls.addWidget(self.analyze_btn)

        controls.addSpacing(Spacing.SM)

        # Add to Sequence button
        self.add_to_seq_btn = QPushButton("Add to Sequence")
        self.add_to_seq_btn.setToolTip("Add selected frames to the sequence timeline")
        self.add_to_seq_btn.setEnabled(False)
        self.add_to_seq_btn.clicked.connect(self._on_add_to_sequence_clicked)
        controls.addWidget(self.add_to_seq_btn)

        controls.addStretch()

        # Zoom slider
        zoom_label = QLabel("Zoom:")
        controls.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(5)
        self.zoom_slider.setValue(3)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(1)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.setToolTip("Thumbnail zoom level")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        controls.addWidget(self.zoom_slider)

        # Selection count label
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        controls.addWidget(self.selection_label)

        return controls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, project):
        """Store a reference to the active project."""
        self._project = project

    def set_sources(self, sources: list):
        """Update the source selector combo with available sources.

        Args:
            sources: List of Source objects from the project.
        """
        self._sources = sources
        self.source_combo.blockSignals(True)
        current_text = self.source_combo.currentText()
        self.source_combo.clear()
        for source in sources:
            self.source_combo.addItem(source.filename, source.id)
        # Restore previous selection if still available
        idx = self.source_combo.findText(current_text)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        self.source_combo.blockSignals(False)

    def update_frame_browser(self):
        """Refresh the frame browser with frames from the project."""
        if not self._project:
            return

        frames = self._project.frames
        if frames:
            self.frame_browser.set_frames(frames)
            self.state_stack.setCurrentIndex(self.STATE_FRAMES)
        else:
            self.frame_browser.clear()
            self.state_stack.setCurrentIndex(self.STATE_EMPTY)

    def on_tab_activated(self):
        """Refresh data when tab becomes visible."""
        self.update_frame_browser()
        # Refresh source list from project
        if self._project:
            self.set_sources(self._project.sources)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int):
        """Update interval spinner suffix based on extraction mode."""
        if index == 0:
            self.interval_spin.setSuffix(" frames")
        else:
            self.interval_spin.setSuffix(" sec")

    def _on_extract_clicked(self):
        """Handle Extract Frames button click."""
        source_id = self.source_combo.currentData()
        if not source_id:
            return
        mode = "frames" if self.mode_combo.currentIndex() == 0 else "seconds"
        interval = self.interval_spin.value()
        self.extract_frames_requested.emit(source_id, mode, interval)

    def _on_import_clicked(self):
        """Handle Import Images button click."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
        )
        if file_paths:
            paths = [Path(p) for p in file_paths]
            self.import_images_requested.emit(paths)

    def _on_analyze_clicked(self):
        """Handle Analyze Selected button click."""
        ids = self.frame_browser.get_selected_frame_ids()
        if ids:
            self.analyze_frames_requested.emit(ids)

    def _on_add_to_sequence_clicked(self):
        """Handle Add to Sequence button click."""
        ids = self.frame_browser.get_selected_frame_ids()
        if ids:
            self.add_to_sequence_requested.emit(ids)

    def _on_frames_selected(self, frame_ids: list):
        """Handle selection changes from the frame browser."""
        count = len(frame_ids)
        has_selection = count > 0
        self.analyze_btn.setEnabled(has_selection)
        self.add_to_seq_btn.setEnabled(has_selection)

        if count == 0:
            self.selection_label.setText("")
        elif count == 1:
            self.selection_label.setText("1 frame selected")
        else:
            self.selection_label.setText(f"{count} frames selected")

        self.frames_selected.emit(frame_ids)

    def _on_frame_double_clicked(self, frame_id: str):
        """Handle double-click on a frame (placeholder for future detail view)."""
        logger.debug("Frame double-clicked: %s", frame_id)

    def _on_zoom_changed(self, level: int):
        """Handle zoom slider changes."""
        self.frame_browser.set_zoom(level)
