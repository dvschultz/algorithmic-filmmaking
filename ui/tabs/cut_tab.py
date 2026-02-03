"""Cut tab for scene detection and clip browsing."""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QStackedWidget,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser
from ui.widgets import EmptyStateWidget
from ui.widgets.styled_slider import StyledSlider
from ui.theme import UISizes


class CutTab(BaseTab):
    """Tab for scene detection and clip browsing.

    Signals:
        detect_requested: Emitted when detection is requested (mode: str, config: dict)
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
        clip_dragged_to_timeline: Emitted when a clip is dragged to timeline (clip: Clip)
        clips_sent_to_analyze: Emitted when clips are sent to Analyze tab (clip_ids: list[str])
    """

    detect_requested = Signal(str, dict)  # mode, config dict
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip
    clips_sent_to_analyze = Signal(list)  # list[str] clip IDs
    selection_changed = Signal(list)  # list[str] selected clip IDs

    # State constants for stacked widget
    STATE_NO_VIDEO = 0
    STATE_NO_CLIPS = 1
    STATE_CLIPS = 2

    def __init__(self, parent=None):
        # References to be set by MainWindow
        self._current_source = None
        self._clips = []
        self._settings = None
        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Cut tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top controls bar
        controls = self._create_controls()
        layout.addLayout(controls)

        # Stacked widget for different states
        self.state_stack = QStackedWidget()

        # State 0: No video loaded
        self.no_video_widget = EmptyStateWidget(
            "No Video Loaded",
            "Import a video in the Collect tab first"
        )
        self.state_stack.addWidget(self.no_video_widget)

        # State 1: Video loaded but no clips detected
        self.no_clips_widget = EmptyStateWidget(
            "Ready to Cut",
            "Click 'Detect Scenes' to find clips in your video"
        )
        self.state_stack.addWidget(self.no_clips_widget)

        # State 2: Clips detected - main content
        self.content_widget = self._create_content_area()
        self.state_stack.addWidget(self.content_widget)

        layout.addWidget(self.state_stack)

        # Start with no video state
        self.state_stack.setCurrentIndex(self.STATE_NO_VIDEO)

    def _create_controls(self) -> QHBoxLayout:
        """Create the top controls bar."""
        controls = QHBoxLayout()
        controls.setContentsMargins(10, 10, 10, 10)

        # Detection mode dropdown
        controls.addWidget(QLabel("Mode:"))
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.mode_dropdown.addItem("Visual (Adaptive)", "adaptive")
        self.mode_dropdown.addItem("Visual (Content)", "content")
        self.mode_dropdown.addItem("Text/Karaoke", "karaoke")
        self.mode_dropdown.setToolTip("Detection mode: Visual detects scene changes, Text/Karaoke detects text changes")
        self.mode_dropdown.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_dropdown)

        controls.addSpacing(10)

        # Visual mode: Sensitivity slider (shown by default)
        self.sensitivity_widget = QWidget()
        sensitivity_layout = QHBoxLayout(self.sensitivity_widget)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_layout.addWidget(QLabel("Sensitivity:"))

        self.sensitivity_slider = StyledSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)  # 1.0 to 10.0
        self.sensitivity_slider.setValue(30)  # Default 3.0
        self.sensitivity_slider.setMaximumWidth(150)
        self.sensitivity_slider.setToolTip("Lower = more scenes detected")
        sensitivity_layout.addWidget(self.sensitivity_slider)

        self.sensitivity_label = QLabel("3.0")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v/10:.1f}")
        )
        sensitivity_layout.addWidget(self.sensitivity_label)
        controls.addWidget(self.sensitivity_widget)

        # Karaoke mode: Options (hidden by default)
        self.karaoke_widget = QWidget()
        karaoke_layout = QHBoxLayout(self.karaoke_widget)
        karaoke_layout.setContentsMargins(0, 0, 0, 0)

        # ROI Top (0.0 = full frame, 0.75 = bottom 25%)
        karaoke_layout.addWidget(QLabel("ROI Top:"))
        self.roi_top_spin = QDoubleSpinBox()
        self.roi_top_spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.roi_top_spin.setRange(0.0, 0.95)
        self.roi_top_spin.setValue(0.0)  # Full frame by default
        self.roi_top_spin.setSingleStep(0.05)
        self.roi_top_spin.setToolTip("Top of text region (0.0=full frame, 0.75=bottom 25%)")
        karaoke_layout.addWidget(self.roi_top_spin)

        karaoke_layout.addSpacing(5)

        # Text similarity threshold
        karaoke_layout.addWidget(QLabel("Threshold:"))
        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.text_threshold_spin.setRange(10.0, 100.0)
        self.text_threshold_spin.setValue(60.0)  # Tuned default
        self.text_threshold_spin.setSingleStep(5.0)
        self.text_threshold_spin.setToolTip("Text similarity threshold (lower = more cuts)")
        karaoke_layout.addWidget(self.text_threshold_spin)

        karaoke_layout.addSpacing(5)

        # Confirm frames
        karaoke_layout.addWidget(QLabel("Confirm:"))
        self.confirm_frames_spin = QSpinBox()
        self.confirm_frames_spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.confirm_frames_spin.setRange(1, 10)
        self.confirm_frames_spin.setValue(3)  # Reduces OCR jitter
        self.confirm_frames_spin.setToolTip("Frames to confirm text change (reduces false positives)")
        karaoke_layout.addWidget(self.confirm_frames_spin)

        karaoke_layout.addSpacing(5)

        # Cut offset
        karaoke_layout.addWidget(QLabel("Offset:"))
        self.cut_offset_spin = QSpinBox()
        self.cut_offset_spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.cut_offset_spin.setRange(0, 30)
        self.cut_offset_spin.setValue(5)  # Compensates for fade-in delay
        self.cut_offset_spin.setToolTip("Shift cuts backward to catch fade-in starts")
        karaoke_layout.addWidget(self.cut_offset_spin)

        controls.addWidget(self.karaoke_widget)
        self.karaoke_widget.setVisible(False)  # Hidden by default

        controls.addSpacing(10)

        # Detect button
        self.detect_btn = QPushButton("Detect Scenes")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self._on_detect_click)
        controls.addWidget(self.detect_btn)

        controls.addSpacing(20)

        # Analyze Selected button
        self.analyze_btn = QPushButton("Analyze Selected")
        self.analyze_btn.setToolTip("Send selected clips to Analyze tab")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze_click)
        controls.addWidget(self.analyze_btn)

        controls.addStretch()

        # Selection count label
        self.selection_label = QLabel("")
        controls.addWidget(self.selection_label)

        controls.addSpacing(10)

        # Clip count label
        self.clip_count_label = QLabel("")
        controls.addWidget(self.clip_count_label)

        return controls

    def _on_mode_changed(self, index):
        """Handle detection mode change."""
        mode = self.mode_dropdown.currentData()
        is_karaoke = mode == "karaoke"
        self.sensitivity_widget.setVisible(not is_karaoke)
        self.karaoke_widget.setVisible(is_karaoke)
        # Update button text
        if is_karaoke:
            self.detect_btn.setText("Detect Text Changes")
        else:
            self.detect_btn.setText("Detect Scenes")

    def _create_content_area(self) -> QWidget:
        """Create the main content area with clip browser and video player."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Clip browser (full width - video preview is in clip details sidebar)
        self.clip_browser = ClipBrowser()
        self.clip_browser.set_drag_enabled(True)
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged)
        self.clip_browser.filters_changed.connect(self._on_filters_changed)
        layout.addWidget(self.clip_browser)

        return content

    def _on_detect_click(self):
        """Handle detect button click."""
        mode = self.mode_dropdown.currentData()

        if mode == "karaoke":
            config = {
                "roi_top_percent": self.roi_top_spin.value(),
                "text_similarity_threshold": self.text_threshold_spin.value(),
                "confirm_frames": self.confirm_frames_spin.value(),
                "cut_offset": self.cut_offset_spin.value(),
            }
        else:
            config = {
                "threshold": self.sensitivity_slider.value() / 10.0,
                "use_adaptive": mode == "adaptive",
            }

        self.detect_requested.emit(mode, config)

    def _on_analyze_click(self):
        """Handle analyze selected button click."""
        selected_clips = self.clip_browser.get_selected_clips()
        if selected_clips:
            clip_ids = [clip.id for clip in selected_clips]
            self.clips_sent_to_analyze.emit(clip_ids)

    def _on_clip_selected(self, clip):
        """Handle clip selection."""
        self.clip_selected.emit(clip)
        self._update_selection_ui()

    def _on_clip_double_clicked(self, clip):
        """Handle clip double-click."""
        self.clip_double_clicked.emit(clip)

    def _on_clip_dragged(self, clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    def _on_filters_changed(self):
        """Handle filter changes - clear selection and update UI."""
        self.clear_selection()
        visible = self.clip_browser.get_visible_clip_count()
        total = len(self.clip_browser.thumbnails)
        if self.clip_browser.has_active_filters():
            self.clip_count_label.setText(f"{visible}/{total} clips")
        else:
            self.clip_count_label.setText(f"{total} clips")

    def _update_selection_ui(self):
        """Update selection count and button state."""
        selected = self.clip_browser.get_selected_clips()
        count = len(selected)
        if count > 0:
            self.selection_label.setText(f"{count} selected")
            self.analyze_btn.setEnabled(True)
        else:
            self.selection_label.setText("")
            self.analyze_btn.setEnabled(False)
        
        # Notify parent of selection change
        self.selection_changed.emit([c.id for c in selected])

    # Public methods for MainWindow to call

    def set_source(self, source):
        """Set the current video source."""
        self._current_source = source
        if source:
            self.detect_btn.setEnabled(True)
            # Only show "no clips" state if we don't already have clips visible
            if not self._clips and not self.clip_browser.thumbnails:
                self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)
        else:
            self.detect_btn.setEnabled(False)
            self.state_stack.setCurrentIndex(self.STATE_NO_VIDEO)

    def set_clips(self, clips):
        """Set the detected clips."""
        self._clips = clips
        if clips:
            self.clip_count_label.setText(f"{len(clips)} clips")
            self.state_stack.setCurrentIndex(self.STATE_CLIPS)
        else:
            self.clip_count_label.setText("")
            if self._current_source:
                self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)
            else:
                self.state_stack.setCurrentIndex(self.STATE_NO_VIDEO)
        self._update_selection_ui()

    def add_clip(self, clip, source):
        """Add a single clip to the browser (called during thumbnail generation)."""
        # Switch to clips state if not already showing clips
        if self.state_stack.currentIndex() != self.STATE_CLIPS:
            self.state_stack.setCurrentIndex(self.STATE_CLIPS)
        self.clip_browser.add_clip(clip, source)
        # Track clip and update count
        if clip not in self._clips:
            self._clips.append(clip)
            self.clip_count_label.setText(f"{len(self._clips)} clips")

    def clear_clips(self):
        """Clear all clips from the browser."""
        self.clip_browser.clear()
        self._clips = []
        self.clip_count_label.setText("")
        self._update_selection_ui()

    def remove_clips_for_source(self, source_id: str):
        """Remove clips for a specific source (used when re-analyzing)."""
        self.clip_browser.remove_clips_for_source(source_id)
        # Update internal clip list
        self._clips = [c for c in self._clips if c.source_id != source_id]
        # Update count label
        if self._clips:
            self.clip_count_label.setText(f"{len(self._clips)} clips")
        else:
            self.clip_count_label.setText("")
        self._update_selection_ui()

    def update_clip_colors(self, clip_id: str, colors: list):
        """Update colors for a clip."""
        self.clip_browser.update_clip_colors(clip_id, colors)

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update shot type for a clip."""
        self.clip_browser.update_clip_shot_type(clip_id, shot_type)

    def update_clip_thumbnail(self, clip_id: str, thumb_path):
        """Update thumbnail for a clip."""
        self.clip_browser.update_clip_thumbnail(clip_id, thumb_path)

    def set_sensitivity(self, value: float):
        """Set the sensitivity slider value."""
        self.sensitivity_slider.setValue(int(value * 10))

    def set_detecting(self, is_detecting: bool):
        """Update UI state during detection."""
        self.detect_btn.setEnabled(not is_detecting)
        self.mode_dropdown.setEnabled(not is_detecting)
        self.analyze_btn.setEnabled(not is_detecting and len(self.clip_browser.get_selected_clips()) > 0)

        mode = self.mode_dropdown.currentData()
        if is_detecting:
            self.detect_btn.setText("Detecting...")
        elif mode == "karaoke":
            self.detect_btn.setText("Detect Text Changes")
        else:
            self.detect_btn.setText("Detect Scenes")

    def get_detection_mode(self) -> str:
        """Get the current detection mode."""
        return self.mode_dropdown.currentData()

    def set_detection_mode(self, mode: str):
        """Set the detection mode programmatically."""
        for i in range(self.mode_dropdown.count()):
            if self.mode_dropdown.itemData(i) == mode:
                self.mode_dropdown.setCurrentIndex(i)
                break

    def update_clip_transcript(self, clip_id: str, segments: list):
        """Update transcript for a clip."""
        self.clip_browser.update_clip_transcript(clip_id, segments)

    def clear_selection(self):
        """Clear the current selection."""
        self.clip_browser.selected_clips.clear()
        for thumb in self.clip_browser.thumbnails:
            thumb.set_selected(False)
        self._update_selection_ui()

    def get_active_filters(self) -> dict:
        """Get the current filter state from the clip browser.

        Returns:
            Dict with filter names and their current values
        """
        return self.clip_browser.get_active_filters()

    def has_active_filters(self) -> bool:
        """Check if any filters are currently active.

        Returns:
            True if at least one filter is set
        """
        return self.clip_browser.has_active_filters()
