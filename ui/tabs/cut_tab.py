"""Cut tab for scene detection and clip browsing."""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser
from ui.widgets import EmptyStateWidget
from ui.widgets.styled_slider import StyledSlider


class CutTab(BaseTab):
    """Tab for scene detection and clip browsing.

    Signals:
        detect_requested: Emitted when detection is requested (threshold: float)
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
        clip_dragged_to_timeline: Emitted when a clip is dragged to timeline (clip: Clip)
        clips_sent_to_analyze: Emitted when clips are sent to Analyze tab (clip_ids: list[str])
    """

    detect_requested = Signal(float)  # threshold
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

        # Sensitivity slider
        controls.addWidget(QLabel("Sensitivity:"))

        self.sensitivity_slider = StyledSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)  # 1.0 to 10.0
        self.sensitivity_slider.setValue(30)  # Default 3.0
        self.sensitivity_slider.setMaximumWidth(150)
        self.sensitivity_slider.setToolTip("Lower = more scenes detected")
        controls.addWidget(self.sensitivity_slider)

        self.sensitivity_label = QLabel("3.0")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v/10:.1f}")
        )
        controls.addWidget(self.sensitivity_label)

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
        threshold = self.sensitivity_slider.value() / 10.0
        self.detect_requested.emit(threshold)

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
        self.analyze_btn.setEnabled(not is_detecting and len(self.clip_browser.get_selected_clips()) > 0)
        if is_detecting:
            self.detect_btn.setText("Detecting...")
        else:
            self.detect_btn.setText("Detect Scenes")

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
