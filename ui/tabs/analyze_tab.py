"""Analyze tab for scene detection and clip browsing."""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser
from ui.video_player import VideoPlayer


class EmptyStateWidget(QWidget):
    """Widget showing empty state messages."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self._setup_ui(title, message)

    def _setup_ui(self, title: str, message: str):
        layout = QVBoxLayout(self)
        layout.addStretch()

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #666;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setStyleSheet("color: #888;")
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)

        layout.addStretch()


class AnalyzeTab(BaseTab):
    """Tab for detecting scenes and browsing detected clips.

    Signals:
        detect_requested: Emitted when detection is requested (threshold: float)
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
        clip_dragged_to_timeline: Emitted when a clip is dragged to timeline (clip: Clip)
    """

    detect_requested = Signal(float)  # threshold
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip

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
        """Set up the Analyze tab UI."""
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
            "Ready to Analyze",
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

        self.sensitivity_slider = QSlider(Qt.Horizontal)
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

        controls.addStretch()

        # Clip count label
        self.clip_count_label = QLabel("")
        controls.addWidget(self.clip_count_label)

        return controls

    def _create_content_area(self) -> QWidget:
        """Create the main content area with clip browser and video player."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal splitter for browser and player
        splitter = QSplitter(Qt.Horizontal)

        # Left: Clip browser
        self.clip_browser = ClipBrowser()
        self.clip_browser.set_drag_enabled(True)
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged)
        splitter.addWidget(self.clip_browser)

        # Right: Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        return content

    def _on_detect_click(self):
        """Handle detect button click."""
        threshold = self.sensitivity_slider.value() / 10.0
        self.detect_requested.emit(threshold)

    def _on_clip_selected(self, clip):
        """Handle clip selection."""
        self.clip_selected.emit(clip)
        if self._current_source:
            start_time = clip.start_time(self._current_source.fps)
            self.video_player.seek_to(start_time)

    def _on_clip_double_clicked(self, clip):
        """Handle clip double-click."""
        self.clip_double_clicked.emit(clip)
        if self._current_source:
            start_time = clip.start_time(self._current_source.fps)
            end_time = clip.end_time(self._current_source.fps)
            self.video_player.play_range(start_time, end_time)

    def _on_clip_dragged(self, clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    # Public methods for MainWindow to call

    def set_source(self, source):
        """Set the current video source."""
        self._current_source = source
        if source:
            self.detect_btn.setEnabled(True)
            self.video_player.load_video(source.file_path)
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

    def add_clip(self, clip, source):
        """Add a single clip to the browser (called during thumbnail generation)."""
        self.clip_browser.add_clip(clip, source)

    def clear_clips(self):
        """Clear all clips from the browser."""
        self.clip_browser.clear()
        self._clips = []
        self.clip_count_label.setText("")

    def update_clip_colors(self, clip_id: str, colors: list):
        """Update colors for a clip."""
        self.clip_browser.update_clip_colors(clip_id, colors)

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update shot type for a clip."""
        self.clip_browser.update_clip_shot_type(clip_id, shot_type)

    def set_sensitivity(self, value: float):
        """Set the sensitivity slider value."""
        self.sensitivity_slider.setValue(int(value * 10))

    def set_detecting(self, is_detecting: bool):
        """Update UI state during detection."""
        self.detect_btn.setEnabled(not is_detecting)
        if is_detecting:
            self.detect_btn.setText("Detecting...")
        else:
            self.detect_btn.setText("Detect Scenes")
