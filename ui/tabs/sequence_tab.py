"""Sequence tab for timeline editing and playback."""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QSplitter,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.widgets import EmptyStateWidget


class SequenceTab(BaseTab):
    """Tab for arranging clips on the timeline and previewing.

    Signals:
        playback_requested: Emitted when playback is requested (start_frame: int)
        stop_requested: Emitted when stop is requested
        export_requested: Emitted when export is requested
        clip_added: Emitted when a clip is added to timeline (clip: Clip, source: Source)
    """

    playback_requested = Signal(int)  # start_frame
    stop_requested = Signal()
    export_requested = Signal()
    clip_added = Signal(object, object)  # Clip, Source

    # State constants
    STATE_NO_CLIPS = 0
    STATE_TIMELINE = 1

    def __init__(self, parent=None):
        self._current_source = None
        self._clips = []
        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Sequence tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for different states
        self.state_stack = QStackedWidget()

        # State 0: No clips available
        self.no_clips_widget = EmptyStateWidget(
            "No Clips Available",
            "Detect scenes in the Analyze tab first, then drag clips here"
        )
        self.state_stack.addWidget(self.no_clips_widget)

        # State 1: Timeline content
        self.content_widget = self._create_content_area()
        self.state_stack.addWidget(self.content_widget)

        layout.addWidget(self.state_stack)

        # Start with no clips state
        self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)

    def _create_content_area(self) -> QWidget:
        """Create the main content area with video player and timeline."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter for player and timeline
        splitter = QSplitter(Qt.Vertical)

        # Top: Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        # Bottom: Timeline
        self.timeline = TimelineWidget()
        self.timeline.playhead_changed.connect(self._on_playhead_changed)
        self.timeline.playback_requested.connect(self._on_playback_requested)
        self.timeline.stop_requested.connect(self._on_stop_requested)
        self.timeline.export_requested.connect(self._on_export_requested)
        splitter.addWidget(self.timeline)

        splitter.setSizes([400, 250])
        layout.addWidget(splitter)

        return content

    def _on_playhead_changed(self, time_seconds: float):
        """Handle playhead position change."""
        # This is handled by MainWindow for cross-component coordination
        pass

    def _on_playback_requested(self, start_frame: int):
        """Handle playback request."""
        self.playback_requested.emit(start_frame)

    def _on_stop_requested(self):
        """Handle stop request."""
        self.stop_requested.emit()

    def _on_export_requested(self):
        """Handle export request."""
        self.export_requested.emit()

    # Public methods for MainWindow to call

    def set_source(self, source):
        """Set the current video source."""
        self._current_source = source
        if source:
            self.video_player.load_video(source.file_path)

    def set_clips_available(self, clips, source):
        """Set available clips for the timeline."""
        self._clips = clips
        self._current_source = source
        if clips:
            self.timeline.set_fps(source.fps)
            self.timeline.set_available_clips(clips, source)
            self.state_stack.setCurrentIndex(self.STATE_TIMELINE)
            # Ensure video player has the source loaded
            self.video_player.load_video(source.file_path)
        else:
            self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)

    def add_clip_to_timeline(self, clip, source):
        """Add a clip to the timeline."""
        self.timeline.set_fps(source.fps)
        self.timeline.add_clip(clip, source)
        self.clip_added.emit(clip, source)

    def get_sequence(self):
        """Get the current sequence from timeline."""
        return self.timeline.get_sequence()

    def get_clip_at_playhead(self):
        """Get clip at current playhead position."""
        return self.timeline.get_clip_at_playhead()

    def set_playhead_time(self, time_seconds: float):
        """Set the playhead position."""
        self.timeline.set_playhead_time(time_seconds)

    def get_playhead_time(self) -> float:
        """Get current playhead time in seconds."""
        return self.timeline.get_playhead_time()

    def set_playing(self, is_playing: bool):
        """Update UI for playing state."""
        self.timeline.set_playing(is_playing)

    def seek_video_to(self, time_seconds: float):
        """Seek the video player to a position."""
        self.video_player.seek_to(time_seconds)

    def play_video_range(self, start_seconds: float, end_seconds: float):
        """Play a range in the video player."""
        self.video_player.play_range(start_seconds, end_seconds)

    def load_video(self, file_path):
        """Load a video file into the player."""
        self.video_player.load_video(file_path)
