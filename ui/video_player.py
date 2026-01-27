"""Video player component using QMediaPlayer."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStyle,
)
from ui.widgets.styled_slider import StyledSlider
from PySide6.QtCore import Qt, QUrl, Slot, Signal
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from ui.theme import theme


class VideoPlayer(QWidget):
    """Video player with playback controls."""

    # Signals
    position_updated = Signal(int)  # position in milliseconds

    def __init__(self):
        super().__init__()
        # Range playback (clip mode)
        self._clip_start_ms: Optional[int] = None  # Clip start in milliseconds
        self._clip_end_ms: Optional[int] = None  # Clip end in milliseconds
        self._loop_playback: bool = True  # Loop within clip range

        self._setup_ui()
        self._setup_player()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Preview")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        layout.addWidget(header)

        # Video widget (always black background for video)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.video_widget, 1)

        # Controls
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 8, 8, 8)

        # Play/Pause button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setFixedSize(44, 32)
        self.play_btn.setAccessibleName("Play")
        self.play_btn.setToolTip("Play/Pause video")
        self.play_btn.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setFixedSize(44, 32)
        self.stop_btn.setAccessibleName("Stop")
        self.stop_btn.setToolTip("Stop video")
        self.stop_btn.clicked.connect(self._stop)
        controls.addWidget(self.stop_btn)

        # Position slider
        self.position_slider = StyledSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setAccessibleName("Video position")
        self.position_slider.sliderMoved.connect(self._set_position)
        controls.addWidget(self.position_slider, 1)

        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace;")
        controls.addWidget(self.time_label)

        layout.addLayout(controls)

    def _setup_player(self):
        """Set up media player."""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

    def _connect_signals(self):
        """Connect player signals."""
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

    def load_video(self, path: Path):
        """Load a video file."""
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self._clip_start_ms = None
        self._clip_end_ms = None

    def seek_to(self, seconds: float):
        """Seek to a position in seconds."""
        self.player.setPosition(int(seconds * 1000))

    def set_clip_range(self, start_seconds: float, end_seconds: float):
        """Set playback range to a specific clip.

        This constrains:
        - Playback to loop/stop at clip boundaries
        - Slider range to only show clip portion
        - Time display to show clip-relative time

        Args:
            start_seconds: Clip start time in seconds
            end_seconds: Clip end time in seconds
        """
        self._clip_start_ms = int(start_seconds * 1000)
        self._clip_end_ms = int(end_seconds * 1000)
        # Update slider range to clip duration
        clip_duration = self._clip_end_ms - self._clip_start_ms
        self.position_slider.setRange(0, clip_duration)
        # Seek to clip start
        self.player.setPosition(self._clip_start_ms)
        # Update time display
        self._update_time_label(self._clip_start_ms)

    def clear_clip_range(self):
        """Clear clip range, allowing full video playback."""
        self._clip_start_ms = None
        self._clip_end_ms = None
        # Reset slider to full video
        self.position_slider.setRange(0, self.player.duration())

    def play_range(self, start_seconds: float, end_seconds: float):
        """Play a specific range of the video (legacy method)."""
        self.set_clip_range(start_seconds, end_seconds)
        self.player.play()

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            # If in clip mode and at end, restart from clip start
            if self._clip_start_ms is not None and self._clip_end_ms is not None:
                current_pos = self.player.position()
                if current_pos >= self._clip_end_ms:
                    self.player.setPosition(self._clip_start_ms)
            self.player.play()

    def _stop(self):
        """Stop playback and return to clip/video start."""
        self.player.stop()
        # If in clip mode, seek back to clip start
        if self._clip_start_ms is not None:
            self.player.setPosition(self._clip_start_ms)

    def _set_position(self, position: int):
        """Set playback position from slider.

        In clip mode, slider position is relative to clip start.
        """
        if self._clip_start_ms is not None:
            # Convert slider position (relative to clip) to absolute position
            absolute_position = self._clip_start_ms + position
            self.player.setPosition(absolute_position)
        else:
            self.player.setPosition(position)

    @Slot(int)
    def _on_position_changed(self, position: int):
        """Handle position change."""
        self.position_updated.emit(position)

        if self._clip_start_ms is not None and self._clip_end_ms is not None:
            # Clip mode: show position relative to clip
            relative_position = position - self._clip_start_ms
            clip_duration = self._clip_end_ms - self._clip_start_ms

            # Clamp to valid range
            relative_position = max(0, min(relative_position, clip_duration))
            self.position_slider.setValue(relative_position)

            # Update time label with clip-relative time
            self._update_time_label(position)

            # Check for clip end
            if position >= self._clip_end_ms:
                if self._loop_playback:
                    # Loop back to clip start
                    self.player.setPosition(self._clip_start_ms)
                else:
                    self.player.pause()
        else:
            # Full video mode
            self.position_slider.setValue(position)
            current = self._format_time(position)
            total = self._format_time(self.player.duration())
            self.time_label.setText(f"{current} / {total}")

    def _update_time_label(self, absolute_position: int):
        """Update time label for clip mode."""
        if self._clip_start_ms is not None and self._clip_end_ms is not None:
            relative_pos = absolute_position - self._clip_start_ms
            clip_duration = self._clip_end_ms - self._clip_start_ms
            # Clamp to valid range for display
            relative_pos = max(0, min(relative_pos, clip_duration))
            current = self._format_time(relative_pos)
            total = self._format_time(clip_duration)
            self.time_label.setText(f"{current} / {total}")

    @Slot(int)
    def _on_duration_changed(self, duration: int):
        """Handle duration change."""
        # Only update slider range if not in clip mode
        if self._clip_start_ms is None:
            self.position_slider.setRange(0, duration)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle playback state change."""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as MM:SS."""
        seconds = ms // 1000
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
