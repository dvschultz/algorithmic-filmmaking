"""Video player component using QMediaPlayer."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QStyle,
)
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
        self.end_position: Optional[int] = None  # For range playback

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
        self.play_btn.setFixedSize(40, 30)
        self.play_btn.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setFixedSize(40, 30)
        self.stop_btn.clicked.connect(self._stop)
        controls.addWidget(self.stop_btn)

        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
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
        self.end_position = None

    def seek_to(self, seconds: float):
        """Seek to a position in seconds."""
        self.player.setPosition(int(seconds * 1000))

    def play_range(self, start_seconds: float, end_seconds: float):
        """Play a specific range of the video."""
        self.end_position = int(end_seconds * 1000)
        self.player.setPosition(int(start_seconds * 1000))
        self.player.play()

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        """Stop playback."""
        self.player.stop()
        self.end_position = None

    def _set_position(self, position: int):
        """Set playback position from slider."""
        self.player.setPosition(position)

    @Slot(int)
    def _on_position_changed(self, position: int):
        """Handle position change."""
        self.position_slider.setValue(position)
        self.position_updated.emit(position)

        # Update time label
        current = self._format_time(position)
        total = self._format_time(self.player.duration())
        self.time_label.setText(f"{current} / {total}")

        # Check for range end
        if self.end_position and position >= self.end_position:
            self.player.pause()
            self.end_position = None

    @Slot(int)
    def _on_duration_changed(self, duration: int):
        """Handle duration change."""
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
