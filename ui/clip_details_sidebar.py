"""Clip details sidebar widget.

Provides a dismissable sidebar displaying detailed clip information:
- Video preview at top
- Clip title and metadata
- Analysis data (colors, shot type, transcript)
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models.clip import Clip, Source
from ui.theme import theme
from ui.video_player import VideoPlayer

logger = logging.getLogger(__name__)


class ClipDetailsSidebar(QDockWidget):
    """Sidebar displaying detailed clip information.

    Opens on left side of app to show:
    - Video preview with playback controls
    - Clip metadata (title, duration, frames, resolution)
    - Analysis data (colors, shot type, transcript)
    """

    # Signals
    clip_shown = Signal(str)  # clip_id when shown
    sidebar_closed = Signal()  # sidebar was closed

    def __init__(self, parent=None):
        """Create the clip details sidebar.

        Args:
            parent: Parent widget
        """
        super().__init__("Clip Details", parent)
        self.setObjectName("ClipDetailsSidebar")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(400)
        self.setMaximumWidth(550)

        # State references (not copies - per documented learnings)
        self._clip_ref: Optional[Clip] = None
        self._source_ref: Optional[Source] = None
        self._loading = False  # Guard flag for duplicate signals
        self._pending_seek: Optional[float] = None  # Seek position to apply when media loads

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Build the sidebar UI."""
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video preview section - square aspect for both horizontal and vertical video
        self.video_player = VideoPlayer()
        # Make video area square (works well for both landscape and portrait video)
        self.video_player.video_widget.setMinimumSize(350, 350)
        self.video_player.video_widget.setMaximumSize(500, 500)
        self.video_player.setMinimumHeight(400)
        self.video_player.setMaximumHeight(550)
        main_layout.addWidget(self.video_player)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {theme().border_secondary};")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)

        # Scroll area for metadata and analysis
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {theme().background_secondary};
                border: none;
            }}
        """)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        # Title section
        self.title_label = QLabel("No clip selected")
        self.title_label.setWordWrap(True)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._apply_title_style()
        content_layout.addWidget(self.title_label)

        # Metadata section
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self._apply_secondary_style(self.metadata_label)
        content_layout.addWidget(self.metadata_label)

        # Colors section header
        self.colors_header = QLabel("Dominant Colors")
        self._apply_section_header_style(self.colors_header)
        content_layout.addWidget(self.colors_header)

        # Color swatches container
        self.color_swatches = QWidget()
        self.color_swatches_layout = QHBoxLayout(self.color_swatches)
        self.color_swatches_layout.setContentsMargins(0, 0, 0, 0)
        self.color_swatches_layout.setSpacing(8)
        content_layout.addWidget(self.color_swatches)

        # Shot type section
        self.shot_type_header = QLabel("Shot Type")
        self._apply_section_header_style(self.shot_type_header)
        content_layout.addWidget(self.shot_type_header)

        self.shot_type_label = QLabel("")
        self._apply_secondary_style(self.shot_type_label)
        content_layout.addWidget(self.shot_type_label)

        # Transcript section header
        self.transcript_header = QLabel("Transcript")
        self._apply_section_header_style(self.transcript_header)
        content_layout.addWidget(self.transcript_header)

        # Transcript text
        self.transcript_text = QLabel("")
        self.transcript_text.setWordWrap(True)
        self.transcript_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._apply_secondary_style(self.transcript_text)
        content_layout.addWidget(self.transcript_text)

        # Stretch to push content to top
        content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

        self.setWidget(container)

        # Start hidden
        self.hide()

    def _connect_signals(self):
        """Connect signals."""
        theme().changed.connect(self._refresh_theme)
        self.visibilityChanged.connect(self._on_visibility_changed)
        # Connect to media status to seek after video loads
        self.video_player.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _apply_title_style(self):
        """Apply title label styling."""
        self.title_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: {theme().text_primary};
        """)

    def _apply_section_header_style(self, label: QLabel):
        """Apply section header styling."""
        label.setStyleSheet(f"""
            font-size: 12px;
            font-weight: bold;
            color: {theme().text_muted};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)

    def _apply_secondary_style(self, label: QLabel):
        """Apply secondary text styling."""
        label.setStyleSheet(f"color: {theme().text_secondary}; line-height: 1.4;")

    @Slot()
    def _refresh_theme(self):
        """Update colors on theme change."""
        self._apply_title_style()
        self._apply_section_header_style(self.colors_header)
        self._apply_section_header_style(self.shot_type_header)
        self._apply_section_header_style(self.transcript_header)
        self._apply_secondary_style(self.metadata_label)
        self._apply_secondary_style(self.shot_type_label)
        self._apply_secondary_style(self.transcript_text)

        # Re-render color swatches with current clip
        if self._clip_ref:
            self._update_colors(self._clip_ref.dominant_colors)

    @Slot(bool)
    def _on_visibility_changed(self, visible: bool):
        """Handle visibility changes."""
        if not visible:
            self.sidebar_closed.emit()
            # Stop video playback when hidden
            if hasattr(self.video_player, 'player'):
                self.video_player.player.pause()

    @Slot(QMediaPlayer.MediaStatus)
    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus):
        """Handle media status changes to seek after video loads."""
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self._pending_seek is not None:
                self.video_player.seek_to(self._pending_seek)
                self._pending_seek = None

    def show_clip(self, clip: Clip, source: Source):
        """Display details for the given clip.

        Args:
            clip: The clip to display
            source: The source video for this clip
        """
        if self._loading:
            logger.debug("show_clip called while loading, ignoring")
            return  # Guard against duplicate signal delivery

        self._loading = True
        logger.info(f"Showing clip details: {clip.id}")

        # Store references (not copies)
        self._clip_ref = clip
        self._source_ref = source

        # Title: filename - timecode
        start_time = clip.start_time(source.fps)
        title = f"{source.filename} - {self._format_time(start_time)}"
        self.title_label.setText(title)

        # Metadata
        duration = clip.duration_seconds(source.fps)
        metadata_lines = [
            f"Duration: {self._format_time(duration, include_fraction=True)}",
            f"Frames: {clip.start_frame} - {clip.end_frame}",
            f"Resolution: {source.width}x{source.height}",
            f"FPS: {source.fps:.2f}",
        ]
        self.metadata_label.setText("\n".join(metadata_lines))

        # Colors
        self._update_colors(clip.dominant_colors)

        # Shot type
        if clip.shot_type:
            self.shot_type_label.setText(clip.shot_type.title())
            self.shot_type_header.show()
            self.shot_type_label.show()
        else:
            self.shot_type_label.setText("Not analyzed")
            self.shot_type_header.show()
            self.shot_type_label.show()

        # Transcript
        if clip.transcript:
            transcript_text = clip.get_transcript_text()
            self.transcript_text.setText(transcript_text if transcript_text else "No speech detected")
            self.transcript_header.show()
            self.transcript_text.show()
        else:
            self.transcript_text.setText("No transcript available")
            self.transcript_header.show()
            self.transcript_text.show()

        # Load video preview
        if source.file_path.exists():
            # Store seek position - will be applied when media loads
            self._pending_seek = start_time
            self.video_player.load_video(source.file_path)
        else:
            logger.warning(f"Source file not found: {source.file_path}")
            self._show_missing_file_state()

        # Show the sidebar
        self.show()
        self.raise_()

        self.clip_shown.emit(clip.id)
        self._loading = False

    def _update_colors(self, colors: Optional[list[tuple[int, int, int]]]):
        """Update the color swatches display.

        Args:
            colors: List of RGB tuples, or None if not analyzed
        """
        # Clear existing swatches
        while self.color_swatches_layout.count():
            item = self.color_swatches_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not colors:
            no_colors = QLabel("Not analyzed")
            no_colors.setStyleSheet(f"color: {theme().text_muted}; font-style: italic;")
            self.color_swatches_layout.addWidget(no_colors)
            self.color_swatches_layout.addStretch()
            return

        # Display up to 5 color swatches
        for r, g, b in colors[:5]:
            swatch = QFrame()
            swatch.setFixedSize(40, 40)
            swatch.setStyleSheet(f"""
                background-color: rgb({r}, {g}, {b});
                border-radius: 4px;
                border: 1px solid {theme().border_secondary};
            """)
            swatch.setToolTip(f"RGB({r}, {g}, {b})")
            self.color_swatches_layout.addWidget(swatch)

        self.color_swatches_layout.addStretch()

    def _show_missing_file_state(self):
        """Show error state when source file is missing."""
        # Clear video widget - we can't actually clear it but we can show text
        self.title_label.setText(self.title_label.text() + "\n(Source file not found)")

    def _format_time(self, seconds: float, include_fraction: bool = False) -> str:
        """Format seconds as HH:MM:SS.ff or MM:SS.ff.

        Args:
            seconds: Time in seconds
            include_fraction: If True, include fractional seconds

        Returns:
            Formatted time string
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if include_fraction:
            if h > 0:
                return f"{h}:{m:02d}:{s:05.2f}"
            return f"{m}:{s:05.2f}"
        else:
            s = int(s)
            if h > 0:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m}:{s:02d}"

    def clear(self):
        """Clear the sidebar content."""
        self._clip_ref = None
        self._source_ref = None
        self.title_label.setText("No clip selected")
        self.metadata_label.setText("")
        self._update_colors(None)
        self.shot_type_label.setText("")
        self.transcript_text.setText("")
        if hasattr(self.video_player, 'player'):
            self.video_player.player.stop()

    def keyPressEvent(self, event):
        """Handle key press events.

        Args:
            event: The key event
        """
        if event.key() == Qt.Key_Escape:
            self.hide()
            event.accept()
        else:
            super().keyPressEvent(event)
