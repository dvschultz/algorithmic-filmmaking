"""Clip details sidebar widget.

Provides a dismissable sidebar displaying detailed clip information:
- Video preview at top
- Editable clip name
- Clip metadata (read-only: duration, frames, resolution)
- Editable shot type dropdown
- Editable transcript segments with timestamps
- Dominant colors (read-only)
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

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
from ui.widgets.editable_label import EditableLabel
from ui.widgets.shot_type_dropdown import ShotTypeDropdown
from ui.widgets.editable_transcript import EditableTranscriptWidget

if TYPE_CHECKING:
    from core.transcription import TranscriptSegment

logger = logging.getLogger(__name__)


class ClipDetailsSidebar(QDockWidget):
    """Sidebar displaying detailed clip information with editable fields.

    Opens on left side of app to show:
    - Video preview with playback controls
    - Editable clip name
    - Clip metadata (read-only: duration, frames, resolution)
    - Editable shot type dropdown
    - Editable transcript with timestamps
    - Dominant colors display
    """

    # Signals
    clip_shown = Signal(str)  # clip_id when shown
    sidebar_closed = Signal()  # sidebar was closed
    clip_edited = Signal(object)  # Clip - emitted when clip is edited

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
        self._pending_clip_range: Optional[tuple[float, float]] = None  # (start, end) seconds to apply when media loads

        # Edit guard flags (prevent duplicate execution per documented learnings)
        self._name_change_in_progress = False
        self._shot_type_change_in_progress = False
        self._transcript_change_in_progress = False

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

        # Clip name section (EDITABLE)
        self.name_header = QLabel("Clip Name")
        self._apply_section_header_style(self.name_header)
        content_layout.addWidget(self.name_header)

        self.name_edit = EditableLabel("", placeholder="Enter clip name...")
        self.name_edit.value_changed.connect(self._on_name_changed)
        content_layout.addWidget(self.name_edit)

        # Source info (read-only)
        self.source_label = QLabel("")
        self._apply_muted_style(self.source_label)
        content_layout.addWidget(self.source_label)

        # Metadata section (read-only)
        self.metadata_header = QLabel("Details")
        self._apply_section_header_style(self.metadata_header)
        content_layout.addWidget(self.metadata_header)

        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self._apply_secondary_style(self.metadata_label)
        content_layout.addWidget(self.metadata_label)

        # Shot type section (EDITABLE)
        self.shot_type_header = QLabel("Shot Type")
        self._apply_section_header_style(self.shot_type_header)
        content_layout.addWidget(self.shot_type_header)

        self.shot_type_dropdown = ShotTypeDropdown()
        self.shot_type_dropdown.value_changed.connect(self._on_shot_type_changed)
        content_layout.addWidget(self.shot_type_dropdown)

        # Colors section (read-only)
        self.colors_header = QLabel("Dominant Colors")
        self._apply_section_header_style(self.colors_header)
        content_layout.addWidget(self.colors_header)

        self.color_swatches = QWidget()
        self.color_swatches_layout = QHBoxLayout(self.color_swatches)
        self.color_swatches_layout.setContentsMargins(0, 0, 0, 0)
        self.color_swatches_layout.setSpacing(8)
        content_layout.addWidget(self.color_swatches)

        # Transcript section (EDITABLE)
        self.transcript_header = QLabel("Transcript")
        self._apply_section_header_style(self.transcript_header)
        content_layout.addWidget(self.transcript_header)

        self.transcript_edit = EditableTranscriptWidget()
        self.transcript_edit.segments_changed.connect(self._on_transcript_changed)
        content_layout.addWidget(self.transcript_edit)

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

    def _apply_muted_style(self, label: QLabel):
        """Apply muted text styling."""
        label.setStyleSheet(f"color: {theme().text_muted}; font-size: 12px;")

    @Slot()
    def _refresh_theme(self):
        """Update colors on theme change."""
        self._apply_section_header_style(self.name_header)
        self._apply_section_header_style(self.metadata_header)
        self._apply_section_header_style(self.colors_header)
        self._apply_section_header_style(self.shot_type_header)
        self._apply_section_header_style(self.transcript_header)
        self._apply_secondary_style(self.metadata_label)
        self._apply_muted_style(self.source_label)

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
        """Handle media status changes to set clip range after video loads."""
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self._pending_clip_range is not None:
                start_time, end_time = self._pending_clip_range
                self.video_player.set_clip_range(start_time, end_time)
                self._pending_clip_range = None

    @Slot(str)
    def _on_name_changed(self, new_name: str):
        """Handle clip name change."""
        if self._name_change_in_progress or self._loading:
            return
        if not self._clip_ref:
            return
        self._name_change_in_progress = True

        self._clip_ref.name = new_name
        self.clip_edited.emit(self._clip_ref)

        self._name_change_in_progress = False

    @Slot(str)
    def _on_shot_type_changed(self, new_shot_type: str):
        """Handle shot type change."""
        if self._shot_type_change_in_progress or self._loading:
            return
        if not self._clip_ref:
            return
        self._shot_type_change_in_progress = True

        # Store None if empty, otherwise the shot type string
        self._clip_ref.shot_type = new_shot_type if new_shot_type else None
        self.clip_edited.emit(self._clip_ref)

        self._shot_type_change_in_progress = False

    @Slot(list)
    def _on_transcript_changed(self, segments: list):
        """Handle transcript segment change."""
        if self._transcript_change_in_progress or self._loading:
            return
        if not self._clip_ref:
            return
        self._transcript_change_in_progress = True

        # Segments are already updated in-place by the widget
        self._clip_ref.transcript = segments if segments else None
        self.clip_edited.emit(self._clip_ref)

        self._transcript_change_in_progress = False

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

        # Block signals while updating UI (per documented learnings)
        self.name_edit.blockSignals(True)
        self.shot_type_dropdown.blockSignals(True)
        self.transcript_edit.blockSignals(True)

        # Clip name (editable)
        display_name = clip.display_name(source.filename, source.fps)
        self.name_edit.setText(clip.name)  # Show actual name, not fallback
        self.name_edit.setPlaceholder(display_name)  # Use fallback as placeholder

        # Source info (read-only)
        start_time = clip.start_time(source.fps)
        self.source_label.setText(f"{source.filename} at {self._format_time(start_time)}")

        # Metadata (read-only)
        duration = clip.duration_seconds(source.fps)
        metadata_lines = [
            f"Duration: {self._format_time(duration, include_fraction=True)}",
            f"Frames: {clip.start_frame} - {clip.end_frame}",
            f"Resolution: {source.width}x{source.height}",
            f"FPS: {source.fps:.2f}",
        ]
        self.metadata_label.setText("\n".join(metadata_lines))

        # Shot type (editable)
        self.shot_type_dropdown.setValue(clip.shot_type)

        # Colors (read-only)
        self._update_colors(clip.dominant_colors)

        # Transcript (editable)
        self.transcript_edit.setSegments(clip.transcript)

        # Unblock signals
        self.name_edit.blockSignals(False)
        self.shot_type_dropdown.blockSignals(False)
        self.transcript_edit.blockSignals(False)

        # Enable editing
        self._set_editing_enabled(True)

        # Load video preview with clip range
        if source.file_path.exists():
            # Store clip range - will be applied when media loads
            end_time = clip.end_time(source.fps)
            self._pending_clip_range = (start_time, end_time)
            self.video_player.load_video(source.file_path)
        else:
            logger.warning(f"Source file not found: {source.file_path}")
            self._show_missing_file_state()

        # Show the sidebar
        self.show()
        self.raise_()

        self.clip_shown.emit(clip.id)
        self._loading = False

    def show_multi_selection(self, count: int):
        """Display multi-selection state.

        Args:
            count: Number of clips selected
        """
        self._loading = True
        self._clip_ref = None
        self._source_ref = None

        # Block signals
        self.name_edit.blockSignals(True)
        self.shot_type_dropdown.blockSignals(True)
        self.transcript_edit.blockSignals(True)

        # Show selection count
        self.name_edit.setText("")
        self.name_edit.setPlaceholder(f"{count} clips selected")
        self.source_label.setText("Multiple clips selected")
        self.metadata_label.setText("Select a single clip to view details")

        # Clear other fields
        self.shot_type_dropdown.setValue(None)
        self._update_colors(None)
        self.transcript_edit.setSegments(None)

        # Unblock signals
        self.name_edit.blockSignals(False)
        self.shot_type_dropdown.blockSignals(False)
        self.transcript_edit.blockSignals(False)

        # Disable editing
        self._set_editing_enabled(False)

        # Stop video
        if hasattr(self.video_player, 'player'):
            self.video_player.player.stop()

        self.show()
        self._loading = False

    def _set_editing_enabled(self, enabled: bool):
        """Enable or disable editing of all editable fields.

        Args:
            enabled: Whether editing should be enabled
        """
        self.name_edit.setEnabled(enabled)
        self.shot_type_dropdown.setEnabled(enabled)
        self.transcript_edit.setEnabled(enabled)

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
        self.source_label.setText(self.source_label.text() + " (Source file not found)")

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

        # Block signals
        self.name_edit.blockSignals(True)
        self.shot_type_dropdown.blockSignals(True)
        self.transcript_edit.blockSignals(True)

        self.name_edit.setText("")
        self.name_edit.setPlaceholder("No clip selected")
        self.source_label.setText("")
        self.metadata_label.setText("")
        self._update_colors(None)
        self.shot_type_dropdown.setValue(None)
        self.transcript_edit.setSegments(None)

        # Unblock signals
        self.name_edit.blockSignals(False)
        self.shot_type_dropdown.blockSignals(False)
        self.transcript_edit.blockSignals(False)

        # Disable editing
        self._set_editing_enabled(False)

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
