"""Timeline preview widget showing sorted clips in sequence."""

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont

from ui.theme import theme


class ClipPreviewItem(QFrame):
    """Individual clip thumbnail in the preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setFixedSize(80, 60)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the item UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(76, 43)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")
        layout.addWidget(self.thumbnail_label)

        # Duration
        self.duration_label = QLabel()
        self.duration_label.setAlignment(Qt.AlignCenter)
        self.duration_label.setStyleSheet(f"font-size: 9px; color: {theme().text_muted};")
        layout.addWidget(self.duration_label)

        self._update_style()

    def set_clip(self, clip, source):
        """Set the clip to display.

        Args:
            clip: Clip object with thumbnail_path and frame info
            source: Source object with fps
        """
        # Load thumbnail
        if clip.thumbnail_path and clip.thumbnail_path.exists():
            pixmap = QPixmap(str(clip.thumbnail_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    76, 43,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.thumbnail_label.setPixmap(scaled)
        else:
            self.thumbnail_label.setText("...")

        # Set duration
        duration = clip.duration_seconds(source.fps)
        self.duration_label.setText(self._format_duration(duration))

    def _format_duration(self, seconds: float) -> str:
        """Format duration as M:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _update_style(self):
        """Update visual style."""
        self.setStyleSheet(f"""
            ClipPreviewItem {{
                background-color: {theme().card_background};
                border: 1px solid {theme().card_border};
                border-radius: 4px;
            }}
        """)


class TimelinePreview(QWidget):
    """Read-only preview of sorted clips.

    Displays a horizontal strip of clip thumbnails showing the
    current sort order without editing capabilities.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clip_items: list[ClipPreviewItem] = []
        self._setup_ui()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the preview UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()

        header_label = QLabel("Preview")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setStyleSheet(f"color: {theme().text_secondary};")
        header_layout.addWidget(header_label)

        self.count_label = QLabel()
        self.count_label.setStyleSheet(f"color: {theme().text_muted};")
        header_layout.addWidget(self.count_label)

        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        self._header_label = header_label

        # Scroll area for clips
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(90)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
            }}
        """)

        # Container for clip items
        self.clips_container = QWidget()
        self.clips_layout = QHBoxLayout(self.clips_container)
        self.clips_layout.setContentsMargins(8, 8, 8, 8)
        self.clips_layout.setSpacing(8)
        self.clips_layout.addStretch()

        self.scroll_area.setWidget(self.clips_container)
        main_layout.addWidget(self.scroll_area)

        # Loading overlay
        self.loading_label = QLabel("Generating preview...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet(f"""
            color: {theme().text_muted};
            background-color: {theme().background_secondary};
            border: 1px solid {theme().border_secondary};
            border-radius: 4px;
            padding: 20px;
        """)
        self.loading_label.setFixedHeight(90)
        self.loading_label.setVisible(False)
        main_layout.addWidget(self.loading_label)

        # Empty state
        self.empty_label = QLabel("No clips to preview")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {theme().text_muted};")
        self.empty_label.setFixedHeight(90)
        self.empty_label.setVisible(False)
        main_layout.addWidget(self.empty_label)

    def set_clips(self, clips: list, sources: dict):
        """Update preview with new clip order.

        Args:
            clips: List of (Clip, Source) tuples in display order
            sources: Dict mapping source_id to Source (for multi-source support)
        """
        # Clear existing items
        self._clear_clips()

        if not clips:
            self.scroll_area.setVisible(False)
            self.loading_label.setVisible(False)
            self.empty_label.setVisible(True)
            self.count_label.setText("")
            return

        self.scroll_area.setVisible(True)
        self.loading_label.setVisible(False)
        self.empty_label.setVisible(False)

        # Update count
        self.count_label.setText(f"({len(clips)} clips)")

        # Remove the stretch at the end
        if self.clips_layout.count() > 0:
            item = self.clips_layout.takeAt(self.clips_layout.count() - 1)

        # Add clip items
        for clip, source in clips:
            item = ClipPreviewItem()
            item.set_clip(clip, source)
            self.clips_layout.addWidget(item)
            self._clip_items.append(item)

        # Add stretch back
        self.clips_layout.addStretch()

    def _clear_clips(self):
        """Clear all clip items."""
        for item in self._clip_items:
            self.clips_layout.removeWidget(item)
            item.deleteLater()
        self._clip_items.clear()

    def set_loading(self, loading: bool):
        """Show/hide loading indicator during sort computation."""
        if loading:
            self.scroll_area.setVisible(False)
            self.empty_label.setVisible(False)
            self.loading_label.setVisible(True)
        else:
            self.loading_label.setVisible(False)
            # set_clips will handle showing the appropriate view

    def clear(self):
        """Clear the preview."""
        self._clear_clips()
        self.scroll_area.setVisible(False)
        self.loading_label.setVisible(False)
        self.empty_label.setVisible(True)
        self.count_label.setText("")

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._header_label.setStyleSheet(f"color: {theme().text_secondary};")
        self.count_label.setStyleSheet(f"color: {theme().text_muted};")
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
            }}
        """)
        self.loading_label.setStyleSheet(f"""
            color: {theme().text_muted};
            background-color: {theme().background_secondary};
            border: 1px solid {theme().border_secondary};
            border-radius: 4px;
            padding: 20px;
        """)
        self.empty_label.setStyleSheet(f"color: {theme().text_muted};")

        # Refresh clip items
        for item in self._clip_items:
            item._update_style()
            item.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")
            item.duration_label.setStyleSheet(f"font-size: 9px; color: {theme().text_muted};")
