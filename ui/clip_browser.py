"""Clip browser with thumbnail grid view."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGridLayout,
    QLabel,
    QFrame,
    QApplication,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QPixmap, QDrag, QPainter, QColor

from models.clip import Clip, Source
from core.analysis.color import get_primary_hue


class ColorSwatchBar(QWidget):
    """Widget that displays dominant colors as horizontal stripes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors: list[tuple[int, int, int]] = []
        self.setFixedSize(160, 10)

    def set_colors(self, colors: list[tuple[int, int, int]]):
        """Set the colors to display."""
        self.colors = colors
        self.update()

    def paintEvent(self, event):
        """Paint the color stripes."""
        if not self.colors:
            return

        painter = QPainter(self)
        width = self.width()
        height = self.height()

        # Calculate stripe width based on number of colors
        n_colors = len(self.colors)
        stripe_width = width / n_colors

        for i, rgb in enumerate(self.colors):
            color = QColor(rgb[0], rgb[1], rgb[2])
            painter.fillRect(
                int(i * stripe_width),
                0,
                int(stripe_width) + 1,  # +1 to avoid gaps
                height,
                color,
            )

        painter.end()


class ClipThumbnail(QFrame):
    """Individual clip thumbnail widget."""

    clicked = Signal(object)  # Clip
    double_clicked = Signal(object)  # Clip
    drag_started = Signal(object)  # Clip

    def __init__(self, clip: Clip, source: Source, drag_enabled: bool = False):
        super().__init__()
        self.clip = clip
        self.source = source
        self.selected = False
        self._drag_enabled = drag_enabled
        self._drag_start_pos = None

        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(2)
        self.setFixedSize(180, 145)  # Increased height for color bar
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail image
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(160, 90)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("background-color: #333;")

        if clip.thumbnail_path and clip.thumbnail_path.exists():
            self._load_thumbnail(clip.thumbnail_path)
        else:
            self.thumbnail_label.setText("Loading...")

        layout.addWidget(self.thumbnail_label)

        # Color swatch bar
        self.color_bar = ColorSwatchBar()
        if clip.dominant_colors:
            self.color_bar.set_colors(clip.dominant_colors)
        layout.addWidget(self.color_bar)

        # Duration label
        duration = clip.duration_seconds(source.fps)
        self.duration_label = QLabel(self._format_duration(duration))
        self.duration_label.setAlignment(Qt.AlignCenter)
        self.duration_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.duration_label)

        self._update_style()

    def _load_thumbnail(self, path: Path):
        """Load thumbnail image."""
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                160, 90,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS.ms"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self._update_style()

    def _update_style(self):
        """Update visual style based on state."""
        if self.selected:
            self.setStyleSheet("""
                ClipThumbnail {
                    background-color: #4a90d9;
                    border: 2px solid #2d6cb5;
                }
            """)
        else:
            self.setStyleSheet("""
                ClipThumbnail {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                }
                ClipThumbnail:hover {
                    background-color: #e8e8e8;
                    border: 1px solid #bbb;
                }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
        self.clicked.emit(self.clip)

    def mouseMoveEvent(self, event):
        if not self._drag_enabled or not self._drag_start_pos:
            return

        # Check if drag threshold met
        if (event.pos() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        # Start drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-clip-id", self.clip.id.encode())
        drag.setMimeData(mime_data)

        # Set drag pixmap (thumbnail)
        if self.thumbnail_label.pixmap():
            drag.setPixmap(self.thumbnail_label.pixmap().scaled(80, 45, Qt.KeepAspectRatio))

        # Execute drag
        result = drag.exec_(Qt.CopyAction)
        if result == Qt.CopyAction:
            self.drag_started.emit(self.clip)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.clip)

    def set_drag_enabled(self, enabled: bool):
        """Enable or disable dragging."""
        self._drag_enabled = enabled

    def set_colors(self, colors: list[tuple[int, int, int]]):
        """Set the dominant colors for this clip."""
        self.color_bar.set_colors(colors)


class ClipBrowser(QWidget):
    """Grid browser for viewing detected clips."""

    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip

    COLUMNS = 4

    def __init__(self):
        super().__init__()
        self.thumbnails: list[ClipThumbnail] = []
        self.selected_clips: set[str] = set()  # clip ids
        self._drag_enabled = False
        self._source_lookup: dict[str, Source] = {}  # clip_id -> Source

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with sort dropdown
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 4)

        header_label = QLabel("Detected Scenes")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # Sort dropdown
        sort_label = QLabel("Order:")
        sort_label.setStyleSheet("font-size: 12px; color: #666;")
        header_layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Timeline", "Color", "Duration"])
        self.sort_combo.setFixedWidth(100)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        header_layout.addWidget(self.sort_combo)

        layout.addLayout(header_layout)

        # Scroll area for thumbnails
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Container for grid
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        # Placeholder for empty state
        self.empty_label = QLabel("No scenes detected yet.\nDrop a video or click Import.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #999; font-size: 12px;")
        self.grid.addWidget(self.empty_label, 0, 0, 1, self.COLUMNS)

    def add_clip(self, clip: Clip, source: Source):
        """Add a clip to the browser."""
        # Remove empty placeholder if present
        if self.empty_label.isVisible():
            self.empty_label.setVisible(False)

        # Store source reference
        self._source_lookup[clip.id] = source

        # Create thumbnail widget
        thumb = ClipThumbnail(clip, source, drag_enabled=self._drag_enabled)
        thumb.clicked.connect(self._on_thumbnail_clicked)
        thumb.double_clicked.connect(self._on_thumbnail_double_clicked)
        thumb.drag_started.connect(self._on_drag_started)

        self.thumbnails.append(thumb)

        # Add to grid
        row = (len(self.thumbnails) - 1) // self.COLUMNS
        col = (len(self.thumbnails) - 1) % self.COLUMNS
        self.grid.addWidget(thumb, row, col)

    def clear(self):
        """Clear all clips."""
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()

        self.thumbnails = []
        self.selected_clips = set()
        self._source_lookup = {}
        self.empty_label.setVisible(True)

    def set_drag_enabled(self, enabled: bool):
        """Enable or disable dragging clips to timeline."""
        self._drag_enabled = enabled
        for thumb in self.thumbnails:
            thumb.set_drag_enabled(enabled)

    def get_selected_clips(self) -> list[Clip]:
        """Get list of selected clips."""
        return [t.clip for t in self.thumbnails if t.clip.id in self.selected_clips]

    def _on_thumbnail_clicked(self, clip: Clip):
        """Handle thumbnail click."""
        # Toggle selection
        if clip.id in self.selected_clips:
            self.selected_clips.remove(clip.id)
        else:
            self.selected_clips.add(clip.id)

        # Update all thumbnail states
        for thumb in self.thumbnails:
            thumb.set_selected(thumb.clip.id in self.selected_clips)

        self.clip_selected.emit(clip)

    def _on_thumbnail_double_clicked(self, clip: Clip):
        """Handle thumbnail double-click."""
        self.clip_double_clicked.emit(clip)

    def _on_drag_started(self, clip: Clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    def get_source_for_clip(self, clip_id: str) -> Optional[Source]:
        """Get the source for a clip by ID."""
        return self._source_lookup.get(clip_id)

    def update_clip_colors(self, clip_id: str, colors: list[tuple[int, int, int]]):
        """Update the colors for a specific clip thumbnail."""
        for thumb in self.thumbnails:
            if thumb.clip.id == clip_id:
                thumb.set_colors(colors)
                break

    def _on_sort_changed(self, sort_option: str):
        """Handle sort dropdown change."""
        if sort_option == "Timeline":
            self._sort_by_timeline()
        elif sort_option == "Color":
            self._sort_by_color()
        elif sort_option == "Duration":
            self._sort_by_duration()

    def _sort_by_timeline(self):
        """Sort clips by timeline order (start frame)."""
        self.thumbnails.sort(key=lambda t: t.clip.start_frame)
        self._rebuild_grid()

    def _sort_by_color(self):
        """Sort clips by primary hue (HSV color wheel order)."""
        def get_hue(thumb: ClipThumbnail) -> float:
            if thumb.clip.dominant_colors:
                return get_primary_hue(thumb.clip.dominant_colors)
            return 0.0

        self.thumbnails.sort(key=get_hue)
        self._rebuild_grid()

    def _sort_by_duration(self):
        """Sort clips by duration (longest first)."""
        self.thumbnails.sort(
            key=lambda t: t.clip.duration_seconds(t.source.fps),
            reverse=True,
        )
        self._rebuild_grid()

    def _rebuild_grid(self):
        """Rebuild the grid layout with current thumbnail order."""
        # Remove all thumbnails from grid
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)

        # Re-add in new order
        for i, thumb in enumerate(self.thumbnails):
            row = i // self.COLUMNS
            col = i % self.COLUMNS
            self.grid.addWidget(thumb, row, col)
