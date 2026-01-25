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
    QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QPixmap, QDrag, QPainter, QColor

from models.clip import Clip, Source
from core.analysis.color import get_primary_hue, classify_color_palette, get_palette_display_name, COLOR_PALETTES
from core.analysis.shots import get_display_name, SHOT_TYPES


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
        self.setFixedSize(180, 160)  # Increased height for shot type label
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail container (for overlay positioning)
        self.thumb_container = QWidget()
        self.thumb_container.setFixedSize(160, 90)
        thumb_layout = QVBoxLayout(self.thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        # Thumbnail image
        self.thumbnail_label = QLabel(self.thumb_container)
        self.thumbnail_label.setFixedSize(160, 90)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("background-color: #333;")

        if clip.thumbnail_path and clip.thumbnail_path.exists():
            self._load_thumbnail(clip.thumbnail_path)
        else:
            self.thumbnail_label.setText("Loading...")

        # Transcript overlay (positioned on top of thumbnail)
        self.transcript_overlay = QLabel(self.thumb_container)
        self.transcript_overlay.setFixedSize(160, 90)
        self.transcript_overlay.setAlignment(Qt.AlignCenter)
        self.transcript_overlay.setWordWrap(True)
        self.transcript_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.85); "
            "color: white; "
            "font-size: 10px; "
            "padding: 6px; "
            "border-radius: 0px;"
        )
        self.transcript_overlay.setGeometry(0, 0, 160, 90)
        self.transcript_overlay.raise_()  # Ensure overlay is on top
        self.transcript_overlay.setVisible(False)
        self._update_transcript_overlay()

        layout.addWidget(self.thumb_container)

        # Color swatch bar
        self.color_bar = ColorSwatchBar()
        if clip.dominant_colors:
            self.color_bar.set_colors(clip.dominant_colors)
        layout.addWidget(self.color_bar)

        # Info row: duration and shot type
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)

        # Duration label
        duration = clip.duration_seconds(source.fps)
        self.duration_label = QLabel(self._format_duration(duration))
        self.duration_label.setAlignment(Qt.AlignLeft)
        self.duration_label.setStyleSheet("font-size: 11px; color: #666;")
        info_layout.addWidget(self.duration_label)

        info_layout.addStretch()

        # Shot type label
        self.shot_type_label = QLabel()
        self.shot_type_label.setAlignment(Qt.AlignRight)
        self.shot_type_label.setStyleSheet(
            "font-size: 10px; color: #fff; background-color: #666; "
            "border-radius: 3px; padding: 1px 4px;"
        )
        if clip.shot_type:
            self.shot_type_label.setText(get_display_name(clip.shot_type))
        else:
            self.shot_type_label.setVisible(False)
        info_layout.addWidget(self.shot_type_label)

        layout.addLayout(info_layout)

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

    def set_thumbnail(self, path: Path):
        """Update the thumbnail image."""
        self.clip.thumbnail_path = path
        self._load_thumbnail(path)

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

    def set_shot_type(self, shot_type: str):
        """Set the shot type for this clip."""
        self.shot_type_label.setText(get_display_name(shot_type))
        self.shot_type_label.setVisible(True)

    def set_transcript(self, segments: list):
        """Set the transcript segments for this clip."""
        self.clip.transcript = segments
        self._update_transcript_overlay()

    def _update_transcript_overlay(self):
        """Update the transcript overlay text."""
        if self.clip.transcript:
            # Get first ~100 chars of transcript
            full_text = self.clip.get_transcript_text()
            if len(full_text) > 100:
                display_text = full_text[:100] + "..."
            else:
                display_text = full_text
            self.transcript_overlay.setText(f'"{display_text}"')
        else:
            self.transcript_overlay.setText("")

    def enterEvent(self, event):
        """Show transcript overlay on hover if transcript exists."""
        if self.clip.transcript and self.clip.get_transcript_text():
            self.transcript_overlay.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide transcript overlay when not hovering."""
        self.transcript_overlay.setVisible(False)
        super().leaveEvent(event)


class ClipBrowser(QWidget):
    """Grid browser for viewing detected clips."""

    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip

    COLUMNS = 4

    def __init__(self):
        super().__init__()
        self.thumbnails: list[ClipThumbnail] = []
        self._thumbnail_by_id: dict[str, ClipThumbnail] = {}  # O(1) lookup by clip_id
        self.selected_clips: set[str] = set()  # clip ids
        self._drag_enabled = False
        self._source_lookup: dict[str, Source] = {}  # clip_id -> Source
        self._current_filter = "All"  # Current shot type filter
        self._current_color_filter = "All"  # Current color palette filter
        self._current_search_query = ""  # Current transcript search query

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with filter and sort dropdowns
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 4)

        header_label = QLabel("Detected Scenes")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # Shot type filter dropdown
        filter_label = QLabel("Shot:")
        filter_label.setStyleSheet("font-size: 12px; color: #666;")
        header_layout.addWidget(filter_label)

        self.filter_combo = QComboBox()
        filter_options = ["All"] + [get_display_name(st) for st in SHOT_TYPES]
        self.filter_combo.addItems(filter_options)
        self.filter_combo.setFixedWidth(100)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        header_layout.addWidget(self.filter_combo)

        header_layout.addSpacing(8)

        # Color palette filter dropdown
        color_label = QLabel("Color:")
        color_label.setStyleSheet("font-size: 12px; color: #666;")
        header_layout.addWidget(color_label)

        self.color_filter_combo = QComboBox()
        color_filter_options = ["All"] + [get_palette_display_name(cp) for cp in COLOR_PALETTES]
        self.color_filter_combo.addItems(color_filter_options)
        self.color_filter_combo.setFixedWidth(80)
        self.color_filter_combo.currentTextChanged.connect(self._on_color_filter_changed)
        header_layout.addWidget(self.color_filter_combo)

        header_layout.addSpacing(8)

        # Transcript search input
        search_label = QLabel("Search:")
        search_label.setStyleSheet("font-size: 12px; color: #666;")
        header_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcripts...")
        self.search_input.setFixedWidth(120)
        self.search_input.setToolTip("Filter clips by transcript content")
        self.search_input.textChanged.connect(self._on_search_changed)
        header_layout.addWidget(self.search_input)

        header_layout.addSpacing(16)

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
        self._thumbnail_by_id[clip.id] = thumb  # O(1) lookup

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
        self._thumbnail_by_id = {}
        self.selected_clips = set()
        self._source_lookup = {}
        self.empty_label.setVisible(True)

    def remove_clips_for_source(self, source_id: str):
        """Remove all clips for a specific source (used when re-analyzing)."""
        # Separate into keep and remove in single pass (O(n) instead of O(n²))
        keep = []
        remove = []
        for thumb in self.thumbnails:
            if thumb.source.id == source_id:
                remove.append(thumb)
            else:
                keep.append(thumb)

        # Clean up removed widgets
        for thumb in remove:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()
            # Remove from selection if selected
            self.selected_clips.discard(thumb.clip.id)
            # Remove from lookups
            self._thumbnail_by_id.pop(thumb.clip.id, None)
            self._source_lookup.pop(thumb.clip.id, None)

        # Replace list in one operation (avoids O(n) list.remove() calls)
        if remove:
            self.thumbnails = keep
            self._rebuild_grid()

        # Show empty state if no clips left
        if not self.thumbnails:
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
        """Update the colors for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_colors(colors)

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update the shot type for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_shot_type(shot_type)

    def update_clip_transcript(self, clip_id: str, segments: list):
        """Update the transcript for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_transcript(segments)

    def update_clip_thumbnail(self, clip_id: str, thumb_path: Path):
        """Update the thumbnail image for a specific clip (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_thumbnail(thumb_path)

    def _on_filter_changed(self, filter_option: str):
        """Handle shot type filter dropdown change."""
        self._current_filter = filter_option
        self._rebuild_grid()

    def _on_color_filter_changed(self, filter_option: str):
        """Handle color palette filter dropdown change."""
        self._current_color_filter = filter_option
        self._rebuild_grid()

    def _on_search_changed(self, search_text: str):
        """Handle transcript search input change."""
        self._current_search_query = search_text.lower().strip()
        self._rebuild_grid()

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
        """Rebuild the grid layout with current thumbnail order and filter."""
        # Remove all thumbnails from grid
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
            thumb.setVisible(False)

        # Filter thumbnails based on current filter
        visible_thumbs = []
        for thumb in self.thumbnails:
            if self._matches_filter(thumb):
                visible_thumbs.append(thumb)

        # Re-add visible thumbnails in order
        for i, thumb in enumerate(visible_thumbs):
            row = i // self.COLUMNS
            col = i % self.COLUMNS
            self.grid.addWidget(thumb, row, col)
            thumb.setVisible(True)

    def _matches_filter(self, thumb: ClipThumbnail) -> bool:
        """Check if a thumbnail matches all filters (AND logic)."""
        # Check shot type filter
        if self._current_filter != "All":
            shot_type = thumb.clip.shot_type
            if not shot_type:
                return False
            if get_display_name(shot_type) != self._current_filter:
                return False

        # Check color palette filter
        if self._current_color_filter != "All":
            colors = thumb.clip.dominant_colors
            if not colors:
                return False
            palette = classify_color_palette(colors)
            if get_palette_display_name(palette) != self._current_color_filter:
                return False

        # Check transcript search
        if self._current_search_query:
            transcript_text = thumb.clip.get_transcript_text().lower()
            if self._current_search_query not in transcript_text:
                return False

        return True
