"""Source browser with thumbnail grid view for video library."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QGridLayout,
    QLabel,
    QFrame,
    QPushButton,
    QFileDialog,
    QApplication,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent, QPalette

from models.clip import Source
from ui.source_thumbnail import SourceThumbnail
from ui.theme import theme, UISizes


class AddVideoCard(QFrame):
    """Card widget for adding new videos to the library."""

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

    files_dropped = Signal(list)  # List of Paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(2)
        self.setFixedSize(UISizes.GRID_CARD_MAX_WIDTH, 200)  # Match SourceThumbnail
        self.setCursor(Qt.PointingHandCursor)

        self._setup_ui()
        self._apply_theme()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignCenter)

        # Plus icon
        self.icon_label = QLabel("+")
        icon_font = QFont()
        icon_font.setPointSize(36)
        icon_font.setBold(True)
        self.icon_label.setFont(icon_font)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Text
        self.text_label = QLabel("Add Video")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        # Subtext
        self.sub_label = QLabel("Drop or click")
        self.sub_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sub_label)

    def _apply_theme(self, dragging: bool = False):
        """Apply theme-aware styles."""
        if dragging:
            self.setStyleSheet(f"""
                AddVideoCard {{
                    background-color: rgba(76, 175, 80, 0.1);
                    border: 2px dashed {theme().accent_green};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                AddVideoCard {{
                    background-color: {theme().card_background};
                    border: 2px dashed {theme().border_primary};
                }}
                AddVideoCard:hover {{
                    background-color: {theme().card_hover};
                    border-color: {theme().border_focus};
                }}
            """)
        self.icon_label.setStyleSheet(f"color: {theme().text_secondary};")
        self.text_label.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {theme().text_secondary};")
        self.sub_label.setStyleSheet(f"font-size: 10px; color: {theme().text_muted};")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._browse_for_files()

    def _browse_for_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Video Files",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
        )
        if file_paths:
            self.files_dropped.emit([Path(p) for p in file_paths])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        self._apply_theme(dragging=True)
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._apply_theme(dragging=False)

    def dropEvent(self, event: QDropEvent):
        self._apply_theme(dragging=False)
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    paths.append(path)
        if paths:
            self.files_dropped.emit(paths)


class SourceBrowser(QWidget):
    """Grid browser for viewing video sources in the library."""

    source_selected = Signal(object)  # Source
    source_double_clicked = Signal(object)  # Source
    files_dropped = Signal(list)  # List of Paths from add card

    COLUMNS = 4

    def __init__(self):
        super().__init__()
        self.thumbnails: list[SourceThumbnail] = []
        self.selected_source_ids: set[str] = set()  # Multi-selection support
        self._sources_by_id: dict[str, Source] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for thumbnails
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Container for grid
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(UISizes.GRID_GUTTER)
        self.grid.setContentsMargins(UISizes.GRID_MARGIN, UISizes.GRID_MARGIN, UISizes.GRID_MARGIN, UISizes.GRID_MARGIN)
        # Align grid to top-left
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        # Add video card (always first in grid)
        self.add_card = AddVideoCard()
        self.add_card.files_dropped.connect(self.files_dropped.emit)
        self._rebuild_grid()

    def add_source(self, source: Source):
        """Add a source to the browser."""
        # Avoid duplicates
        if source.id in self._sources_by_id:
            return

        # Store reference
        self._sources_by_id[source.id] = source

        # Create thumbnail widget
        thumb = SourceThumbnail(source)
        thumb.clicked.connect(self._on_thumbnail_clicked)
        thumb.double_clicked.connect(self._on_thumbnail_double_clicked)

        self.thumbnails.append(thumb)

        # Add to grid
        self._rebuild_grid()

    def remove_source(self, source_id: str):
        """Remove a source from the browser."""
        if source_id not in self._sources_by_id:
            return

        # Find and remove thumbnail
        for thumb in self.thumbnails:
            if thumb.source.id == source_id:
                self.grid.removeWidget(thumb)
                thumb.deleteLater()
                self.thumbnails.remove(thumb)
                break

        del self._sources_by_id[source_id]

        # Clear selection if removed
        self.selected_source_ids.discard(source_id)

        self._rebuild_grid()

    def clear(self):
        """Clear all sources."""
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()

        self.thumbnails = []
        self.selected_source_ids.clear()
        self._sources_by_id = {}
        # Add card stays at position 0
        self._rebuild_grid()

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return self._sources_by_id.get(source_id)

    def get_all_sources(self) -> list[Source]:
        """Get all sources in the browser."""
        return list(self._sources_by_id.values())

    def get_unanalyzed_sources(self) -> list[Source]:
        """Get all sources that haven't been analyzed."""
        return [s for s in self._sources_by_id.values() if not s.analyzed]

    def update_source_thumbnail(self, source_id: str, thumb_path: Path):
        """Update the thumbnail for a specific source."""
        for thumb in self.thumbnails:
            if thumb.source.id == source_id:
                thumb.set_thumbnail(thumb_path)
                break

    def update_source_analyzed(self, source_id: str, analyzed: bool = True):
        """Update the analyzed status for a specific source."""
        source = self._sources_by_id.get(source_id)
        if source:
            source.analyzed = analyzed
        for thumb in self.thumbnails:
            if thumb.source.id == source_id:
                thumb.set_analyzed(analyzed)
                break

    def _on_thumbnail_clicked(self, source: Source):
        """Handle thumbnail click - toggle selection."""
        # Toggle selection
        if source.id in self.selected_source_ids:
            self.selected_source_ids.discard(source.id)
        else:
            self.selected_source_ids.add(source.id)

        # Update all thumbnail states
        for thumb in self.thumbnails:
            thumb.set_selected(thumb.source.id in self.selected_source_ids)

        self.source_selected.emit(source)

    def select_all(self) -> None:
        """Select all sources."""
        self.selected_source_ids = set(thumb.source.id for thumb in self.thumbnails)

        # Update all thumbnail states
        for thumb in self.thumbnails:
            thumb.set_selected(True)

    def clear_selection(self) -> None:
        """Clear all selections."""
        self.selected_source_ids.clear()

        # Update all thumbnail states
        for thumb in self.thumbnails:
            thumb.set_selected(False)

    def get_selected_sources(self) -> list[Source]:
        """Get list of selected sources."""
        return [thumb.source for thumb in self.thumbnails if thumb.source.id in self.selected_source_ids]

    def _on_thumbnail_double_clicked(self, source: Source):
        """Handle thumbnail double-click."""
        self.source_double_clicked.emit(source)

    def _rebuild_grid(self):
        """Rebuild the grid layout with add card first, left-aligned."""
        # Remove add card and all thumbnails from grid
        self.grid.removeWidget(self.add_card)
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)

        # Add card is always at position 0, aligned top-left
        self.grid.addWidget(self.add_card, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        # Re-add thumbnails starting at position 1
        for i, thumb in enumerate(self.thumbnails):
            pos = i + 1  # Offset by 1 for add card
            row = pos // self.COLUMNS
            col = pos % self.COLUMNS
            self.grid.addWidget(thumb, row, col, Qt.AlignTop | Qt.AlignLeft)
            thumb.setVisible(True)

    def source_count(self) -> int:
        """Get the number of sources in the browser."""
        return len(self.thumbnails)

    def unanalyzed_count(self) -> int:
        """Get the number of unanalyzed sources."""
        return len(self.get_unanalyzed_sources())
