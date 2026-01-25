"""Expandable YouTube search panel for Collect tab."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtNetwork import QNetworkAccessManager

from ui.theme import theme
from ui.youtube_result_thumbnail import YouTubeResultThumbnail
from core.youtube_api import YouTubeVideo


class YouTubeSearchPanel(QWidget):
    """Collapsible panel for YouTube search and results."""

    # Signals
    search_requested = Signal(str)  # query
    download_requested = Signal(list)  # list of YouTubeVideo
    error_occurred = Signal(str)

    THUMB_WIDTH = 180
    THUMB_SPACING = 8
    MIN_COLUMNS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._results: list[YouTubeVideo] = []
        self._thumbnails: list[YouTubeResultThumbnail] = []
        self._selected_videos: set[str] = set()  # video_ids
        self._columns = self.MIN_COLUMNS

        # Shared network manager for thumbnail downloads (efficient resource usage)
        self._network_manager = QNetworkAccessManager(self)

        self._setup_ui()
        self._connect_theme()

    def _setup_ui(self):
        """Set up the panel UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Toggle button
        self.toggle_btn = QPushButton("Search YouTube")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._on_toggle)
        self.main_layout.addWidget(self.toggle_btn)

        # Content container (for animation)
        self.content = QFrame()
        self.content.setMaximumHeight(0)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)

        # Search row
        search_row = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search YouTube videos...")
        self.search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self.search_input, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search)
        search_row.addWidget(self.search_btn)

        self.content_layout.addLayout(search_row)

        # Results scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(200)
        self.scroll.setMaximumHeight(400)

        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(8)
        self.results_grid.setContentsMargins(0, 0, 0, 0)

        self.scroll.setWidget(self.results_container)
        self.content_layout.addWidget(self.scroll)

        # Status/action row
        action_row = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._on_select_all)
        action_row.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_selection)
        action_row.addWidget(self.clear_btn)

        self.status_label = QLabel("")
        action_row.addWidget(self.status_label)

        action_row.addStretch()

        self.download_btn = QPushButton("Download Selected")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._on_download)
        action_row.addWidget(self.download_btn)

        self.content_layout.addLayout(action_row)

        self.main_layout.addWidget(self.content)

        # Animation
        self._animation = QPropertyAnimation(self.content, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

    @Slot(bool)
    def _on_toggle(self, checked: bool):
        """Handle panel expand/collapse."""
        self._expanded = checked

        if checked:
            self.toggle_btn.setText("Search YouTube")
            self._animation.setStartValue(0)
            self._animation.setEndValue(500)  # Max expanded height
        else:
            self.toggle_btn.setText("Search YouTube")
            self._animation.setStartValue(self.content.height())
            self._animation.setEndValue(0)

        self._animation.start()

    @Slot()
    def _on_search(self):
        """Emit search request."""
        query = self.search_input.text().strip()
        if query:
            self.search_requested.emit(query)

    def display_results(self, videos: list[YouTubeVideo]):
        """Display search results in the grid."""
        self._clear_results()
        self._results = videos

        # Calculate columns based on available width
        self._calculate_columns()

        for i, video in enumerate(videos):
            thumb = YouTubeResultThumbnail(video, self._network_manager)
            thumb.selection_changed.connect(self._on_selection_changed)
            self._thumbnails.append(thumb)

        self._reflow_grid()
        self._update_status()

    def _calculate_columns(self):
        """Calculate number of columns based on scroll area width."""
        available_width = self.scroll.viewport().width()
        if available_width <= 0:
            available_width = self.scroll.width() - 20  # Account for scrollbar

        # Calculate how many thumbnails fit
        cols = max(self.MIN_COLUMNS, available_width // (self.THUMB_WIDTH + self.THUMB_SPACING))
        self._columns = cols

    def _reflow_grid(self):
        """Reposition thumbnails in the grid based on current column count."""
        # Remove all widgets from grid (but don't delete them)
        for thumb in self._thumbnails:
            self.results_grid.removeWidget(thumb)

        # Re-add in correct positions
        for i, thumb in enumerate(self._thumbnails):
            row = i // self._columns
            col = i % self._columns
            self.results_grid.addWidget(thumb, row, col)

    def resizeEvent(self, event):
        """Handle resize to reflow grid."""
        super().resizeEvent(event)
        if self._thumbnails:
            # Delay reflow slightly to get accurate size
            QTimer.singleShot(0, self._on_resize_reflow)

    def _on_resize_reflow(self):
        """Reflow grid after resize."""
        old_cols = self._columns
        self._calculate_columns()
        if self._columns != old_cols:
            self._reflow_grid()

    def _clear_results(self):
        """Clear all result thumbnails."""
        for thumb in self._thumbnails:
            self.results_grid.removeWidget(thumb)
            thumb.deleteLater()
        self._thumbnails = []
        self._results = []
        self._selected_videos = set()
        self._update_status()

    @Slot(str, bool)
    def _on_selection_changed(self, video_id: str, selected: bool):
        """Handle thumbnail selection change."""
        if selected:
            self._selected_videos.add(video_id)
        else:
            self._selected_videos.discard(video_id)
        self._update_status()

    @Slot()
    def _on_select_all(self):
        """Select all results."""
        for thumb in self._thumbnails:
            thumb.set_selected(True)

    @Slot()
    def _on_clear_selection(self):
        """Clear all selections."""
        for thumb in self._thumbnails:
            thumb.set_selected(False)

    @Slot()
    def _on_download(self):
        """Request download of selected videos."""
        selected = [v for v in self._results if v.video_id in self._selected_videos]
        if selected:
            self.download_requested.emit(selected)

    def _update_status(self):
        """Update status label and button state."""
        count = len(self._selected_videos)
        if count == 0:
            self.status_label.setText("")
            self.download_btn.setEnabled(False)
        else:
            self.status_label.setText(f"{count} selected")
            self.download_btn.setEnabled(True)

    def set_searching(self, searching: bool):
        """Update UI state during search."""
        self.search_btn.setEnabled(not searching)
        self.search_input.setEnabled(not searching)
        if searching:
            self.search_btn.setText("Searching...")
        else:
            self.search_btn.setText("Search")

    def set_downloading(self, downloading: bool):
        """Update UI state during download."""
        self.download_btn.setEnabled(not downloading and len(self._selected_videos) > 0)
        if downloading:
            self.download_btn.setText("Downloading...")
        else:
            self.download_btn.setText("Download Selected")

    def _connect_theme(self):
        """Connect to theme changes."""
        if theme().changed:
            theme().changed.connect(self._apply_theme)
        self._apply_theme()

    @Slot()
    def _apply_theme(self):
        """Apply current theme colors."""
        self.content.setStyleSheet(
            f"""
            QFrame {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
            }}
        """
        )
        self.status_label.setStyleSheet(f"color: {theme().text_secondary};")

    def get_search_query(self) -> str:
        """Get the current search query."""
        return self.search_input.text().strip()
