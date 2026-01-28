"""Expandable YouTube search panel for Collect tab."""

import logging
from typing import Optional

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
    QComboBox,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, Slot, QPropertyAnimation, QEasingCurve, QTimer, QThread, QObject
from PySide6.QtNetwork import QNetworkAccessManager

from ui.theme import theme, UISizes
from ui.youtube_result_thumbnail import YouTubeResultThumbnail
from core.youtube_api import YouTubeVideo, ASPECT_RATIO_RANGES, RESOLUTION_THRESHOLDS, SIZE_LIMITS

logger = logging.getLogger(__name__)


class MetadataFetchWorker(QObject):
    """Worker to fetch detailed video metadata via yt-dlp in background."""

    progress = Signal(int, int)  # current, total
    video_updated = Signal(str, dict)  # video_id, metadata dict
    finished = Signal()
    error = Signal(str)

    def __init__(self, videos: list[YouTubeVideo]):
        super().__init__()
        self._videos = videos
        self._cancelled = False

    def cancel(self):
        """Cancel the fetch operation."""
        self._cancelled = True

    def run(self):
        """Fetch metadata for each video."""
        try:
            from core.downloader import VideoDownloader
            downloader = VideoDownloader()
        except Exception as e:
            self.error.emit(f"Failed to initialize downloader: {e}")
            self.finished.emit()
            return

        total = len(self._videos)
        for i, video in enumerate(self._videos):
            if self._cancelled:
                break

            self.progress.emit(i, total)

            try:
                info = downloader.get_video_info(
                    video.youtube_url,
                    include_format_details=True
                )
                metadata = {
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "aspect_ratio": info.get("aspect_ratio"),
                    "filesize_approx": info.get("filesize_approx"),
                }
                self.video_updated.emit(video.video_id, metadata)
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for {video.video_id}: {e}")

        self.progress.emit(total, total)
        self.finished.emit()


class YouTubeSearchPanel(QWidget):
    """Collapsible panel for YouTube search and results."""

    # Signals
    search_requested = Signal(str)  # query
    download_requested = Signal(list)  # list of YouTubeVideo
    error_occurred = Signal(str)

    THUMB_WIDTH = UISizes.GRID_CARD_MAX_WIDTH
    THUMB_SPACING = UISizes.GRID_GUTTER
    MIN_COLUMNS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._results: list[YouTubeVideo] = []
        self._thumbnails: list[YouTubeResultThumbnail] = []
        self._selected_videos: set[str] = set()  # video_ids
        self._columns = self.MIN_COLUMNS
        self._metadata_thread: Optional[QThread] = None
        self._metadata_worker: Optional[MetadataFetchWorker] = None

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

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(12)

        # Aspect ratio filter
        aspect_label = QLabel("Aspect:")
        aspect_label.setStyleSheet(f"color: {theme().text_secondary};")
        filter_row.addWidget(aspect_label)

        self.aspect_combo = QComboBox()
        self.aspect_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.aspect_combo.addItem("Any", "any")
        for name in ASPECT_RATIO_RANGES.keys():
            self.aspect_combo.addItem(name, name.lower().replace(":", "-"))
        self.aspect_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.aspect_combo)

        # Resolution filter
        res_label = QLabel("Resolution:")
        res_label.setStyleSheet(f"color: {theme().text_secondary};")
        filter_row.addWidget(res_label)

        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.resolution_combo.addItem("Any", "any")
        self.resolution_combo.addItem("4K (2160p+)", "4k")
        self.resolution_combo.addItem("1080p+", "1080p")
        self.resolution_combo.addItem("720p+", "720p")
        self.resolution_combo.addItem("480p+", "480p")
        self.resolution_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.resolution_combo)

        # Max size filter
        size_label = QLabel("Max Size:")
        size_label.setStyleSheet(f"color: {theme().text_secondary};")
        filter_row.addWidget(size_label)

        self.size_combo = QComboBox()
        self.size_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.size_combo.addItem("Any", "any")
        self.size_combo.addItem("< 100 MB", "100mb")
        self.size_combo.addItem("< 500 MB", "500mb")
        self.size_combo.addItem("< 1 GB", "1gb")
        self.size_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.size_combo)

        filter_row.addStretch()

        self.content_layout.addLayout(filter_row)

        # Metadata fetch progress bar
        self.metadata_progress = QProgressBar()
        self.metadata_progress.setMaximumHeight(16)
        self.metadata_progress.setTextVisible(True)
        self.metadata_progress.setFormat("Fetching video details... %v/%m")
        self.metadata_progress.hide()
        self.content_layout.addWidget(self.metadata_progress)

        # Results scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(200)
        self.scroll.setMaximumHeight(400)

        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(UISizes.GRID_GUTTER)
        self.results_grid.setContentsMargins(0, 0, 0, 0)
        self.results_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

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

        # Start fetching metadata in background
        self._start_metadata_fetch()

    def _start_metadata_fetch(self):
        """Start background worker to fetch detailed metadata for all results."""
        if not self._results:
            return

        # Cancel any existing fetch
        self._cancel_metadata_fetch()

        # Show progress bar
        self.metadata_progress.setMaximum(len(self._results))
        self.metadata_progress.setValue(0)
        self.metadata_progress.show()

        # Create worker and thread
        self._metadata_thread = QThread()
        self._metadata_worker = MetadataFetchWorker(self._results)
        self._metadata_worker.moveToThread(self._metadata_thread)

        # Connect signals
        self._metadata_thread.started.connect(self._metadata_worker.run)
        self._metadata_worker.progress.connect(self._on_metadata_progress)
        self._metadata_worker.video_updated.connect(self._on_video_metadata_updated)
        self._metadata_worker.finished.connect(self._on_metadata_fetch_complete)
        self._metadata_worker.error.connect(self._on_metadata_error)

        # Connect thread finished to cleanup - ensures proper cleanup after thread stops
        self._metadata_thread.finished.connect(self._on_metadata_thread_finished)

        # Start
        self._metadata_thread.start()

    def _cancel_metadata_fetch(self):
        """Cancel any running metadata fetch."""
        if self._metadata_worker:
            self._metadata_worker.cancel()
        if self._metadata_thread and self._metadata_thread.isRunning():
            self._metadata_thread.quit()
            # Wait for thread to finish - this ensures clean shutdown
            if not self._metadata_thread.wait(2000):
                logger.warning("Metadata fetch thread did not stop in time, terminating")
                self._metadata_thread.terminate()
                self._metadata_thread.wait()

    @Slot()
    def _on_metadata_thread_finished(self):
        """Handle thread cleanup after it has fully stopped."""
        if self._metadata_worker:
            self._metadata_worker.deleteLater()
            self._metadata_worker = None
        if self._metadata_thread:
            self._metadata_thread.deleteLater()
            self._metadata_thread = None

    @Slot(int, int)
    def _on_metadata_progress(self, current: int, total: int):
        """Update metadata fetch progress bar."""
        self.metadata_progress.setValue(current)

    @Slot(str, dict)
    def _on_video_metadata_updated(self, video_id: str, metadata: dict):
        """Handle metadata update for a single video."""
        # Update the video object
        for video in self._results:
            if video.video_id == video_id:
                video.width = metadata.get("width")
                video.height = metadata.get("height")
                video.aspect_ratio = metadata.get("aspect_ratio")
                video.filesize_approx = metadata.get("filesize_approx")
                video.has_detailed_info = True
                break

        # Update the corresponding thumbnail
        for thumb in self._thumbnails:
            if thumb.video.video_id == video_id:
                thumb.update_metadata_display()
                break

        # Re-apply filters to show/hide based on new metadata
        self._apply_filters()

    @Slot()
    def _on_metadata_fetch_complete(self):
        """Handle metadata fetch completion."""
        self.metadata_progress.hide()
        self._apply_filters()
        # Tell thread to quit - cleanup happens in _on_metadata_thread_finished
        if self._metadata_thread:
            self._metadata_thread.quit()

    @Slot(str)
    def _on_metadata_error(self, error: str):
        """Handle metadata fetch error."""
        logger.error(f"Metadata fetch error: {error}")
        self.metadata_progress.hide()

    @Slot()
    def _apply_filters(self):
        """Apply current filter settings to show/hide thumbnails."""
        aspect_filter = self.aspect_combo.currentData()
        resolution_filter = self.resolution_combo.currentData()
        size_filter = self.size_combo.currentData()

        # Map combo data back to filter values
        if aspect_filter and aspect_filter != "any":
            # Convert "16-9" back to "16:9"
            aspect_filter = aspect_filter.replace("-", ":")
        else:
            aspect_filter = "any"

        visible_count = 0
        for thumb in self._thumbnails:
            video = thumb.video

            # If no detailed info yet, show the video (pending metadata)
            if not video.has_detailed_info:
                thumb.setVisible(True)
                visible_count += 1
                continue

            # Apply filters
            matches = (
                video.matches_aspect_ratio(aspect_filter) and
                video.matches_resolution(resolution_filter or "any") and
                video.matches_max_size(size_filter or "any")
            )
            thumb.setVisible(matches)
            if matches:
                visible_count += 1

        # Reflow grid with only visible items
        self._reflow_visible_grid()

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
            self.results_grid.addWidget(thumb, row, col, Qt.AlignTop | Qt.AlignLeft)

    def _reflow_visible_grid(self):
        """Reposition only visible thumbnails in the grid."""
        # Remove all widgets from grid (but don't delete them)
        for thumb in self._thumbnails:
            self.results_grid.removeWidget(thumb)

        # Re-add only visible ones in correct positions
        visible_index = 0
        for thumb in self._thumbnails:
            if thumb.isVisible():
                row = visible_index // self._columns
                col = visible_index % self._columns
                self.results_grid.addWidget(thumb, row, col, Qt.AlignTop | Qt.AlignLeft)
                visible_index += 1

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
        self._cancel_metadata_fetch()
        self.metadata_progress.hide()
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
