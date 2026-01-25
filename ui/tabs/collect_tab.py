"""Collect tab for importing and managing video library."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QInputDialog,
    QLineEdit,
)
from PySide6.QtCore import Signal

from .base_tab import BaseTab
from models.clip import Source
from ui.source_browser import SourceBrowser


class CollectTab(BaseTab):
    """Tab for importing local videos and managing video library.

    Signals:
        videos_added: Emitted when videos are added to library (paths: list[Path])
        analyze_requested: Emitted when analysis is requested (source_ids: list[str])
            If empty list, analyze all unanalyzed sources.
        download_requested: Emitted when URL download is requested (url: str)
        source_selected: Emitted when a source is selected (source: Source)
    """

    videos_added = Signal(list)  # list of Paths
    analyze_requested = Signal(list)  # list of source IDs (empty = all unanalyzed)
    download_requested = Signal(str)  # URL
    source_selected = Signal(object)  # Source

    def _setup_ui(self):
        """Set up the Collect tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top toolbar
        toolbar = self._create_toolbar()
        layout.addLayout(toolbar)

        # Main content: source browser grid (with built-in add card)
        self.source_browser = SourceBrowser()
        self.source_browser.source_selected.connect(self._on_source_selected)
        self.source_browser.source_double_clicked.connect(self._on_source_double_clicked)
        self.source_browser.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.source_browser, 1)  # Stretch factor 1

    def _create_toolbar(self) -> QHBoxLayout:
        """Create the top toolbar."""
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Video Library")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        toolbar.addWidget(title)

        toolbar.addSpacing(16)

        # Video count label
        self.count_label = QLabel("0 videos")
        self.count_label.setStyleSheet("color: #666;")
        toolbar.addWidget(self.count_label)

        toolbar.addStretch()

        # Import from URL button
        self.url_btn = QPushButton("Import from URL...")
        self.url_btn.setToolTip("Download video from YouTube or Vimeo")
        self.url_btn.clicked.connect(self._on_url_click)
        toolbar.addWidget(self.url_btn)

        toolbar.addSpacing(8)

        # Cut New button
        self.analyze_btn = QPushButton("Cut New Videos")
        self.analyze_btn.setToolTip("Detect scenes in all unanalyzed videos")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze_click)
        toolbar.addWidget(self.analyze_btn)

        return toolbar

    def _on_files_dropped(self, paths: list[Path]):
        """Handle files dropped on add card."""
        self.videos_added.emit(paths)

    def _on_analyze_click(self):
        """Handle Cut New Videos button click."""
        # Emit with empty list to analyze all unanalyzed
        self.analyze_requested.emit([])

    def _on_url_click(self):
        """Handle URL import button click."""
        url, ok = QInputDialog.getText(
            self,
            "Import from URL",
            "Enter YouTube or Vimeo URL:",
            QLineEdit.Normal,
            "",
        )
        if ok and url.strip():
            self.download_requested.emit(url.strip())

    def _on_source_selected(self, source: Source):
        """Handle source selection in browser."""
        self.source_selected.emit(source)

    def _on_source_double_clicked(self, source: Source):
        """Handle source double-click - switch to analyze."""
        self.source_selected.emit(source)

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        count = self.source_browser.source_count()
        unanalyzed = self.source_browser.unanalyzed_count()

        # Update count label
        if count == 0:
            self.count_label.setText("0 videos")
        elif count == 1:
            self.count_label.setText("1 video")
        else:
            self.count_label.setText(f"{count} videos")

        # Update cut button
        if unanalyzed > 0:
            self.analyze_btn.setEnabled(True)
            if unanalyzed == 1:
                self.analyze_btn.setText("Cut 1 New Video")
            else:
                self.analyze_btn.setText(f"Cut {unanalyzed} New Videos")
        else:
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Cut New Videos")

    # Public methods for MainWindow to call

    def add_source(self, source: Source):
        """Add a source to the library grid."""
        self.source_browser.add_source(source)
        self._update_ui_state()

    def remove_source(self, source_id: str):
        """Remove a source from the library."""
        self.source_browser.remove_source(source_id)
        self._update_ui_state()

    def clear(self):
        """Clear all sources from the library."""
        self.source_browser.clear()
        self._update_ui_state()

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return self.source_browser.get_source(source_id)

    def get_all_sources(self) -> list[Source]:
        """Get all sources in the library."""
        return self.source_browser.get_all_sources()

    def get_unanalyzed_sources(self) -> list[Source]:
        """Get all unanalyzed sources."""
        return self.source_browser.get_unanalyzed_sources()

    def update_source_thumbnail(self, source_id: str, thumb_path: Path):
        """Update thumbnail for a source."""
        self.source_browser.update_source_thumbnail(source_id, thumb_path)

    def update_source_analyzed(self, source_id: str, analyzed: bool = True):
        """Mark a source as analyzed."""
        self.source_browser.update_source_analyzed(source_id, analyzed)
        self._update_ui_state()

    def set_downloading(self, is_downloading: bool):
        """Update UI state during download."""
        self.url_btn.setEnabled(not is_downloading)
        if is_downloading:
            self.url_btn.setText("Downloading...")
        else:
            self.url_btn.setText("Import from URL...")
