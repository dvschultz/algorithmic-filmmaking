"""Collect tab for importing and downloading videos."""

from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtCore import Signal

from .base_tab import BaseTab


class CollectTab(BaseTab):
    """Tab for importing local videos and downloading from URLs.

    Signals:
        video_imported: Emitted when a video is imported (path: Path)
        download_requested: Emitted when URL download is requested (url: str)
    """

    video_imported = Signal(object)  # Path
    download_requested = Signal(str)  # URL

    def _setup_ui(self):
        """Set up the Collect tab UI."""
        layout = self._create_placeholder(
            "Collect",
            "Import videos from local files or download from URLs"
        )
        self.setLayout(layout)
