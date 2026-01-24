"""Collect tab for importing and downloading videos."""

from pathlib import Path

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QFileDialog,
    QInputDialog,
    QLineEdit,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent

from .base_tab import BaseTab


class DropZone(QFrame):
    """Drag-and-drop zone for video files."""

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

    file_dropped = Signal(object)  # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()
        self._update_style(dragging=False)

    def _setup_ui(self):
        """Set up the drop zone UI."""
        self.setMinimumSize(400, 200)
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Icon/emoji
        icon_label = QLabel("\U0001F4C1")  # Folder emoji
        icon_font = QFont()
        icon_font.setPointSize(48)
        icon_label.setFont(icon_font)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # Main text
        main_label = QLabel("Drop video here")
        main_font = QFont()
        main_font.setPointSize(16)
        main_font.setBold(True)
        main_label.setFont(main_font)
        main_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(main_label)

        # Sub text
        sub_label = QLabel("or click to browse")
        sub_label.setStyleSheet("color: #888;")
        sub_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub_label)

        # Supported formats
        formats_label = QLabel("Supported: MP4, MKV, MOV, AVI, WebM")
        formats_label.setStyleSheet("color: #999; margin-top: 10px;")
        formats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(formats_label)

    def _update_style(self, dragging: bool):
        """Update visual style based on drag state."""
        if dragging:
            self.setStyleSheet("""
                DropZone {
                    border: 3px dashed #4CAF50;
                    border-radius: 10px;
                    background-color: rgba(76, 175, 80, 0.1);
                }
            """)
        else:
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #666;
                    border-radius: 10px;
                    background-color: transparent;
                }
                DropZone:hover {
                    border-color: #888;
                    background-color: rgba(255, 255, 255, 0.05);
                }
            """)

    def mousePressEvent(self, event):
        """Handle click to open file dialog."""
        if event.button() == Qt.LeftButton:
            self._browse_for_file()

    def _browse_for_file(self):
        """Open file dialog to select video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
        )
        if file_path:
            self.file_dropped.emit(Path(file_path))

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        self._update_style(dragging=True)
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self._update_style(dragging=False)

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        self._update_style(dragging=False)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    self.file_dropped.emit(path)
                    return


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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.addStretch()

        # Center container
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        center_layout.addWidget(self.drop_zone)

        # Separator
        separator_layout = QHBoxLayout()
        separator_layout.addStretch()
        separator_label = QLabel("— or —")
        separator_label.setStyleSheet("color: #888; margin: 20px 0;")
        separator_layout.addWidget(separator_label)
        separator_layout.addStretch()
        center_layout.addLayout(separator_layout)

        # Import URL button
        self.url_btn = QPushButton("Import from URL...")
        self.url_btn.setMinimumWidth(200)
        self.url_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 24px;
                font-size: 14px;
            }
        """)
        self.url_btn.clicked.connect(self._on_url_click)
        url_layout = QHBoxLayout()
        url_layout.addStretch()
        url_layout.addWidget(self.url_btn)
        url_layout.addStretch()
        center_layout.addLayout(url_layout)

        # Supported URL sources
        url_info = QLabel("Supported: YouTube, Vimeo")
        url_info.setStyleSheet("color: #999; margin-top: 10px;")
        url_info.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(url_info)

        layout.addLayout(center_layout)
        layout.addStretch()

    def _on_file_dropped(self, path: Path):
        """Handle file dropped or selected via browse."""
        self.video_imported.emit(path)

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
