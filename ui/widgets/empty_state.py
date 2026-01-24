"""Empty state widget for displaying placeholder messages."""

from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class EmptyStateWidget(QWidget):
    """Widget showing empty state message with title and description."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self._setup_ui(title, message)

    def _setup_ui(self, title: str, message: str):
        layout = QVBoxLayout(self)
        layout.addStretch()

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #666;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setStyleSheet("color: #888;")
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)

        layout.addStretch()
