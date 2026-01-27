"""Empty state widget for displaying placeholder messages."""

from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ui.theme import theme


class EmptyStateWidget(QWidget):
    """Widget showing empty state message with title and description."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self._setup_ui(title, message)

    def _setup_ui(self, title: str, message: str):
        layout = QVBoxLayout(self)
        layout.addStretch()

        self._title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f"color: {theme().text_secondary};")
        self._title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title_label)

        self._message_label = QLabel(message)
        self._message_label.setStyleSheet(f"color: {theme().text_muted};")
        self._message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._message_label)

        layout.addStretch()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._title_label.setStyleSheet(f"color: {theme().text_secondary};")
        self._message_label.setStyleSheet(f"color: {theme().text_muted};")
