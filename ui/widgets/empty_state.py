"""Empty state widget for displaying placeholder messages."""

from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QColor, QLinearGradient, QBrush

from ui.theme import theme, TypeScale, Spacing


class EmptyStateWidget(QWidget):
    """Widget showing empty state message with title and description.

    Features a subtle gradient wash backdrop for visual interest.
    """

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self._setup_ui(title, message)

    def _setup_ui(self, title: str, message: str):
        layout = QVBoxLayout(self)
        layout.addStretch()

        self._title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(TypeScale.XL)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f"color: {theme().text_secondary}; background: transparent;")
        self._title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title_label)

        self._message_label = QLabel(message)
        self._message_label.setStyleSheet(f"color: {theme().text_muted}; background: transparent;")
        self._message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._message_label)

        layout.addStretch()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def paintEvent(self, event):
        """Paint a subtle gradient wash backdrop."""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(99, 102, 241, 30))   # Very faint indigo
        gradient.setColorAt(0.5, QColor(139, 92, 246, 20))    # Very faint violet
        gradient.setColorAt(1.0, QColor(59, 130, 246, 25))    # Very faint blue
        painter.fillRect(self.rect(), QBrush(gradient))
        painter.end()
        super().paintEvent(event)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._title_label.setStyleSheet(f"color: {theme().text_secondary}; background: transparent;")
        self._message_label.setStyleSheet(f"color: {theme().text_muted}; background: transparent;")
        self.update()
