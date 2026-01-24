"""Base class for workflow tabs."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class BaseTab(QWidget):
    """Base class for all workflow tabs.

    Provides common signals and lifecycle methods for tab activation/deactivation.
    """

    # Common signals
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the tab UI. Override in subclasses."""
        pass

    def on_tab_activated(self):
        """Called when this tab becomes visible.

        Override to refresh data or update UI state.
        """
        pass

    def on_tab_deactivated(self):
        """Called when switching away from this tab.

        Override to pause operations or save state.
        """
        pass

    def _create_placeholder(self, title: str, message: str) -> QVBoxLayout:
        """Create a centered placeholder layout with title and message.

        Args:
            title: Large title text
            message: Smaller description text

        Returns:
            QVBoxLayout with centered placeholder content
        """
        layout = QVBoxLayout()
        layout.addStretch()

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(24)
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
        return layout
