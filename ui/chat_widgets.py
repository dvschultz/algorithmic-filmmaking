"""Chat UI widgets for message display.

Provides:
- MessageBubble: Static message display (user or assistant)
- StreamingBubble: Accumulates streaming text from LLM
- ToolIndicator: Shows tool execution status
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class MessageBubble(QFrame):
    """Single message bubble for chat display."""

    def __init__(self, text: str, is_user: bool, parent=None):
        """Create a message bubble.

        Args:
            text: Message text (supports Markdown)
            is_user: True for user messages, False for assistant
            parent: Parent widget
        """
        super().__init__(parent)
        self._is_user = is_user
        self._setup_ui(text)

    def _setup_ui(self, text: str):
        # Set object name for styling
        self.setObjectName("userBubble" if self._is_user else "assistantBubble")

        # Style the bubble
        if self._is_user:
            self.setStyleSheet("""
                QFrame#userBubble {
                    background-color: #0084ff;
                    border-radius: 12px;
                    margin-left: 40px;
                }
                QFrame#userBubble QLabel {
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#assistantBubble {
                    background-color: #e4e6eb;
                    border-radius: 12px;
                    margin-right: 40px;
                }
                QFrame#assistantBubble QLabel {
                    color: #050505;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.MarkdownText)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.label)


class StreamingBubble(QFrame):
    """Message bubble that accumulates streaming text."""

    def __init__(self, parent=None):
        """Create a streaming bubble for assistant responses.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._text = ""
        self._is_finished = False
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("assistantBubble")
        self.setStyleSheet("""
            QFrame#assistantBubble {
                background-color: #e4e6eb;
                border-radius: 12px;
                margin-right: 40px;
            }
            QFrame#assistantBubble QLabel {
                color: #050505;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel("...")
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.PlainText)  # Use plain text during streaming
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.label)

    def append_text(self, chunk: str):
        """Append a chunk of text to the bubble.

        Args:
            chunk: Text chunk from streaming response
        """
        if self._is_finished:
            return

        self._text += chunk
        # Use plain text during streaming for performance
        self.label.setText(self._text)

    def finish(self):
        """Mark the bubble as finished and render final Markdown."""
        if self._is_finished:
            return

        self._is_finished = True
        # Render final text as Markdown
        self.label.setTextFormat(Qt.MarkdownText)
        self.label.setText(self._text)

    @property
    def text(self) -> str:
        """Get the accumulated text."""
        return self._text

    def clear_text(self):
        """Clear the accumulated text."""
        self._text = ""
        self.label.setText("...")
        self._is_finished = False
        self.label.setTextFormat(Qt.PlainText)


class ToolIndicator(QFrame):
    """Shows tool execution status inline in chat."""

    def __init__(self, tool_name: str, status: str = "running", parent=None):
        """Create a tool indicator.

        Args:
            tool_name: Name of the tool being executed
            status: Initial status ("running", "complete", "error")
            parent: Parent widget
        """
        super().__init__(parent)
        self._tool_name = tool_name
        self._status = status
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("toolIndicator")
        self.setStyleSheet("""
            QFrame#toolIndicator {
                background-color: #f0f2f5;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
                padding: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Status icon
        self.status_icon = QLabel(self._get_status_icon())
        self.status_icon.setFixedWidth(20)
        layout.addWidget(self.status_icon)

        # Tool name and status
        self.name_label = QLabel(self._get_status_text())
        self.name_label.setStyleSheet("color: #65676b; font-size: 12px;")
        layout.addWidget(self.name_label)

        layout.addStretch()

    def _get_status_icon(self) -> str:
        """Get icon for current status."""
        icons = {
            "running": "\u23f3",  # Hourglass
            "complete": "\u2714",  # Check mark
            "error": "\u2718",     # X mark
        }
        return icons.get(self._status, "\u2753")  # Question mark fallback

    def _get_status_text(self) -> str:
        """Get status text."""
        prefixes = {
            "running": "Running:",
            "complete": "Completed:",
            "error": "Failed:",
        }
        prefix = prefixes.get(self._status, "")
        return f"{prefix} {self._tool_name}"

    def set_status(self, status: str):
        """Update the status.

        Args:
            status: New status ("running", "complete", "error")
        """
        self._status = status
        self.status_icon.setText(self._get_status_icon())
        self.name_label.setText(self._get_status_text())

        # Update color based on status
        if status == "complete":
            self.status_icon.setStyleSheet("color: #00a400;")
        elif status == "error":
            self.status_icon.setStyleSheet("color: #ff0000;")

    def set_complete(self, success: bool = True):
        """Mark the tool as complete.

        Args:
            success: True if successful, False if failed
        """
        self.set_status("complete" if success else "error")
