"""Chat panel widget for agent interaction.

Provides:
- Message history display with user/assistant bubbles
- Input area with send/cancel buttons
- Provider selection dropdown
- Streaming state management
"""

from PySide6.QtCore import Qt, Signal, Slot, QTimer, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.chat_widgets import MessageBubble, StreamingBubble, ToolIndicator


class ChatPanel(QWidget):
    """Collapsible chat panel for agent interaction."""

    # Signals
    message_sent = Signal(str)  # User message text
    cancel_requested = Signal()  # Cancel button clicked
    provider_changed = Signal(str)  # Provider selection changed

    def __init__(self, parent=None):
        """Create the chat panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._is_streaming = False
        self._response_finished_handled = False
        self._current_bubble = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with provider selector
        header = QHBoxLayout()
        header_label = QLabel("Agent Chat")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(header_label)
        header.addStretch()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "Local (Ollama)",
            "OpenAI",
            "Anthropic",
            "Gemini",
            "OpenRouter"
        ])
        self.provider_combo.setToolTip("Select LLM provider")
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        header.addWidget(self.provider_combo)

        layout.addLayout(header)

        # Message history scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                background-color: white;
            }
        """)

        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(8)
        self.messages_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_area.setWidget(self.messages_widget)

        layout.addWidget(self.scroll_area, 1)

        # Input area with send/cancel buttons
        input_layout = QHBoxLayout()

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Describe what you want to create...")
        self.input_field.setMaximumHeight(80)
        self.input_field.setStyleSheet("""
            QTextEdit {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        # Handle Enter key to send (Shift+Enter for newline)
        self.input_field.installEventFilter(self)
        input_layout.addWidget(self.input_field, 1)

        # Button container
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0084ff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0073e6;
            }
            QPushButton:disabled {
                background-color: #b0b0b0;
            }
        """)
        self.send_button.clicked.connect(self._on_send_clicked)
        button_layout.addWidget(self.send_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #ff2222;
            }
        """)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setVisible(False)
        button_layout.addWidget(self.cancel_button)

        input_layout.addLayout(button_layout)
        layout.addLayout(input_layout)

    def eventFilter(self, obj, event):
        """Handle Enter key in input field."""
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key_Return and not key_event.modifiers() & Qt.ShiftModifier:
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    def _on_send_clicked(self):
        """Handle send button click."""
        if self._is_streaming:
            return  # Ignore if already streaming

        message = self.input_field.toPlainText().strip()
        if not message:
            return

        self.input_field.clear()
        self._add_user_message(message)
        self._set_streaming_state(True)
        self.message_sent.emit(message)

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()
        self._set_streaming_state(False)
        self.add_assistant_message("*Cancelled*")

    def _on_provider_changed(self, text: str):
        """Handle provider selection change."""
        # Map display name to provider key
        provider_map = {
            "Local (Ollama)": "local",
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Gemini": "gemini",
            "OpenRouter": "openrouter",
        }
        provider = provider_map.get(text, "local")
        self.provider_changed.emit(provider)

    def _set_streaming_state(self, is_streaming: bool):
        """Update UI state for streaming.

        Args:
            is_streaming: True if currently streaming a response
        """
        self._is_streaming = is_streaming
        self._response_finished_handled = False  # Reset guard

        self.input_field.setEnabled(not is_streaming)
        self.send_button.setVisible(not is_streaming)
        self.cancel_button.setVisible(is_streaming)

    def _add_user_message(self, text: str):
        """Add a user message bubble.

        Args:
            text: Message text
        """
        bubble = MessageBubble(text, is_user=True)
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def add_assistant_message(self, text: str):
        """Add a complete assistant message.

        Args:
            text: Message text (supports Markdown)
        """
        bubble = MessageBubble(text, is_user=False)
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def start_streaming_response(self) -> StreamingBubble:
        """Start a streaming response bubble.

        Returns:
            StreamingBubble that can receive text chunks
        """
        bubble = StreamingBubble()
        self._current_bubble = bubble
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()
        return bubble

    def add_tool_indicator(self, tool_name: str, status: str = "running") -> ToolIndicator:
        """Add a tool execution indicator.

        Args:
            tool_name: Name of the tool being executed
            status: Initial status

        Returns:
            ToolIndicator that can be updated
        """
        indicator = ToolIndicator(tool_name, status)
        self.messages_layout.addWidget(indicator)
        self._scroll_to_bottom()
        return indicator

    def _scroll_to_bottom(self):
        """Scroll the message area to the bottom."""
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    @Slot(str)
    def on_stream_chunk(self, chunk: str):
        """Handle streaming chunk - with guard.

        Args:
            chunk: Text chunk from streaming response
        """
        if self._current_bubble and not self._response_finished_handled:
            self._current_bubble.append_text(chunk)
            self._scroll_to_bottom()

    @Slot()
    def on_stream_complete(self):
        """Handle stream completion - with guard."""
        if self._response_finished_handled:
            return
        self._response_finished_handled = True

        if self._current_bubble:
            self._current_bubble.finish()
            self._current_bubble = None

        self._set_streaming_state(False)

    def on_stream_error(self, error: str):
        """Handle stream error.

        Args:
            error: Error message
        """
        self._response_finished_handled = True

        if self._current_bubble:
            self._current_bubble.finish()
            self._current_bubble = None

        self.add_assistant_message(f"**Error:** {error}")
        self._set_streaming_state(False)

    def clear_messages(self):
        """Clear all messages from the chat."""
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_provider(self, provider: str):
        """Set the selected provider.

        Args:
            provider: Provider key (local, openai, anthropic, gemini, openrouter)
        """
        # Map provider key to display name
        display_map = {
            "local": "Local (Ollama)",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Gemini",
            "openrouter": "OpenRouter",
        }
        display_name = display_map.get(provider, "Local (Ollama)")
        index = self.provider_combo.findText(display_name)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)

    def get_provider(self) -> str:
        """Get the currently selected provider key.

        Returns:
            Provider key (local, openai, anthropic, gemini, openrouter)
        """
        text = self.provider_combo.currentText()
        provider_map = {
            "Local (Ollama)": "local",
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Gemini": "gemini",
            "OpenRouter": "openrouter",
        }
        return provider_map.get(text, "local")
