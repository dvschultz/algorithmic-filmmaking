"""Chat panel widget for agent interaction.

Provides:
- Message history display with user/assistant bubbles
- Input area with send/cancel buttons
- Provider selection dropdown
- Streaming state management
"""

from PySide6.QtCore import Qt, Signal, Slot, QTimer, QEvent
from PySide6.QtGui import QKeyEvent, QStandardItem, QPalette, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.chat_widgets import MessageBubble, StreamingBubble, ToolIndicator, ExamplePromptsWidget
from core.settings import (
    get_anthropic_api_key,
    get_openai_api_key,
    get_gemini_api_key,
    get_openrouter_api_key,
)


class DisabledItemDelegate(QStyledItemDelegate):
    """Custom delegate that visually grays out disabled combo box items."""

    def paint(self, painter, option, index):
        # Check if item is disabled
        if not (index.flags() & Qt.ItemIsEnabled):
            # Gray out the text for disabled items
            option = QStyleOptionViewItem(option)
            option.palette.setColor(QPalette.Text, QColor(160, 160, 160))
            option.palette.setColor(QPalette.HighlightedText, QColor(160, 160, 160))
        super().paint(painter, option, index)


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
        self._example_prompts_visible = True
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

        # Use custom delegate to gray out disabled items
        self.provider_combo.setItemDelegate(DisabledItemDelegate(self.provider_combo))

        header.addWidget(self.provider_combo)

        # Disable providers without API keys (deferred to allow full init)
        QTimer.singleShot(0, self.update_provider_availability)

        layout.addLayout(header)

        # Message history scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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

        # Example prompts (shown when chat is empty)
        self.example_prompts = ExamplePromptsWidget()
        self.example_prompts.prompt_clicked.connect(self._on_example_prompt_clicked)
        self.messages_layout.addWidget(self.example_prompts)

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

    def _on_example_prompt_clicked(self, prompt_text: str):
        """Handle example prompt click - fill input field."""
        self.input_field.setText(prompt_text)
        self.input_field.setFocus()
        # Move cursor to end
        cursor = self.input_field.textCursor()
        cursor.movePosition(cursor.End)
        self.input_field.setTextCursor(cursor)

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

        # Disable example prompts during streaming
        if self._example_prompts_visible:
            self.example_prompts.setEnabled(not is_streaming)

    def _add_user_message(self, text: str):
        """Add a user message bubble.

        Args:
            text: Message text
        """
        # Hide example prompts after first message
        if self._example_prompts_visible:
            self.example_prompts.hide()
            self._example_prompts_visible = False

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
        # Force layout update to recalculate sizes
        self.messages_widget.updateGeometry()
        self.messages_widget.adjustSize()
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    @Slot()
    def on_clear_bubble(self):
        """Clear the current streaming bubble content."""
        if self._current_bubble:
            self._current_bubble.clear_text()

    @Slot(str)
    def on_stream_chunk(self, chunk: str):
        """Handle streaming chunk - with guard.

        Args:
            chunk: Text chunk from streaming response
        """
        if self._current_bubble and not self._response_finished_handled:
            self._current_bubble.append_text(chunk)
            self._scroll_to_bottom()

    @Slot(str)
    def on_tool_result_formatted(self, formatted_text: str):
        """Display a formatted tool result.

        Args:
            formatted_text: Human-readable tool result
        """
        # Clear any junk in the current bubble and show the formatted result
        if self._current_bubble:
            self._current_bubble.clear_text()
            self._current_bubble.append_text(formatted_text)
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
        # Remove all widgets except the example prompts
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget and widget is not self.example_prompts:
                widget.deleteLater()

        # Restore example prompts
        self.messages_layout.addWidget(self.example_prompts)
        self.example_prompts.show()
        self.example_prompts.reset_guard()
        self._example_prompts_visible = True

    def update_project_state(
        self,
        has_sources: bool = False,
        clip_count: int = 0,
        has_analyzed: bool = False,
        sequence_length: int = 0
    ):
        """Update project state to show relevant example prompts.

        Args:
            has_sources: Whether project has video sources
            clip_count: Number of clips in project
            has_analyzed: Whether clips have been analyzed
            sequence_length: Number of clips in the sequence
        """
        self.example_prompts.update_project_state(
            has_sources=has_sources,
            clip_count=clip_count,
            has_analyzed=has_analyzed,
            sequence_length=sequence_length
        )

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

    def update_provider_availability(self):
        """Update provider dropdown to disable providers without API keys.

        Local (Ollama) is always enabled as it doesn't require an API key.
        Other providers are disabled if no API key is configured.
        """
        # Map combo index to API key getter (index 0 = Local, no key needed)
        api_key_checks = {
            1: get_openai_api_key,      # OpenAI
            2: get_anthropic_api_key,   # Anthropic
            3: get_gemini_api_key,      # Gemini
            4: get_openrouter_api_key,  # OpenRouter
        }

        model = self.provider_combo.model()
        current_index = self.provider_combo.currentIndex()
        current_disabled = False

        for index, get_key in api_key_checks.items():
            item = model.item(index)
            if item:
                has_key = bool(get_key())
                if has_key:
                    # Enable the item
                    item.setFlags(item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setToolTip("")
                else:
                    # Disable the item
                    item.setFlags(item.flags() & ~(Qt.ItemIsEnabled | Qt.ItemIsSelectable))
                    item.setToolTip("API key not configured. Add it in Settings > API Keys.")

                    # Check if current selection is now disabled
                    if index == current_index:
                        current_disabled = True

        # If current selection is disabled, switch to Local
        if current_disabled:
            self.provider_combo.setCurrentIndex(0)  # Local (Ollama)
