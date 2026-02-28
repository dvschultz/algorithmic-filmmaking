"""Chat panel widget for agent interaction.

Provides:
- Message history display with user/assistant bubbles
- Input area with send/cancel buttons
- Provider selection dropdown
- Streaming state management
"""

from PySide6.QtCore import Qt, Signal, Slot, QTimer, QEvent
from PySide6.QtGui import QKeyEvent, QStandardItem, QPalette, QColor, QTextCursor
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

from ui.chat_widgets import (
    MessageBubble, StreamingBubble, ToolIndicator, ThinkingIndicator,
    ExamplePromptsWidget, PlanWidget
)
from ui.theme import theme, TypeScale, Spacing, Radii
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
    clear_requested = Signal()  # Clear chat history requested
    export_requested = Signal()  # Export chat history requested

    # Plan-related signals
    plan_confirmed = Signal(object)  # Emits Plan object when confirmed
    plan_cancelled = Signal()  # Plan cancelled before execution
    plan_retry_requested = Signal(int)  # Retry step at index
    plan_stop_requested = Signal()  # Stop execution after failure

    def __init__(self, parent=None):
        """Create the chat panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._is_streaming = False
        self._response_finished_handled = False
        self._current_bubble = None
        self._thinking_indicator = None
        self._example_prompts_visible = True
        self._current_plan_widget = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with provider selector
        header = QHBoxLayout()
        header_label = QLabel("Agent Chat")
        header_label.setStyleSheet(f"font-weight: bold; font-size: {TypeScale.MD}px;")
        header.addWidget(header_label)
        header.addStretch()

        # Header button style (shared)
        t = theme()
        header_button_style = f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {t.border_secondary};
                border-radius: {Radii.SM}px;
                padding: {Spacing.XS}px {Spacing.SM}px;
                color: {t.text_secondary};
                font-size: {TypeScale.SM}px;
            }}
            QPushButton:hover {{
                background-color: {t.background_tertiary};
                border-color: {t.text_muted};
            }}
            QPushButton:pressed {{
                background-color: {t.background_elevated};
            }}
            QPushButton:disabled {{
                color: {t.text_muted};
                border-color: {t.border_secondary};
            }}
        """

        # Export chat button
        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("Export chat history to file")
        self.export_button.setStyleSheet(header_button_style)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)  # Disabled until messages exist
        header.addWidget(self.export_button)

        # Clear chat button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear chat history (helps avoid rate limits)")
        self.clear_button.setStyleSheet(header_button_style)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        header.addWidget(self.clear_button)

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
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {t.border_secondary};
                border-radius: {Radii.LG}px;
                background-color: {t.background_primary};
            }}
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
        self.input_field.setPlaceholderText("Try 'detect scenes', 'analyze colors', 'build a sequence'...")
        self.input_field.setMaximumHeight(80)
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {t.border_secondary};
                border-radius: {Radii.LG}px;
                padding: {Spacing.SM}px;
                background-color: {t.background_primary};
                color: {t.text_primary};
            }}
        """)
        # Handle Enter key to send (Shift+Enter for newline)
        self.input_field.installEventFilter(self)
        input_layout.addWidget(self.input_field, 1)

        # Button container
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.accent_blue};
                color: {t.text_inverted};
                border: none;
                border-radius: {Radii.SM}px;
                padding: {Spacing.SM}px {Spacing.LG}px;
            }}
            QPushButton:hover {{
                background-color: {t.accent_blue_hover};
            }}
            QPushButton:disabled {{
                background-color: {t.text_muted};
            }}
        """)
        self.send_button.clicked.connect(self._on_send_clicked)
        button_layout.addWidget(self.send_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.accent_red};
                color: {t.text_inverted};
                border: none;
                border-radius: {Radii.SM}px;
                padding: {Spacing.SM}px {Spacing.LG}px;
            }}
            QPushButton:hover {{
                background-color: {t.accent_red};
            }}
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

        # Intercept slash commands before sending to LLM
        if message.startswith("/"):
            self._add_user_message(message)
            if self._handle_slash_command(message):
                return
            # Slash command not handled - pass through to LLM
            self._set_streaming_state(True)
            self.message_sent.emit(message)
            return

        self._add_user_message(message)
        self._set_streaming_state(True)
        self.message_sent.emit(message)

    def _handle_slash_command(self, message: str) -> bool:
        """Handle slash commands locally without sending to LLM.

        Args:
            message: The slash command message

        Returns:
            True if command was handled, False to pass through to LLM
        """
        command = message.split()[0].lower()

        if command == "/help":
            help_text = (
                "**Agent Capabilities**\n\n"
                "**Import & Sources**\n"
                "- `search_youtube` - Search for videos on YouTube\n"
                "- `download_video` - Download a video from URL\n"
                "- `list_sources` - List all source videos in the project\n"
                "- `remove_source` - Remove a source video\n"
                "- `update_source` - Update source metadata\n\n"
                "**Scene Detection**\n"
                "- `detect_scenes_live` - Detect scenes in a single video\n"
                "- `detect_all_unanalyzed` - Detect scenes in all new videos\n"
                "- `check_detection_status` - Check detection progress\n\n"
                "**Analysis**\n"
                "- `start_clip_analysis` - Run analysis operations on clips\n"
                "- `analyze_all_live` - Run full analysis pipeline\n"
                "- Colors, shot types, transcription, description, classification, objects\n\n"
                "**Sequence & Timeline**\n"
                "- `add_to_sequence` - Add clips to the timeline\n"
                "- `remove_from_sequence` - Remove clips from timeline\n"
                "- `reorder_sequence` - Reorder timeline clips\n"
                "- `update_sequence_clip` - Trim or reposition a clip\n"
                "- `clear_sequence` - Clear the entire timeline\n\n"
                "**Export**\n"
                "- `export_sequence` - Export timeline as MP4 video\n"
                "- `export_edl` - Export as EDL for external editors\n"
                "- `export_clips` - Export individual clips\n\n"
                "**Navigation & Filters**\n"
                "- `navigate_to_tab` - Switch between tabs\n"
                "- `apply_filters` - Filter clips by duration, shot type, etc.\n"
                "- `set_clip_sort_order` - Sort clips by timeline, color, or duration\n\n"
                "**Planning**\n"
                "- Describe a complex workflow and the agent will create a step-by-step plan\n"
                "- Plans can be confirmed, edited, or cancelled before execution\n\n"
                "Type a natural language request to get started!"
            )
            self.add_assistant_message(help_text)
            return True

        # Unknown slash command - pass through to LLM
        return False

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()
        self._set_streaming_state(False)
        self.add_assistant_message("*Cancelled*")

    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_messages()
        self.clear_requested.emit()

    def _on_export_clicked(self):
        """Handle export button click."""
        self.export_requested.emit()

    def _on_example_prompt_clicked(self, prompt_text: str):
        """Handle example prompt click - fill input field."""
        self.input_field.setText(prompt_text)
        self.input_field.setFocus()
        # Move cursor to end
        cursor = self.input_field.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
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

        # Disable export during streaming
        if is_streaming:
            self.export_button.setEnabled(False)
            self.export_button.setToolTip("Wait for response to complete")

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

        # Enable export button now that we have messages
        self._update_export_button_state(has_messages=True)

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
        # Show thinking indicator while waiting for response
        self._show_thinking_indicator()

        bubble = StreamingBubble()
        self._current_bubble = bubble
        # Don't add bubble to layout yet - it will replace thinking indicator when text arrives
        return bubble

    def _show_thinking_indicator(self):
        """Show the thinking indicator."""
        if self._thinking_indicator is None:
            self._thinking_indicator = ThinkingIndicator()
            self.messages_layout.addWidget(self._thinking_indicator)
            self._scroll_to_bottom()

    def _hide_thinking_indicator(self):
        """Hide and remove the thinking indicator."""
        if self._thinking_indicator is not None:
            self._thinking_indicator.stop()
            self._thinking_indicator.hide()
            self.messages_layout.removeWidget(self._thinking_indicator)
            self._thinking_indicator.deleteLater()
            self._thinking_indicator = None

    def add_tool_indicator(self, tool_name: str, status: str = "running") -> ToolIndicator:
        """Add a tool execution indicator.

        Args:
            tool_name: Name of the tool being executed
            status: Initial status

        Returns:
            ToolIndicator that can be updated
        """
        # Hide thinking indicator - tool execution is more specific feedback
        self._hide_thinking_indicator()

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
            # First chunk - hide thinking indicator and show the bubble
            if self._thinking_indicator is not None:
                self._hide_thinking_indicator()
                self.messages_layout.addWidget(self._current_bubble)

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
            # Ensure bubble is in layout (may not be if only tools have run)
            if self._current_bubble.parent() is None:
                self.messages_layout.addWidget(self._current_bubble)
            self._current_bubble.clear_text()
            self._current_bubble.append_text(formatted_text)
            self._scroll_to_bottom()

        # Show thinking indicator - LLM is processing after tool result
        self._show_thinking_indicator()

    @Slot()
    def on_stream_complete(self):
        """Handle stream completion - with guard."""
        if self._response_finished_handled:
            return
        self._response_finished_handled = True

        # Hide thinking indicator
        self._hide_thinking_indicator()

        if self._current_bubble:
            # Ensure bubble is in layout before finishing
            if self._current_bubble.parent() is None:
                self.messages_layout.addWidget(self._current_bubble)
            self._current_bubble.finish()
            self._current_bubble = None

        self._set_streaming_state(False)

        # Re-enable export button after streaming completes
        self._update_export_button_state(has_messages=True)

    def on_stream_error(self, error: str):
        """Handle stream error.

        Args:
            error: Error message
        """
        self._response_finished_handled = True

        # Hide thinking indicator
        self._hide_thinking_indicator()

        if self._current_bubble:
            self._current_bubble.finish()
            self._current_bubble = None

        self.add_assistant_message(f"**Error:** {error}")
        self._set_streaming_state(False)

    def clear_messages(self):
        """Clear all messages from the chat."""
        # Reset streaming state if active
        if self._is_streaming:
            self._set_streaming_state(False)

        # Clean up thinking indicator if present
        self._hide_thinking_indicator()

        # Clean up plan widget if present
        self._current_plan_widget = None

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

        # Disable export button - no messages to export
        self._update_export_button_state(has_messages=False)

    def _update_export_button_state(self, has_messages: bool):
        """Update export button enabled state.

        Args:
            has_messages: Whether there are messages to export
        """
        if self._is_streaming:
            # Don't enable during streaming
            return

        self.export_button.setEnabled(has_messages)
        if has_messages:
            self.export_button.setToolTip("Export chat history to file")
        else:
            self.export_button.setToolTip("No messages to export")

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

    # =========================================================================
    # Plan Widget Methods
    # =========================================================================

    def show_plan_widget(self, plan):
        """Display a plan widget inline in the chat.

        Args:
            plan: Plan object to display
        """
        # Hide example prompts if still visible
        if self._example_prompts_visible:
            self.example_prompts.hide()
            self._example_prompts_visible = False

        # Remove any existing plan widget
        if self._current_plan_widget is not None:
            self._remove_plan_widget()

        # Create and add new plan widget
        self._current_plan_widget = PlanWidget(plan)
        self._current_plan_widget.confirmed.connect(self._on_plan_confirmed)
        self._current_plan_widget.cancelled.connect(self._on_plan_cancelled)
        self._current_plan_widget.retry_requested.connect(self._on_plan_retry)
        self._current_plan_widget.stop_requested.connect(self._on_plan_stop)

        self.messages_layout.addWidget(self._current_plan_widget)
        self._scroll_to_bottom()

    def _remove_plan_widget(self):
        """Remove the current plan widget."""
        if self._current_plan_widget is not None:
            self.messages_layout.removeWidget(self._current_plan_widget)
            self._current_plan_widget.deleteLater()
            self._current_plan_widget = None

    def _on_plan_confirmed(self, plan):
        """Handle plan confirmation from widget."""
        self.plan_confirmed.emit(plan)

    def _on_plan_cancelled(self):
        """Handle plan cancellation from widget."""
        self._remove_plan_widget()
        self.plan_cancelled.emit()

    def _on_plan_retry(self, step_index: int):
        """Handle retry request from widget."""
        self.plan_retry_requested.emit(step_index)

    def _on_plan_stop(self):
        """Handle stop request from widget."""
        self.plan_stop_requested.emit()

    def update_plan_step_status(self, step_index: int, status: str, error: str = None):
        """Update a plan step's status.

        Args:
            step_index: Index of the step to update
            status: New status (pending, running, completed, failed)
            error: Error message if status is 'failed'
        """
        if self._current_plan_widget is not None:
            self._current_plan_widget.update_step_status(step_index, status, error)
            self._scroll_to_bottom()

    def set_plan_executing(self, executing: bool):
        """Set whether the plan is currently executing.

        Args:
            executing: True if plan is being executed
        """
        if self._current_plan_widget is not None:
            self._current_plan_widget.set_executing(executing)

    def mark_plan_completed(self):
        """Mark the current plan as fully completed."""
        if self._current_plan_widget is not None:
            self._current_plan_widget.mark_completed()

    def get_current_plan(self):
        """Get the current plan object if a plan widget is displayed.

        Returns:
            Plan object or None
        """
        if self._current_plan_widget is not None:
            return self._current_plan_widget.plan
        return None

    def has_pending_plan(self) -> bool:
        """Check if there's a plan widget waiting for confirmation.

        Returns:
            True if a plan is displayed and in draft status
        """
        if self._current_plan_widget is not None:
            plan = self._current_plan_widget.plan
            return plan.status == "draft"
        return False
