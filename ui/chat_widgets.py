"""Chat UI widgets for message display.

Provides:
- MessageBubble: Static message display (user or assistant)
- StreamingBubble: Accumulates streaming text from LLM
- ToolIndicator: Shows tool execution status
- ExamplePromptsWidget: Clickable example prompts with random cycling
- PlanStepWidget: Individual step in an editable plan
- PlanWidget: Inline editable plan with steps
"""

import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QTextBrowser,
    QVBoxLayout, QWidget
)

from ui.theme import theme, TypeScale, Spacing, Radii


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
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Use QTextBrowser for assistant messages (better sizing for long content)
        # Use QLabel for user messages (simpler, shorter content)
        if self._is_user:
            self.label = QLabel(text)
            self.label.setWordWrap(True)
            self.label.setTextFormat(Qt.MarkdownText)
            self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addWidget(self.label)
        else:
            self.text_browser = QTextBrowser()
            self.text_browser.setMarkdown(text)
            self.text_browser.setOpenExternalLinks(True)
            self.text_browser.setFrameShape(QFrame.NoFrame)
            # Make background transparent to show bubble background
            self._apply_text_browser_style()
            # Let the browser expand to fit content
            self.text_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            layout.addWidget(self.text_browser)

            # Schedule size adjustment after layout is complete
            QTimer.singleShot(0, self._adjust_text_browser_height)

        # Allow bubble to expand for content
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _apply_style(self):
        """Apply theme-based styling to the bubble."""
        t = theme()
        if self._is_user:
            self.setStyleSheet(f"""
                QFrame#userBubble {{
                    background-color: {t.chat_user_bubble};
                    border-radius: 12px;
                    margin-left: 40px;
                }}
                QFrame#userBubble QLabel {{
                    color: {t.chat_user_text};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame#assistantBubble {{
                    background-color: {t.chat_assistant_bubble};
                    border-radius: 12px;
                    margin-right: 40px;
                }}
                QFrame#assistantBubble QLabel {{
                    color: {t.chat_assistant_text};
                }}
            """)

    def _apply_text_browser_style(self):
        """Apply theme-based styling to text browser."""
        if hasattr(self, 'text_browser') and self.text_browser:
            t = theme()
            self.text_browser.setStyleSheet(f"""
                QTextBrowser {{
                    background: transparent;
                    border: none;
                    color: {t.chat_assistant_text};
                }}
            """)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._apply_style()
        if hasattr(self, 'text_browser'):
            self._apply_text_browser_style()

    def _adjust_text_browser_height(self):
        """Adjust text browser height to fit content."""
        if hasattr(self, 'text_browser'):
            doc = self.text_browser.document()
            doc.setTextWidth(self.text_browser.viewport().width())
            height = int(doc.size().height() + 10)
            self.text_browser.setMinimumHeight(height)
            self.text_browser.setMaximumHeight(height)


class StreamingBubble(QFrame):
    """Message bubble that accumulates streaming text."""

    def __init__(self, parent=None):
        """Create a streaming bubble for assistant responses.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._text = ""
        self._chunk_buffer = ""
        self._is_finished = False

        # Timer for batched UI updates (every 100ms)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(100)
        self._update_timer.timeout.connect(self._flush_buffer)

        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("assistantBubble")
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel("...")
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.PlainText)  # Use plain text during streaming
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Allow label to expand vertically for long content
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.label)

        # Allow bubble to expand for content
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _apply_style(self):
        """Apply theme-based styling."""
        t = theme()
        self.setStyleSheet(f"""
            QFrame#assistantBubble {{
                background-color: {t.chat_assistant_bubble};
                border-radius: 12px;
                margin-right: 40px;
            }}
            QFrame#assistantBubble QLabel {{
                color: {t.chat_assistant_text};
            }}
        """)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._apply_style()
        if hasattr(self, 'text_browser') and self.text_browser:
            t = theme()
            self.text_browser.setStyleSheet(f"""
                QTextBrowser {{
                    background: transparent;
                    border: none;
                    color: {t.chat_assistant_text};
                }}
            """)

    def append_text(self, chunk: str):
        """Append a chunk of text to the bubble.

        Chunks are buffered and flushed to the UI periodically
        for better performance with rapid streaming.

        Args:
            chunk: Text chunk from streaming response
        """
        if self._is_finished:
            return

        self._chunk_buffer += chunk

        # Start timer if not already running
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _flush_buffer(self):
        """Flush buffered chunks to the label."""
        if not self._chunk_buffer:
            self._update_timer.stop()
            return

        self._text += self._chunk_buffer
        self._chunk_buffer = ""
        self.label.setText(self._text)

    def finish(self):
        """Mark the bubble as finished and render final Markdown."""
        if self._is_finished:
            return

        # Stop timer and flush any remaining buffer
        self._update_timer.stop()
        if self._chunk_buffer:
            self._text += self._chunk_buffer
            self._chunk_buffer = ""

        self._is_finished = True

        # Replace QLabel with QTextBrowser for better Markdown sizing
        self.label.hide()
        self.label.deleteLater()

        t = theme()
        self.text_browser = QTextBrowser()
        self.text_browser.setMarkdown(self._text)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setFrameShape(QFrame.NoFrame)
        self.text_browser.setStyleSheet(f"""
            QTextBrowser {{
                background: transparent;
                border: none;
                color: {t.chat_assistant_text};
            }}
        """)
        self.text_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout().addWidget(self.text_browser)

        # Schedule height adjustment
        QTimer.singleShot(0, self._adjust_text_browser_height)

    def _adjust_text_browser_height(self):
        """Adjust text browser height to fit content."""
        if hasattr(self, 'text_browser') and self.text_browser:
            doc = self.text_browser.document()
            doc.setTextWidth(self.text_browser.viewport().width())
            height = int(doc.size().height() + 10)
            self.text_browser.setMinimumHeight(height)
            self.text_browser.setMaximumHeight(height)

    @property
    def text(self) -> str:
        """Get the accumulated text."""
        return self._text

    def clear_text(self):
        """Clear the accumulated text."""
        self._update_timer.stop()
        self._text = ""
        self._chunk_buffer = ""
        self._is_finished = False

        # If we have a text_browser (from finish()), remove it and recreate label
        if hasattr(self, 'text_browser') and self.text_browser:
            self.text_browser.hide()
            self.text_browser.deleteLater()
            self.text_browser = None

            self.label = QLabel("...")
            self.label.setWordWrap(True)
            self.label.setTextFormat(Qt.PlainText)
            self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.layout().addWidget(self.label)
        else:
            self.label.setText("...")
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
        self._apply_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Status icon
        self.status_icon = QLabel(self._get_status_icon())
        self.status_icon.setFixedWidth(20)
        layout.addWidget(self.status_icon)

        # Tool name and status
        self.name_label = QLabel(self._get_status_text())
        self.name_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        layout.addWidget(self.name_label)

        layout.addStretch()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _apply_style(self):
        """Apply theme-based styling."""
        t = theme()
        self.setStyleSheet(f"""
            QFrame#toolIndicator {{
                background-color: {t.background_tertiary};
                border-radius: 8px;
                border: 1px solid {t.border_secondary};
                padding: 4px;
            }}
        """)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._apply_style()
        self.name_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")

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
        t = theme()
        if status == "complete":
            self.status_icon.setStyleSheet(f"color: {t.accent_green};")
        elif status == "error":
            self.status_icon.setStyleSheet(f"color: {t.accent_red};")

    def set_complete(self, success: bool = True):
        """Mark the tool as complete.

        Args:
            success: True if successful, False if failed
        """
        self.set_status("complete" if success else "error")


class ThinkingIndicator(QFrame):
    """Shows an animated 'thinking' indicator while the LLM is processing."""

    def __init__(self, parent=None):
        """Create a thinking indicator.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._dot_count = 0
        self._setup_ui()

        # Animation timer for pulsing dots
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(400)
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start()

    def _setup_ui(self):
        self.setObjectName("thinkingIndicator")
        self._apply_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Thinking text with animated dots
        self.label = QLabel("Thinking")
        self.label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.BASE}px; font-style: italic;")
        layout.addWidget(self.label)
        layout.addStretch()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _apply_style(self):
        """Apply theme-based styling."""
        t = theme()
        self.setStyleSheet(f"""
            QFrame#thinkingIndicator {{
                background-color: {t.chat_assistant_bubble};
                border-radius: 12px;
                margin-right: 40px;
            }}
        """)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._apply_style()
        self.label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.BASE}px; font-style: italic;")

    def _animate(self):
        """Animate the thinking dots."""
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self.label.setText(f"Thinking{dots}")

    def stop(self):
        """Stop the animation."""
        self._animation_timer.stop()

    def hideEvent(self, event):
        """Stop animation when hidden."""
        self._animation_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        """Start animation when shown."""
        self._animation_timer.start()
        super().showEvent(event)


class ExamplePromptsWidget(QWidget):
    """Displays clickable example prompts when chat is empty.

    Shows a random selection of prompts from a pool, cycling through
    new prompts each time the widget is reset. Prompts are filtered
    based on current project state to show relevant suggestions.
    """

    prompt_clicked = Signal(str)  # Emits the prompt text when clicked

    # How many prompts to show at once
    NUM_DISPLAY = 4

    # Project state requirements for prompt categories
    # "empty" = no sources, "has_sources" = sources but no clips,
    # "has_clips" = clips exist, "analyzed" = clips have analysis,
    # "has_sequence" = sequence has clips, "always" = show anytime
    PROMPTS_BY_STATE = {
        # === Empty Project - Import/Search ===
        "empty": [
            "Search YouTube for 'nature documentary b-roll'",
            "Find videos about 'cinematic transitions'",
            "Search for 'drone footage mountains'",
            "Look up 'slow motion water' on YouTube",
            "Find 'urban timelapse' videos",
            "Download this YouTube video: [paste URL]",
        ],

        # === Has Sources - Need Scene Detection ===
        "has_sources": [
            "Detect scenes in my video",
            "Split this video into clips",
            "Run scene detection with high sensitivity",
            "Find all the cuts in my footage",
            "Show me the project summary",
            "List all my video sources",
        ],

        # === Has Clips - Can Analyze or Filter ===
        "has_clips": [
            "Show me all clips in this project",
            "How many clips do I have?",
            "What's the total duration of my project?",
            "Find clips longer than 10 seconds",
            "Show me all short clips under 3 seconds",
            "Find clips between 5 and 15 seconds",
            "Show me the longest clips in my project",
            "Add the first 10 clips to the timeline",
            # Analysis prompts
            "Analyze colors in my project",
            "What are the dominant colors in my clips?",
            "Run color analysis on this project",
            "Classify the shot types in my project",
            "Run shot analysis on my clips",
            "Transcribe the speech in my clips",
            "Run speech recognition on my video",
        ],

        # === Analyzed Clips - Full Filtering ===
        "analyzed": [
            "Find all the close-up shots",
            "Show me all wide shots",
            "Find medium shots in my project",
            "List all establishing shots",
            "Show me the extreme close-ups",
            "Find clips with speech",
            "Show me clips without any dialogue",
            "Find all talking head shots",
            "Find close-up shots with speech",
            "Show me wide shots longer than 5 seconds",
            "Find medium shots without dialogue",
            "Show me short close-ups under 3 seconds",
            "Add all wide shots to the sequence",
            "Put the close-ups in the timeline",
            "Build a sequence from all speech clips",
            "Find close-up shots and add them to the timeline",
            "Find speech clips and export them",
        ],

        # === Has Sequence - Export/Modify ===
        "has_sequence": [
            "Export my sequence clips",
            "Show me what's in the sequence",
            "Clear the sequence and start over",
        ],

        # === Always Available ===
        "always": [
            "Show me the project summary",
            "Export all clips to my desktop",
            "Search YouTube for b-roll and download the best result",
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._click_handled = False
        # Project state tracking
        self._has_sources = False
        self._clip_count = 0
        self._has_analyzed = False
        self._sequence_length = 0
        self._selected_prompts = self._select_random_prompts()
        self._prompts_container = None
        self._setup_ui()

    def _get_available_prompts(self) -> list[str]:
        """Get prompts relevant to current project state."""
        prompts = list(self.PROMPTS_BY_STATE["always"])  # Always include these

        if not self._has_sources:
            # Empty project - show import/search prompts
            prompts.extend(self.PROMPTS_BY_STATE["empty"])
        elif self._clip_count == 0:
            # Has sources but no clips - need scene detection
            prompts.extend(self.PROMPTS_BY_STATE["has_sources"])
        else:
            # Has clips
            prompts.extend(self.PROMPTS_BY_STATE["has_clips"])

            if self._has_analyzed:
                # Clips are analyzed - show filtering prompts
                prompts.extend(self.PROMPTS_BY_STATE["analyzed"])

            if self._sequence_length > 0:
                # Has sequence - show sequence-related prompts
                prompts.extend(self.PROMPTS_BY_STATE["has_sequence"])

        return prompts

    def _select_random_prompts(self) -> list[str]:
        """Select random prompts from available pool based on project state."""
        available = self._get_available_prompts()
        return random.sample(available, min(self.NUM_DISPLAY, len(available)))

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Add stretch to center vertically
        layout.addStretch(1)

        # Header text
        self._header = QLabel("Try asking...")
        self._header.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.MD}px; font-weight: 500;")
        self._header.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._header)

        # Prompts container
        self._prompts_container = QWidget()
        self._rebuild_prompt_buttons()
        layout.addWidget(self._prompts_container)

        # Add stretch to center vertically
        layout.addStretch(1)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._header.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.MD}px; font-weight: 500;")
        self._rebuild_prompt_buttons()

    def _rebuild_prompt_buttons(self):
        """Rebuild the prompt buttons with current selection."""
        # Clear existing layout if present
        if self._prompts_container.layout():
            old_layout = self._prompts_container.layout()
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QWidget().setLayout(old_layout)  # Orphan the old layout

        prompts_layout = QVBoxLayout(self._prompts_container)
        prompts_layout.setContentsMargins(0, 0, 0, 0)
        prompts_layout.setSpacing(8)

        t = theme()
        for prompt_text in self._selected_prompts:
            btn = QPushButton(prompt_text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t.background_tertiary};
                    border: 1px solid {t.border_secondary};
                    border-radius: 12px;
                    padding: 10px 16px;
                    color: {t.text_primary};
                    text-align: left;
                    font-size: {TypeScale.BASE}px;
                }}
                QPushButton:hover {{
                    background-color: {t.background_elevated};
                }}
                QPushButton:pressed {{
                    background-color: {t.accent_blue};
                }}
                QPushButton:disabled {{
                    background-color: {t.background_secondary};
                    color: {t.text_muted};
                    border-color: {t.border_secondary};
                }}
            """)
            btn.clicked.connect(lambda checked, text=prompt_text: self._on_button_clicked(text))
            prompts_layout.addWidget(btn)

    def _on_button_clicked(self, prompt_text: str):
        """Handle prompt button click with debounce to prevent rapid double-clicks."""
        if self._click_handled:
            return
        self._click_handled = True
        self.prompt_clicked.emit(prompt_text)
        # Reset guard after short delay to allow subsequent clicks
        QTimer.singleShot(200, self._reset_click_guard)

    def _reset_click_guard(self):
        """Reset the click guard after debounce period."""
        self._click_handled = False

    def _refresh_prompts(self):
        """Select new random prompts and rebuild buttons."""
        self._selected_prompts = self._select_random_prompts()
        self._rebuild_prompt_buttons()

    def update_project_state(
        self,
        has_sources: bool = False,
        clip_count: int = 0,
        has_analyzed: bool = False,
        sequence_length: int = 0
    ):
        """Update project state to filter relevant prompts.

        Args:
            has_sources: Whether project has video sources
            clip_count: Number of clips in project
            has_analyzed: Whether clips have been analyzed (shot type, speech, etc.)
            sequence_length: Number of clips in the sequence
        """
        self._has_sources = has_sources
        self._clip_count = clip_count
        self._has_analyzed = has_analyzed
        self._sequence_length = sequence_length

    def reset_guard(self):
        """Reset the click guard and refresh prompts."""
        self._click_handled = False
        self._refresh_prompts()


class PlanStepWidget(QFrame):
    """Single step row in a plan widget with status, text, and controls."""

    # Signals
    text_edited = Signal(int, str)  # (step_index, new_text)
    delete_requested = Signal(int)  # step_index
    move_up_requested = Signal(int)  # step_index
    move_down_requested = Signal(int)  # step_index

    # Status icons
    STATUS_ICONS = {
        "pending": "\u2610",    # Empty box ☐
        "running": "\u23f3",    # Hourglass ⏳
        "completed": "\u2714",  # Check mark ✔
        "failed": "\u2718",     # X mark ✘
    }

    def __init__(self, index: int, text: str, status: str = "pending", parent=None):
        """Create a plan step widget.

        Args:
            index: Step index (0-based)
            text: Step description text
            status: Step status (pending, running, completed, failed)
            parent: Parent widget
        """
        super().__init__(parent)
        self._index = index
        self._text = text
        self._status = status
        self._is_editing = False
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("planStep")
        self._update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Status icon
        self.status_icon = QLabel(self.STATUS_ICONS.get(self._status, "\u2610"))
        self.status_icon.setFixedWidth(20)
        self.status_icon.setAlignment(Qt.AlignTop)
        self.status_icon.setStyleSheet(f"font-size: {TypeScale.MD}px;")
        layout.addWidget(self.status_icon)

        t = theme()

        # Step number
        self.number_label = QLabel(f"{self._index + 1}.")
        self.number_label.setFixedWidth(28)
        self.number_label.setAlignment(Qt.AlignTop)
        self.number_label.setStyleSheet(f"font-weight: bold; color: {t.text_secondary};")
        layout.addWidget(self.number_label)

        # Step text (label for display, line edit for editing)
        self.text_label = QLabel(self._text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.PlainText)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.text_label.setStyleSheet(f"color: {t.text_primary}; line-height: 1.3;")
        self.text_label.mouseDoubleClickEvent = self._on_double_click
        layout.addWidget(self.text_label, 1)

        # Edit field (hidden initially)
        self.text_edit = QLineEdit(self._text)
        self.text_edit.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {t.accent_blue};
                border-radius: 4px;
                padding: 4px;
                background: {t.background_primary};
                color: {t.text_primary};
            }}
        """)
        self.text_edit.returnPressed.connect(self._finish_editing)
        self.text_edit.hide()
        layout.addWidget(self.text_edit, 1)

        # Move up button - increased size for accessibility (44x32 minimum touch target)
        self.up_btn = QPushButton("\u25b2")  # ▲
        self.up_btn.setFixedSize(32, 32)
        self.up_btn.setToolTip("Move up")
        self.up_btn.setAccessibleName("Move step up")
        self.up_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.XS}px;
            }}
            QPushButton:hover {{ color: {t.accent_blue}; }}
        """)
        self.up_btn.clicked.connect(lambda: self.move_up_requested.emit(self._index))
        layout.addWidget(self.up_btn)

        # Move down button
        self.down_btn = QPushButton("\u25bc")  # ▼
        self.down_btn.setFixedSize(32, 32)
        self.down_btn.setToolTip("Move down")
        self.down_btn.setAccessibleName("Move step down")
        self.down_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.XS}px;
            }}
            QPushButton:hover {{ color: {t.accent_blue}; }}
        """)
        self.down_btn.clicked.connect(lambda: self.move_down_requested.emit(self._index))
        layout.addWidget(self.down_btn)

        # Delete button
        self.delete_btn = QPushButton("\u2715")  # ✕
        self.delete_btn.setFixedSize(32, 32)
        self.delete_btn.setToolTip("Delete step")
        self.delete_btn.setAccessibleName("Delete step")
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.SM}px;
            }}
            QPushButton:hover {{ color: {t.accent_red}; }}
        """)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._index))
        layout.addWidget(self.delete_btn)

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._update_style()
        t = theme()
        self.number_label.setStyleSheet(f"font-weight: bold; color: {t.text_secondary};")
        self.text_label.setStyleSheet(f"color: {t.text_primary}; line-height: 1.3;")
        self.up_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.XS}px;
            }}
            QPushButton:hover {{ color: {t.accent_blue}; }}
        """)
        self.down_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.XS}px;
            }}
            QPushButton:hover {{ color: {t.accent_blue}; }}
        """)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: {TypeScale.SM}px;
            }}
            QPushButton:hover {{ color: {t.accent_red}; }}
        """)

    def _update_style(self):
        """Update widget style based on status."""
        t = theme()
        if self._status == "running":
            bg_color = t.plan_running_bg
            border_color = t.plan_running_border
        elif self._status == "completed":
            bg_color = t.plan_completed_bg
            border_color = t.plan_completed_border
        elif self._status == "failed":
            bg_color = t.plan_failed_bg
            border_color = t.plan_failed_border
        else:  # pending
            bg_color = t.plan_pending_bg
            border_color = t.plan_pending_border

        self.setStyleSheet(f"""
            QFrame#planStep {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 6px;
            }}
        """)

    def _on_double_click(self, event):
        """Handle double-click to start editing."""
        if self._status == "pending":  # Only allow editing pending steps
            self._start_editing()

    def _start_editing(self):
        """Enter edit mode."""
        if self._is_editing:
            return
        self._is_editing = True
        self.text_label.hide()
        self.text_edit.setText(self._text)
        self.text_edit.show()
        self.text_edit.setFocus()
        self.text_edit.selectAll()

    def _finish_editing(self):
        """Exit edit mode and save changes."""
        if not self._is_editing:
            return
        self._is_editing = False
        new_text = self.text_edit.text().strip()
        if new_text and new_text != self._text:
            self._text = new_text
            self.text_label.setText(new_text)
            self.text_edited.emit(self._index, new_text)
        self.text_edit.hide()
        self.text_label.show()

    def focusOutEvent(self, event):
        """Handle focus loss to finish editing."""
        if self._is_editing:
            self._finish_editing()
        super().focusOutEvent(event)

    def set_status(self, status: str):
        """Update step status.

        Args:
            status: New status (pending, running, completed, failed)
        """
        self._status = status
        self.status_icon.setText(self.STATUS_ICONS.get(status, "\u2610"))
        self._update_style()

        # Update icon color based on status
        t = theme()
        if status == "completed":
            self.status_icon.setStyleSheet(f"font-size: {TypeScale.MD}px; color: {t.plan_completed_border};")
        elif status == "failed":
            self.status_icon.setStyleSheet(f"font-size: {TypeScale.MD}px; color: {t.plan_failed_border};")
        elif status == "running":
            self.status_icon.setStyleSheet(f"font-size: {TypeScale.MD}px; color: {t.plan_running_border};")
        else:
            self.status_icon.setStyleSheet(f"font-size: {TypeScale.MD}px;")

        # Hide controls for non-pending steps (gives text more room)
        editable = status == "pending"
        self.up_btn.setVisible(editable)
        self.down_btn.setVisible(editable)
        self.delete_btn.setVisible(editable)

    def set_index(self, index: int):
        """Update step index (for reordering)."""
        self._index = index
        self.number_label.setText(f"{index + 1}.")

    @property
    def text(self) -> str:
        return self._text

    @property
    def status(self) -> str:
        return self._status


class PlanWidget(QFrame):
    """Editable plan widget shown inline in chat.

    Displays a list of steps that can be edited, reordered, and deleted
    before confirmation. During execution, shows status updates.
    """

    # Signals
    confirmed = Signal(object)  # Emits Plan object with current steps
    cancelled = Signal()
    retry_requested = Signal(int)  # step_index
    stop_requested = Signal()

    def __init__(self, plan, parent=None):
        """Create a plan widget.

        Args:
            plan: Plan object with steps
            parent: Parent widget
        """
        super().__init__(parent)
        self._plan = plan
        self._step_widgets: list[PlanStepWidget] = []
        self._is_executing = False
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("planWidget")
        t = theme()
        self.setStyleSheet(f"""
            QFrame#planWidget {{
                background-color: {t.background_primary};
                border: 2px solid {t.accent_blue};
                border-radius: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        header_icon = QLabel("\U0001f4cb")  # 📋 clipboard
        header_icon.setStyleSheet(f"font-size: {TypeScale.XL}px;")
        header_layout.addWidget(header_icon)

        self._header_text = QLabel(f"Plan: {self._plan.summary}")
        self._header_text.setStyleSheet(f"font-weight: bold; font-size: {TypeScale.MD}px; color: {t.text_primary};")
        self._header_text.setWordWrap(True)
        header_layout.addWidget(self._header_text, 1)

        layout.addLayout(header_layout)

        # Separator
        self._separator1 = QFrame()
        self._separator1.setFrameShape(QFrame.HLine)
        self._separator1.setStyleSheet(f"background-color: {t.border_secondary};")
        self._separator1.setFixedHeight(1)
        layout.addWidget(self._separator1)

        # Steps container
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(6)

        # Add step widgets
        for i, step in enumerate(self._plan.steps):
            self._add_step_widget(i, step.description, step.status)

        layout.addWidget(self.steps_container)

        # Separator before buttons
        self._separator2 = QFrame()
        self._separator2.setFrameShape(QFrame.HLine)
        self._separator2.setStyleSheet(f"background-color: {t.border_secondary};")
        self._separator2.setFixedHeight(1)
        layout.addWidget(self._separator2)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.background_tertiary};
                color: {t.text_secondary};
                border: 1px solid {t.border_secondary};
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {t.background_elevated};
            }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addSpacing(8)

        # Confirm button
        self.confirm_btn = QPushButton("\u2714 Confirm Plan")  # ✔
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.accent_blue};
                color: {t.text_inverted};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {t.accent_blue_hover};
            }}
            QPushButton:disabled {{
                background-color: {t.text_muted};
            }}
        """)
        self.confirm_btn.clicked.connect(self._on_confirm)
        button_layout.addWidget(self.confirm_btn)

        # Stop button (hidden initially, shown during execution on failure)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.accent_red};
                color: {t.text_inverted};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {t.accent_red};
            }}
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.hide()
        button_layout.addWidget(self.stop_btn)

        # Retry button (hidden initially, shown on step failure)
        self.retry_btn = QPushButton("Retry")
        self.retry_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.accent_orange};
                color: {t.text_inverted};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {t.accent_orange};
            }}
        """)
        self.retry_btn.clicked.connect(self._on_retry)
        self.retry_btn.hide()
        button_layout.addWidget(self.retry_btn)

        layout.addLayout(button_layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        """Handle theme changes."""
        t = theme()
        self.setStyleSheet(f"""
            QFrame#planWidget {{
                background-color: {t.background_primary};
                border: 2px solid {t.accent_blue};
                border-radius: 12px;
            }}
        """)
        self._header_text.setStyleSheet(f"font-weight: bold; font-size: {TypeScale.MD}px; color: {t.text_primary};")
        self._separator1.setStyleSheet(f"background-color: {t.border_secondary};")
        self._separator2.setStyleSheet(f"background-color: {t.border_secondary};")
        # Button styles would need to be refreshed too but they're less critical

    def _add_step_widget(self, index: int, text: str, status: str):
        """Add a step widget to the list."""
        step_widget = PlanStepWidget(index, text, status)
        step_widget.text_edited.connect(self._on_step_edited)
        step_widget.delete_requested.connect(self._on_step_delete)
        step_widget.move_up_requested.connect(self._on_step_move_up)
        step_widget.move_down_requested.connect(self._on_step_move_down)
        self._step_widgets.append(step_widget)
        self.steps_layout.addWidget(step_widget)

    def _reindex_steps(self):
        """Update step indices after reorder/delete."""
        for i, widget in enumerate(self._step_widgets):
            widget.set_index(i)

    def _on_step_edited(self, index: int, new_text: str):
        """Handle step text edit."""
        if 0 <= index < len(self._plan.steps):
            self._plan.steps[index].description = new_text

    def _on_step_delete(self, index: int):
        """Handle step deletion."""
        if len(self._step_widgets) <= 1:
            return  # Can't delete last step

        if 0 <= index < len(self._step_widgets):
            # Remove widget
            widget = self._step_widgets.pop(index)
            self.steps_layout.removeWidget(widget)
            widget.deleteLater()

            # Remove from plan
            if 0 <= index < len(self._plan.steps):
                self._plan.steps.pop(index)

            self._reindex_steps()

    def _on_step_move_up(self, index: int):
        """Handle step move up."""
        if index <= 0:
            return

        # Swap widgets
        self._step_widgets[index], self._step_widgets[index - 1] = \
            self._step_widgets[index - 1], self._step_widgets[index]

        # Swap in plan
        self._plan.steps[index], self._plan.steps[index - 1] = \
            self._plan.steps[index - 1], self._plan.steps[index]

        # Rebuild layout
        self._rebuild_steps_layout()

    def _on_step_move_down(self, index: int):
        """Handle step move down."""
        if index >= len(self._step_widgets) - 1:
            return

        # Swap widgets
        self._step_widgets[index], self._step_widgets[index + 1] = \
            self._step_widgets[index + 1], self._step_widgets[index]

        # Swap in plan
        self._plan.steps[index], self._plan.steps[index + 1] = \
            self._plan.steps[index + 1], self._plan.steps[index]

        # Rebuild layout
        self._rebuild_steps_layout()

    def _rebuild_steps_layout(self):
        """Rebuild steps layout after reorder."""
        # Remove all widgets from layout (but don't delete them)
        while self.steps_layout.count():
            self.steps_layout.takeAt(0)

        # Re-add in new order
        for i, widget in enumerate(self._step_widgets):
            widget.set_index(i)
            self.steps_layout.addWidget(widget)

    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancelled.emit()

    def _on_confirm(self):
        """Handle confirm button click."""
        # Update plan steps from widgets (in case of edits)
        for i, widget in enumerate(self._step_widgets):
            if i < len(self._plan.steps):
                self._plan.steps[i].description = widget.text

        self._plan.confirm()
        self.confirmed.emit(self._plan)

    def _on_retry(self):
        """Handle retry button click."""
        self.retry_btn.hide()
        self.stop_btn.hide()
        self.retry_requested.emit(self._plan.current_step_index)

    def _on_stop(self):
        """Handle stop button click."""
        self.retry_btn.hide()
        self.stop_btn.hide()
        self.stop_requested.emit()

    def set_executing(self, executing: bool):
        """Set execution state.

        Args:
            executing: Whether plan is currently executing
        """
        self._is_executing = executing

        # Disable editing controls during execution
        for widget in self._step_widgets:
            widget.up_btn.setEnabled(not executing and widget.status == "pending")
            widget.down_btn.setEnabled(not executing and widget.status == "pending")
            widget.delete_btn.setEnabled(not executing and widget.status == "pending")

        # Update buttons
        self.confirm_btn.setEnabled(not executing)
        self.cancel_btn.setText("Stop" if executing else "Cancel")

    def update_step_status(self, step_index: int, status: str, error: str = None):
        """Update a step's status.

        Args:
            step_index: Index of step to update
            status: New status (pending, running, completed, failed)
            error: Error message if failed
        """
        if 0 <= step_index < len(self._step_widgets):
            self._step_widgets[step_index].set_status(status)

            # Also update plan model
            if 0 <= step_index < len(self._plan.steps):
                self._plan.steps[step_index].status = status
                if error:
                    self._plan.steps[step_index].error = error

            # Show retry/stop buttons on failure
            if status == "failed":
                self.retry_btn.show()
                self.stop_btn.show()
                self.confirm_btn.hide()
                self.cancel_btn.hide()

    def mark_completed(self):
        """Mark the plan as fully completed."""
        self._is_executing = False
        self.confirm_btn.hide()
        self.cancel_btn.hide()
        self.retry_btn.hide()
        self.stop_btn.hide()

    def get_steps(self) -> list[str]:
        """Get current step descriptions."""
        return [w.text for w in self._step_widgets]

    @property
    def plan(self):
        """Get the plan object."""
        return self._plan
