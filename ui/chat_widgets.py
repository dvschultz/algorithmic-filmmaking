"""Chat UI widgets for message display.

Provides:
- MessageBubble: Static message display (user or assistant)
- StreamingBubble: Accumulates streaming text from LLM
- ToolIndicator: Shows tool execution status
- ExamplePromptsWidget: Clickable example prompts with random cycling
"""

import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextBrowser, QVBoxLayout,
    QWidget
)


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
            self.text_browser.setStyleSheet("""
                QTextBrowser {
                    background: transparent;
                    border: none;
                    color: #050505;
                }
            """)
            # Let the browser expand to fit content
            self.text_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            layout.addWidget(self.text_browser)

            # Schedule size adjustment after layout is complete
            QTimer.singleShot(0, self._adjust_text_browser_height)

        # Allow bubble to expand for content
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

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
        # Allow label to expand vertically for long content
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.label)

        # Allow bubble to expand for content
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

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

        self.text_browser = QTextBrowser()
        self.text_browser.setMarkdown(self._text)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setFrameShape(QFrame.NoFrame)
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                border: none;
                color: #050505;
            }
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
        header = QLabel("Try asking...")
        header.setStyleSheet("color: #65676b; font-size: 14px; font-weight: 500;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Prompts container
        self._prompts_container = QWidget()
        self._rebuild_prompt_buttons()
        layout.addWidget(self._prompts_container)

        # Add stretch to center vertically
        layout.addStretch(1)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

        for prompt_text in self._selected_prompts:
            btn = QPushButton(prompt_text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f2f5;
                    border: 1px solid #d0d0d0;
                    border-radius: 12px;
                    padding: 10px 16px;
                    color: #333333;
                    text-align: left;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #e4e6eb;
                }
                QPushButton:pressed {
                    background-color: #d8dadf;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    color: #a0a0a0;
                    border-color: #e0e0e0;
                }
            """)
            btn.clicked.connect(lambda checked, text=prompt_text: self._on_button_clicked(text))
            prompts_layout.addWidget(btn)

    def _on_button_clicked(self, prompt_text: str):
        """Handle prompt button click with guard pattern."""
        if self._click_handled:
            return
        self._click_handled = True
        self.prompt_clicked.emit(prompt_text)

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
