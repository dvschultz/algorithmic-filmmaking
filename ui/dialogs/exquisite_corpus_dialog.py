"""Multi-step dialog for Exquisite Corpus workflow.

This dialog guides the user through:
1. Entering a mood/vibe prompt
2. Extracting text from clips (with progress)
3. Previewing and reordering the generated poem
4. Creating the final sequence
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QTextEdit,
    QPushButton,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ui.theme import theme

logger = logging.getLogger(__name__)


class ExquisiteCorpusDialog(QDialog):
    """Multi-page dialog for Exquisite Corpus workflow.

    Guides user through text extraction and poem generation,
    allowing preview and reordering before applying the sequence.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source) tuples when complete
    """

    sequence_ready = Signal(list)  # List of (Clip, Source) tuples

    # Page indices
    PAGE_MOOD = 0
    PAGE_PROGRESS = 1
    PAGE_PREVIEW = 2

    def __init__(self, clips, sources_by_id, project, parent=None):
        """Initialize the dialog.

        Args:
            clips: List of Clip objects to process
            sources_by_id: Dict mapping source_id to Source objects
            project: Project object (for access to clips_by_id if needed)
            parent: Parent widget
        """
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self.extraction_results = {}
        self.poem_lines = []
        self.worker = None

        self.setWindowTitle("Exquisite Corpus")
        self.setMinimumSize(600, 500)
        self.setModal(True)

        self._setup_ui()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 1: Mood prompt
        self.mood_page = self._create_mood_page()
        self.stack.addWidget(self.mood_page)

        # Page 2: Extraction progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Page 3: Poem preview
        self.preview_page = self._create_preview_page()
        self.stack.addWidget(self.preview_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setVisible(False)

        self.next_btn = QPushButton("Extract Text")
        self.next_btn.clicked.connect(self._go_next)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

    def _create_mood_page(self) -> QWidget:
        """Create the mood prompt input page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Exquisite Corpus")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._mood_header = header

        # Description
        desc = QLabel(
            f"Selected {len(self.clips)} clips. Text will be extracted from each clip "
            "and used to generate a poem. The clips will be sequenced to match the poem."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        self._mood_desc = desc

        layout.addSpacing(20)

        # Mood prompt input
        prompt_label = QLabel("Enter the mood or vibe for your poem:")
        layout.addWidget(prompt_label)
        self._prompt_label = prompt_label

        self.mood_input = QTextEdit()
        self.mood_input.setPlaceholderText(
            "e.g., melancholic and introspective, chaotic urban energy, "
            "dreamlike and surreal, contemplative silence, absurdist comedy..."
        )
        self.mood_input.setMaximumHeight(100)
        layout.addWidget(self.mood_input)

        # Info about what happens
        info = QLabel(
            "💡 The LLM will arrange your clips' on-screen text into a poem "
            "that evokes this mood. Phrases will be used exactly as extracted."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self._mood_info = info

        layout.addStretch()
        return page

    def _create_progress_page(self) -> QWidget:
        """Create the extraction progress page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Extracting Text from Clips...")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._progress_header = header

        layout.addSpacing(20)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("Starting...")
        layout.addWidget(self.progress_label)

        layout.addSpacing(20)

        # Results summary
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()
        return page

    def _create_preview_page(self) -> QWidget:
        """Create the poem preview page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Generated Poem")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._preview_header = header

        # Instruction
        instruction = QLabel("Drag lines to reorder. Each line will become one clip in the sequence.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self._preview_instruction = instruction

        layout.addSpacing(10)

        # Poem line list (drag-drop enabled)
        self.poem_list = QListWidget()
        self.poem_list.setDragDropMode(QListWidget.InternalMove)
        self.poem_list.setDefaultDropAction(Qt.MoveAction)
        self.poem_list.setAlternatingRowColors(True)
        layout.addWidget(self.poem_list)

        # Button row
        btn_layout = QHBoxLayout()

        self.regen_btn = QPushButton("🎲 Regenerate")
        self.regen_btn.setToolTip("Generate a new poem with the same mood")
        self.regen_btn.clicked.connect(self._regenerate_poem)
        btn_layout.addWidget(self.regen_btn)

        btn_layout.addStretch()

        # Mood indicator
        self.mood_indicator = QLabel("")
        btn_layout.addWidget(self.mood_indicator)

        layout.addLayout(btn_layout)

        return page

    def _apply_theme(self):
        """Apply theme colors to the dialog."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme().background_primary};
            }}
            QLabel {{
                color: {theme().text_primary};
            }}
            QTextEdit {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                padding: 8px;
            }}
            QListWidget {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {theme().border_primary};
            }}
            QListWidget::item:selected {{
                background-color: {theme().accent_blue}33;
            }}
            QListWidget::item:alternate {{
                background-color: {theme().background_secondary};
            }}
            QPushButton {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {theme().background_elevated};
            }}
            QPushButton:disabled {{
                background-color: {theme().background_secondary};
                color: {theme().text_muted};
            }}
            QProgressBar {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme().accent_blue};
                border-radius: 3px;
            }}
        """)

        # Update specific labels
        if hasattr(self, '_mood_info'):
            self._mood_info.setStyleSheet(f"color: {theme().text_muted}; font-size: 11px;")
        if hasattr(self, '_preview_instruction'):
            self._preview_instruction.setStyleSheet(f"color: {theme().text_secondary};")

    def _go_back(self):
        """Navigate to previous page."""
        current = self.stack.currentIndex()
        if current == self.PAGE_PREVIEW:
            # Can go back to mood page
            self.stack.setCurrentIndex(self.PAGE_MOOD)
            self._update_nav_buttons()

    def _go_next(self):
        """Navigate to next page or finish."""
        current = self.stack.currentIndex()

        if current == self.PAGE_MOOD:
            # Validate mood input
            mood = self.mood_input.toPlainText().strip()
            if not mood:
                QMessageBox.warning(
                    self,
                    "Mood Required",
                    "Please enter a mood or vibe for your poem."
                )
                return
            # Go to progress page and start extraction
            self.stack.setCurrentIndex(self.PAGE_PROGRESS)
            self._update_nav_buttons()
            self._start_extraction()

        elif current == self.PAGE_PREVIEW:
            # Finish and create sequence
            self._finish()

    def _update_nav_buttons(self):
        """Update navigation button states based on current page."""
        current = self.stack.currentIndex()

        if current == self.PAGE_MOOD:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Extract Text")
            self.next_btn.setEnabled(True)
        elif current == self.PAGE_PROGRESS:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Please wait...")
            self.next_btn.setEnabled(False)
        elif current == self.PAGE_PREVIEW:
            self.back_btn.setVisible(True)
            self.next_btn.setText("Create Sequence")
            self.next_btn.setEnabled(True)

    def _start_extraction(self):
        """Start the text extraction worker."""
        from ui.workers.text_extraction_worker import TextExtractionWorker

        logger.info(f"Starting text extraction for {len(self.clips)} clips")

        self.worker = TextExtractionWorker(
            clips=self.clips,
            sources_by_id=self.sources_by_id,
            num_keyframes=3,
            use_vlm_fallback=True,
            parent=self,
        )
        self.worker.progress.connect(self._on_extraction_progress, Qt.UniqueConnection)
        self.worker.finished.connect(self._on_extraction_finished, Qt.UniqueConnection)
        self.worker.error.connect(self._on_extraction_error, Qt.UniqueConnection)
        self.worker.start()

    def _on_extraction_progress(self, current: int, total: int, clip_id: str):
        """Update progress display."""
        progress_pct = int(current / total * 100)
        self.progress_bar.setValue(progress_pct)
        self.progress_label.setText(f"Processing clip {current}/{total}...")

    def _on_extraction_finished(self, results: dict):
        """Handle extraction completion."""
        self.extraction_results = results

        # Count clips with text
        clips_with_text = sum(1 for texts in results.values() if texts)
        total_clips = len(self.clips)

        self.results_label.setText(
            f"✓ Found text in {clips_with_text}/{total_clips} clips"
        )

        if clips_with_text < 2:
            self.results_label.setText(
                f"⚠️ Only {clips_with_text} clip(s) have text. Need at least 2 for a poem.\n"
                "Try selecting different clips or using videos with on-screen text."
            )
            self.results_label.setStyleSheet(f"color: {theme().accent_orange};")
            # Enable back button to try again
            self.back_btn.setVisible(True)
            self.next_btn.setText("Back to Start")
            self.next_btn.setEnabled(True)
            self.next_btn.clicked.disconnect()
            self.next_btn.clicked.connect(lambda: self.stack.setCurrentIndex(self.PAGE_MOOD))
            return

        # Store extracted text in clip objects for persistence
        clips_by_id = {c.id: c for c in self.clips}
        for clip_id, texts in results.items():
            if clip_id in clips_by_id:
                clips_by_id[clip_id].extracted_texts = texts

        # Generate poem
        self._generate_poem()

    def _on_extraction_error(self, error_msg: str):
        """Handle extraction error."""
        logger.error(f"Extraction error: {error_msg}")
        self.progress_label.setText(f"⚠️ Error: {error_msg}")

    def _generate_poem(self):
        """Generate poem from extracted text."""
        from core.remix.exquisite_corpus import generate_poem

        self.progress_label.setText("Generating poem...")
        self.progress_bar.setValue(95)

        # Build clips_with_text list
        clips_with_text = []
        clips_by_id = {c.id: c for c in self.clips}

        for clip_id, texts in self.extraction_results.items():
            if texts:
                clip = clips_by_id.get(clip_id)
                if clip:
                    # Combine text from all extracted frames
                    combined = " | ".join(t.text for t in texts)
                    clips_with_text.append((clip, combined))

        mood = self.mood_input.toPlainText().strip()

        try:
            self.poem_lines = generate_poem(clips_with_text, mood)
            self._display_poem()
            self.stack.setCurrentIndex(self.PAGE_PREVIEW)
            self._update_nav_buttons()
            self.mood_indicator.setText(f"Mood: {mood[:30]}{'...' if len(mood) > 30 else ''}")

        except Exception as e:
            logger.error(f"Poem generation failed: {e}")
            self.progress_label.setText(f"⚠️ Poem generation failed: {e}")
            self.progress_label.setStyleSheet(f"color: {theme().accent_orange};")
            # Enable retry
            self.back_btn.setVisible(True)
            self.next_btn.setText("Try Again")
            self.next_btn.setEnabled(True)

    def _display_poem(self):
        """Display the generated poem in the list widget."""
        self.poem_list.clear()
        for line in self.poem_lines:
            item = QListWidgetItem(f"{line.line_number}. {line.text}")
            item.setData(Qt.UserRole, line.clip_id)
            item.setData(Qt.UserRole + 1, line.text)  # Store original text
            self.poem_list.addItem(item)

    def _regenerate_poem(self):
        """Regenerate the poem with the same mood."""
        self.regen_btn.setEnabled(False)
        self.regen_btn.setText("Generating...")

        try:
            self._generate_poem()
        finally:
            self.regen_btn.setEnabled(True)
            self.regen_btn.setText("🎲 Regenerate")

    def _finish(self):
        """Finish the workflow and create the sequence."""
        from core.remix.exquisite_corpus import PoemLine, sequence_by_poem

        # Get reordered poem from list widget
        reordered_lines = []
        for i in range(self.poem_list.count()):
            item = self.poem_list.item(i)
            clip_id = item.data(Qt.UserRole)
            text = item.data(Qt.UserRole + 1)
            reordered_lines.append(PoemLine(
                text=text,
                clip_id=clip_id,
                line_number=i + 1,
            ))

        # Create sequence
        clips_by_id = {c.id: c for c in self.clips}
        sequence = sequence_by_poem(
            reordered_lines,
            clips_by_id,
            self.sources_by_id,
        )

        logger.info(f"Created sequence with {len(sequence)} clips")

        self.sequence_ready.emit(sequence)
        self.accept()

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        self.reject()

    def closeEvent(self, event):
        """Handle dialog close (X button)."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        event.accept()
