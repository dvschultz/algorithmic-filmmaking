"""Multi-step dialog for Storyteller narrative workflow.

This dialog guides the user through:
1. Configuring narrative parameters (theme, structure, duration)
2. LLM processing (with progress)
3. Previewing and reordering the generated sequence
4. Creating the final timeline
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
    QComboBox,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from ui.theme import theme, UISizes

logger = logging.getLogger(__name__)


class StorytellerWorker(QThread):
    """Background worker for LLM narrative generation."""

    progress = Signal(str)  # status message
    finished_narrative = Signal(list)  # list of NarrativeLine
    error = Signal(str)

    def __init__(
        self,
        clips_with_descriptions: list,
        target_duration_minutes: Optional[int],
        narrative_structure: str,
        theme: Optional[str],
        parent=None,
    ):
        super().__init__(parent)
        self.clips_with_descriptions = clips_with_descriptions
        self.target_duration_minutes = target_duration_minutes
        self.narrative_structure = narrative_structure
        self.theme = theme
        self._cancelled = False

    def run(self):
        """Run the narrative generation."""
        from core.remix.storyteller import generate_narrative

        try:
            self.progress.emit("Analyzing clip descriptions...")

            narrative_lines = generate_narrative(
                clips_with_descriptions=self.clips_with_descriptions,
                target_duration_minutes=self.target_duration_minutes,
                narrative_structure=self.narrative_structure,
                theme=self.theme,
            )

            if not self._cancelled:
                self.finished_narrative.emit(narrative_lines)

        except Exception as e:
            if not self._cancelled:
                logger.error(f"Narrative generation error: {e}")
                self.error.emit(str(e))

    def cancel(self):
        """Cancel the worker."""
        self._cancelled = True


class StorytellerDialog(QDialog):
    """Multi-page dialog for Storyteller narrative workflow.

    Guides user through narrative configuration and generation,
    allowing preview and reordering before applying the sequence.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source) tuples when complete
    """

    sequence_ready = Signal(list)  # List of (Clip, Source) tuples

    # Page indices
    PAGE_CONFIG = 0
    PAGE_PROGRESS = 1
    PAGE_PREVIEW = 2

    def __init__(
        self,
        clips,
        sources_by_id,
        project,
        parent=None,
        initial_duration: str = None,
        initial_structure: str = None,
        initial_theme: str = None,
    ):
        """Initialize the dialog.

        Args:
            clips: List of Clip objects to process
            sources_by_id: Dict mapping source_id to Source objects
            project: Project object
            parent: Parent widget
            initial_duration: Optional pre-selected duration ("10min", "30min", "1hr", "90min", "all")
            initial_structure: Optional pre-selected structure ("three_act", "chronological", "thematic", "auto")
            initial_theme: Optional pre-filled theme text
        """
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self.narrative_lines = []
        self.worker = None
        self._initial_duration = initial_duration
        self._initial_structure = initial_structure
        self._initial_theme = initial_theme

        # Filter clips to those with descriptions
        self.clips_with_descriptions = []
        self.clips_without_descriptions = []
        for clip in clips:
            if clip.description:
                self.clips_with_descriptions.append(clip)
            else:
                self.clips_without_descriptions.append(clip)

        self.setWindowTitle("Storyteller")
        self.setMinimumSize(650, 550)
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

        # Page 1: Configuration
        self.config_page = self._create_config_page()
        self.stack.addWidget(self.config_page)

        # Page 2: Progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Page 3: Preview
        self.preview_page = self._create_preview_page()
        self.stack.addWidget(self.preview_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setVisible(False)

        self.next_btn = QPushButton("Generate Narrative")
        self.next_btn.clicked.connect(self._go_next)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

    def _create_config_page(self) -> QWidget:
        """Create the configuration input page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Storyteller")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._config_header = header

        # Description
        desc_text = f"Create a narrative sequence from {len(self.clips_with_descriptions)} clips with descriptions."
        if self.clips_without_descriptions:
            desc_text += f"\n({len(self.clips_without_descriptions)} clips without descriptions will be excluded)"
        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        self._config_desc = desc

        layout.addSpacing(20)

        # Theme input (optional)
        theme_label = QLabel("Theme or focus (optional):")
        layout.addWidget(theme_label)
        self._theme_label = theme_label

        self.theme_input = QTextEdit()
        self.theme_input.setPlaceholderText(
            "e.g., urban isolation, the passage of time, human connection, "
            "industrial decay, moments of joy..."
        )
        self.theme_input.setMaximumHeight(80)
        if self._initial_theme:
            self.theme_input.setPlainText(self._initial_theme)
        layout.addWidget(self.theme_input)

        layout.addSpacing(16)

        # Narrative structure dropdown
        structure_layout = QHBoxLayout()
        structure_label = QLabel("Narrative structure:")
        structure_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
        structure_layout.addWidget(structure_label)
        self._structure_label = structure_label

        self.structure_combo = QComboBox()
        self.structure_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.structure_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self.structure_combo.addItems([
            "Three-Act (setup, confrontation, resolution)",
            "Chronological (time-based ordering)",
            "Thematic (group by themes)",
            "Auto (LLM chooses best fit)",
        ])
        # Set initial index
        structure_to_index = {"three_act": 0, "chronological": 1, "thematic": 2, "auto": 3}
        initial_idx = structure_to_index.get(self._initial_structure, 3)  # Default to Auto
        self.structure_combo.setCurrentIndex(initial_idx)
        structure_layout.addWidget(self.structure_combo)
        structure_layout.addStretch()
        layout.addLayout(structure_layout)

        layout.addSpacing(8)

        # Duration target dropdown
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Target duration:")
        duration_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
        duration_layout.addWidget(duration_label)
        self._duration_label = duration_label

        self.duration_combo = QComboBox()
        self.duration_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.duration_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self.duration_combo.addItems([
            "Use all clips (no target)",
            "~10 minutes",
            "~30 minutes",
            "~1 hour",
            "~90 minutes",
        ])
        # Set initial index
        duration_to_index = {"all": 0, "10min": 1, "30min": 2, "1hr": 3, "90min": 4}
        initial_dur_idx = duration_to_index.get(self._initial_duration, 0)
        self.duration_combo.setCurrentIndex(initial_dur_idx)
        duration_layout.addWidget(self.duration_combo)
        duration_layout.addStretch()
        layout.addLayout(duration_layout)

        layout.addSpacing(16)

        # Info about what happens
        info = QLabel(
            "The LLM will analyze clip descriptions and arrange them into a "
            "narrative sequence. You can preview and reorder before applying."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self._config_info = info

        layout.addStretch()
        return page

    def _create_progress_page(self) -> QWidget:
        """Create the progress page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Generating Narrative...")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._progress_header = header

        layout.addSpacing(20)

        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("Starting...")
        layout.addWidget(self.progress_label)

        layout.addStretch()
        return page

    def _create_preview_page(self) -> QWidget:
        """Create the preview page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Generated Narrative")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._preview_header = header

        # Stats row
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)
        self._stats_label = self.stats_label

        # Instruction
        instruction = QLabel("Drag clips to reorder. Each item will become one clip in the sequence.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self._preview_instruction = instruction

        layout.addSpacing(10)

        # Narrative list (drag-drop enabled)
        self.narrative_list = QListWidget()
        self.narrative_list.setDragDropMode(QListWidget.InternalMove)
        self.narrative_list.setDefaultDropAction(Qt.MoveAction)
        self.narrative_list.setAlternatingRowColors(True)
        layout.addWidget(self.narrative_list)

        # Button row
        btn_layout = QHBoxLayout()

        self.regen_btn = QPushButton("Regenerate")
        self.regen_btn.setToolTip("Generate a new narrative with the same settings")
        self.regen_btn.clicked.connect(self._regenerate_narrative)
        btn_layout.addWidget(self.regen_btn)

        btn_layout.addStretch()

        # Structure indicator
        self.structure_indicator = QLabel("")
        btn_layout.addWidget(self.structure_indicator)

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
            QComboBox {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 28px;
            }}
            QComboBox:hover {{
                background-color: {theme().background_elevated};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
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
        if hasattr(self, '_config_info'):
            self._config_info.setStyleSheet(f"color: {theme().text_muted}; font-size: 11px;")
        if hasattr(self, '_preview_instruction'):
            self._preview_instruction.setStyleSheet(f"color: {theme().text_secondary};")

    def _go_back(self):
        """Navigate to previous page."""
        current = self.stack.currentIndex()
        if current == self.PAGE_PREVIEW:
            self.stack.setCurrentIndex(self.PAGE_CONFIG)
            self._update_nav_buttons()

    def _go_next(self):
        """Navigate to next page or finish."""
        current = self.stack.currentIndex()

        if current == self.PAGE_CONFIG:
            # Validate we have clips with descriptions
            if not self.clips_with_descriptions:
                QMessageBox.warning(
                    self,
                    "No Descriptions",
                    "None of the selected clips have descriptions. "
                    "Run description analysis in the Analyze tab first."
                )
                return

            # Go to progress page and start generation
            self.stack.setCurrentIndex(self.PAGE_PROGRESS)
            self._update_nav_buttons()
            self._start_generation()

        elif current == self.PAGE_PREVIEW:
            # Finish and create sequence
            self._finish()

    def _update_nav_buttons(self):
        """Update navigation button states based on current page."""
        current = self.stack.currentIndex()

        if current == self.PAGE_CONFIG:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Generate Narrative")
            self.next_btn.setEnabled(True)
        elif current == self.PAGE_PROGRESS:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Please wait...")
            self.next_btn.setEnabled(False)
        elif current == self.PAGE_PREVIEW:
            self.back_btn.setVisible(True)
            self.next_btn.setText("Create Sequence")
            self.next_btn.setEnabled(True)

    def _get_selected_structure(self) -> str:
        """Get the selected narrative structure key."""
        index_to_structure = {0: "three_act", 1: "chronological", 2: "thematic", 3: "auto"}
        return index_to_structure.get(self.structure_combo.currentIndex(), "auto")

    def _get_selected_duration(self) -> Optional[int]:
        """Get the selected target duration in minutes, or None for 'all'."""
        index_to_duration = {0: None, 1: 10, 2: 30, 3: 60, 4: 90}
        return index_to_duration.get(self.duration_combo.currentIndex(), None)

    def _start_generation(self):
        """Start the narrative generation worker."""
        # Build clips_with_descriptions list as tuples
        clips_data = []
        for clip in self.clips_with_descriptions:
            source = self.sources_by_id.get(clip.source_id)
            if source:
                # Store duration for the generator
                clip._duration_seconds = clip.duration_seconds(source.fps)
            clips_data.append((clip, clip.description))

        theme_text = self.theme_input.toPlainText().strip() or None
        structure = self._get_selected_structure()
        duration = self._get_selected_duration()

        logger.info(
            f"Starting Storyteller generation: {len(clips_data)} clips, "
            f"structure={structure}, duration={duration}, theme={theme_text}"
        )

        self.worker = StorytellerWorker(
            clips_with_descriptions=clips_data,
            target_duration_minutes=duration,
            narrative_structure=structure,
            theme=theme_text,
            parent=self,
        )
        self.worker.progress.connect(self._on_progress, Qt.UniqueConnection)
        self.worker.finished_narrative.connect(self._on_generation_finished, Qt.UniqueConnection)
        self.worker.error.connect(self._on_generation_error, Qt.UniqueConnection)
        self.worker.start()

    def _on_progress(self, message: str):
        """Update progress display."""
        self.progress_label.setText(message)

    def _on_generation_finished(self, narrative_lines: list):
        """Handle generation completion."""
        self.narrative_lines = narrative_lines

        # Calculate stats
        total_duration = 0.0
        for line in narrative_lines:
            clip = next((c for c in self.clips if c.id == line.clip_id), None)
            if clip:
                source = self.sources_by_id.get(clip.source_id)
                if source:
                    total_duration += clip.duration_seconds(source.fps)

        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        self.stats_label.setText(
            f"{len(narrative_lines)} clips selected | {minutes}:{seconds:02d} total duration"
        )

        # Display narrative
        self._display_narrative()
        self.stack.setCurrentIndex(self.PAGE_PREVIEW)
        self._update_nav_buttons()

        # Show structure used
        structure = self._get_selected_structure()
        self.structure_indicator.setText(f"Structure: {structure.replace('_', ' ').title()}")

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        logger.error(f"Generation error: {error_msg}")
        self.progress_label.setText(f"Error: {error_msg}")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        # Enable retry
        self.back_btn.setVisible(True)
        self.next_btn.setText("Try Again")
        self.next_btn.setEnabled(True)
        self.next_btn.clicked.disconnect()
        self.next_btn.clicked.connect(lambda: self.stack.setCurrentIndex(self.PAGE_CONFIG))

    def _display_narrative(self):
        """Display the generated narrative in the list widget."""
        self.narrative_list.clear()
        for line in self.narrative_lines:
            # Truncate description for display
            desc = line.description[:80] + "..." if len(line.description) > 80 else line.description
            item_text = f"{line.line_number}. [{line.narrative_role}] {desc}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, line.clip_id)
            item.setData(Qt.UserRole + 1, line.description)
            item.setData(Qt.UserRole + 2, line.narrative_role)
            self.narrative_list.addItem(item)

    def _regenerate_narrative(self):
        """Regenerate the narrative with the same settings."""
        self.regen_btn.setEnabled(False)
        self.regen_btn.setText("Generating...")

        self.stack.setCurrentIndex(self.PAGE_PROGRESS)
        self._update_nav_buttons()
        self._start_generation()

        self.regen_btn.setEnabled(True)
        self.regen_btn.setText("Regenerate")

    def _finish(self):
        """Finish the workflow and create the sequence."""
        from core.remix.storyteller import NarrativeLine, sequence_by_narrative

        # Get reordered narrative from list widget
        reordered_lines = []
        for i in range(self.narrative_list.count()):
            item = self.narrative_list.item(i)
            clip_id = item.data(Qt.UserRole)
            description = item.data(Qt.UserRole + 1)
            role = item.data(Qt.UserRole + 2)
            reordered_lines.append(NarrativeLine(
                clip_id=clip_id,
                description=description,
                narrative_role=role,
                line_number=i + 1,
            ))

        # Create sequence
        clips_by_id = {c.id: c for c in self.clips}
        sequence = sequence_by_narrative(
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


class MissingDescriptionsDialog(QDialog):
    """Dialog shown when clips are missing descriptions.

    Offers user choice to either:
    - Exclude clips without descriptions
    - Navigate to Analyze tab to run description analysis
    """

    # User chose to exclude clips without descriptions
    exclude_selected = Signal()
    # User chose to run analysis
    analyze_selected = Signal()

    def __init__(self, clips_without_descriptions: list, total_clips: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Missing Descriptions")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Message
        msg = QLabel(
            f"{len(clips_without_descriptions)} of {total_clips} clips don't have descriptions.\n\n"
            "Storyteller needs descriptions to create a narrative."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        layout.addSpacing(20)

        # Buttons
        btn_layout = QHBoxLayout()

        exclude_btn = QPushButton("Exclude clips without descriptions")
        exclude_btn.clicked.connect(self._on_exclude)
        btn_layout.addWidget(exclude_btn)

        analyze_btn = QPushButton("Run description analysis first")
        analyze_btn.clicked.connect(self._on_analyze)
        btn_layout.addWidget(analyze_btn)

        layout.addLayout(btn_layout)

        # Cancel
        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_layout.addWidget(cancel_btn)
        layout.addLayout(cancel_layout)

        self._apply_theme()

    def _apply_theme(self):
        """Apply theme colors."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme().background_primary};
            }}
            QLabel {{
                color: {theme().text_primary};
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
        """)

    def _on_exclude(self):
        """User chose to exclude clips."""
        self.exclude_selected.emit()
        self.accept()

    def _on_analyze(self):
        """User chose to run analysis."""
        self.analyze_selected.emit()
        self.accept()
