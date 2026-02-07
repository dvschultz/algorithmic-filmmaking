"""Import dialog for the intention-first workflow.

This dialog appears when a user clicks a sequence card but has no clips.
It allows importing videos via drag-drop or URLs, then shows progress
as the workflow processes the videos.
"""

import logging
from pathlib import Path
from enum import Enum, auto

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QFileDialog,
    QProgressBar,
    QWidget,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent

from ui.theme import theme, TypeScale, Spacing, Radii

logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


class WorkflowStep(Enum):
    """Steps in the intention workflow."""
    DOWNLOADING = auto()
    DETECTING = auto()
    THUMBNAILS = auto()
    ANALYZING = auto()
    BUILDING = auto()


class DropZone(QFrame):
    """Drag-and-drop zone for video files."""

    files_dropped = Signal(list)  # List of Paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self._setup_ui()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel("📁")
        icon_font = QFont()
        icon_font.setPointSize(32)
        self.icon_label.setFont(icon_font)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drag & drop videos here")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.sub_label = QLabel("or click to browse")
        self.sub_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sub_label)

    def _apply_theme(self, dragging: bool = False):
        if dragging:
            self.setStyleSheet(f"""
                DropZone {{
                    background-color: {theme().surface_success};
                    border: 2px dashed {theme().accent_green};
                    border-radius: {Radii.MD}px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                DropZone {{
                    background-color: {theme().card_background};
                    border: 2px dashed {theme().border_primary};
                    border-radius: {Radii.MD}px;
                }}
                DropZone:hover {{
                    background-color: {theme().card_hover};
                    border-color: {theme().border_focus};
                }}
            """)
        self.text_label.setStyleSheet(f"font-size: {TypeScale.MD}px; font-weight: bold; color: {theme().text_secondary};")
        self.sub_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_muted};")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._browse_for_files()

    def _browse_for_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
        )
        if file_paths:
            self.files_dropped.emit([Path(p) for p in file_paths])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.is_dir() or path.suffix.lower() in VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        self._apply_theme(dragging=True)
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._apply_theme(dragging=False)

    def dropEvent(self, event: QDropEvent):
        self._apply_theme(dragging=False)
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    paths.extend(self._scan_folder(path))
                elif path.suffix.lower() in VIDEO_EXTENSIONS:
                    paths.append(path)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()

    def _scan_folder(self, folder: Path) -> list[Path]:
        """Recursively scan folder for video files."""
        videos = []
        for item in folder.rglob("*"):
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(item)
        return sorted(videos)


class IntentionImportDialog(QDialog):
    """Dialog for importing videos in the intention-first workflow.

    Two views:
    1. Import view: drag-drop zone + URL input + pending list
    2. Progress view: step indicators + progress bar

    Signals:
        import_requested: Emitted with (local_paths, urls, algorithm, ...) when Start Import clicked
        cancelled: Emitted when Cancel clicked during import or progress
    """

    # local_paths, urls, algorithm, direction, shot_type, poem_length, storyteller_duration, storyteller_structure, storyteller_theme
    import_requested = Signal(list, list, str, str, str, str, str, str, str)
    cancelled = Signal()

    # View indices
    VIEW_IMPORT = 0
    VIEW_PROGRESS = 1

    def __init__(self, algorithm: str, parent=None):
        super().__init__(parent)
        self._algorithm = algorithm
        self._local_paths: list[Path] = []
        self._urls: list[str] = []
        self._current_step: WorkflowStep | None = None
        self._step_progress: dict[WorkflowStep, tuple[bool, float]] = {}  # step -> (complete, progress)

        self.setWindowTitle(f"Create {algorithm.capitalize()} Sequence")
        self.setMinimumSize(500, 450)
        self.setModal(True)

        self._setup_ui()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Stack for switching between import and progress views
        self.view_stack = QStackedWidget()

        # Import view
        import_widget = self._create_import_view()
        self.view_stack.addWidget(import_widget)

        # Progress view
        progress_widget = self._create_progress_view()
        self.view_stack.addWidget(progress_widget)

        layout.addWidget(self.view_stack)

        # Start in import view
        self.view_stack.setCurrentIndex(self.VIEW_IMPORT)

    def _create_import_view(self) -> QWidget:
        """Create the import view with drag-drop, URL input, and pending list."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel(f"Import videos for {self._algorithm.capitalize()} sequence")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)
        self._import_header = header

        # Direction selector (for Duration and Color algorithms)
        self._direction_container = QWidget()
        direction_layout = QHBoxLayout(self._direction_container)
        direction_layout.setContentsMargins(0, 0, 0, 8)

        direction_label = QLabel("Sort order:")
        direction_label.setStyleSheet(f"color: {theme().text_secondary};")
        direction_layout.addWidget(direction_label)
        self._direction_label = direction_label

        self.direction_dropdown = QComboBox()
        self.direction_dropdown.setMinimumWidth(160)

        # Populate based on algorithm
        if self._algorithm.lower() == "duration":
            self.direction_dropdown.addItems(["Shortest First", "Longest First"])
        elif self._algorithm.lower() == "color":
            self.direction_dropdown.addItems(["Rainbow", "Warm to Cool", "Cool to Warm"])

        direction_layout.addWidget(self.direction_dropdown)
        direction_layout.addStretch()

        layout.addWidget(self._direction_container)

        # Hide if not applicable
        if self._algorithm.lower() not in ("duration", "color"):
            self._direction_container.hide()

        # Shot type filter (available for all algorithms)
        self._shot_type_container = QWidget()
        shot_type_layout = QHBoxLayout(self._shot_type_container)
        shot_type_layout.setContentsMargins(0, 0, 0, 8)

        shot_type_label = QLabel("Filter by shot type:")
        shot_type_label.setStyleSheet(f"color: {theme().text_secondary};")
        shot_type_layout.addWidget(shot_type_label)
        self._shot_type_label = shot_type_label

        self.shot_type_dropdown = QComboBox()
        self.shot_type_dropdown.setMinimumWidth(160)
        self.shot_type_dropdown.addItems([
            "All",
            "Wide Shot",
            "Full Shot",
            "Medium Shot",
            "Close-up",
            "Extreme Close-up",
        ])
        shot_type_layout.addWidget(self.shot_type_dropdown)
        shot_type_layout.addStretch()

        layout.addWidget(self._shot_type_container)

        # Shot type filter is available for all algorithms (cross-cutting filter)

        # Poem length selector (for Exquisite Corpus algorithm only)
        # Use a simple horizontal layout directly instead of a container widget
        # to avoid background color issues in dark mode
        if self._algorithm.lower() == "exquisite_corpus":
            poem_length_layout = QHBoxLayout()
            poem_length_layout.setContentsMargins(0, 0, 0, 8)

            poem_length_label = QLabel("Poem length:")
            poem_length_layout.addWidget(poem_length_label)
            self._poem_length_label = poem_length_label

            self.poem_length_dropdown = QComboBox()
            self.poem_length_dropdown.setMinimumWidth(160)
            self.poem_length_dropdown.addItems([
                "Short (up to 11 lines)",
                "Medium (12-25 lines)",
                "Long (26+ lines)",
            ])
            self.poem_length_dropdown.setCurrentIndex(1)  # Default to Medium
            poem_length_layout.addWidget(self.poem_length_dropdown)
            poem_length_layout.addStretch()

            layout.addLayout(poem_length_layout)
        else:
            # Create dummy dropdown so _get_poem_length doesn't fail
            self.poem_length_dropdown = QComboBox()
            self.poem_length_dropdown.setCurrentIndex(1)

        # Storyteller configuration (for storyteller algorithm only)
        if self._algorithm.lower() == "storyteller":
            # Theme input (optional)
            theme_layout = QHBoxLayout()
            theme_layout.setContentsMargins(0, 0, 0, 8)

            theme_label = QLabel("Theme (optional):")
            theme_label.setStyleSheet(f"color: {theme().text_secondary};")
            theme_layout.addWidget(theme_label)
            self._storyteller_theme_label = theme_label

            self.storyteller_theme_input = QTextEdit()
            self.storyteller_theme_input.setPlaceholderText("e.g., urban isolation, joy, transformation...")
            self.storyteller_theme_input.setMaximumHeight(50)
            theme_layout.addWidget(self.storyteller_theme_input)

            layout.addLayout(theme_layout)

            # Structure dropdown
            structure_layout = QHBoxLayout()
            structure_layout.setContentsMargins(0, 0, 0, 8)

            structure_label = QLabel("Narrative structure:")
            structure_label.setStyleSheet(f"color: {theme().text_secondary};")
            structure_layout.addWidget(structure_label)
            self._storyteller_structure_label = structure_label

            self.storyteller_structure_dropdown = QComboBox()
            self.storyteller_structure_dropdown.setMinimumWidth(200)
            self.storyteller_structure_dropdown.addItems([
                "Auto (LLM chooses)",
                "Three-Act (setup, conflict, resolution)",
                "Chronological (time-based)",
                "Thematic (grouped by theme)",
            ])
            self.storyteller_structure_dropdown.setCurrentIndex(0)  # Default to Auto
            structure_layout.addWidget(self.storyteller_structure_dropdown)
            structure_layout.addStretch()

            layout.addLayout(structure_layout)

            # Duration dropdown
            duration_layout = QHBoxLayout()
            duration_layout.setContentsMargins(0, 0, 0, 8)

            duration_label = QLabel("Target duration:")
            duration_label.setStyleSheet(f"color: {theme().text_secondary};")
            duration_layout.addWidget(duration_label)
            self._storyteller_duration_label = duration_label

            self.storyteller_duration_dropdown = QComboBox()
            self.storyteller_duration_dropdown.setMinimumWidth(160)
            self.storyteller_duration_dropdown.addItems([
                "Use all clips",
                "~10 minutes",
                "~30 minutes",
                "~1 hour",
                "~90 minutes",
            ])
            self.storyteller_duration_dropdown.setCurrentIndex(0)  # Default to all
            duration_layout.addWidget(self.storyteller_duration_dropdown)
            duration_layout.addStretch()

            layout.addLayout(duration_layout)
        else:
            # Create dummy widgets so getters don't fail
            self.storyteller_theme_input = QTextEdit()
            self.storyteller_structure_dropdown = QComboBox()
            self.storyteller_structure_dropdown.setCurrentIndex(0)
            self.storyteller_duration_dropdown = QComboBox()
            self.storyteller_duration_dropdown.setCurrentIndex(0)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # URL section
        url_label = QLabel("Or enter YouTube/Vimeo URLs (one per line):")
        layout.addWidget(url_label)
        self._url_label = url_label

        self.url_input = QTextEdit()
        self.url_input.setPlaceholderText("https://youtube.com/watch?v=...\nhttps://vimeo.com/...")
        self.url_input.setMaximumHeight(80)
        self.url_input.textChanged.connect(self._update_pending_list)
        layout.addWidget(self.url_input)

        # Pending imports list
        pending_label = QLabel("Pending imports:")
        layout.addWidget(pending_label)
        self._pending_label = pending_label

        self.pending_list = QListWidget()
        self.pending_list.setMaximumHeight(120)
        layout.addWidget(self.pending_list)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        self.start_btn = QPushButton("Start Import")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        button_layout.addWidget(self.start_btn)

        layout.addLayout(button_layout)

        return widget

    def _create_progress_view(self) -> QWidget:
        """Create the progress view with step indicators."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        self.progress_header = QLabel(f"Creating your {self._algorithm.capitalize()} sequence...")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        self.progress_header.setFont(header_font)
        layout.addWidget(self.progress_header)

        layout.addSpacing(20)

        # Step indicators
        self.step_labels: dict[WorkflowStep, QLabel] = {}
        self.step_progress_bars: dict[WorkflowStep, QProgressBar] = {}

        steps = [
            (WorkflowStep.DOWNLOADING, "Downloading videos"),
            (WorkflowStep.DETECTING, "Detecting scenes"),
            (WorkflowStep.THUMBNAILS, "Generating thumbnails"),
            (WorkflowStep.ANALYZING, "Analyzing clips"),
            (WorkflowStep.BUILDING, "Building sequence"),
        ]

        for step, label_text in steps:
            step_layout = QHBoxLayout()

            # Status indicator
            indicator = QLabel("○")
            indicator.setFixedWidth(24)
            step_layout.addWidget(indicator)
            self.step_labels[step] = indicator

            # Label
            label = QLabel(label_text)
            label.setMinimumWidth(150)
            step_layout.addWidget(label)

            # Progress bar
            progress = QProgressBar()
            progress.setMaximum(100)
            progress.setValue(0)
            progress.setVisible(False)
            step_layout.addWidget(progress)
            self.step_progress_bars[step] = progress

            layout.addLayout(step_layout)

        layout.addStretch()

        # Status message
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.progress_cancel_btn = QPushButton("Cancel")
        self.progress_cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.progress_cancel_btn)

        layout.addLayout(button_layout)

        return widget

    def _apply_theme(self):
        """Apply theme colors."""
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
                border-radius: {Radii.SM}px;
                padding: {Spacing.SM}px;
            }}
            QComboBox {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: {Radii.SM}px;
                padding: 6px {Spacing.MD}px;
                min-height: 24px;
            }}
            QComboBox:hover {{
                background-color: {theme().background_elevated};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: {Spacing.SM}px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                selection-background-color: {theme().accent_blue};
            }}
            QListWidget {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: {Radii.SM}px;
            }}
            QListWidget::item {{
                padding: {Spacing.XS}px;
            }}
            QPushButton {{
                background-color: {theme().background_tertiary};
                color: {theme().text_primary};
                border: 1px solid {theme().border_primary};
                border-radius: {Radii.SM}px;
                padding: {Spacing.SM}px {Spacing.LG}px;
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
                border-radius: {Radii.SM}px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme().accent_blue};
                border-radius: 3px;
            }}
        """)

        if hasattr(self, '_import_header'):
            self._import_header.setStyleSheet(f"color: {theme().text_primary};")
        if hasattr(self, '_url_label'):
            self._url_label.setStyleSheet(f"color: {theme().text_secondary};")
        if hasattr(self, '_pending_label'):
            self._pending_label.setStyleSheet(f"color: {theme().text_secondary};")

    def _on_files_dropped(self, paths: list[Path]):
        """Handle dropped files."""
        for path in paths:
            if path not in self._local_paths:
                self._local_paths.append(path)
        self._update_pending_list()

    def _update_pending_list(self):
        """Update the pending imports list."""
        self.pending_list.clear()

        # Add local files
        for path in self._local_paths:
            item = QListWidgetItem(f"📁 {path.name}")
            item.setData(Qt.UserRole, ("file", path))
            self.pending_list.addItem(item)

        # Add URLs from text input
        self._urls = []
        url_text = self.url_input.toPlainText().strip()
        if url_text:
            for line in url_text.split("\n"):
                url = line.strip()
                if url and (url.startswith("http://") or url.startswith("https://")):
                    self._urls.append(url)
                    # Determine icon based on domain
                    if "youtube.com" in url or "youtu.be" in url:
                        icon = "🎬"
                    elif "vimeo.com" in url:
                        icon = "🎥"
                    else:
                        icon = "🔗"
                    item = QListWidgetItem(f"{icon} {url[:50]}...")
                    item.setData(Qt.UserRole, ("url", url))
                    self.pending_list.addItem(item)

        # Enable/disable start button
        has_items = bool(self._local_paths or self._urls)
        self.start_btn.setEnabled(has_items)

    def _get_direction(self) -> str | None:
        """Get the selected direction based on algorithm and dropdown."""
        if self._algorithm.lower() == "duration":
            text = self.direction_dropdown.currentText()
            if text == "Longest First":
                return "long_first"
            return "short_first"
        elif self._algorithm.lower() == "color":
            text = self.direction_dropdown.currentText()
            if text == "Warm to Cool":
                return "warm_to_cool"
            elif text == "Cool to Warm":
                return "cool_to_warm"
            return "rainbow"
        return None

    def _get_shot_type(self) -> str | None:
        """Get the selected shot type filter."""
        text = self.shot_type_dropdown.currentText()
        if text == "All":
            return None
        # Convert display text to internal format (lowercase)
        return text.lower()

    def _get_poem_length(self) -> str | None:
        """Get the selected poem length (for exquisite_corpus algorithm)."""
        if self._algorithm.lower() != "exquisite_corpus":
            return None
        # Map dropdown index to length key
        length_map = {0: "short", 1: "medium", 2: "long"}
        return length_map.get(self.poem_length_dropdown.currentIndex(), "medium")

    def _get_storyteller_duration(self) -> str | None:
        """Get the selected target duration (for storyteller algorithm)."""
        if self._algorithm.lower() != "storyteller":
            return None
        # Map dropdown index to duration key
        duration_map = {0: "all", 1: "10min", 2: "30min", 3: "1hr", 4: "90min"}
        return duration_map.get(self.storyteller_duration_dropdown.currentIndex(), "all")

    def _get_storyteller_structure(self) -> str | None:
        """Get the selected narrative structure (for storyteller algorithm)."""
        if self._algorithm.lower() != "storyteller":
            return None
        # Map dropdown index to structure key
        structure_map = {0: "auto", 1: "three_act", 2: "chronological", 3: "thematic"}
        return structure_map.get(self.storyteller_structure_dropdown.currentIndex(), "auto")

    def _get_storyteller_theme(self) -> str | None:
        """Get the theme text (for storyteller algorithm)."""
        if self._algorithm.lower() != "storyteller":
            return None
        text = self.storyteller_theme_input.toPlainText().strip()
        return text if text else None

    def _on_start(self):
        """Handle Start Import click."""
        if not self._local_paths and not self._urls:
            return

        # Switch to progress view
        self.view_stack.setCurrentIndex(self.VIEW_PROGRESS)

        # Initialize step states
        for step in WorkflowStep:
            self._step_progress[step] = (False, 0.0)

        # Get algorithm-specific parameters
        direction = self._get_direction()
        shot_type = self._get_shot_type()
        poem_length = self._get_poem_length()
        storyteller_duration = self._get_storyteller_duration()
        storyteller_structure = self._get_storyteller_structure()
        storyteller_theme = self._get_storyteller_theme()

        # Emit signal to start workflow
        self.import_requested.emit(
            self._local_paths.copy(),
            self._urls.copy(),
            self._algorithm,
            direction,
            shot_type,
            poem_length,
            storyteller_duration,
            storyteller_structure,
            storyteller_theme,
        )

    def _on_cancel(self):
        """Handle Cancel click."""
        self.cancelled.emit()
        self.reject()

    def closeEvent(self, event):
        """Handle dialog close (X button)."""
        # If we're in progress view and not complete, treat as cancel
        if self.stack.currentIndex() == 1:  # Progress view
            if self.progress_cancel_btn.text() != "Close":
                self.cancelled.emit()
        event.accept()

    # --- Progress update methods (called by coordinator) ---

    def show_progress(self):
        """Switch to the progress view.

        Called externally when the workflow starts processing.
        """
        self.view_stack.setCurrentIndex(self.VIEW_PROGRESS)

        # Initialize step states
        for step in WorkflowStep:
            self._step_progress[step] = (False, 0.0)

    def set_step_active(self, step: WorkflowStep, message: str = ""):
        """Mark a step as currently active."""
        self._current_step = step

        for s, label in self.step_labels.items():
            if s == step:
                label.setText("●")
                label.setStyleSheet(f"color: {theme().accent_blue}; font-size: 16px;")
                self.step_progress_bars[s].setVisible(True)
            elif self._step_progress.get(s, (False, 0))[0]:
                label.setText("✓")
                label.setStyleSheet(f"color: {theme().accent_green}; font-size: 16px;")
                self.step_progress_bars[s].setVisible(False)
            else:
                label.setText("○")
                label.setStyleSheet(f"color: {theme().text_muted}; font-size: 16px;")
                self.step_progress_bars[s].setVisible(False)

        if message:
            self.status_label.setText(message)

    def set_step_progress(self, step: WorkflowStep, progress: float, message: str = ""):
        """Update progress for a step (0-100)."""
        self._step_progress[step] = (False, progress)
        if step in self.step_progress_bars:
            self.step_progress_bars[step].setValue(int(progress))
        if message:
            self.status_label.setText(message)

    def set_step_complete(self, step: WorkflowStep, message: str = ""):
        """Mark a step as complete."""
        self._step_progress[step] = (True, 100.0)

        if step in self.step_labels:
            self.step_labels[step].setText("✓")
            self.step_labels[step].setStyleSheet(f"color: {theme().accent_green}; font-size: 16px;")
        if step in self.step_progress_bars:
            self.step_progress_bars[step].setValue(100)
            self.step_progress_bars[step].setVisible(False)
        if message:
            self.status_label.setText(message)

    def set_step_skipped(self, step: WorkflowStep, message: str = ""):
        """Mark a step as skipped (e.g., no analysis needed)."""
        self._step_progress[step] = (True, 100.0)

        if step in self.step_labels:
            self.step_labels[step].setText("—")
            self.step_labels[step].setStyleSheet(f"color: {theme().text_muted}; font-size: 16px;")
        if step in self.step_progress_bars:
            self.step_progress_bars[step].setVisible(False)
        if message:
            self.status_label.setText(message)

    def set_error(self, message: str):
        """Show an error message."""
        self.status_label.setText(f"⚠️ {message}")
        self.status_label.setStyleSheet(f"color: {theme().accent_orange};")

    def set_complete(
        self,
        clip_count: int = 0,
        sources_processed: int = 0,
        sources_failed: int = 0,
    ):
        """Mark workflow as complete with summary.

        Args:
            clip_count: Number of clips created
            sources_processed: Number of sources successfully processed
            sources_failed: Number of sources that failed
        """
        self.progress_header.setText("Complete!")

        # Build summary message
        if sources_failed > 0:
            message = f"Created {clip_count} clips from {sources_processed} of {sources_processed + sources_failed} videos"
        else:
            message = f"Created {clip_count} clips from {sources_processed} videos"

        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {theme().accent_green};")
        self.progress_cancel_btn.setText("Close")
