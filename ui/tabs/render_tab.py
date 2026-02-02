"""Render tab for export configuration and rendering."""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.widgets import EmptyStateWidget


class RenderTab(BaseTab):
    """Tab for configuring and exporting the final sequence.

    Signals:
        export_sequence_requested: Emitted when sequence export is requested
        export_clips_requested: Emitted when selected clips export is requested
        export_all_clips_requested: Emitted when all clips export is requested
        export_dataset_requested: Emitted when dataset export is requested
    """

    export_sequence_requested = Signal()
    export_clips_requested = Signal()
    export_all_clips_requested = Signal()
    export_dataset_requested = Signal()

    # State constants
    STATE_NO_CONTENT = 0
    STATE_READY = 1

    def __init__(self, parent=None):
        self._sequence_duration = 0
        self._sequence_clip_count = 0
        self._detected_clip_count = 0
        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Render tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for different states
        self.state_stack = QStackedWidget()

        # State 0: No content to export
        self.no_content_widget = EmptyStateWidget(
            "Nothing to Export",
            "Add clips to the timeline in the Sequence tab first"
        )
        self.state_stack.addWidget(self.no_content_widget)

        # State 1: Ready to export
        self.content_widget = self._create_content_area()
        self.state_stack.addWidget(self.content_widget)

        layout.addWidget(self.state_stack)

        # Start with no content state
        self.state_stack.setCurrentIndex(self.STATE_NO_CONTENT)

    def _create_content_area(self) -> QWidget:
        """Create the main content area with export options."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Export Settings group
        settings_group = QGroupBox("Export Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Quality row
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "High (best quality, larger file)",
            "Medium (balanced)",
            "Low (smaller file, faster)",
        ])
        self.quality_combo.setCurrentIndex(1)  # Default to medium
        self.quality_combo.setMinimumWidth(250)
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        settings_layout.addLayout(quality_layout)

        # Resolution row
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Original",
            "1080p (1920x1080)",
            "720p (1280x720)",
            "480p (854x480)",
        ])
        self.resolution_combo.setCurrentIndex(0)  # Default to original
        self.resolution_combo.setMinimumWidth(250)
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        settings_layout.addLayout(resolution_layout)

        # Frame rate row
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems([
            "Original",
            "60 fps",
            "30 fps",
            "24 fps",
        ])
        self.fps_combo.setCurrentIndex(0)  # Default to original
        self.fps_combo.setMinimumWidth(250)
        fps_layout.addWidget(self.fps_combo)
        fps_layout.addStretch()
        settings_layout.addLayout(fps_layout)

        layout.addWidget(settings_group)

        # Sequence Export group
        sequence_group = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout(sequence_group)

        # Sequence info
        self.sequence_info_label = QLabel("Duration: 00:00:00    Clips: 0")
        sequence_layout.addWidget(self.sequence_info_label)

        # Export sequence button
        self.export_sequence_btn = QPushButton("Export Sequence")
        self.export_sequence_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.export_sequence_btn.clicked.connect(self._on_export_sequence_click)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_sequence_btn)
        btn_layout.addStretch()
        sequence_layout.addLayout(btn_layout)

        layout.addWidget(sequence_group)

        # Other Exports group
        other_group = QGroupBox("Other Exports")
        other_layout = QHBoxLayout(other_group)
        other_layout.addStretch()

        # Export selected clips button
        self.export_clips_btn = QPushButton("Export Selected Clips")
        self.export_clips_btn.clicked.connect(self._on_export_clips_click)
        other_layout.addWidget(self.export_clips_btn)

        # Export all clips button
        self.export_all_clips_btn = QPushButton("Export All Clips")
        self.export_all_clips_btn.clicked.connect(self._on_export_all_clips_click)
        other_layout.addWidget(self.export_all_clips_btn)

        # Export dataset button
        self.export_dataset_btn = QPushButton("Export Dataset (JSON)")
        self.export_dataset_btn.setToolTip("Export clip metadata to JSON file")
        self.export_dataset_btn.clicked.connect(self._on_export_dataset_click)
        other_layout.addWidget(self.export_dataset_btn)

        other_layout.addStretch()
        layout.addWidget(other_group)

        layout.addStretch()
        return content

    def _on_export_sequence_click(self):
        """Handle export sequence button click."""
        self.export_sequence_requested.emit()

    def _on_export_clips_click(self):
        """Handle export selected clips button click."""
        self.export_clips_requested.emit()

    def _on_export_all_clips_click(self):
        """Handle export all clips button click."""
        self.export_all_clips_requested.emit()

    def _on_export_dataset_click(self):
        """Handle export dataset button click."""
        self.export_dataset_requested.emit()

    # Public methods for MainWindow to call

    def set_sequence_info(self, duration_seconds: float, clip_count: int):
        """Update sequence information display."""
        self._sequence_duration = duration_seconds
        self._sequence_clip_count = clip_count

        # Format duration as HH:MM:SS
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        self.sequence_info_label.setText(f"Duration: {duration_str}    Clips: {clip_count}")
        self.export_sequence_btn.setEnabled(clip_count > 0)

        # Update state
        if clip_count > 0 or self._detected_clip_count > 0:
            self.state_stack.setCurrentIndex(self.STATE_READY)
        else:
            self.state_stack.setCurrentIndex(self.STATE_NO_CONTENT)

    def set_detected_clips_count(self, count: int):
        """Update detected clips count for enabling clip export buttons."""
        self._detected_clip_count = count
        has_clips = count > 0

        self.export_clips_btn.setEnabled(has_clips)
        self.export_all_clips_btn.setEnabled(has_clips)
        self.export_dataset_btn.setEnabled(has_clips)

        # Update state
        if has_clips or self._sequence_clip_count > 0:
            self.state_stack.setCurrentIndex(self.STATE_READY)
        else:
            self.state_stack.setCurrentIndex(self.STATE_NO_CONTENT)

    def get_quality_setting(self) -> str:
        """Get selected quality setting."""
        index = self.quality_combo.currentIndex()
        return ["high", "medium", "low"][index]

    def get_resolution_setting(self) -> tuple:
        """Get selected resolution setting as (width, height) or None for original."""
        index = self.resolution_combo.currentIndex()
        resolutions = [
            None,  # Original
            (1920, 1080),
            (1280, 720),
            (854, 480),
        ]
        return resolutions[index]

    def get_fps_setting(self) -> int:
        """Get selected FPS setting or None for original."""
        index = self.fps_combo.currentIndex()
        fps_values = [None, 60, 30, 24]
        return fps_values[index]

    def clear(self):
        """Clear render tab state for new project."""
        # Reset counts
        self._sequence_duration = 0
        self._sequence_clip_count = 0
        self._detected_clip_count = 0

        # Reset dropdowns to defaults
        self.quality_combo.setCurrentIndex(1)  # Medium
        self.resolution_combo.setCurrentIndex(0)  # Original
        self.fps_combo.setCurrentIndex(0)  # Original

        # Update display
        self.sequence_info_label.setText("Duration: 00:00:00    Clips: 0")
        self.export_sequence_btn.setEnabled(False)
        self.export_clips_btn.setEnabled(False)
        self.export_all_clips_btn.setEnabled(False)
        self.export_dataset_btn.setEnabled(False)

        # Show empty state
        self.state_stack.setCurrentIndex(self.STATE_NO_CONTENT)
