"""Analyze tab for scene detection and clip browsing."""

from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtCore import Signal

from .base_tab import BaseTab


class AnalyzeTab(BaseTab):
    """Tab for detecting scenes and browsing detected clips.

    Signals:
        detect_requested: Emitted when detection is requested (threshold: float)
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
    """

    detect_requested = Signal(float)  # threshold
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip

    def _setup_ui(self):
        """Set up the Analyze tab UI."""
        layout = self._create_placeholder(
            "Analyze",
            "Detect scenes and browse clips"
        )
        self.setLayout(layout)
