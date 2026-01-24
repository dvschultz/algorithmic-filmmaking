"""Sequence tab for timeline editing and playback."""

from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtCore import Signal

from .base_tab import BaseTab


class SequenceTab(BaseTab):
    """Tab for arranging clips on the timeline and previewing.

    Signals:
        playback_requested: Emitted when playback is requested (start_frame: int)
        stop_requested: Emitted when stop is requested
        export_requested: Emitted when export is requested
    """

    playback_requested = Signal(int)  # start_frame
    stop_requested = Signal()
    export_requested = Signal()

    def _setup_ui(self):
        """Set up the Sequence tab UI."""
        layout = self._create_placeholder(
            "Sequence",
            "Arrange clips on the timeline"
        )
        self.setLayout(layout)
