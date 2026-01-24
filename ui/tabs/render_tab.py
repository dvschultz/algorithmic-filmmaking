"""Render tab for export configuration and rendering."""

from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtCore import Signal

from .base_tab import BaseTab


class RenderTab(BaseTab):
    """Tab for configuring and exporting the final sequence.

    Signals:
        export_sequence_requested: Emitted when sequence export is requested
        export_clips_requested: Emitted when clip export is requested
        export_dataset_requested: Emitted when dataset export is requested
    """

    export_sequence_requested = Signal()
    export_clips_requested = Signal()
    export_dataset_requested = Signal()

    def _setup_ui(self):
        """Set up the Render tab UI."""
        layout = self._create_placeholder(
            "Render",
            "Configure export settings and render"
        )
        self.setLayout(layout)
