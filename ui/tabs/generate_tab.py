"""Generate tab for algorithmic remix features (stub)."""

from PySide6.QtWidgets import QVBoxLayout, QLabel
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from .base_tab import BaseTab
from ui.theme import theme


class GenerateTab(BaseTab):
    """Placeholder tab for future algorithmic remix features.

    This tab will eventually contain:
    - Shuffle with constraints
    - Similarity chaining
    - Beat-synced editing
    - Color-based sequencing
    """

    def _setup_ui(self):
        """Set up the Generate tab UI with 'Coming Soon' placeholder."""
        layout = QVBoxLayout()
        layout.addStretch()

        # Coming Soon title
        title_label = QLabel("Coming Soon")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {theme().text_secondary};")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Algorithmic remix features will appear here.")
        desc_label.setStyleSheet(f"color: {theme().text_muted}; margin-top: 10px;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Planned features list
        features_text = """
Planned capabilities:
• Shuffle with constraints
• Similarity chaining
• Beat-synced editing
• Color-based sequencing
        """
        features_label = QLabel(features_text.strip())
        features_label.setStyleSheet(f"color: {theme().text_muted}; margin-top: 20px;")
        features_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(features_label)

        layout.addStretch()
        self.setLayout(layout)
