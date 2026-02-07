"""Grid of sorting algorithm cards for the Sequence tab."""

from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ui.theme import theme, Spacing, TypeScale
from ui.widgets.sorting_card import SortingCard


# Algorithm definitions: key -> (icon, title, description)
ALGORITHMS = {
    "color": (
        "🎨",
        "Chromatic Flow",
        "Arrange clips along a color gradient"
    ),
    "color_cycle": (
        "🌈",
        "Color Cycle",
        "Curate clips with strong color identity and cycle through the spectrum"
    ),
    "duration": (
        "⏱️",
        "Tempo Shift",
        "Order clips from shortest to longest (or reverse)"
    ),
    "brightness": (
        "🌗",
        "Into the Dark",
        "Arrange clips from light to shadow, or shadow to light"
    ),
    "volume": (
        "🔊",
        "Crescendo",
        "Build from silence to thunder, or thunder to silence"
    ),
    "shuffle": (
        "🎲",
        "Dice Roll",
        "Randomly shuffle clips into a new order"
    ),
    "sequential": (
        "📋",
        "Time Capsule",
        "Keep clips in their original order"
    ),
    "shot_type": (
        "🎬",
        "Focal Ladder",
        "Arrange clips by camera shot scale"
    ),
    "proximity": (
        "🔭",
        "Up Close and Personal",
        "Glide from distant vistas to intimate close-ups"
    ),
    "similarity_chain": (
        "🔗",
        "Human Centipede",
        "Chain clips together by visual similarity — each cut flows into the next"
    ),
    "match_cut": (
        "✂️",
        "Match Cut",
        "Find hidden connections between clips — where one ending meets another's beginning"
    ),
    "exquisite_corpus": (
        "📝",
        "Exquisite Corpus",
        "Generate a poem from on-screen text"
    ),
    "storyteller": (
        "📖",
        "Storyteller",
        "Create a narrative from clip descriptions"
    ),
}


class SortingCardGrid(QWidget):
    """Grid of sorting algorithm cards.

    Displays available sorting algorithms as clickable cards in a 2x2 grid.
    Emits signal when an algorithm is selected.

    Signals:
        algorithm_selected: Emitted with algorithm key when a card is clicked
    """

    algorithm_selected = Signal(str)  # algorithm key

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, SortingCard] = {}
        self._selected_key: str | None = None
        self._setup_ui()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the grid UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)

        # Header
        header = QLabel("Choose a Sorting Method")
        header_font = QFont()
        header_font.setPointSize(TypeScale.XL)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {theme().text_primary}; margin-bottom: {Spacing.LG}px;")
        main_layout.addWidget(header)
        self._header = header

        # Subheader
        subheader = QLabel("Select how you want to arrange your clips")
        subheader.setAlignment(Qt.AlignCenter)
        subheader.setStyleSheet(f"color: {theme().text_secondary}; margin-bottom: {Spacing.XL}px;")
        main_layout.addWidget(subheader)
        self._subheader = subheader

        main_layout.addStretch()

        # Grid container (centered)
        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(Spacing.XL)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Create cards in grid layout (13 algorithms, 4 columns)
        # Row 0: basic sorting
        # Row 1: color/brightness/volume
        # Row 2: shot analysis + similarity
        # Row 3: text/narrative
        positions = [
            ("shuffle", 0, 0),
            ("sequential", 0, 1),
            ("duration", 0, 2),
            ("color", 0, 3),
            ("color_cycle", 1, 0),
            ("brightness", 1, 1),
            ("volume", 1, 2),
            ("shot_type", 1, 3),
            ("proximity", 2, 0),
            ("similarity_chain", 2, 1),
            ("match_cut", 2, 2),
            ("exquisite_corpus", 2, 3),
            ("storyteller", 3, 0),
        ]

        for key, row, col in positions:
            icon, title, description = ALGORITHMS[key]
            card = SortingCard(key, icon, title, description)
            card.clicked.connect(self._on_card_clicked)
            grid_layout.addWidget(card, row, col, Qt.AlignCenter)
            self._cards[key] = card

        # Center the grid
        center_layout = QVBoxLayout()
        center_layout.addWidget(grid_container, 0, Qt.AlignCenter)
        main_layout.addLayout(center_layout)

        main_layout.addStretch()

    def _on_card_clicked(self, key: str):
        """Handle card click."""
        # Deselect previous
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)

        # Select new
        self._selected_key = key
        if key in self._cards:
            self._cards[key].set_selected(True)

        self.algorithm_selected.emit(key)

    def set_algorithm_availability(self, available: dict[str, bool | tuple[bool, str]]):
        """Enable/disable cards based on clip analysis state.

        Args:
            available: Dict mapping algorithm key to either:
                - bool: True if available, False otherwise
                - tuple[bool, str]: (available, reason) where reason explains
                  why the algorithm is unavailable
        """
        for key, status in available.items():
            if key in self._cards:
                if isinstance(status, tuple):
                    enabled, reason = status
                    self._cards[key].set_enabled(enabled, reason)
                else:
                    self._cards[key].set_enabled(status)

    def get_selected(self) -> str | None:
        """Get the currently selected algorithm key."""
        return self._selected_key

    def clear_selection(self):
        """Clear the current selection."""
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)
        self._selected_key = None

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._header.setStyleSheet(f"color: {theme().text_primary}; margin-bottom: {Spacing.LG}px;")
        self._subheader.setStyleSheet(f"color: {theme().text_secondary}; margin-bottom: {Spacing.XL}px;")
