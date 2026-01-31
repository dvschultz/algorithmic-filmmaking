"""Grid of sorting algorithm cards for the Sequence tab."""

from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ui.theme import theme
from ui.widgets.sorting_card import SortingCard


# Algorithm definitions: key -> (icon, title, description)
ALGORITHMS = {
    "color": (
        "🎨",
        "Color",
        "Sort clips by dominant color along the color wheel"
    ),
    "duration": (
        "⏱️",
        "Duration",
        "Sort clips by length (shortest or longest first)"
    ),
    "shuffle": (
        "🎲",
        "Shuffle",
        "Randomize clip order with no repeating sources"
    ),
    "sequential": (
        "📋",
        "Sequential",
        "Keep clips in their original detection order"
    ),
    "exquisite_corpus": (
        "📝",
        "Exquisite Corpus",
        "Create a poem from on-screen text"
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
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Choose a Sorting Method")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {theme().text_primary}; margin-bottom: 16px;")
        main_layout.addWidget(header)
        self._header = header

        # Subheader
        subheader = QLabel("Select how you want to arrange your clips")
        subheader.setAlignment(Qt.AlignCenter)
        subheader.setStyleSheet(f"color: {theme().text_secondary}; margin-bottom: 24px;")
        main_layout.addWidget(subheader)
        self._subheader = subheader

        main_layout.addStretch()

        # Grid container (centered)
        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Create cards in 3x2 grid (5 algorithms)
        positions = [
            ("color", 0, 0),
            ("duration", 0, 1),
            ("shuffle", 0, 2),
            ("sequential", 1, 0),
            ("exquisite_corpus", 1, 1),
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
        self._header.setStyleSheet(f"color: {theme().text_primary}; margin-bottom: 16px;")
        self._subheader.setStyleSheet(f"color: {theme().text_secondary}; margin-bottom: 24px;")
