"""Grid of sorting algorithm cards for the Sequence tab.

Displays 19 sorting algorithms as clickable cards, filterable by category
via a pill bar.
"""

from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QMouseEvent

from ui.theme import theme, Spacing, TypeScale
from ui.widgets.sorting_card import SortingCard
from ui.widgets.category_pill_bar import CategoryPillBar
from ui.algorithm_config import ALGORITHM_CONFIG


class SortingCardGrid(QWidget):
    """Grid of sorting algorithm cards with category filtering.

    Displays available sorting algorithms as clickable cards in a 4-column
    grid.  A pill bar above the grid lets users filter by category.

    Signals:
        algorithm_selected: Emitted with algorithm key when a card is clicked
        category_changed: Emitted with category name when the user switches
    """

    algorithm_selected = Signal(str)  # algorithm key
    category_changed = Signal(str)  # category name

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

        # Category pill bar (replaces the old subheader)
        self._pill_bar = CategoryPillBar()
        self._pill_bar.category_changed.connect(self._on_category_changed)
        main_layout.addWidget(self._pill_bar)

        main_layout.addStretch()

        # Grid container (centered)
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(Spacing.XL)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)

        # Master position order — defines canonical card ordering
        self._positions = [
            "shuffle", "sequential", "duration", "color",
            "brightness", "volume", "shot_type", "proximity",
            "similarity_chain", "match_cut", "exquisite_corpus", "storyteller",
            "free_association", "reference_guided", "signature_style",
            "rose_hobart", "staccato",
            "gaze_sort", "gaze_consistency", "eyes_without_a_face",
        ]

        # Create all cards (but don't add to layout yet)
        for key in self._positions:
            cfg = ALGORITHM_CONFIG[key]
            icon, title, description = cfg["icon"], cfg["label"], cfg["description"]
            card = SortingCard(key, icon, title, description)
            card.clicked.connect(self._on_card_clicked)
            self._cards[key] = card

        # Let clicks on empty grid space propagate to this widget
        self._grid_container.mousePressEvent = lambda e: e.ignore()

        # Center the grid
        center_layout = QVBoxLayout()
        center_layout.addWidget(self._grid_container, 0, Qt.AlignCenter)
        main_layout.addLayout(center_layout)

        main_layout.addStretch()

        # Initial grid build shows all cards
        self._rebuild_grid("All")

    # ── Category filtering ──────────────────────────────────────────

    def _on_category_changed(self, category: str):
        """Handle pill bar category change from user interaction."""
        self._rebuild_grid(category)
        self.category_changed.emit(category)

    def _rebuild_grid(self, category: str):
        """Clear and rebuild the grid layout for the given category.

        Cards are persistent objects reused across rebuilds — never
        call deleteLater() on them.
        """
        # Determine visible keys
        if category == "All":
            visible_keys = set(self._positions)
        else:
            cat_lower = category.lower()
            visible_keys = {
                key for key in self._positions
                if cat_lower in ALGORITHM_CONFIG[key].get("categories", [])
            }

        # Clear selection if the selected card is not in the new set
        if self._selected_key and self._selected_key not in visible_keys:
            self.clear_selection()

        # Drain the layout without destroying widgets
        while self._grid_layout.count():
            self._grid_layout.takeAt(0)

        # Re-add visible cards in sequential 4-column positions
        i = 0
        for key in self._positions:
            card = self._cards[key]
            if key in visible_keys:
                card.setVisible(True)
                self._grid_layout.addWidget(card, i // 6, i % 6, Qt.AlignCenter)
                i += 1
            else:
                card.setVisible(False)

    def set_category(self, category: str):
        """Programmatically set the active category.

        Delegates to the pill bar (which does not emit on programmatic set)
        and rebuilds the grid.
        """
        self._pill_bar.set_category(category)
        self._rebuild_grid(category)

    # ── Card interaction ────────────────────────────────────────────

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

        Updates ALL cards regardless of current category filter so that
        switching categories always shows correct enabled/disabled state.
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

    def mousePressEvent(self, event: QMouseEvent):
        """Clicking empty space deselects the current card."""
        if event.button() == Qt.LeftButton:
            self.clear_selection()
        super().mousePressEvent(event)

    def clear_selection(self):
        """Clear the current selection."""
        if self._selected_key and self._selected_key in self._cards:
            self._cards[self._selected_key].set_selected(False)
        self._selected_key = None

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._header.setStyleSheet(f"color: {theme().text_primary}; margin-bottom: {Spacing.LG}px;")
