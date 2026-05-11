"""Tests for SortingCardGrid category filtering and reflow."""

import sys
import pytest
from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication

from ui.algorithm_config import ALGORITHM_CONFIG
from ui.widgets.sorting_card_grid import SortingCardGrid

# Ensure a QApplication exists for widget tests
app = QApplication.instance() or QApplication(sys.argv)


EXPECTED_COUNTS = {
    "All": 23,
    "Arrange": 10,  # +1 for cassette_tape
    "Find": 4,
    "Connect": 6,  # +1 for free_association
    "Audio": 4,  # +1 for cassette_tape
    "Text": 6,  # exquisite_corpus, storyteller, free_association, reference_guided, word_sequencer, word_llm_composer
}

ARRANGE_KEYS = {
    "shuffle", "sequential", "duration", "color",
    "brightness", "volume", "shot_type", "proximity", "gaze_sort",
    "cassette_tape",
}

TEXT_KEYS = {
    "exquisite_corpus",
    "storyteller",
    "free_association",
    "reference_guided",
    "word_sequencer",
    "word_llm_composer",
}


def _visible_keys(grid: SortingCardGrid) -> set[str]:
    """Return set of algorithm keys whose cards are currently in the grid layout."""
    in_layout = set()
    for i in range(grid._grid_layout.count()):
        item = grid._grid_layout.itemAt(i)
        if item and item.widget():
            in_layout.add(item.widget().key)
    return in_layout


class TestCategoryFiltering:

    def setup_method(self):
        self.grid = SortingCardGrid()

    def test_all_shows_every_registered_algorithm(self):
        self.grid.set_category("All")
        assert _visible_keys(self.grid) == set(ALGORITHM_CONFIG.keys())

    def test_arrange_shows_expected_cards(self):
        self.grid.set_category("Arrange")
        visible = _visible_keys(self.grid)
        assert visible == ARRANGE_KEYS
        assert len(visible) == 10

    def test_text_shows_2_cards(self):
        self.grid.set_category("Text")
        assert _visible_keys(self.grid) == TEXT_KEYS

    def test_multi_tagged_algorithm_in_both_categories(self):
        self.grid.set_category("Arrange")
        assert "volume" in _visible_keys(self.grid)
        self.grid.set_category("Audio")
        assert "volume" in _visible_keys(self.grid)

    def test_category_card_counts(self):
        for category, expected in EXPECTED_COUNTS.items():
            self.grid.set_category(category)
            assert len(_visible_keys(self.grid)) == expected, (
                f"{category} expected {expected} cards"
            )

    def test_card_order_preserved(self):
        self.grid.set_category("Arrange")
        # Get the visible cards in grid position order
        positions = []
        for i in range(self.grid._grid_layout.count()):
            item = self.grid._grid_layout.itemAt(i)
            if item and item.widget():
                positions.append(item.widget().key)
        # Should match the order from _positions, filtered
        expected_order = [k for k in self.grid._positions if k in ARRANGE_KEYS]
        assert positions == expected_order

    def test_reflow_text_to_all(self):
        self.grid.set_category("Text")
        assert len(_visible_keys(self.grid)) == 6
        self.grid.set_category("All")
        assert len(_visible_keys(self.grid)) == 23

    def test_selection_cleared_on_hidden_card(self):
        # Select a card in Arrange
        self.grid._on_card_clicked("volume")
        assert self.grid._selected_key == "volume"
        # Switch to Text — volume is not in Text
        self.grid.set_category("Text")
        assert self.grid._selected_key is None

    def test_selection_preserved_when_card_stays_visible(self):
        self.grid._on_card_clicked("volume")
        # Switch to Audio — volume IS in Audio
        self.grid.set_category("Audio")
        assert self.grid._selected_key == "volume"


class TestAvailabilityWithCategories:

    def setup_method(self):
        self.grid = SortingCardGrid()

    def test_availability_persists_across_category_switch(self):
        self.grid.set_algorithm_availability({"volume": (False, "needs analysis")})
        assert not self.grid._cards["volume"].is_enabled()
        self.grid.set_category("Audio")
        assert not self.grid._cards["volume"].is_enabled()
        self.grid.set_category("All")
        assert not self.grid._cards["volume"].is_enabled()

    def test_disabled_cards_shown_not_hidden(self):
        """Text category with all disabled algorithms shows disabled cards in layout."""
        self.grid.set_algorithm_availability({
            "exquisite_corpus": (False, "needs extract_text"),
            "storyteller": (False, "needs describe"),
        })
        self.grid.set_category("Text")
        in_layout = _visible_keys(self.grid)
        assert "exquisite_corpus" in in_layout
        assert "storyteller" in in_layout
        assert not self.grid._cards["exquisite_corpus"].is_enabled()
        assert not self.grid._cards["storyteller"].is_enabled()


class TestSignals:

    def setup_method(self):
        self.grid = SortingCardGrid()

    def test_card_click_emits_algorithm_selected(self):
        handler = MagicMock()
        self.grid.algorithm_selected.connect(handler)
        self.grid._on_card_clicked("shuffle")
        handler.assert_called_once_with("shuffle")

    def test_category_changed_emits_on_pill_click(self):
        handler = MagicMock()
        self.grid.category_changed.connect(handler)
        # Simulate user clicking "Text" pill
        self.grid._pill_bar._current = "All"  # ensure starting state
        self.grid._pill_bar._on_group_clicked(5)  # Text is index 5
        handler.assert_called_once_with("Text")
