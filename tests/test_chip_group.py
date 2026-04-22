"""Tests for ChipGroup widget."""

import os

import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_toggle_chip_updates_selection(qapp):
    from ui.widgets.chip_group import ChipGroup
    group = ChipGroup()
    group.set_options([("a", "A"), ("b", "B")])

    emissions: list[set] = []
    group.selection_changed.connect(lambda s: emissions.append(set(s)))

    # Simulate user toggling chip "a"
    group._buttons["a"].setChecked(True)
    qapp.processEvents()

    assert group.selected_values() == {"a"}
    assert emissions[-1] == {"a"}


def test_set_selected_checks_chips_without_loop(qapp):
    from ui.widgets.chip_group import ChipGroup
    group = ChipGroup()
    group.set_options([("a", "A"), ("b", "B"), ("c", "C")])

    emissions: list[set] = []
    group.selection_changed.connect(lambda s: emissions.append(set(s)))

    group.set_selected({"a", "b"})
    qapp.processEvents()

    assert group.selected_values() == {"a", "b"}
    assert emissions == [{"a", "b"}]
    assert group._buttons["a"].isChecked()
    assert group._buttons["b"].isChecked()
    assert not group._buttons["c"].isChecked()


def test_set_selected_unknown_values_silently_skipped(qapp):
    from ui.widgets.chip_group import ChipGroup
    group = ChipGroup()
    group.set_options([("a", "A")])

    emissions: list[set] = []
    group.selection_changed.connect(lambda s: emissions.append(set(s)))

    group.set_selected({"nonexistent"})
    qapp.processEvents()
    # No valid match → no state change → no emission
    assert group.selected_values() == set()
    assert emissions == []


def test_clear_selection_emits_only_when_selection_existed(qapp):
    from ui.widgets.chip_group import ChipGroup
    group = ChipGroup()
    group.set_options([("a", "A"), ("b", "B")])

    emissions: list[set] = []
    group.selection_changed.connect(lambda s: emissions.append(set(s)))

    group.clear_selection()
    qapp.processEvents()
    assert emissions == []  # nothing was selected

    group.set_selected({"a"})
    qapp.processEvents()
    group.clear_selection()
    qapp.processEvents()
    assert emissions[-1] == set()


def test_set_options_after_selection_clears_selection(qapp):
    from ui.widgets.chip_group import ChipGroup
    group = ChipGroup()
    group.set_options([("a", "A")])
    group.set_selected({"a"})
    qapp.processEvents()
    assert group.selected_values() == {"a"}

    group.set_options([("x", "X"), ("y", "Y")])
    qapp.processEvents()
    assert group.selected_values() == set()


def test_theme_refresh_does_not_crash(qapp):
    from ui.widgets.chip_group import ChipGroup
    from ui.theme import theme
    group = ChipGroup()
    group.set_options([("a", "A")])
    # Just trigger the refresh; assert stylesheet non-empty
    group._refresh_theme()
    assert "ChipGroup QPushButton" in group.styleSheet()
