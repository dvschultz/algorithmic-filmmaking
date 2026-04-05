"""Tests for CategoryPillBar widget."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_creates_buttons_matching_category_order(qapp):
    """Widget creates 6 buttons matching CATEGORY_ORDER."""
    from ui.algorithm_config import CATEGORY_ORDER
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()
    assert len(bar._buttons) == len(CATEGORY_ORDER)
    for name in CATEGORY_ORDER:
        assert name in bar._buttons
        assert bar._buttons[name].isCheckable()


def test_clicking_pill_emits_category_changed(qapp):
    """Clicking a pill emits category_changed with the category name."""
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()
    received: list[str] = []
    bar.category_changed.connect(received.append)

    # Click the "Audio" pill.
    bar._buttons["Audio"].click()

    assert received == ["Audio"]


def test_set_category_does_not_emit(qapp):
    """set_category('Audio') selects the Audio pill WITHOUT emitting."""
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()
    received: list[str] = []
    bar.category_changed.connect(received.append)

    bar.set_category("Audio")

    assert bar._buttons["Audio"].isChecked()
    assert received == []


def test_clicking_already_selected_pill_does_not_reemit(qapp):
    """Clicking the already-selected pill does not re-emit category_changed."""
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()
    received: list[str] = []

    # First select "Audio" via click.
    bar._buttons["Audio"].click()
    bar.category_changed.connect(received.append)

    # Click "Audio" again — should NOT emit.
    bar._buttons["Audio"].click()

    assert received == []


def test_only_one_pill_checked_at_a_time(qapp):
    """Only one pill is checked at a time (exclusive selection)."""
    from ui.algorithm_config import CATEGORY_ORDER
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()

    bar._buttons["Audio"].click()

    checked = [name for name in CATEGORY_ORDER if bar._buttons[name].isChecked()]
    assert checked == ["Audio"]


def test_set_category_nonexistent_falls_back_to_all(qapp):
    """set_category('nonexistent') falls back to 'All'."""
    from ui.widgets.category_pill_bar import CategoryPillBar

    bar = CategoryPillBar()
    bar._buttons["Audio"].click()  # move away from All first

    bar.set_category("nonexistent")

    assert bar._buttons["All"].isChecked()
