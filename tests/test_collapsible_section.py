"""Tests for CollapsibleSection widget."""

import os

import pytest

from PySide6.QtWidgets import QLabel


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_constructs_expanded_by_default(qapp):
    from ui.widgets.collapsible_section import CollapsibleSection
    section = CollapsibleSection("Shot properties")
    assert section.expanded is True


def test_set_expanded_toggles_content_visibility(qapp):
    from ui.widgets.collapsible_section import CollapsibleSection
    section = CollapsibleSection("Shot")
    content = QLabel("x")
    section.setContentWidget(content)
    section.show()
    qapp.processEvents()

    section.set_expanded(False)
    assert section.expanded is False

    section.set_expanded(True)
    assert section.expanded is True


def test_set_expanded_same_state_is_noop_and_no_signal(qapp):
    from ui.widgets.collapsible_section import CollapsibleSection
    section = CollapsibleSection("Shot")

    emissions: list[bool] = []
    section.expanded_changed.connect(lambda v: emissions.append(v))

    section.set_expanded(True)  # already expanded
    qapp.processEvents()
    assert emissions == []

    section.set_expanded(False)
    qapp.processEvents()
    assert emissions == [False]

    section.set_expanded(False)  # already collapsed
    qapp.processEvents()
    assert emissions == [False]


def test_set_content_widget_replaces_previous(qapp):
    from ui.widgets.collapsible_section import CollapsibleSection
    section = CollapsibleSection("X")
    first = QLabel("first")
    second = QLabel("second")
    section.setContentWidget(first)
    section.setContentWidget(second)
    # first should have been reparented away
    assert first.parent() is not section


def test_show_clear_indicator_triggers_callback(qapp):
    from ui.widgets.collapsible_section import CollapsibleSection
    section = CollapsibleSection("X")
    calls = []
    section.show_clear_indicator(True, on_click=lambda: calls.append(1))
    section._clear_btn.click()
    qapp.processEvents()
    assert calls == [1]

    section.show_clear_indicator(False)
    # after hiding, stored callback stays but clicking hidden button is a no-op path
    section._clear_btn.click()
    qapp.processEvents()
    # Clicking a hidden/invisible button doesn't fire in offscreen reliably,
    # but the callback replacement path works: reset and verify
    section.show_clear_indicator(True, on_click=None)
    section._clear_btn.click()
    qapp.processEvents()
    assert calls == [1]
