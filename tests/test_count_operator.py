"""Tests for CountOperator widget."""

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


def test_default_state_is_inactive(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    assert widget.value() is None


def test_set_value_activates_and_emits(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()

    emitted: list[tuple] = []
    widget.value_changed.connect(lambda op, n: emitted.append((op, n)))

    widget.set_value(">", 5)
    qapp.processEvents()

    assert widget.value() == (">", 5)
    assert emitted == [(">", 5)]


def test_clear_resets_and_emits_none(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    widget.set_value(">", 3)

    emitted: list[tuple] = []
    widget.value_changed.connect(lambda op, n: emitted.append((op, n)))

    widget.clear()
    qapp.processEvents()

    assert widget.value() is None
    assert emitted == [(None, None)]


def test_clear_when_already_inactive_is_noop(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    emitted: list[tuple] = []
    widget.value_changed.connect(lambda op, n: emitted.append((op, n)))

    widget.clear()
    qapp.processEvents()
    assert emitted == []


def test_set_value_with_none_clears(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    widget.set_value(">", 2)

    widget.set_value(None, None)
    assert widget.value() is None


def test_invalid_operator_coerces_to_default(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    widget.set_value("??", 1)
    assert widget.value() == (">", 1)


def test_user_edit_via_spinbox_activates(qapp):
    from ui.widgets.count_operator import CountOperator
    widget = CountOperator()
    emitted: list[tuple] = []
    widget.value_changed.connect(lambda op, n: emitted.append((op, n)))

    # Simulate user changing the spin value
    widget._spin.setValue(4)
    qapp.processEvents()

    assert widget.value() == (">", 4)
    assert emitted[-1] == (">", 4)
