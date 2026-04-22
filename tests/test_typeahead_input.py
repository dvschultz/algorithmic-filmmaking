"""Tests for TypeaheadInput widget."""

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


def test_commits_vocabulary_match_via_return(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary(["dog", "cat", "bird"])

    emitted: list[str] = []
    widget.value_selected.connect(lambda v: emitted.append(v))

    widget._line.setText("dog")
    widget._line.returnPressed.emit()
    qapp.processEvents()

    assert emitted == ["dog"]
    assert widget._line.text() == ""


def test_case_insensitive_match_emits_canonical(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary(["dog", "cat"])

    emitted: list[str] = []
    widget.value_selected.connect(lambda v: emitted.append(v))

    widget._line.setText("DOG")
    widget._line.returnPressed.emit()
    qapp.processEvents()
    assert emitted == ["dog"]


def test_non_vocabulary_match_does_not_emit(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary(["dog", "cat"])

    emitted: list[str] = []
    widget.value_selected.connect(lambda v: emitted.append(v))

    widget._line.setText("nonexistent")
    widget._line.returnPressed.emit()
    qapp.processEvents()
    assert emitted == []
    # Input retains text so the user can see what they typed
    assert widget._line.text() == "nonexistent"


def test_empty_input_does_not_emit(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary(["dog"])

    emitted: list[str] = []
    widget.value_selected.connect(lambda v: emitted.append(v))

    widget._line.setText("")
    widget._line.returnPressed.emit()
    qapp.processEvents()
    assert emitted == []


def test_empty_vocabulary_never_emits(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary([])

    emitted: list[str] = []
    widget.value_selected.connect(lambda v: emitted.append(v))

    widget._line.setText("anything")
    widget._line.returnPressed.emit()
    qapp.processEvents()
    assert emitted == []


def test_set_vocabulary_updates_completer(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget.set_vocabulary(["dog"])
    assert widget._model.stringList() == ["dog"]
    widget.set_vocabulary(["cat", "bird"])
    assert widget._model.stringList() == ["cat", "bird"]


def test_clear_input(qapp):
    from ui.widgets.typeahead_input import TypeaheadInput
    widget = TypeaheadInput()
    widget._line.setText("hello")
    widget.clear_input()
    assert widget._line.text() == ""
