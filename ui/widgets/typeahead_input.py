"""Typeahead input backed by QCompleter over a QStringListModel.

Used for searchable vocabularies (e.g. ImageNet's 1000 classes) where a flat
combo box would be unusable. Emits `value_selected` with the vocabulary's
canonical form when the user commits a match via the completion popup or
Enter. Does not emit on partial typing.
"""

from typing import Optional

from PySide6.QtCore import QStringListModel, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLineEdit,
    QWidget,
)

from ui.theme import Spacing, TypeScale, UISizes, theme


class TypeaheadInput(QWidget):
    """QLineEdit + QCompleter over a user-provided vocabulary.

    Signals:
        value_selected(str): emitted when the user commits a vocabulary
            entry (via popup click or Enter). Input is cleared after emit.
    """

    value_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._vocab: list[str] = []
        self._lookup: dict[str, str] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)

        self._line = QLineEdit()
        self._line.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)

        self._model = QStringListModel([], self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._line.setCompleter(self._completer)

        self._completer.activated[str].connect(self._on_activated)
        self._line.returnPressed.connect(self._on_return_pressed)

        layout.addWidget(self._line)

        self._refresh_theme()
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    # ── Public API ───────────────────────────────────────────────────

    def set_vocabulary(self, items: list[str]) -> None:
        self._vocab = list(items)
        self._lookup = {v.casefold(): v for v in self._vocab}
        self._model.setStringList(self._vocab)

    def clear_input(self) -> None:
        self._line.clear()

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 Qt style
        self._line.setPlaceholderText(text)

    # ── Slots ────────────────────────────────────────────────────────

    @Slot(str)
    def _on_activated(self, text: str) -> None:
        canonical = self._lookup.get(text.casefold())
        if canonical is None:
            return
        self._line.clear()
        self.value_selected.emit(canonical)

    @Slot()
    def _on_return_pressed(self) -> None:
        text = self._line.text().strip()
        if not text:
            return
        canonical = self._lookup.get(text.casefold())
        if canonical is None:
            return
        self._line.clear()
        self.value_selected.emit(canonical)

    # ── Theming ──────────────────────────────────────────────────────

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self._line.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {c.background_tertiary};
                color: {c.text_primary};
                border: 1px solid {c.border_secondary};
                border-radius: 4px;
                padding: 0 {Spacing.SM}px;
                font-size: {TypeScale.SM}px;
            }}
            QLineEdit:focus {{
                border: 1px solid {c.border_focus};
            }}
            """
        )
