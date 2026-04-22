"""Non-exclusive multi-select chip bar.

Used by FilterSidebar to expose enum-valued multi-select filters (shot type,
aspect ratio, gaze direction, etc.). Chips are pill-shaped pushbuttons; the
selected set is exposed via selected_values() and emitted on
selection_changed.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import Radii, Spacing, TypeScale, theme


class ChipGroup(QWidget):
    """Multi-select chip bar backed by a non-exclusive QButtonGroup.

    Signals:
        selection_changed(set): emitted with the new selection only when
            the selected values actually change.
    """

    selection_changed = Signal(set)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._selected: set[str] = set()
        self._emitting = False

        self._group = QButtonGroup(self)
        self._group.setExclusive(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.XXS)

        self._row = QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(Spacing.XS)
        self._row.addStretch()
        outer.addLayout(self._row)

        self._empty_label = QLabel("(no options)")
        self._empty_label.setVisible(False)
        outer.addWidget(self._empty_label)

        self._refresh_theme()
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    # ── Public API ───────────────────────────────────────────────────

    def set_options(self, options: list[tuple[str, str]]) -> None:
        """Replace chips. Clears existing selection (emits if non-empty)."""
        # Remove existing buttons
        for btn in list(self._buttons.values()):
            self._group.removeButton(btn)
            self._row.removeWidget(btn)
            btn.setParent(None)
        self._buttons.clear()

        had_selection = bool(self._selected)
        self._selected.clear()

        # Insert new chips before the trailing stretch
        for idx, (value, label) in enumerate(options):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("chip_value", value)
            btn.toggled.connect(self._on_chip_toggled)
            self._group.addButton(btn)
            self._buttons[value] = btn
            self._row.insertWidget(idx, btn)

        self._empty_label.setVisible(not options)
        self._refresh_theme()

        if had_selection and not self._emitting:
            self.selection_changed.emit(set())

    def set_selected(self, values: set[str]) -> None:
        """Programmatically check chips. Only emits if the set changes."""
        new = {v for v in values if v in self._buttons}
        if new == self._selected:
            return
        self._emitting = True
        try:
            for value, btn in self._buttons.items():
                btn.setChecked(value in new)
        finally:
            self._emitting = False
        self._selected = new
        self.selection_changed.emit(set(self._selected))

    def selected_values(self) -> set[str]:
        return set(self._selected)

    def clear_selection(self) -> None:
        if not self._selected:
            return
        self.set_selected(set())

    # ── Slots ────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_chip_toggled(self, checked: bool) -> None:
        if self._emitting:
            return
        btn = self.sender()
        if btn is None:
            return
        value = btn.property("chip_value")
        if not isinstance(value, str):
            return
        if checked:
            self._selected.add(value)
        else:
            self._selected.discard(value)
        self.selection_changed.emit(set(self._selected))

    # ── Theming ──────────────────────────────────────────────────────

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self.setStyleSheet(
            f"""
            ChipGroup QPushButton {{
                background-color: {c.background_tertiary};
                color: {c.text_primary};
                border: 1px solid {c.border_secondary};
                border-radius: {Radii.FULL}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-size: {TypeScale.XS}px;
                min-height: 0;
            }}
            ChipGroup QPushButton:hover {{
                border: 1px solid {c.accent_blue};
            }}
            ChipGroup QPushButton:checked {{
                background-color: {c.badge_analyzed};
                color: {c.badge_analyzed_text};
                border: 1px solid {c.badge_analyzed};
            }}
            ChipGroup QLabel {{
                color: {c.text_muted};
                font-size: {TypeScale.SM}px;
                font-style: italic;
            }}
            """
        )
