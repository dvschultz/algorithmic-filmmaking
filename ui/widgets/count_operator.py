"""Compound widget for discrete count filters: operator picker + int input.

Used for filters like "person count > 1" or "object count = 3". Combines a
QComboBox of operators (>, =, <) with a QSpinBox for the threshold and a
Clear button that resets both.
"""

from typing import Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ui.theme import Spacing, TypeScale, UISizes, theme


OPERATORS = (">", "=", "<")


class CountOperator(QWidget):
    """Operator + integer compound. value() returns ``(op, n)`` or ``None``.

    Signals:
        value_changed(object, object): ``(op, n)`` or ``(None, None)``.
    """

    value_changed = Signal(object, object)

    def __init__(
        self,
        min_value: int = 0,
        max_value: int = 999,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._active = False
        self._emitting_internal = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)

        self._op_combo = QComboBox()
        self._op_combo.addItems(OPERATORS)
        self._op_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._op_combo.setFixedWidth(56)
        self._op_combo.setCurrentIndex(0)
        self._op_combo.currentTextChanged.connect(self._on_user_edit)
        layout.addWidget(self._op_combo)

        self._spin = QSpinBox()
        self._spin.setRange(min_value, max_value)
        self._spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._spin.valueChanged.connect(self._on_user_edit)
        layout.addWidget(self._spin)

        self._clear_btn = QPushButton("×")
        self._clear_btn.setFixedWidth(24)
        self._clear_btn.setToolTip("Clear")
        self._clear_btn.clicked.connect(self.clear)
        layout.addWidget(self._clear_btn)

        self._refresh_theme()
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    # ── Public API ───────────────────────────────────────────────────

    def value(self) -> Optional[tuple[str, int]]:
        if not self._active:
            return None
        return (self._op_combo.currentText(), self._spin.value())

    def set_value(self, op: Optional[str], n: Optional[int]) -> None:
        if op is None or n is None:
            self.clear()
            return
        if op not in OPERATORS:
            op = ">"
        self._emitting_internal = True
        try:
            self._op_combo.setCurrentText(op)
            self._spin.setValue(n)
        finally:
            self._emitting_internal = False
        was_active = self._active
        self._active = True
        if not was_active or self.value() != (op, n):
            self.value_changed.emit(op, n)

    @Slot()
    def clear(self) -> None:
        was_active = self._active
        self._active = False
        self._emitting_internal = True
        try:
            self._op_combo.setCurrentIndex(0)
            self._spin.setValue(self._spin.minimum())
        finally:
            self._emitting_internal = False
        if was_active:
            self.value_changed.emit(None, None)

    # ── Slots ────────────────────────────────────────────────────────

    @Slot()
    def _on_user_edit(self, *_args) -> None:
        if self._emitting_internal:
            return
        self._active = True
        op, n = self._op_combo.currentText(), self._spin.value()
        self.value_changed.emit(op, n)

    # ── Theming ──────────────────────────────────────────────────────

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self.setStyleSheet(
            f"""
            QComboBox, QSpinBox {{
                background-color: {c.background_tertiary};
                color: {c.text_primary};
                border: 1px solid {c.border_secondary};
                border-radius: 4px;
                padding: 0 {Spacing.XS}px;
                font-size: {TypeScale.SM}px;
            }}
            QComboBox:focus, QSpinBox:focus {{
                border: 1px solid {c.border_focus};
            }}
            QPushButton {{
                background: {c.background_tertiary};
                color: {c.text_muted};
                border: 1px solid {c.border_secondary};
                border-radius: 4px;
                font-size: {TypeScale.SM}px;
            }}
            QPushButton:hover {{
                background: {c.accent_blue};
                color: {c.text_inverted};
                border: 1px solid {c.accent_blue};
            }}
            """
        )
