"""Horizontal pill bar for filtering algorithms by category."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QButtonGroup
from PySide6.QtCore import Signal

from ui.algorithm_config import CATEGORY_ORDER
from ui.theme import theme, Radii, Spacing, TypeScale


class CategoryPillBar(QWidget):
    """Horizontal bar of pill-shaped category buttons.

    Displays one pill per entry in CATEGORY_ORDER. Only one pill can be
    selected at a time (exclusive QButtonGroup).

    Signals:
        category_changed: Emitted with the category name when the user
            clicks a *different* pill.  NOT emitted by programmatic
            ``set_category()`` calls.
    """

    category_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._current: str = CATEGORY_ORDER[0]
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._setup_ui()
        self._refresh_theme()

        # React to theme switches.
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    # ── Setup ────────────────────────────────────────────────────────

    def _setup_ui(self):
        """Build the pill layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # Center-align: stretch on both sides.
        layout.addStretch()

        for idx, name in enumerate(CATEGORY_ORDER):
            btn = QPushButton(name)
            btn.setCheckable(True)
            self._group.addButton(btn, idx)
            self._buttons[name] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Default selection: first pill ("All").
        first = CATEGORY_ORDER[0]
        self._buttons[first].setChecked(True)

        # Connect *after* initial check so we don't emit during init.
        self._group.idClicked.connect(self._on_group_clicked)

    # ── Interaction ──────────────────────────────────────────────────

    def _on_group_clicked(self, button_id: int):
        """Handle user click on a pill button."""
        name = CATEGORY_ORDER[button_id]
        if name == self._current:
            return
        self._current = name
        self.category_changed.emit(name)

    # ── Programmatic API ─────────────────────────────────────────────

    def set_category(self, name: str):
        """Select a pill without emitting ``category_changed``.

        If *name* is not a recognised category, falls back to ``"All"``.
        """
        if name not in self._buttons:
            name = "All"

        self._current = name
        self._group.blockSignals(True)
        self._buttons[name].setChecked(True)
        self._group.blockSignals(False)

    # ── Theming ──────────────────────────────────────────────────────

    def _refresh_theme(self):
        """Apply pill styling using current theme colours."""
        c = theme()
        self.setStyleSheet(f"""
            CategoryPillBar QPushButton {{
                background-color: {c.background_secondary};
                color: {c.text_secondary};
                border: 1px solid {c.border_secondary};
                border-radius: {Radii.FULL}px;
                padding: {Spacing.XS}px {Spacing.SM}px;
                font-size: {TypeScale.SM}px;
                min-height: 0;
            }}
            CategoryPillBar QPushButton:hover {{
                background-color: {c.background_tertiary};
            }}
            CategoryPillBar QPushButton:checked {{
                background-color: {c.accent_blue};
                color: {c.text_inverted};
                border: 1px solid {c.accent_blue};
            }}
            CategoryPillBar QPushButton:focus {{
                border: 1px solid {c.border_focus};
            }}
        """)
