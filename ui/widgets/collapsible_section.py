"""Collapsible section widget with a header toggle and content container.

Used by FilterSidebar to group filters into expandable categories. The header
is a QToolButton with a rotating chevron; the content is a QFrame whose
visibility mirrors the expanded state.
"""

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import Radii, Spacing, TypeScale, theme


class CollapsibleSection(QWidget):
    """Titled section with a toggle that shows/hides its content widget.

    Signals:
        expanded_changed(bool): emitted only when state actually changes.
    """

    expanded_changed = Signal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = title
        self._expanded = True
        self._clear_callback: Optional[Callable[[], None]] = None
        self._setup_ui()
        self._refresh_theme()

        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.XXS)

        # Header row
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(Spacing.XS)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.DownArrow)
        self._toggle_btn.setText(self._title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.setAutoRaise(True)
        self._toggle_btn.clicked.connect(self._on_header_clicked)
        header_layout.addWidget(self._toggle_btn, 1)

        self._clear_btn = QPushButton("×")
        self._clear_btn.setFixedSize(18, 18)
        self._clear_btn.setVisible(False)
        self._clear_btn.setToolTip("Clear this section")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        header_layout.addWidget(self._clear_btn)

        outer.addWidget(header)

        # Content frame
        self._content_frame = QFrame()
        content_layout = QVBoxLayout(self._content_frame)
        content_layout.setContentsMargins(Spacing.MD, Spacing.XS, 0, Spacing.SM)
        content_layout.setSpacing(Spacing.SM)
        outer.addWidget(self._content_frame)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._toggle_btn.setChecked(expanded)
        self._toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._content_frame.setVisible(expanded)
        self.expanded_changed.emit(expanded)

    def setContentWidget(self, widget: QWidget) -> None:
        layout = self._content_frame.layout()
        # Remove any previous widgets
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        layout.addWidget(widget)

    def show_clear_indicator(
        self, visible: bool, on_click: Optional[Callable[[], None]] = None
    ) -> None:
        self._clear_callback = on_click
        self._clear_btn.setVisible(visible)

    # ── Slots ────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_header_clicked(self, checked: bool) -> None:
        self.set_expanded(checked)

    @Slot()
    def _on_clear_clicked(self) -> None:
        if self._clear_callback is not None:
            self._clear_callback()

    # ── Theming ──────────────────────────────────────────────────────

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self.setStyleSheet(
            f"""
            CollapsibleSection QToolButton {{
                color: {c.text_primary};
                background: transparent;
                border: none;
                font-size: {TypeScale.MD}px;
                font-weight: 600;
                padding: {Spacing.XS}px 0;
                text-align: left;
            }}
            CollapsibleSection QToolButton:hover {{
                color: {c.accent_blue};
            }}
            CollapsibleSection QPushButton {{
                background: {c.background_tertiary};
                color: {c.text_muted};
                border: none;
                border-radius: {Radii.SM}px;
                font-size: {TypeScale.XS}px;
                font-weight: 700;
                padding: 0;
            }}
            CollapsibleSection QPushButton:hover {{
                background: {c.accent_blue};
                color: {c.text_inverted};
            }}
            """
        )
