"""Sorting algorithm card widget for the Sequence tab."""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt, QRectF
from PySide6.QtGui import QFont, QKeyEvent, QPainter, QColor

from ui.theme import theme, TypeScale, Spacing, Radii
from ui.gradient_glow import paint_gradient_glow, paint_card_body


class SortingCard(QFrame):
    """A clickable card for selecting a sorting algorithm.

    Displays an icon, title, and description for a sorting algorithm.
    Supports hover, selected, and disabled states with gradient glow.
    Keyboard accessible via Tab/Enter/Space.

    Signals:
        clicked: Emitted with the algorithm key when card is clicked
    """

    clicked = Signal(str)  # algorithm key

    def __init__(
        self,
        key: str,
        icon: str,
        title: str,
        description: str,
        parent=None
    ):
        super().__init__(parent)
        self.key = key
        self._icon = icon
        self._title = title
        self._description = description
        self._selected = False
        self._enabled = True
        self._disabled_reason = ""
        self._hovered = False

        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedSize(200, 150)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Accessibility
        self.setAccessibleName(title)
        self.setAccessibleDescription(description)

        self._setup_ui()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.SM)

        # Icon
        self.icon_label = QLabel(self._icon)
        icon_font = QFont()
        icon_font.setPointSize(TypeScale.XXXL)
        self.icon_label.setFont(icon_font)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Title
        self.title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(TypeScale.MD)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Description
        self.desc_label = QLabel(self._description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignCenter)
        desc_font = QFont()
        desc_font.setPointSize(TypeScale.SM)
        self.desc_label.setFont(desc_font)
        layout.addWidget(self.desc_label)

        layout.addStretch()

    def paintEvent(self, event):
        """Custom paint for gradient glow and rounded card body."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        card_rect = QRectF(self.rect())
        grad = theme().gradient

        if not self._enabled:
            # Disabled: no glow, muted card body
            paint_card_body(
                painter, card_rect,
                QColor(theme().background_tertiary),
                QColor(theme().border_secondary),
                Radii.MD,
            )
        elif self._selected:
            # Selected: electric multicolor glow
            paint_gradient_glow(
                painter, card_rect,
                grad.active_colors,
                grad.active_opacity,
                grad.glow_spread,
                Radii.MD,
            )
            paint_card_body(
                painter, card_rect,
                QColor(theme().card_background),
                QColor(theme().accent_blue),
                Radii.MD, 2,
            )
        elif self._hovered or self.hasFocus():
            # Hover/Focus: slightly brighter default glow
            brighter_opacity = min(grad.default_opacity + 40, 255)
            paint_gradient_glow(
                painter, card_rect,
                grad.default_colors,
                brighter_opacity,
                grad.glow_spread,
                Radii.MD,
            )
            paint_card_body(
                painter, card_rect,
                QColor(theme().card_hover),
                QColor(theme().border_focus),
                Radii.MD,
            )
        else:
            # Normal: cool purple-blue glow
            paint_gradient_glow(
                painter, card_rect,
                grad.default_colors,
                grad.default_opacity,
                grad.glow_spread,
                Radii.MD,
            )
            paint_card_body(
                painter, card_rect,
                QColor(theme().card_background),
                QColor(theme().card_border),
                Radii.MD,
            )

        painter.end()

    def _update_label_colors(self):
        """Update label colors based on state."""
        if not self._enabled:
            color = theme().text_muted
        elif self._selected:
            color = theme().text_primary
        else:
            color = theme().text_primary

        sec_color = theme().text_muted if not self._enabled else theme().text_secondary

        self.icon_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.title_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.desc_label.setStyleSheet(f"color: {sec_color}; background: transparent;")

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self._update_label_colors()
        self.update()

    def set_enabled(self, enabled: bool, reason: str = ""):
        """Enable or disable the card."""
        self._enabled = enabled
        self._disabled_reason = reason
        if reason and not enabled:
            self.setToolTip(reason)
        else:
            self.setToolTip("")
        self.setCursor(Qt.ForbiddenCursor if not enabled else Qt.PointingHandCursor)
        self._update_label_colors()
        self.update()

    def is_enabled(self) -> bool:
        """Check if the card is enabled."""
        return self._enabled

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton and self._enabled:
            self.clicked.emit(self.key)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for accessibility."""
        if self._enabled and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.key)
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event):
        """Handle focus gained."""
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        """Handle focus lost."""
        super().focusOutEvent(event)
        self.update()

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._update_label_colors()
        self.update()
