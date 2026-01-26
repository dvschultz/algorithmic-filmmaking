"""Sorting algorithm card widget for the Sequence tab."""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QKeyEvent

from ui.theme import theme


class SortingCard(QFrame):
    """A clickable card for selecting a sorting algorithm.

    Displays an icon, title, and description for a sorting algorithm.
    Supports hover, selected, and disabled states.
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
        """Initialize the sorting card.

        Args:
            key: Unique algorithm identifier (e.g., "color", "shuffle")
            icon: Emoji or text icon to display
            title: Display name of the algorithm
            description: Brief description of what the algorithm does
            parent: Parent widget
        """
        super().__init__(parent)
        self.key = key
        self._icon = icon
        self._title = title
        self._description = description
        self._selected = False
        self._enabled = True
        self._disabled_reason = ""

        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(1)
        self.setFixedSize(200, 150)
        self.setCursor(Qt.PointingHandCursor)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        self._setup_ui()
        self._update_style()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # Icon
        self.icon_label = QLabel(self._icon)
        icon_font = QFont()
        icon_font.setPointSize(32)
        self.icon_label.setFont(icon_font)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Title
        self.title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Description
        self.desc_label = QLabel(self._description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.desc_label)

        layout.addStretch()

    def _update_style(self):
        """Update visual style based on state."""
        if not self._enabled:
            # Disabled state
            self.setStyleSheet(f"""
                SortingCard {{
                    background-color: {theme().background_tertiary};
                    border: 1px solid {theme().border_secondary};
                }}
            """)
            self.icon_label.setStyleSheet(f"color: {theme().text_muted};")
            self.title_label.setStyleSheet(f"color: {theme().text_muted};")
            self.desc_label.setStyleSheet(f"font-size: 11px; color: {theme().text_muted};")
            self.setCursor(Qt.ForbiddenCursor)
        elif self._selected:
            # Selected state
            self.setStyleSheet(f"""
                SortingCard {{
                    background-color: {theme().accent_blue};
                    border: 2px solid {theme().accent_blue_hover};
                }}
            """)
            self.icon_label.setStyleSheet(f"color: {theme().text_inverted};")
            self.title_label.setStyleSheet(f"color: {theme().text_inverted};")
            self.desc_label.setStyleSheet(f"font-size: 11px; color: {theme().text_inverted};")
            self.setCursor(Qt.PointingHandCursor)
        else:
            # Normal state (hover handled in stylesheet)
            self.setStyleSheet(f"""
                SortingCard {{
                    background-color: {theme().card_background};
                    border: 1px solid {theme().card_border};
                }}
                SortingCard:hover {{
                    background-color: {theme().card_hover};
                    border: 1px solid {theme().border_focus};
                }}
            """)
            self.icon_label.setStyleSheet(f"color: {theme().text_primary};")
            self.title_label.setStyleSheet(f"color: {theme().text_primary};")
            self.desc_label.setStyleSheet(f"font-size: 11px; color: {theme().text_secondary};")
            self.setCursor(Qt.PointingHandCursor)

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self._update_style()

    def set_enabled(self, enabled: bool, reason: str = ""):
        """Enable or disable the card.

        Args:
            enabled: Whether the card should be clickable
            reason: Tooltip text explaining why the card is disabled
        """
        self._enabled = enabled
        self._disabled_reason = reason
        if reason and not enabled:
            self.setToolTip(reason)
        else:
            self.setToolTip("")
        self._update_style()

    def is_enabled(self) -> bool:
        """Check if the card is enabled."""
        return self._enabled

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
        """Handle focus gained - add visual indicator."""
        super().focusInEvent(event)
        if self._enabled and not self._selected:
            # Add focus ring
            self.setStyleSheet(f"""
                SortingCard {{
                    background-color: {theme().card_hover};
                    border: 2px solid {theme().border_focus};
                }}
            """)

    def focusOutEvent(self, event):
        """Handle focus lost - remove visual indicator."""
        super().focusOutEvent(event)
        self._update_style()

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._update_style()
