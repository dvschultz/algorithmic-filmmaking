"""Collapsible source group header for ClipBrowser."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QKeyEvent

from ui.theme import theme, UISizes


class SourceGroupHeader(QFrame):
    """Clickable header for a source group in ClipBrowser.

    Displays a chevron icon, source filename, and clip count.
    Clicking toggles the expanded/collapsed state.

    Signals:
        toggled: Emitted with (source_id, is_expanded) when header is clicked
    """

    toggled = Signal(str, bool)  # (source_id, is_expanded)

    def __init__(
        self,
        source_id: str,
        filename: str,
        clip_count: int,
        parent=None
    ):
        """Initialize the source group header.

        Args:
            source_id: Unique identifier for the source
            filename: Display name (source video filename)
            clip_count: Total number of clips in this group
            parent: Parent widget
        """
        super().__init__(parent)
        self.source_id = source_id
        self._filename = filename
        self._total_clip_count = clip_count
        self._visible_clip_count = clip_count  # For filtered display
        self._selected_count = 0  # Number of selected clips in this group
        self._is_expanded = True

        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedHeight(UISizes.SOURCE_HEADER_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        # Accessibility
        self.setAccessibleName(f"{filename} source group")
        self.setAccessibleDescription(
            f"Collapsible group containing {clip_count} clips. "
            "Press Enter or Space to toggle."
        )

        self._setup_ui()
        self._update_style()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the header UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Chevron icon (▼ expanded, ▶ collapsed)
        self.chevron_label = QLabel()
        self.chevron_label.setFixedWidth(16)
        self._update_chevron()
        layout.addWidget(self.chevron_label)

        # Filename
        self.filename_label = QLabel(self._filename)
        self.filename_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.filename_label)

        layout.addStretch()

        # Clip count badge
        self.count_label = QLabel()
        self._update_count_label()
        layout.addWidget(self.count_label)

    def _update_chevron(self):
        """Update chevron icon based on expanded state."""
        icon = "▼" if self._is_expanded else "▶"
        self.chevron_label.setText(icon)

    def _update_count_label(self):
        """Update the clip count label text."""
        if self._visible_clip_count == self._total_clip_count:
            # No filter active - show total
            count_text = f"({self._total_clip_count} clips)"
        else:
            # Filter active - show "X of Y"
            count_text = f"({self._visible_clip_count} of {self._total_clip_count})"

        # Add selection indicator if collapsed and has selections
        if not self._is_expanded and self._selected_count > 0:
            count_text = f"({self._total_clip_count} clips \u2022 {self._selected_count} selected)"

        self.count_label.setText(count_text)

    def _update_style(self):
        """Update visual style based on state."""
        self.setStyleSheet(f"""
            SourceGroupHeader {{
                background-color: {theme().background_tertiary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
            }}
            SourceGroupHeader:hover {{
                background-color: {theme().card_hover};
                border-color: {theme().border_focus};
            }}
        """)
        self.chevron_label.setStyleSheet(f"color: {theme().text_secondary};")
        self.filename_label.setStyleSheet(
            f"font-weight: bold; color: {theme().text_primary};"
        )
        self.count_label.setStyleSheet(
            f"font-size: 11px; color: {theme().text_muted};"
        )

    @property
    def is_expanded(self) -> bool:
        """Get the current expanded state."""
        return self._is_expanded

    def set_expanded(self, expanded: bool):
        """Set the expanded state without emitting signal."""
        if self._is_expanded != expanded:
            self._is_expanded = expanded
            self._update_chevron()
            self._update_count_label()

    def toggle(self):
        """Toggle expanded state and emit signal."""
        self._is_expanded = not self._is_expanded
        self._update_chevron()
        self._update_count_label()
        self.toggled.emit(self.source_id, self._is_expanded)

    def set_clip_counts(
        self,
        total: int,
        visible: int,
        selected: int = 0
    ):
        """Update the clip counts displayed.

        Args:
            total: Total clips in this source
            visible: Clips currently visible (after filtering)
            selected: Number of selected clips in this group
        """
        self._total_clip_count = total
        self._visible_clip_count = visible
        self._selected_count = selected
        self._update_count_label()

    def mousePressEvent(self, event):
        """Handle mouse press to toggle."""
        if event.button() == Qt.LeftButton:
            self.toggle()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for accessibility."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.toggle()
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event):
        """Handle focus gained - add visual indicator."""
        super().focusInEvent(event)
        self.setStyleSheet(f"""
            SourceGroupHeader {{
                background-color: {theme().card_hover};
                border: 2px solid {theme().border_focus};
                border-radius: 4px;
            }}
        """)

    def focusOutEvent(self, event):
        """Handle focus lost - remove visual indicator."""
        super().focusOutEvent(event)
        self._update_style()

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._update_style()
