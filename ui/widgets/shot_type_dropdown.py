"""Shot type dropdown widget.

A styled QComboBox for selecting shot type with immediate save on selection.
"""

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QComboBox

from core.analysis.shots import SHOT_TYPES, SHOT_TYPE_DISPLAY
from ui.theme import theme


# Special value for unset shot type
NOT_SET_VALUE = ""
NOT_SET_DISPLAY = "(Not set)"


class ShotTypeDropdown(QComboBox):
    """Dropdown for selecting shot type.

    Signals:
        value_changed(str): Emitted when selection changes (value is shot type string or empty)
    """

    value_changed = Signal(str)

    def __init__(self, parent=None):
        """Create shot type dropdown.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_value = NOT_SET_VALUE
        self._change_in_progress = False

        self._setup_items()
        self._apply_style()

        self.currentIndexChanged.connect(self._on_selection_changed)
        theme().changed.connect(self._apply_style)

    def _setup_items(self):
        """Populate dropdown with shot type options."""
        # Add "Not set" option first
        self.addItem(NOT_SET_DISPLAY, NOT_SET_VALUE)

        # Add shot types with display names
        for shot_type in SHOT_TYPES:
            display_name = SHOT_TYPE_DISPLAY.get(shot_type, shot_type.title())
            self.addItem(display_name, shot_type)

    def _apply_style(self):
        """Apply theme-aware styling."""
        self.setStyleSheet(f"""
            QComboBox {{
                color: {theme().text_primary};
                background-color: {theme().background_tertiary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 120px;
            }}
            QComboBox:hover {{
                border-color: {theme().border_focus};
            }}
            QComboBox:focus {{
                border: 2px solid {theme().accent_blue};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {theme().text_secondary};
            }}
            QComboBox QAbstractItemView {{
                color: {theme().text_primary};
                background-color: {theme().background_elevated};
                border: 1px solid {theme().border_primary};
                selection-background-color: {theme().accent_blue};
                selection-color: {theme().text_inverted};
            }}
        """)

    @Slot(int)
    def _on_selection_changed(self, index: int):
        """Handle selection change."""
        if self._change_in_progress:
            return
        self._change_in_progress = True

        new_value = self.itemData(index)
        if new_value != self._current_value:
            self._current_value = new_value
            self.value_changed.emit(new_value)

        self._change_in_progress = False

    def setValue(self, shot_type: str | None):
        """Set current value programmatically (doesn't emit signal).

        Args:
            shot_type: Shot type string or None/empty for "Not set"
        """
        self._change_in_progress = True

        value = shot_type or NOT_SET_VALUE
        self._current_value = value

        # Find matching item
        for i in range(self.count()):
            if self.itemData(i) == value:
                self.setCurrentIndex(i)
                break
        else:
            # Value not found, default to "Not set"
            self.setCurrentIndex(0)

        self._change_in_progress = False

    def value(self) -> str:
        """Get current shot type value.

        Returns:
            Shot type string, or empty string if not set
        """
        return self._current_value
