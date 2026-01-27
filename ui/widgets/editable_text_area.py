"""Inline editable multi-line text widget.

A text area that can be clicked to edit inline, with auto-save on blur/Ctrl+Enter
and Escape to cancel. Uses dual-widget approach (QLabel + QTextEdit).
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QSizePolicy

from ui.theme import theme


class EditableTextArea(QWidget):
    """Multi-line text that can be clicked to edit inline.

    Signals:
        value_changed(str): Emitted when text is edited and saved
    """

    value_changed = Signal(str)

    def __init__(self, text: str = "", placeholder: str = "", parent=None):
        """Create an editable text area.

        Args:
            text: Initial text value
            placeholder: Placeholder shown when text is empty
            parent: Parent widget
        """
        super().__init__(parent)
        self._text = text
        self._placeholder = placeholder
        self._is_editing = False

        self._setup_ui()
        self._apply_style()
        theme().changed.connect(self._apply_style)

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Display label (scrollable via word wrap)
        self.label = QLabel(self._text or self._placeholder)
        self.label.setCursor(Qt.PointingHandCursor)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.installEventFilter(self)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.label)

        # Edit field (hidden initially)
        self.edit = QTextEdit()
        self.edit.setPlaceholderText(self._placeholder)
        self.edit.setText(self._text)
        self.edit.installEventFilter(self)
        self.edit.setMinimumHeight(80)  # ~3 lines
        self.edit.setMaximumHeight(200)  # ~8 lines
        self.edit.hide()
        layout.addWidget(self.edit)

    def _apply_style(self):
        """Apply theme-aware styling."""
        # Label style - use muted color for placeholder state
        text_color = theme().text_muted if not self._text else theme().text_primary
        font_style = "italic" if not self._text else "normal"

        self.label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-style: {font_style};
                padding: 4px 8px;
                border-radius: 4px;
            }}
            QLabel:hover {{
                background-color: {theme().background_tertiary};
            }}
        """)

        # Edit field style
        self.edit.setStyleSheet(f"""
            QTextEdit {{
                color: {theme().text_primary};
                background-color: {theme().background_tertiary};
                border: 1px solid {theme().border_focus};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QTextEdit:focus {{
                border: 2px solid {theme().accent_blue};
            }}
        """)

    def eventFilter(self, obj, event):
        """Handle events for label and edit widgets."""
        if obj == self.label:
            if event.type() == event.Type.MouseButtonPress:
                # Only start editing on click, not on text selection
                if not self.label.hasSelectedText():
                    self._start_editing()
                return True
        elif obj == self.edit:
            if event.type() == event.Type.FocusOut:
                self._finish_editing()
                return False
            elif event.type() == event.Type.KeyPress:
                key_event = event
                if key_event.key() == Qt.Key_Escape:
                    self._cancel_editing()
                    return True
                # Ctrl+Enter to save
                if (key_event.key() == Qt.Key_Return and
                        key_event.modifiers() & Qt.ControlModifier):
                    self._finish_editing()
                    return True
        return super().eventFilter(obj, event)

    def _start_editing(self):
        """Enter edit mode."""
        if self._is_editing or not self.isEnabled():
            return
        self._is_editing = True

        self.label.hide()
        self.edit.setText(self._text)
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()

    def _finish_editing(self):
        """Exit edit mode and save changes."""
        if not self._is_editing:
            return
        self._is_editing = False

        new_text = self.edit.toPlainText().strip()
        if new_text != self._text:
            self._text = new_text
            self.label.setText(new_text or self._placeholder)
            self._apply_style()  # Update placeholder styling
            self.value_changed.emit(new_text)

        self.edit.hide()
        self.label.show()

    def _cancel_editing(self):
        """Cancel edit and revert to previous value."""
        if not self._is_editing:
            return
        self._is_editing = False

        self.edit.hide()
        self.label.show()

    def setText(self, text: str):
        """Set text programmatically (doesn't emit signal).

        Args:
            text: New text value
        """
        self._text = text
        self.label.setText(text or self._placeholder)
        self.edit.setText(text)
        self._apply_style()

    def text(self) -> str:
        """Get current text value."""
        return self._text

    def setPlaceholder(self, placeholder: str):
        """Set placeholder text.

        Args:
            placeholder: Placeholder shown when text is empty
        """
        self._placeholder = placeholder
        self.edit.setPlaceholderText(placeholder)
        if not self._text:
            self.label.setText(placeholder)
        self._apply_style()

    def setEnabled(self, enabled: bool):
        """Enable or disable editing.

        Args:
            enabled: Whether editing is enabled
        """
        super().setEnabled(enabled)
        if enabled:
            self.label.setCursor(Qt.PointingHandCursor)
        else:
            self.label.setCursor(Qt.ArrowCursor)
            if self._is_editing:
                self._cancel_editing()
