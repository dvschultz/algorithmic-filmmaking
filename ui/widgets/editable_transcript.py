"""Editable transcript widget.

Displays transcript segments with timestamps and allows editing the text
portion while preserving the timing information.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QFrame,
)

from ui.theme import theme

# Import for type hints only to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.transcription import TranscriptSegment


class TranscriptSegmentWidget(QFrame):
    """Single transcript segment with timestamp and editable text."""

    text_changed = Signal(int, str)  # (segment_index, new_text)

    def __init__(self, index: int, segment: "TranscriptSegment", parent=None):
        """Create a transcript segment widget.

        Args:
            index: Segment index in the transcript list
            segment: The transcript segment data
            parent: Parent widget
        """
        super().__init__(parent)
        self._index = index
        self._segment = segment
        self._is_editing = False

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Build the segment UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Timestamp label (read-only)
        time_str = self._format_time_range(self._segment.start_time, self._segment.end_time)
        self.time_label = QLabel(time_str)
        self.time_label.setFixedWidth(100)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.time_label)

        # Separator
        separator = QLabel(":")
        separator.setFixedWidth(10)
        layout.addWidget(separator)

        # Text label (display mode)
        self.text_label = QLabel(self._segment.text)
        self.text_label.setWordWrap(True)
        self.text_label.setCursor(Qt.PointingHandCursor)
        self.text_label.installEventFilter(self)
        layout.addWidget(self.text_label, 1)

        # Text edit (edit mode, hidden initially)
        self.text_edit = QLineEdit(self._segment.text)
        self.text_edit.returnPressed.connect(self._finish_editing)
        self.text_edit.installEventFilter(self)
        self.text_edit.hide()
        layout.addWidget(self.text_edit, 1)

    def _apply_style(self):
        """Apply theme-aware styling."""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: transparent;
                border-radius: 4px;
            }}
            QFrame:hover {{
                background-color: {theme().background_tertiary};
            }}
        """)

        self.time_label.setStyleSheet(f"""
            color: {theme().text_muted};
            font-family: monospace;
            font-size: 11px;
        """)

        self.text_label.setStyleSheet(f"""
            color: {theme().text_secondary};
            padding: 2px;
        """)

        self.text_edit.setStyleSheet(f"""
            QLineEdit {{
                color: {theme().text_primary};
                background-color: {theme().background_tertiary};
                border: 1px solid {theme().border_focus};
                border-radius: 4px;
                padding: 2px 4px;
            }}
            QLineEdit:focus {{
                border: 2px solid {theme().accent_blue};
            }}
        """)

    def _format_time_range(self, start: float, end: float) -> str:
        """Format time range as mm:ss - mm:ss."""
        def fmt(t):
            m = int(t // 60)
            s = int(t % 60)
            return f"{m}:{s:02d}"
        return f"{fmt(start)} - {fmt(end)}"

    def eventFilter(self, obj, event):
        """Handle events for label and edit widgets."""
        if obj == self.text_label:
            if event.type() == event.Type.MouseButtonPress:
                self._start_editing()
                return True
        elif obj == self.text_edit:
            if event.type() == event.Type.FocusOut:
                self._finish_editing()
                return False
            elif event.type() == event.Type.KeyPress:
                if event.key() == Qt.Key_Escape:
                    self._cancel_editing()
                    return True
        return super().eventFilter(obj, event)

    def _start_editing(self):
        """Enter edit mode."""
        if self._is_editing:
            return
        self._is_editing = True

        self.text_label.hide()
        self.text_edit.setText(self._segment.text)
        self.text_edit.show()
        self.text_edit.setFocus()
        self.text_edit.selectAll()

    def _finish_editing(self):
        """Exit edit mode and save changes."""
        if not self._is_editing:
            return
        self._is_editing = False

        new_text = self.text_edit.text().strip()
        if new_text and new_text != self._segment.text:
            self._segment.text = new_text
            self.text_label.setText(new_text)
            self.text_changed.emit(self._index, new_text)

        self.text_edit.hide()
        self.text_label.show()

    def _cancel_editing(self):
        """Cancel edit and revert to previous value."""
        if not self._is_editing:
            return
        self._is_editing = False

        self.text_edit.hide()
        self.text_label.show()

    def setEnabled(self, enabled: bool):
        """Enable or disable editing."""
        super().setEnabled(enabled)
        if enabled:
            self.text_label.setCursor(Qt.PointingHandCursor)
        else:
            self.text_label.setCursor(Qt.ArrowCursor)
            if self._is_editing:
                self._cancel_editing()


class EditableTranscriptWidget(QWidget):
    """Widget displaying editable transcript segments.

    Signals:
        segments_changed(list): Emitted when any segment text is edited
    """

    segments_changed = Signal(list)  # list[TranscriptSegment]

    def __init__(self, parent=None):
        """Create editable transcript widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._segments: list["TranscriptSegment"] = []
        self._segment_widgets: list[TranscriptSegmentWidget] = []
        self._change_in_progress = False

        self._setup_ui()
        theme().changed.connect(self._apply_style)

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container for segment widgets
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(2)

        # Empty state label
        self.empty_label = QLabel("No transcript available")
        self._apply_empty_style()
        self.container_layout.addWidget(self.empty_label)

        self.container_layout.addStretch()

        layout.addWidget(self.container)

    def _apply_style(self):
        """Apply theme-aware styling."""
        self._apply_empty_style()
        for widget in self._segment_widgets:
            widget._apply_style()

    def _apply_empty_style(self):
        """Apply styling to empty label."""
        self.empty_label.setStyleSheet(f"""
            color: {theme().text_muted};
            font-style: italic;
            padding: 8px;
        """)

    def setSegments(self, segments: Optional[list["TranscriptSegment"]]):
        """Set transcript segments (doesn't emit signal).

        Args:
            segments: List of transcript segments, or None
        """
        self._change_in_progress = True

        # Clear existing widgets
        for widget in self._segment_widgets:
            widget.deleteLater()
        self._segment_widgets.clear()

        self._segments = segments or []

        if not self._segments:
            self.empty_label.show()
        else:
            self.empty_label.hide()
            # Create segment widgets
            for i, segment in enumerate(self._segments):
                widget = TranscriptSegmentWidget(i, segment)
                widget.text_changed.connect(self._on_segment_changed)
                self.container_layout.insertWidget(i, widget)
                self._segment_widgets.append(widget)

        self._change_in_progress = False

    def segments(self) -> list["TranscriptSegment"]:
        """Get current transcript segments."""
        return self._segments

    @Slot(int, str)
    def _on_segment_changed(self, index: int, new_text: str):
        """Handle segment text change."""
        if self._change_in_progress:
            return
        self._change_in_progress = True

        # Segment is already updated in-place by the widget
        self.segments_changed.emit(self._segments)

        self._change_in_progress = False

    def setEnabled(self, enabled: bool):
        """Enable or disable editing."""
        super().setEnabled(enabled)
        for widget in self._segment_widgets:
            widget.setEnabled(enabled)
