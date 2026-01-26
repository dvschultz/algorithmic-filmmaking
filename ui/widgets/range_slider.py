"""Custom dual-handle range slider widget."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics

from ui.theme import theme


class RangeSlider(QWidget):
    """A slider with two handles for selecting a range.

    Emits range_changed(min_value, max_value) when either handle moves.
    """

    range_changed = Signal(float, float)  # (min_value, max_value)

    HANDLE_WIDTH = 12
    HANDLE_HEIGHT = 20
    TRACK_HEIGHT = 6
    LABEL_HEIGHT = 16
    PADDING = 6

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data range (actual min/max of the data)
        self._data_min: float = 0.0
        self._data_max: float = 100.0

        # Selected range (current filter values)
        self._value_min: float = 0.0
        self._value_max: float = 100.0

        # Dragging state
        self._dragging_min = False
        self._dragging_max = False
        self._drag_offset = 0

        # Appearance
        self._suffix = "s"  # e.g., "s" for seconds

        # Calculate minimum height
        min_height = self.LABEL_HEIGHT + self.PADDING + self.HANDLE_HEIGHT + self.PADDING + self.LABEL_HEIGHT
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(200)

        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def set_range(self, min_val: float, max_val: float):
        """Set the data range (min/max possible values).

        Args:
            min_val: Minimum possible value
            max_val: Maximum possible value
        """
        if max_val <= min_val:
            max_val = min_val + 1.0

        self._data_min = min_val
        self._data_max = max_val

        # Clamp current values to new range
        self._value_min = max(self._data_min, min(self._value_min, self._data_max))
        self._value_max = max(self._data_min, min(self._value_max, self._data_max))

        # Ensure min <= max
        if self._value_min > self._value_max:
            self._value_min = self._value_max

        self.update()

    def set_values(self, min_val: float, max_val: float):
        """Set the selected range values.

        Args:
            min_val: Minimum selected value
            max_val: Maximum selected value
        """
        self._value_min = max(self._data_min, min(min_val, self._data_max))
        self._value_max = max(self._data_min, min(max_val, self._data_max))

        # Ensure min <= max
        if self._value_min > self._value_max:
            self._value_min, self._value_max = self._value_max, self._value_min

        self.update()

    def set_suffix(self, suffix: str):
        """Set the suffix for value labels (e.g., 's' for seconds)."""
        self._suffix = suffix
        self.update()

    def values(self) -> tuple[float, float]:
        """Get the current selected range."""
        return (self._value_min, self._value_max)

    def get_data_range(self) -> tuple[float, float]:
        """Get the data range (min/max possible values)."""
        return (self._data_min, self._data_max)

    def reset(self):
        """Reset to full range."""
        self._value_min = self._data_min
        self._value_max = self._data_max
        self.update()
        self.range_changed.emit(self._value_min, self._value_max)

    def _value_to_x(self, value: float) -> int:
        """Convert a value to x position."""
        track_rect = self._get_track_rect()
        if self._data_max == self._data_min:
            return track_rect.left()

        ratio = (value - self._data_min) / (self._data_max - self._data_min)
        return int(track_rect.left() + ratio * track_rect.width())

    def _x_to_value(self, x: int) -> float:
        """Convert x position to a value."""
        track_rect = self._get_track_rect()
        if track_rect.width() == 0:
            return self._data_min

        ratio = (x - track_rect.left()) / track_rect.width()
        ratio = max(0.0, min(1.0, ratio))
        return self._data_min + ratio * (self._data_max - self._data_min)

    def _get_track_rect(self) -> QRect:
        """Get the rectangle for the track."""
        # Track is in the middle vertically, with padding for handles
        y = self.LABEL_HEIGHT + self.PADDING + (self.HANDLE_HEIGHT - self.TRACK_HEIGHT) // 2
        return QRect(
            self.HANDLE_WIDTH // 2 + self.PADDING,
            y,
            self.width() - self.HANDLE_WIDTH - 2 * self.PADDING,
            self.TRACK_HEIGHT
        )

    def _get_min_handle_rect(self) -> QRect:
        """Get the rectangle for the minimum handle."""
        x = self._value_to_x(self._value_min)
        y = self.LABEL_HEIGHT + self.PADDING
        return QRect(
            x - self.HANDLE_WIDTH // 2,
            y,
            self.HANDLE_WIDTH,
            self.HANDLE_HEIGHT
        )

    def _get_max_handle_rect(self) -> QRect:
        """Get the rectangle for the maximum handle."""
        x = self._value_to_x(self._value_max)
        y = self.LABEL_HEIGHT + self.PADDING
        return QRect(
            x - self.HANDLE_WIDTH // 2,
            y,
            self.HANDLE_WIDTH,
            self.HANDLE_HEIGHT
        )

    def paintEvent(self, event):
        """Paint the slider."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_rect = self._get_track_rect()
        min_handle_rect = self._get_min_handle_rect()
        max_handle_rect = self._get_max_handle_rect()

        # Get theme colors
        t = theme()
        track_color = QColor(t.border_secondary)
        selected_color = QColor(t.accent_blue)
        handle_color = QColor(t.card_background)
        handle_border = QColor(t.accent_blue)
        text_color = QColor(t.text_primary)
        muted_color = QColor(t.text_muted)

        # Draw track background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(track_rect, 3, 3)

        # Draw selected range on track
        min_x = self._value_to_x(self._value_min)
        max_x = self._value_to_x(self._value_max)
        selected_rect = QRect(min_x, track_rect.top(), max_x - min_x, track_rect.height())
        painter.setBrush(QBrush(selected_color))
        painter.drawRoundedRect(selected_rect, 3, 3)

        # Draw handles
        for handle_rect in [min_handle_rect, max_handle_rect]:
            painter.setPen(QPen(handle_border, 2))
            painter.setBrush(QBrush(handle_color))
            painter.drawRoundedRect(handle_rect, 3, 3)

        # Draw labels at ends (data range)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(muted_color)

        min_label = f"{self._data_min:.1f}{self._suffix}"
        max_label = f"{self._data_max:.1f}{self._suffix}"

        # Left label (data min)
        painter.drawText(
            self.PADDING,
            self.LABEL_HEIGHT - 2,
            min_label
        )

        # Right label (data max)
        fm = QFontMetrics(font)
        max_width = fm.horizontalAdvance(max_label)
        painter.drawText(
            self.width() - max_width - self.PADDING,
            self.LABEL_HEIGHT - 2,
            max_label
        )

        # Draw value labels on handles
        painter.setPen(text_color)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Min handle label (below)
        min_value_label = f"{self._value_min:.1f}{self._suffix}"
        min_label_width = fm.horizontalAdvance(min_value_label)
        min_label_x = min_handle_rect.center().x() - min_label_width // 2
        # Clamp to widget bounds
        min_label_x = max(self.PADDING, min(min_label_x, self.width() - min_label_width - self.PADDING))
        painter.drawText(
            min_label_x,
            min_handle_rect.bottom() + self.PADDING + self.LABEL_HEIGHT - 4,
            min_value_label
        )

        # Max handle label (below)
        max_value_label = f"{self._value_max:.1f}{self._suffix}"
        max_label_width = fm.horizontalAdvance(max_value_label)
        max_label_x = max_handle_rect.center().x() - max_label_width // 2
        # Clamp to widget bounds
        max_label_x = max(self.PADDING, min(max_label_x, self.width() - max_label_width - self.PADDING))

        # Avoid overlap with min label
        if max_label_x < min_label_x + min_label_width + 4:
            max_label_x = min_label_x + min_label_width + 4

        painter.drawText(
            max_label_x,
            max_handle_rect.bottom() + self.PADDING + self.LABEL_HEIGHT - 4,
            max_value_label
        )

        painter.end()

    def mousePressEvent(self, event):
        """Handle mouse press to start dragging a handle."""
        if event.button() != Qt.LeftButton:
            return

        pos = event.pos()
        min_rect = self._get_min_handle_rect()
        max_rect = self._get_max_handle_rect()

        # Check which handle was clicked (prefer the one on top if overlapping)
        min_dist = abs(pos.x() - min_rect.center().x())
        max_dist = abs(pos.x() - max_rect.center().x())

        if min_rect.contains(pos) and max_rect.contains(pos):
            # Both handles at same position - pick based on which side of center
            if pos.x() <= min_rect.center().x():
                self._dragging_min = True
            else:
                self._dragging_max = True
        elif min_rect.contains(pos):
            self._dragging_min = True
        elif max_rect.contains(pos):
            self._dragging_max = True
        else:
            # Click on track - move nearest handle
            if min_dist < max_dist:
                self._dragging_min = True
            else:
                self._dragging_max = True

            # Jump handle to click position
            new_value = self._x_to_value(pos.x())
            if self._dragging_min:
                self._value_min = min(new_value, self._value_max)
            else:
                self._value_max = max(new_value, self._value_min)

            self.update()
            self.range_changed.emit(self._value_min, self._value_max)

    def mouseMoveEvent(self, event):
        """Handle mouse move to drag handles."""
        pos = event.pos()

        if self._dragging_min or self._dragging_max:
            new_value = self._x_to_value(pos.x())

            if self._dragging_min:
                # Don't let min exceed max
                self._value_min = min(new_value, self._value_max)
                self._value_min = max(self._data_min, self._value_min)
            else:
                # Don't let max go below min
                self._value_max = max(new_value, self._value_min)
                self._value_max = min(self._data_max, self._value_max)

            self.update()
            self.range_changed.emit(self._value_min, self._value_max)
        else:
            # Update cursor based on hover
            min_rect = self._get_min_handle_rect()
            max_rect = self._get_max_handle_rect()

            if min_rect.contains(pos) or max_rect.contains(pos):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging."""
        if event.button() == Qt.LeftButton:
            self._dragging_min = False
            self._dragging_max = False


class DurationRangeSlider(QWidget):
    """A labeled duration range slider widget."""

    range_changed = Signal(float, float)  # (min_seconds, max_seconds)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Label
        self.label = QLabel("Duration:")
        layout.addWidget(self.label)

        # Range slider
        self.slider = RangeSlider()
        self.slider.set_suffix("s")
        self.slider.range_changed.connect(self._on_range_changed)
        layout.addWidget(self.slider, 1)  # Stretch to fill

    def set_range(self, min_val: float, max_val: float):
        """Set the data range."""
        self.slider.set_range(min_val, max_val)

    def set_values(self, min_val: float, max_val: float):
        """Set the selected range."""
        self.slider.set_values(min_val, max_val)

    def values(self) -> tuple[float, float]:
        """Get the current selected range."""
        return self.slider.values()

    def reset(self):
        """Reset to full range."""
        self.slider.reset()

    def _on_range_changed(self, min_val: float, max_val: float):
        """Forward the signal."""
        self.range_changed.emit(min_val, max_val)
