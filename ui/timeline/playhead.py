"""Playhead component for timeline."""

from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsItem,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF


class PlayheadLine(QGraphicsLineItem):
    """Vertical line indicating current playback position."""

    def __init__(self, height: float, parent=None):
        super().__init__(parent)

        self.setLine(0, 0, 0, height)
        self.setPen(QPen(QColor("#ff4444"), 2))
        self.setZValue(1000)  # Always on top

    def set_height(self, height: float):
        """Update line height."""
        self.setLine(0, 0, 0, height)


class PlayheadHandle(QGraphicsPolygonItem):
    """Draggable triangle handle at top of playhead."""

    def __init__(self, parent_line: PlayheadLine, scene_ref):
        super().__init__()

        self.parent_line = parent_line
        self._scene_ref = scene_ref
        self._pixels_per_second = 100.0
        self._fps = 30.0

        # Create downward-pointing triangle
        triangle = QPolygonF(
            [
                QPointF(-10, 0),
                QPointF(10, 0),
                QPointF(0, 15),
            ]
        )
        self.setPolygon(triangle)
        self.setBrush(QBrush(QColor("#ff4444")))
        self.setPen(QPen(Qt.NoPen))
        self.setZValue(1001)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.PointingHandCursor)

    def set_pixels_per_second(self, pps: float):
        """Update zoom level."""
        self._pixels_per_second = pps

    def set_fps(self, fps: float):
        """Update frame rate."""
        self._fps = fps

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # Constrain to horizontal movement, Y stays at top
            new_pos = value
            new_pos.setY(0)
            new_pos.setX(max(0, new_pos.x()))

            # Move the line to match
            self.parent_line.setX(new_pos.x())

            # Emit signal for time update
            if self._scene_ref and hasattr(self._scene_ref, "playhead_moved"):
                time_seconds = new_pos.x() / self._pixels_per_second
                self._scene_ref.playhead_moved.emit(time_seconds)

            return new_pos

        return super().itemChange(change, value)


class Playhead:
    """Composite playhead with line and draggable handle."""

    def __init__(self, scene, height: float, fps: float = 30.0):
        self._scene = scene
        self._fps = fps
        self._pixels_per_second = 100.0

        # Create components
        self.line = PlayheadLine(height)
        self.handle = PlayheadHandle(self.line, scene)
        self.handle.set_fps(fps)

        # Add to scene
        scene.addItem(self.line)
        scene.addItem(self.handle)

        # Initial position
        self.line.setX(0)
        self.handle.setX(0)

        # Register with scene
        scene.set_playhead(self)

    def set_time(self, time_seconds: float):
        """Set playhead position from time value."""
        x = time_seconds * self._pixels_per_second
        self.line.setX(x)
        self.handle.setX(x)

    def set_frame(self, frame: int):
        """Set playhead position from frame number."""
        time_seconds = frame / self._fps
        self.set_time(time_seconds)

    def get_time(self) -> float:
        """Get current time from playhead position."""
        return self.line.x() / self._pixels_per_second

    def get_frame(self) -> int:
        """Get current frame from playhead position."""
        return int(self.get_time() * self._fps)

    def set_pixels_per_second(self, pps: float):
        """Update zoom level and maintain time position."""
        current_time = self.get_time()
        self._pixels_per_second = pps
        self.handle.set_pixels_per_second(pps)
        self.set_time(current_time)

    def set_height(self, height: float):
        """Update line height."""
        self.line.set_height(height)

    def set_fps(self, fps: float):
        """Update frame rate."""
        self._fps = fps
        self.handle.set_fps(fps)
