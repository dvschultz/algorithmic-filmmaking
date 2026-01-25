"""Track item representing a horizontal lane on the timeline."""

from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QPen

from models.sequence import Track
from ui.theme import theme


class TrackItem(QGraphicsRectItem):
    """A horizontal track lane that contains clips."""

    def __init__(
        self,
        track: Track,
        track_index: int,
        y_position: float,
        height: float,
        scene,
        width: float = 10000,
    ):
        super().__init__()

        self.track = track
        self.track_index = track_index
        self.y_position = y_position
        self.height = height
        self._scene = scene

        # Set geometry
        self.setRect(0, y_position, width, height)

        # Visual style
        self.setBrush(QBrush(theme().colors.qcolor('timeline_track')))
        self.setPen(QPen(Qt.NoPen))
        self.setZValue(-1)  # Behind clips

        # Not selectable or movable
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

        # Accept drops
        self.setAcceptDrops(True)

    def set_width(self, width: float):
        """Update track width."""
        rect = self.rect()
        rect.setWidth(width)
        self.setRect(rect)

    def get_insert_position(self, x: float) -> int:
        """Get the frame position for inserting at x coordinate."""
        return self._scene.get_frame_at_x(x)

    def highlight_drop_target(self, highlight: bool):
        """Visual feedback for drop target."""
        if highlight:
            self.setBrush(QBrush(theme().colors.qcolor('timeline_track_highlight')))
        else:
            self.setBrush(QBrush(theme().colors.qcolor('timeline_track')))
