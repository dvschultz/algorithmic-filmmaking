"""Clip item for timeline - draggable, resizable rectangle."""

from pathlib import Path
from dataclasses import replace
from fractions import Fraction
from math import floor
from models.media_time import frame_rate, nearest_source_frame, source_frame_time

from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsItem,
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QPen, QPixmap, QPainter

from models.sequence import SequenceClip
from ui.theme import theme


class ClipItem(QGraphicsRectItem):
    """A draggable, resizable clip on the timeline."""

    EDGE_THRESHOLD = 8  # Pixels from edge to trigger resize
    SNAP_THRESHOLD_FRAMES = 5  # Frames within which to snap

    def __init__(
        self,
        seq_clip: SequenceClip,
        track_item,
        pixels_per_second: float,
        fps: float,
        thumbnail_path: str | None = None,
    ):
        super().__init__()

        self.seq_clip = seq_clip
        self.track_item = track_item
        self.pixels_per_second = pixels_per_second
        self.fps = fps
        self.thumbnail_path = thumbnail_path
        self._thumbnail_pixmap = None

        # Interaction state
        self._edit_source = None
        self._dragging = False
        self._resizing = None  # 'left', 'right', or None
        self._drag_start_pos = None
        self._original_start_frame = None
        self._original_in_point = None
        self._original_out_point = None

        self._thumbnail_loaded = False

        self._setup_item()

    def _setup_item(self):
        """Configure item flags and appearance."""
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)  # Manual movement
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self._update_geometry()
        self._update_appearance()

    def _load_thumbnail(self):
        """Load thumbnail image if available (lazy - called when needed)."""
        if self._thumbnail_loaded:
            return
        self._thumbnail_loaded = True

        if self.thumbnail_path:
            path = Path(self.thumbnail_path)
            if path.exists():
                self._thumbnail_pixmap = QPixmap(str(path))

    def _update_geometry(self):
        """Update position and size based on clip data."""
        x = self._frame_to_x(self.seq_clip.start_frame)
        width = self._frames_to_width(self.seq_clip.duration_frames)

        track_y = self.track_item.y_position
        height = self.track_item.height - 4

        self.setRect(0, 0, width, height)
        self.setPos(x, track_y + 2)

    def _update_appearance(self):
        """Update visual style."""
        if self.isSelected():
            self.setBrush(QBrush(theme().colors.qcolor('timeline_clip_selected')))
            self.setPen(QPen(theme().colors.qcolor('timeline_clip_selected_border'), 2))
        else:
            self.setBrush(QBrush(theme().colors.qcolor('timeline_clip')))
            self.setPen(QPen(theme().colors.qcolor('timeline_clip_border'), 1))

    def _frame_to_x(self, frame: int) -> float:
        """Convert frame number to x coordinate."""
        seconds = frame / self.fps
        return seconds * self.pixels_per_second

    def _x_to_frame(self, x: float) -> int:
        """Convert x coordinate to frame number."""
        seconds = x / self.pixels_per_second
        return int(seconds * self.fps)

    def _frames_to_width(self, frames: int) -> float:
        """Convert frame count to pixel width."""
        seconds = frames / self.fps
        return seconds * self.pixels_per_second

    def set_pixels_per_second(self, pps: float):
        """Update zoom level and recalculate geometry."""
        self.pixels_per_second = pps
        self._update_geometry()

    # --- Painting ---
    def paint(self, painter: QPainter, option, widget=None):
        """Custom paint to include thumbnail."""
        # Draw base rectangle
        super().paint(painter, option, widget)

        rect = self.rect()

        # Lazy load thumbnail on first paint (when visible)
        if not self._thumbnail_loaded:
            self._load_thumbnail()

        # Draw thumbnail if available
        if self._thumbnail_pixmap and not self._thumbnail_pixmap.isNull():
            thumb_rect = QRectF(2, 2, rect.width() - 4, rect.height() - 4)
            painter.setOpacity(0.8)
            painter.drawPixmap(thumb_rect.toRect(), self._thumbnail_pixmap)
            painter.setOpacity(1.0)

        # Draw clip label
        painter.setPen(theme().colors.qcolor('text_inverted'))
        label = f"{self.seq_clip.duration_frames}f"
        painter.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignLeft | Qt.AlignBottom, label)

    # --- Hover for resize cursor ---
    def hoverMoveEvent(self, event):
        pos = event.pos()
        rect = self.rect()

        if pos.x() < self.EDGE_THRESHOLD:
            self.setCursor(Qt.SizeHorCursor)
        elif pos.x() > rect.width() - self.EDGE_THRESHOLD:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    # --- Mouse handling for drag and resize ---
    def mousePressEvent(self, event):
        from ui.project_access import is_read_only
        if is_read_only(self):
            self.setSelected(True)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            event.ignore()
            return

        scene = self.scene()
        if scene and getattr(scene, "uses_history", False):
            from core.commands.sequence_clips import Placement

            self._edit_source = self.seq_clip
            self._edit_before = Placement.capture(self.seq_clip)
            self._edit_project = scene.project
            self._edit_sequence = scene.sequence
            self.seq_clip = replace(self.seq_clip)

        pos = event.pos()
        rect = self.rect()

        self._drag_start_pos = event.scenePos()
        self._original_start_frame = self.seq_clip.start_frame
        self._original_in_point = self.seq_clip.in_point
        self._original_out_point = self.seq_clip.out_point
        self._original_hold_frames = self.seq_clip.hold_frames

        # Determine if resizing or dragging
        if pos.x() < self.EDGE_THRESHOLD:
            self._resizing = "left"
        elif pos.x() > rect.width() - self.EDGE_THRESHOLD:
            self._resizing = "right"
        else:
            self._dragging = True

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        from ui.project_access import is_read_only
        if is_read_only(self):
            event.ignore()
            return
        if not (self._dragging or self._resizing):
            return

        delta_x = event.scenePos().x() - self._drag_start_pos.x()
        delta_frames = self._x_to_frame(delta_x)
        rate = frame_rate(self.seq_clip.source_rate or self.fps)
        timeline_rate = frame_rate(self.fps)
        timestamps = None
        if self._resizing and self.seq_clip.source_presentation is not None:
            scene = self.scene()
            project = getattr(scene, "project", None)
            source = project.sources_by_id.get(self.seq_clip.source_id) if project is not None else None
            if source is None and scene is not None and hasattr(scene.parent(), "get_sources_lookup"):
                source = scene.parent().get_sources_lookup().get(self.seq_clip.source_id)
            timestamps = getattr(source, "frame_timestamps", None)
            if timestamps is None:
                return
        if self._resizing and self.seq_clip.legacy_timing and self.seq_clip.legacy_timing.get("status") == "unresolved":
            return

        if self.seq_clip.is_frame_entry and self._resizing:
            self.seq_clip.hold_duration = None
            if self._resizing == "left":
                shift = max(-self._original_start_frame,
                            min(delta_frames, self._original_hold_frames - 1))
                self.seq_clip.start_frame = self._original_start_frame + shift
                self.seq_clip.hold_frames = self._original_hold_frames - shift
            else:
                self.seq_clip.hold_frames = max(1, self._original_hold_frames + delta_frames)
            self.prepareGeometryChange()
            self._update_geometry()
            return

        if self._resizing == "left":
            # Trim in-point: adjust start_frame and in_point
            original_time = source_frame_time(self._original_in_point, rate, timestamps)
            new_in = nearest_source_frame(original_time + Fraction(delta_frames) / timeline_rate, rate, timestamps)

            # Clamp: can't trim past out_point or before 0
            min_in = 0
            max_in = self._original_out_point - 1  # At least 1 frame

            new_in = max(min_in, min(max_in, new_in))
            shift = (source_frame_time(new_in, rate, timestamps) - original_time) * timeline_rate
            new_start = self._original_start_frame + floor(shift + Fraction(1, 2))

            if new_start >= 0:
                self.seq_clip.in_point = new_in
                self.seq_clip.start_frame = new_start

        elif self._resizing == "right":
            # Trim out-point
            original_time = source_frame_time(self._original_out_point, rate, timestamps)
            new_out = nearest_source_frame(original_time + Fraction(delta_frames) / timeline_rate, rate, timestamps)

            # Clamp: must be after in_point
            min_out = self.seq_clip.in_point + 1
            new_out = max(min_out, new_out)

            self.seq_clip.out_point = new_out

        elif self._dragging:
            # Move entire clip
            new_start = self._original_start_frame + delta_frames
            new_start = max(0, new_start)

            # Snap to edges
            new_start = self._snap_to_edges(new_start)
            self.seq_clip.start_frame = new_start

        self.prepareGeometryChange()
        self._update_geometry()

    def mouseReleaseEvent(self, event):
        edited = self._dragging or self._resizing
        scene = self.scene()
        self._dragging = False
        self._resizing = None
        super().mouseReleaseEvent(event)
        if self._edit_source is not None:
            self._commit_history_edit()
        elif edited and scene and hasattr(scene, "clip_moved"):
            scene.clip_moved.emit(self.seq_clip.id, self.seq_clip.start_frame)

    def _commit_history_edit(self) -> None:
        """Publish a gesture once; the live model never holds preview values."""
        preview = self.seq_clip
        self.seq_clip = self._edit_source
        self._edit_source = None
        self._update_geometry()
        if (not any(s is self._edit_sequence for s in self._edit_project.sequences)
                or not any(c is self.seq_clip for c in self._edit_sequence.get_all_clips())
                or not self._edit_before.matches(self.seq_clip)):
            return  # Another edit won while the gesture was in progress.
        self._edit_project.update_sequence_clip(
            self.seq_clip.id, sequence=self._edit_sequence,
            start_frame=preview.start_frame, in_point=preview.in_point,
            out_point=preview.out_point, hold_frames=preview.hold_frames,
        )

    def _snap_to_edges(self, frame: int) -> int:
        """Snap to adjacent clip edges if within threshold."""
        scene = self.scene()
        if not scene or not hasattr(scene, "get_snap_points"):
            return frame

        snap_points = scene.get_snap_points(exclude_clip_id=self.seq_clip.id)
        duration = self.seq_clip.duration_frames

        # Check start edge snapping
        for snap_frame in snap_points:
            if abs(frame - snap_frame) <= self.SNAP_THRESHOLD_FRAMES:
                return snap_frame

        # Check end edge snapping
        end_frame = frame + duration
        for snap_frame in snap_points:
            if abs(end_frame - snap_frame) <= self.SNAP_THRESHOLD_FRAMES:
                return snap_frame - duration

        return frame

    # --- Selection state ---
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self._update_appearance()
        return super().itemChange(change, value)

    # --- Context menu ---
    def contextMenuEvent(self, event):
        from ui.project_access import is_read_only
        if is_read_only(self):
            event.ignore()
            return
        from PySide6.QtWidgets import QMenu

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec_(event.screenPos())

        if action == delete_action:
            scene = self.scene()
            if scene and hasattr(scene, "remove_clip"):
                scene.remove_clip(self.seq_clip.id)
