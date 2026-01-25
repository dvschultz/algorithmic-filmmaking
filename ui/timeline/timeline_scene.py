"""Timeline scene holding all timeline items."""

from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtCore import Signal, QRectF, QTimer
from PySide6.QtGui import QColor, QPen, QBrush

from models.sequence import Sequence, Track, SequenceClip
from ui.theme import theme


class TimelineScene(QGraphicsScene):
    """Scene containing all timeline elements (tracks, clips, playhead)."""

    # Layout constants
    RULER_HEIGHT = 30
    TRACK_HEIGHT = 60
    TRACK_SPACING = 2
    TRACK_HEADER_WIDTH = 80

    # Signals
    playhead_moved = Signal(float)  # Emits time in seconds
    clip_moved = Signal(str, int)  # clip_id, new_start_frame
    clip_selected = Signal(str)  # clip_id

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sequence: Sequence = Sequence()
        self.pixels_per_second = 100.0  # Zoom level
        self._track_items = []  # TrackItem instances
        self._clip_items = {}  # clip_id -> ClipItem
        self._playhead = None

        # Debounced zoom timer
        self._pending_pps = None
        self._zoom_timer = QTimer()
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(16)  # ~60fps
        self._zoom_timer.timeout.connect(self._apply_pending_zoom)

        self._setup_scene()

    def _setup_scene(self):
        """Initialize scene with default dimensions."""
        self.setBackgroundBrush(QBrush(theme().colors.qcolor('timeline_background')))
        self.rebuild()  # Build track items for default sequence

    def _update_scene_rect(self):
        """Recalculate scene dimensions based on content."""
        # Height: ruler + tracks
        height = self.RULER_HEIGHT + len(self.sequence.tracks) * (
            self.TRACK_HEIGHT + self.TRACK_SPACING
        )
        height = max(height, 200)  # Minimum height

        # Width: based on sequence duration or minimum
        min_width = 1000
        content_width = self.sequence.duration_seconds * self.pixels_per_second
        width = max(min_width, content_width + 200)  # Extra space at end

        self.setSceneRect(0, 0, width, height)

    def set_sequence(self, sequence: Sequence):
        """Set the sequence to display."""
        self.sequence = sequence
        self.rebuild()

    def rebuild(self):
        """Rebuild all items from sequence data."""
        # Clear existing items (except playhead)
        for item in list(self._track_items):
            self.removeItem(item)
        self._track_items.clear()

        for clip_item in list(self._clip_items.values()):
            self.removeItem(clip_item)
        self._clip_items.clear()

        # Rebuild tracks and clips
        from ui.timeline.track_item import TrackItem
        from ui.timeline.clip_item import ClipItem

        for i, track in enumerate(self.sequence.tracks):
            y_pos = self.RULER_HEIGHT + i * (self.TRACK_HEIGHT + self.TRACK_SPACING)
            track_item = TrackItem(track, i, y_pos, self.TRACK_HEIGHT, self)
            self.addItem(track_item)
            self._track_items.append(track_item)

            # Add clips for this track
            for seq_clip in track.clips:
                clip_item = ClipItem(
                    seq_clip,
                    track_item,
                    self.pixels_per_second,
                    self.sequence.fps,
                )
                self.addItem(clip_item)
                self._clip_items[seq_clip.id] = clip_item

        self._update_scene_rect()

    def set_pixels_per_second(self, pps: float):
        """Update zoom level with debouncing for smooth performance."""
        self._pending_pps = pps
        self.pixels_per_second = pps
        self._update_scene_rect()

        # Update playhead immediately (it's just one item)
        if self._playhead:
            self._playhead.set_pixels_per_second(pps)

        # Debounce clip updates for smooth zoom
        if not self._zoom_timer.isActive():
            self._zoom_timer.start()

    def _apply_pending_zoom(self):
        """Apply pending zoom to all clip items."""
        if self._pending_pps is None:
            return

        pps = self._pending_pps
        self._pending_pps = None

        # Only update visible items first if we have views
        views = self.views()
        if views:
            visible_rect = views[0].mapToScene(
                views[0].viewport().rect()
            ).boundingRect()

            # Update visible clips immediately
            for clip_item in self._clip_items.values():
                if clip_item.sceneBoundingRect().intersects(visible_rect):
                    clip_item.set_pixels_per_second(pps)

            # Schedule off-screen updates for next frame
            QTimer.singleShot(0, lambda: self._update_offscreen_clips(pps, visible_rect))
        else:
            # No views, update all
            for clip_item in self._clip_items.values():
                clip_item.set_pixels_per_second(pps)

        self.update()

    def _update_offscreen_clips(self, pps: float, visible_rect):
        """Update clips that were off-screen during zoom."""
        for clip_item in self._clip_items.values():
            if not clip_item.sceneBoundingRect().intersects(visible_rect):
                clip_item.set_pixels_per_second(pps)

    def get_track_at_y(self, y: float) -> int:
        """Get track index at y coordinate, or -1 if none."""
        if y < self.RULER_HEIGHT:
            return -1

        relative_y = y - self.RULER_HEIGHT
        track_index = int(relative_y / (self.TRACK_HEIGHT + self.TRACK_SPACING))

        if 0 <= track_index < len(self.sequence.tracks):
            return track_index
        return -1

    def get_frame_at_x(self, x: float) -> int:
        """Convert x coordinate to frame number."""
        seconds = x / self.pixels_per_second
        return int(seconds * self.sequence.fps)

    def get_x_for_frame(self, frame: int) -> float:
        """Convert frame number to x coordinate."""
        seconds = frame / self.sequence.fps
        return seconds * self.pixels_per_second

    def add_clip_to_track(
        self,
        track_index: int,
        source_clip_id: str,
        source_id: str,
        start_frame: int,
        in_point: int,
        out_point: int,
        thumbnail_path: str = None,
    ) -> SequenceClip:
        """Add a new clip to a track."""
        if track_index < 0 or track_index >= len(self.sequence.tracks):
            return None

        seq_clip = SequenceClip(
            source_clip_id=source_clip_id,
            source_id=source_id,
            track_index=track_index,
            start_frame=start_frame,
            in_point=in_point,
            out_point=out_point,
        )

        track = self.sequence.tracks[track_index]
        track.add_clip(seq_clip)

        # Create visual item
        from ui.timeline.clip_item import ClipItem

        track_item = self._track_items[track_index]
        clip_item = ClipItem(
            seq_clip,
            track_item,
            self.pixels_per_second,
            self.sequence.fps,
            thumbnail_path=thumbnail_path,
        )
        self.addItem(clip_item)
        self._clip_items[seq_clip.id] = clip_item

        self._update_scene_rect()
        return seq_clip

    def remove_clip(self, clip_id: str):
        """Remove a clip from the timeline."""
        if clip_id in self._clip_items:
            clip_item = self._clip_items.pop(clip_id)
            self.removeItem(clip_item)

            # Remove from sequence data
            for track in self.sequence.tracks:
                track.remove_clip(clip_id)

            self._update_scene_rect()

    def clear_all_clips(self):
        """Remove all clips from the timeline."""
        for clip_item in list(self._clip_items.values()):
            self.removeItem(clip_item)
        self._clip_items.clear()

        for track in self.sequence.tracks:
            track.clips.clear()

        self._update_scene_rect()

    def set_playhead(self, playhead):
        """Set the playhead item reference."""
        self._playhead = playhead

    def get_snap_points(self, exclude_clip_id: str = None) -> list[int]:
        """Get list of frame positions to snap to (clip edges)."""
        snap_points = [0]  # Always snap to start

        for track in self.sequence.tracks:
            for clip in track.clips:
                if clip.id != exclude_clip_id:
                    snap_points.append(clip.start_frame)
                    snap_points.append(clip.end_frame())

        return sorted(set(snap_points))
