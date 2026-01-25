"""Timeline view with zoom and scroll support."""

from PySide6.QtWidgets import QGraphicsView
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui.theme import theme


class TimelineView(QGraphicsView):
    """Graphics view for the timeline with zoom and scroll support."""

    zoom_changed = Signal(float)  # Emits pixels_per_second
    position_clicked = Signal(float)  # Emits time in seconds when ruler clicked

    MIN_PIXELS_PER_SECOND = 10
    MAX_PIXELS_PER_SECOND = 500
    DEFAULT_PIXELS_PER_SECOND = 100

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pixels_per_second = self.DEFAULT_PIXELS_PER_SECOND
        self._ruler_height = 30  # Matches TimelineScene.RULER_HEIGHT

        self._setup_view()

    def _setup_view(self):
        """Configure view settings."""
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        # Scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Performance
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheBackground)

        # Drag mode - rubber band for selection
        self.setDragMode(QGraphicsView.RubberBandDrag)

        # Accept drops
        self.setAcceptDrops(True)

        # Style
        self._update_style()

    def _update_style(self):
        """Update style based on current theme."""
        self.setStyleSheet(f"""
            QGraphicsView {{
                background-color: {theme().timeline_background};
                border: none;
            }}
        """)

    def wheelEvent(self, event):
        """Handle zoom with Ctrl+wheel, scroll otherwise."""
        if event.modifiers() & Qt.ControlModifier:
            # Zoom
            delta = event.angleDelta().y()
            zoom_factor = 1.15 if delta > 0 else 1 / 1.15

            new_pps = self.pixels_per_second * zoom_factor
            new_pps = max(
                self.MIN_PIXELS_PER_SECOND,
                min(self.MAX_PIXELS_PER_SECOND, new_pps),
            )

            if new_pps != self.pixels_per_second:
                # Get mouse position in scene before zoom
                mouse_scene_pos = self.mapToScene(event.position().toPoint())
                mouse_time = mouse_scene_pos.x() / self.pixels_per_second

                # Apply zoom
                self.pixels_per_second = new_pps

                # Update scene
                if self.scene():
                    self.scene().set_pixels_per_second(new_pps)

                # Scroll to keep mouse position stable
                new_scene_x = mouse_time * self.pixels_per_second
                self.centerOn(new_scene_x, mouse_scene_pos.y())

                self.zoom_changed.emit(self.pixels_per_second)
        else:
            # Normal horizontal/vertical scroll
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle clicks on ruler to set playhead position."""
        scene_pos = self.mapToScene(event.pos())

        # Check if click is in ruler area
        if scene_pos.y() < self._ruler_height:
            time_seconds = scene_pos.x() / self.pixels_per_second
            time_seconds = max(0, time_seconds)
            self.position_clicked.emit(time_seconds)
            return

        super().mousePressEvent(event)

    def drawForeground(self, painter: QPainter, rect):
        """Draw time ruler that doesn't scale with zoom."""
        super().drawForeground(painter, rect)

        # Map viewport to scene coordinates
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        # Draw ruler background
        ruler_rect = view_rect
        ruler_rect.setHeight(self._ruler_height)
        ruler_rect.setTop(0)
        painter.fillRect(ruler_rect, theme().colors.qcolor('timeline_ruler'))

        # Draw bottom border
        painter.setPen(QPen(theme().colors.qcolor('timeline_ruler_border'), 1))
        painter.drawLine(
            int(view_rect.left()),
            self._ruler_height,
            int(view_rect.right()),
            self._ruler_height,
        )

        # Calculate tick interval based on zoom
        seconds_visible = view_rect.width() / self.pixels_per_second
        tick_interval = self._calculate_tick_interval(seconds_visible)

        # Draw ticks and labels
        start_time = max(0, int(view_rect.left() / self.pixels_per_second))
        end_time = int(view_rect.right() / self.pixels_per_second) + 1

        painter.setPen(theme().colors.qcolor('timeline_ruler_tick'))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for t in range(start_time, end_time):
            x = t * self.pixels_per_second

            if t % tick_interval == 0:
                # Major tick
                painter.setPen(theme().colors.qcolor('timeline_ruler_tick'))
                painter.drawLine(int(x), 20, int(x), self._ruler_height)
                painter.drawText(int(x) + 3, 16, self._format_time(t))
            elif tick_interval <= 5 and t % (tick_interval // 2 or 1) == 0:
                # Minor tick
                painter.setPen(theme().colors.qcolor('timeline_ruler_tick_minor'))
                painter.drawLine(int(x), 25, int(x), self._ruler_height)

    def _calculate_tick_interval(self, seconds_visible: float) -> int:
        """Choose appropriate tick spacing based on zoom level."""
        if seconds_visible < 10:
            return 1
        elif seconds_visible < 30:
            return 5
        elif seconds_visible < 120:
            return 10
        elif seconds_visible < 300:
            return 30
        elif seconds_visible < 600:
            return 60
        else:
            return 300

    def _format_time(self, seconds: int) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def set_zoom_to_fit(self, duration_seconds: float):
        """Zoom to fit entire duration in view."""
        if duration_seconds <= 0:
            return

        available_width = self.viewport().width() - 40  # Margin
        new_pps = available_width / duration_seconds
        new_pps = max(self.MIN_PIXELS_PER_SECOND, new_pps)

        self.pixels_per_second = new_pps
        if self.scene():
            self.scene().set_pixels_per_second(new_pps)
        self.zoom_changed.emit(self.pixels_per_second)

    def reset_zoom(self):
        """Reset to default zoom level."""
        self.pixels_per_second = self.DEFAULT_PIXELS_PER_SECOND
        if self.scene():
            self.scene().set_pixels_per_second(self.pixels_per_second)
        self.zoom_changed.emit(self.pixels_per_second)
