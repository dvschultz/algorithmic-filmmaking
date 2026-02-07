"""Frame browser with virtualized thumbnail grid using QListView.

Uses QListView + QAbstractListModel for virtualized rendering, so only
visible items are painted. Critical for handling 10,000+ frames efficiently.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListView,
    QAbstractItemView,
    QStyledItemDelegate,
    QStyle,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QAbstractListModel,
    QModelIndex,
    QSize,
    QRect,
    QPoint,
)
from PySide6.QtGui import QPixmap, QPixmapCache, QPainter, QColor, QFont, QPen

from models.frame import Frame
from ui.theme import theme, TypeScale, Spacing, Radii

logger = logging.getLogger(__name__)

# Custom data roles
FrameIdRole = Qt.UserRole
AnalyzedRole = Qt.UserRole + 1
ThumbnailPathRole = Qt.UserRole + 2

# Zoom level configuration: level -> (thumb_width, thumb_height, cell_width, cell_height)
_ZOOM_SIZES = {
    1: (80, 60, 100, 90),
    2: (120, 90, 140, 115),
    3: (160, 120, 180, 150),
    4: (200, 150, 220, 180),
    5: (260, 195, 280, 225),
}
_DEFAULT_ZOOM = 3


class FrameBrowserModel(QAbstractListModel):
    """Backing model for the frame browser QListView.

    Stores a flat list of Frame objects and provides data through
    Qt's model/view roles.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list[Frame] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._frames)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._frames):
            return None

        frame = self._frames[index.row()]

        if role == Qt.DisplayRole:
            return frame.display_name()
        elif role == Qt.DecorationRole:
            if frame.thumbnail_path and frame.thumbnail_path.exists():
                return str(frame.thumbnail_path)
            return None
        elif role == FrameIdRole:
            return frame.id
        elif role == AnalyzedRole:
            return frame.analyzed
        elif role == ThumbnailPathRole:
            if frame.thumbnail_path:
                return str(frame.thumbnail_path)
            return None

        return None

    def set_frames(self, frames: list[Frame]):
        """Replace the entire frame list."""
        self.beginResetModel()
        self._frames = list(frames)
        self.endResetModel()

    def get_frame(self, index: QModelIndex) -> Optional[Frame]:
        """Get the Frame object for a model index."""
        if not index.isValid() or index.row() >= len(self._frames):
            return None
        return self._frames[index.row()]

    def frame_count(self) -> int:
        """Return the number of frames in the model."""
        return len(self._frames)


class FrameThumbnailDelegate(QStyledItemDelegate):
    """Custom delegate that paints frame thumbnails with labels.

    Renders:
    - Thumbnail image (or gray placeholder if missing)
    - Frame number/filename text below the thumbnail
    - Small colored indicator dot if analyzed
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_level = _DEFAULT_ZOOM
        self._pixmap_cache: dict[str, QPixmap] = {}

    def set_zoom_level(self, level: int):
        """Set the zoom level (1-5)."""
        self._zoom_level = max(1, min(5, level))

    @property
    def _sizes(self) -> tuple[int, int, int, int]:
        """Return (thumb_w, thumb_h, cell_w, cell_h) for current zoom."""
        return _ZOOM_SIZES[self._zoom_level]

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        _, _, cell_w, cell_h = self._sizes
        return QSize(cell_w, cell_h)

    def _get_pixmap(self, path_str: str, thumb_w: int, thumb_h: int) -> Optional[QPixmap]:
        """Load and cache a thumbnail pixmap scaled to the requested size."""
        cache_key = f"frame_browser:{path_str}:{thumb_w}x{thumb_h}"
        pm = QPixmapCache.find(cache_key)
        if pm and not pm.isNull():
            return pm

        source_pm = QPixmap(path_str)
        if source_pm.isNull():
            return None

        scaled = source_pm.scaled(
            thumb_w, thumb_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        QPixmapCache.insert(cache_key, scaled)
        return scaled

    def paint(self, painter: QPainter, option, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        t = theme()
        thumb_w, thumb_h, cell_w, cell_h = self._sizes
        rect = option.rect

        # Selection / hover background
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hovered = bool(option.state & QStyle.State_MouseOver)

        if is_selected:
            bg_color = QColor(t.accent_blue)
            bg_color.setAlpha(40)
            painter.fillRect(rect, bg_color)
        elif is_hovered:
            bg_color = QColor(t.card_hover)
            painter.fillRect(rect, bg_color)

        # Thumbnail area: centered horizontally within the cell
        thumb_x = rect.x() + (cell_w - thumb_w) // 2
        thumb_y = rect.y() + Spacing.XS
        thumb_rect = QRect(thumb_x, thumb_y, thumb_w, thumb_h)

        # Draw thumbnail or placeholder
        path_str = index.data(ThumbnailPathRole)
        if path_str:
            pm = self._get_pixmap(path_str, thumb_w, thumb_h)
            if pm and not pm.isNull():
                # Center the (potentially smaller) pixmap within thumb_rect
                px = thumb_x + (thumb_w - pm.width()) // 2
                py = thumb_y + (thumb_h - pm.height()) // 2
                painter.drawPixmap(QPoint(px, py), pm)
            else:
                self._paint_placeholder(painter, thumb_rect, t)
        else:
            self._paint_placeholder(painter, thumb_rect, t)

        # Selection border around thumbnail
        if is_selected:
            pen = QPen(QColor(t.accent_blue), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                thumb_rect.adjusted(-1, -1, 1, 1),
                Radii.SM, Radii.SM,
            )

        # Text label below thumbnail
        text = index.data(Qt.DisplayRole) or ""
        text_y = thumb_y + thumb_h + Spacing.XXS
        text_rect = QRect(rect.x(), text_y, cell_w, cell_h - thumb_h - Spacing.XS - Spacing.XXS)

        font = painter.font()
        font.setPixelSize(TypeScale.XS)
        painter.setFont(font)
        painter.setPen(QColor(t.text_secondary))
        painter.drawText(
            text_rect,
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextSingleLine,
            text,
        )

        # Analyzed indicator dot (top-right of thumbnail)
        analyzed = index.data(AnalyzedRole)
        if analyzed:
            dot_radius = 4
            dot_x = thumb_x + thumb_w - dot_radius - 3
            dot_y = thumb_y + 3 + dot_radius
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.accent_green))
            painter.drawEllipse(QPoint(dot_x, dot_y), dot_radius, dot_radius)

        painter.restore()

    def _paint_placeholder(self, painter: QPainter, rect: QRect, t):
        """Paint a gray rounded-rect placeholder for missing thumbnails."""
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t.thumbnail_background))
        painter.drawRoundedRect(rect, Radii.SM, Radii.SM)

        # Draw a small "no image" icon-like indicator
        painter.setPen(QColor(t.text_muted))
        font = painter.font()
        font.setPixelSize(TypeScale.SM)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, "?")


class FrameBrowser(QWidget):
    """Virtualized frame grid browser.

    Uses QListView in IconMode with wrapping for a responsive grid layout.
    Only visible items are rendered, making it efficient for 10,000+ frames.

    Signals:
        frame_selected: Emitted with frame_id when a single frame is clicked.
        frames_selected: Emitted with list of frame_ids when selection changes.
        frame_double_clicked: Emitted with frame_id on double-click.
    """

    frame_selected = Signal(str)
    frames_selected = Signal(list)
    frame_double_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_level = _DEFAULT_ZOOM
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Model
        self._model = FrameBrowserModel(self)

        # Delegate
        self._delegate = FrameThumbnailDelegate(self)

        # List view configured as icon grid
        self._view = QListView()
        self._view.setModel(self._model)
        self._view.setItemDelegate(self._delegate)
        self._view.setViewMode(QListView.ViewMode.IconMode)
        self._view.setWrapping(True)
        self._view.setResizeMode(QListView.ResizeMode.Adjust)
        self._view.setMovement(QListView.Movement.Static)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.setUniformItemSizes(True)
        self._view.setSpacing(Spacing.SM)
        self._view.setFlow(QListView.Flow.LeftToRight)

        # Styling
        t = theme()
        self._view.setStyleSheet(
            f"QListView {{ background-color: {t.background_primary}; border: none; }}"
            f"QListView::item:selected {{ background: transparent; }}"
            f"QListView::item:hover {{ background: transparent; }}"
        )

        # Connect signals
        self._view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._view.doubleClicked.connect(self._on_double_clicked)

        # Apply initial zoom
        self._apply_zoom()

        layout.addWidget(self._view)

    def set_frames(self, frames: list[Frame]):
        """Replace the displayed frames."""
        self._model.set_frames(frames)

    def set_zoom(self, level: int):
        """Set the zoom level (1-5). Controls thumbnail size."""
        level = max(1, min(5, level))
        if level == self._zoom_level:
            return
        self._zoom_level = level
        self._apply_zoom()

    def _apply_zoom(self):
        """Apply the current zoom level to the delegate and view."""
        self._delegate.set_zoom_level(self._zoom_level)
        _, _, cell_w, cell_h = _ZOOM_SIZES[self._zoom_level]
        self._view.setGridSize(QSize(cell_w, cell_h))
        # Force a relayout
        self._view.doItemsLayout()

    def get_selected_frame_ids(self) -> list[str]:
        """Return list of selected frame IDs."""
        ids = []
        for index in self._view.selectionModel().selectedIndexes():
            frame_id = index.data(FrameIdRole)
            if frame_id:
                ids.append(frame_id)
        return ids

    def clear(self):
        """Clear all frames from the browser."""
        self._model.set_frames([])

    def _on_selection_changed(self, selected, deselected):
        """Handle selection changes in the view."""
        ids = self.get_selected_frame_ids()
        self.frames_selected.emit(ids)
        if len(ids) == 1:
            self.frame_selected.emit(ids[0])

    def _on_double_clicked(self, index: QModelIndex):
        """Handle double-click on an item."""
        frame_id = index.data(FrameIdRole)
        if frame_id:
            self.frame_double_clicked.emit(frame_id)
