"""Gradient glow painting helpers for card widgets.

Provides a reusable function for painting soft radial gradient glow effects
around card rectangles. Used by SortingCard and ClipThumbnail for the
premium visual refresh.
"""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtWidgets import QLabel

from ui.theme import Radii


def paint_gradient_glow(
    painter: QPainter,
    rect: QRectF,
    colors: list[tuple[int, int, int]],
    opacity: int = 160,
    spread: int = 6,
    radius: int = Radii.MD,
):
    """Paint a soft radial gradient glow around a card rectangle.

    Args:
        painter: Active QPainter (must already have Antialiasing enabled)
        rect: Card body rectangle
        colors: 2-3 RGB tuples for gradient stops
        opacity: Alpha (0-255) for glow intensity
        spread: Pixels beyond card edge
        radius: Border-radius matching the card
    """
    if not colors:
        return

    glow_rect = rect.adjusted(-spread, -spread, spread, spread)
    center = glow_rect.center()
    gradient = QRadialGradient(
        center, max(glow_rect.width(), glow_rect.height()) / 2
    )

    # Distribute colors across 0.0-0.7, then fade to transparent
    n = min(len(colors), 3)
    for i, (r, g, b) in enumerate(colors[:3]):
        stop = (i / max(n - 1, 1)) * 0.7
        gradient.setColorAt(stop, QColor(r, g, b, opacity))
    gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

    painter.save()
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(gradient))
    painter.drawRoundedRect(glow_rect, radius + spread, radius + spread)
    painter.restore()


def paint_card_body(
    painter: QPainter,
    rect: QRectF,
    background: QColor,
    border: QColor,
    radius: int = Radii.MD,
    border_width: int = 1,
):
    """Paint a solid rounded card body.

    Args:
        painter: Active QPainter
        rect: Card body rectangle
        background: Fill color
        border: Border color
        radius: Border-radius
        border_width: Border width in pixels
    """
    from PySide6.QtGui import QPen

    painter.save()
    painter.setBrush(QBrush(background))
    if border_width > 0:
        painter.setPen(QPen(border, border_width))
    else:
        painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(rect, radius, radius)
    painter.restore()


class RoundedTopLabel(QLabel):
    """QLabel that clips its pixmap with rounded top corners.

    Used for thumbnail images inside rounded cards so the image doesn't
    poke out of the card's top corners.
    """

    def __init__(self, radius: int = Radii.MD, parent=None):
        super().__init__(parent)
        self._radius = radius

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(self.rect())
        r = min(self._radius, rect.width() / 2, rect.height() / 2)

        # Build clip path: rounded top corners, square bottom corners
        path = QPainterPath()
        path.moveTo(rect.left(), rect.top() + r)
        path.arcTo(rect.left(), rect.top(), r * 2, r * 2, 180, -90)
        path.lineTo(rect.right() - r, rect.top())
        path.arcTo(rect.right() - r * 2, rect.top(), r * 2, r * 2, 90, -90)
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()

        painter.setClipPath(path)

        # Draw background first (visible when pixmap doesn't fill entirely)
        painter.fillRect(rect, self.palette().window())

        # Center the pixmap
        px_rect = pixmap.rect()
        x = (rect.width() - px_rect.width()) / 2
        y = (rect.height() - px_rect.height()) / 2
        painter.drawPixmap(int(x), int(y), pixmap)

        painter.end()
