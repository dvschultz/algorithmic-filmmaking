"""Drawing canvas widget for the Signature Style sequencer.

A PySide6 QWidget-based drawing canvas with pen/eraser tools, undo/redo,
image import, and QPainter-based stroke rendering.

Usage:
    from ui.widgets.drawing_canvas import DrawingCanvas

    canvas = DrawingCanvas()
    canvas.set_color(QColor("red"))
    canvas.set_tool("eraser")
    canvas.undo()
    canvas.redo()
    canvas.clear()

    image = canvas.get_image()
"""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, Qt, QPoint, QSize, QRect
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QResizeEvent,
)

from ui.theme import theme

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_UNDO_SNAPSHOTS = 50
_DEFAULT_PEN_WIDTH = 10
_MIN_CANVAS_WIDTH = 400
_MIN_CANVAS_HEIGHT = 300
_MAX_IMAGE_DIMENSION = 4096
_BACKGROUND_COLOR = QColor(Qt.white)

# Supported image formats for load_image
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class DrawingCanvas(QWidget):
    """A drawing canvas that supports pen and eraser tools with undo/redo.

    The canvas maintains an internal QImage that is rendered via QPainter.
    All drawing operations (pen strokes, eraser, image import, clear) are
    reflected on this internal image and can be retrieved with get_image().

    Signals:
        changed: Emitted after each stroke or operation that modifies the canvas.
    """

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Tool state
        self._tool: str = "pen"  # "pen" or "eraser"
        self._pen_color: QColor = QColor(Qt.black)
        self._pen_width: int = _DEFAULT_PEN_WIDTH
        self._drawing: bool = False
        self._last_point: QPoint = QPoint()

        # Canvas image (initialized in _init_canvas_image)
        self._canvas_image: QImage = QImage()

        # Undo / redo stacks (QImage snapshots)
        self._undo_stack: list[QImage] = []
        self._redo_stack: list[QImage] = []

        # Widget configuration
        self.setMinimumSize(_MIN_CANVAS_WIDTH, _MIN_CANVAS_HEIGHT)
        self.setMouseTracking(False)

        # Initialize the canvas image to match current widget size
        self._init_canvas_image(self.size())

        # Theme integration for border styling
        self._apply_border_style()
        if theme().changed:
            theme().changed.connect(self._apply_border_style)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Set the active drawing tool.

        Args:
            tool: Either "pen" or "eraser".

        Raises:
            ValueError: If tool is not "pen" or "eraser".
        """
        if tool not in ("pen", "eraser"):
            raise ValueError(f"Unknown tool: {tool!r}. Use 'pen' or 'eraser'.")
        self._tool = tool

    def tool(self) -> str:
        """Return the current tool name."""
        return self._tool

    def set_color(self, color: QColor) -> None:
        """Set the pen color.

        Args:
            color: The QColor to use for pen strokes.
        """
        self._pen_color = QColor(color)

    def color(self) -> QColor:
        """Return the current pen color."""
        return QColor(self._pen_color)

    def set_pen_width(self, width: int) -> None:
        """Set the pen stroke width in pixels.

        Args:
            width: Stroke width (clamped to 1..100).
        """
        self._pen_width = max(1, min(width, 100))

    def pen_width(self) -> int:
        """Return the current pen width."""
        return self._pen_width

    def get_image(self) -> QImage:
        """Return a copy of the current canvas as a QImage."""
        return self._canvas_image.copy()

    def clear(self) -> None:
        """Clear the canvas to white, pushing current state to undo stack."""
        self._push_undo()
        self._redo_stack.clear()
        self._canvas_image.fill(_BACKGROUND_COLOR)
        self.update()
        self.changed.emit()

    def load_image(self, path: str) -> bool:
        """Load an image file onto the canvas.

        The image is scaled to fit the canvas while maintaining its aspect
        ratio, then centered. The previous canvas state is pushed to the
        undo stack.

        Args:
            path: Path to a PNG, JPG, BMP, or TIFF file.

        Returns:
            True if the image was loaded successfully, False otherwise.
        """
        file_path = Path(path)

        # Validate extension
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return False

        # Validate file exists
        if not file_path.is_file():
            return False

        # Load and validate image
        source_image = QImage(str(file_path))
        if source_image.isNull():
            return False

        # Enforce maximum dimensions
        if (
            source_image.width() > _MAX_IMAGE_DIMENSION
            or source_image.height() > _MAX_IMAGE_DIMENSION
        ):
            source_image = source_image.scaled(
                _MAX_IMAGE_DIMENSION,
                _MAX_IMAGE_DIMENSION,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

        # Push current state for undo
        self._push_undo()
        self._redo_stack.clear()

        # Clear canvas and draw the imported image centered
        self._canvas_image.fill(_BACKGROUND_COLOR)

        # Scale to fit canvas while keeping aspect ratio
        canvas_size = self._canvas_image.size()
        scaled = source_image.scaled(
            canvas_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Center the scaled image on the canvas
        x = (canvas_size.width() - scaled.width()) // 2
        y = (canvas_size.height() - scaled.height()) // 2

        painter = QPainter(self._canvas_image)
        painter.drawImage(QPoint(x, y), scaled)
        painter.end()

        self.update()
        self.changed.emit()
        return True

    def undo(self) -> None:
        """Undo the last operation. No-op if undo stack is empty."""
        if not self._undo_stack:
            return

        # Save current state to redo stack
        self._redo_stack.append(self._canvas_image.copy())

        # Restore previous state
        self._canvas_image = self._undo_stack.pop()
        self.update()
        self.changed.emit()

    def redo(self) -> None:
        """Redo the last undone operation. No-op if redo stack is empty."""
        if not self._redo_stack:
            return

        # Save current state to undo stack
        self._undo_stack.append(self._canvas_image.copy())

        # Restore redo state
        self._canvas_image = self._redo_stack.pop()
        self.update()
        self.changed.emit()

    def can_undo(self) -> bool:
        """Return True if there is an operation to undo."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Return True if there is an operation to redo."""
        return len(self._redo_stack) > 0

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """Render the canvas image onto the widget."""
        painter = QPainter(self)
        painter.drawImage(QPoint(0, 0), self._canvas_image)
        painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Handle widget resize by expanding the canvas image.

        Existing content is preserved in the top-left corner. New space is
        filled with white.
        """
        new_size = event.size()
        old_size = self._canvas_image.size()

        # Only resize if the widget actually got larger in either dimension
        if new_size.width() > old_size.width() or new_size.height() > old_size.height():
            new_width = max(new_size.width(), old_size.width())
            new_height = max(new_size.height(), old_size.height())

            new_image = QImage(
                QSize(new_width, new_height),
                QImage.Format_ARGB32_Premultiplied,
            )
            new_image.fill(_BACKGROUND_COLOR)

            # Copy existing content
            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self._canvas_image)
            painter.end()

            self._canvas_image = new_image

        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Begin a stroke on left-button press."""
        if event.button() == Qt.LeftButton:
            # Save snapshot for undo before drawing begins
            self._push_undo()
            self._redo_stack.clear()

            self._drawing = True
            self._last_point = event.position().toPoint()

            # Draw a single dot at the press location
            self._draw_line_to(self._last_point)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Continue the stroke as the mouse moves."""
        if self._drawing and (event.buttons() & Qt.LeftButton):
            current_point = event.position().toPoint()
            self._draw_line_to(current_point)
            self._last_point = current_point

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Finish the stroke on left-button release."""
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self.changed.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_canvas_image(self, size: QSize) -> None:
        """Create or recreate the canvas image at the given size."""
        width = max(size.width(), _MIN_CANVAS_WIDTH)
        height = max(size.height(), _MIN_CANVAS_HEIGHT)
        self._canvas_image = QImage(
            QSize(width, height),
            QImage.Format_ARGB32_Premultiplied,
        )
        self._canvas_image.fill(_BACKGROUND_COLOR)

    def _push_undo(self) -> None:
        """Push a snapshot of the current canvas onto the undo stack.

        Caps the stack at _MAX_UNDO_SNAPSHOTS by discarding the oldest entry.
        """
        if len(self._undo_stack) >= _MAX_UNDO_SNAPSHOTS:
            self._undo_stack.pop(0)
        self._undo_stack.append(self._canvas_image.copy())

    def _draw_line_to(self, end_point: QPoint) -> None:
        """Draw a line segment from _last_point to end_point on the canvas image."""
        painter = QPainter(self._canvas_image)
        painter.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen()
        pen.setWidth(self._pen_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        if self._tool == "eraser":
            pen.setColor(_BACKGROUND_COLOR)
        else:
            pen.setColor(self._pen_color)

        painter.setPen(pen)
        painter.drawLine(self._last_point, end_point)
        painter.end()

        self.update()

    def _apply_border_style(self) -> None:
        """Apply a themed border to the canvas widget."""
        border_color = theme().colors.border_primary
        self.setStyleSheet(
            f"DrawingCanvas {{ "
            f"background-color: white; "
            f"border: 1px solid {border_color}; "
            f"}}"
        )
