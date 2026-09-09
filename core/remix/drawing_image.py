"""Qt-free drawing image used by Signature Style definitions.

Provides the small pixel-access surface the parametric sampler and the VLM
slicer need (``width``, ``height``, ``pixelColor``, ``copy``) so the same
code runs on a ``QImage`` in the desktop dialog or on a PNG asset headlessly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _Color:
    r: int
    g: int
    b: int

    def red(self) -> int:
        return self.r

    def green(self) -> int:
        return self.g

    def blue(self) -> int:
        return self.b


class DrawingImage:
    """RGB raster with the QImage-shaped accessors the samplers use."""

    def __init__(self, image: Any) -> None:
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("DrawingImage wraps a PIL image")
        self._image = image.convert("RGB")

    @classmethod
    def load(cls, path: str | Path) -> "DrawingImage":
        from PIL import Image

        with Image.open(path) as opened:
            return cls(opened.copy())

    def width(self) -> int:
        return self._image.width

    def height(self) -> int:
        return self._image.height

    def pixelColor(self, x: int, y: int) -> _Color:  # noqa: N802 - QImage-compatible name
        r, g, b = self._image.getpixel((x, y))[:3]  # type: ignore[index]
        return _Color(int(r), int(g), int(b))

    def copy(self, x: int, y: int, w: int, h: int) -> "DrawingImage":
        return DrawingImage(self._image.crop((x, y, x + w, y + h)))

    def to_png_base64(self) -> str:
        buffer = BytesIO()
        self._image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def save(self, path: str | Path) -> None:
        self._image.save(path, format="PNG")


def save_qimage_png(image: Any, path: str | Path) -> Path:
    """Persist a desktop canvas image as a PNG asset the definition can load."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(target), "PNG"):
        raise RuntimeError(f"Could not save drawing to {target}")
    return target
