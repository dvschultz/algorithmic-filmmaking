"""Qt-free drawing raster used by Signature Style."""

from pathlib import Path

import pytest
from PIL import Image

from core.remix.drawing_image import DrawingImage


def _image(mode="RGB"):
    image = Image.new(mode, (10, 4), (255, 255, 255) if mode == "RGB" else (255, 255, 255, 255))
    image.putpixel((2, 1), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 255))
    return image


def test_wraps_pil_images_with_qimage_shaped_accessors():
    drawing = DrawingImage(_image())
    assert (drawing.width(), drawing.height()) == (10, 4)
    color = drawing.pixelColor(2, 1)
    assert (color.red(), color.green(), color.blue()) == (200, 30, 30)
    assert drawing.pixelColor(0, 0).red() == 255


def test_rgba_input_is_flattened_to_rgb():
    drawing = DrawingImage(_image("RGBA"))
    assert drawing.pixelColor(2, 1).blue() == 30


def test_copy_crops_and_png_round_trip(tmp_path):
    drawing = DrawingImage(_image())
    strip = drawing.copy(2, 0, 3, 4)
    assert (strip.width(), strip.height()) == (3, 4) and strip.pixelColor(0, 1).red() == 200
    assert strip.to_png_base64().startswith("iVBOR")
    path = tmp_path / "d.png"
    drawing.save(path)
    loaded = DrawingImage.load(path)
    assert loaded.pixelColor(2, 1).green() == 30


def test_rejects_non_pil_input_and_missing_files(tmp_path):
    with pytest.raises(TypeError):
        DrawingImage(object())
    with pytest.raises(FileNotFoundError):
        DrawingImage.load(Path(tmp_path / "missing.png"))


def test_samplers_accept_the_drawing_image():
    from core.remix.drawing_vlm import _image_to_base64, slice_drawing_adaptive
    from core.remix.signature_style import sample_drawing_parametric

    image = Image.new("RGB", (120, 30), "white")
    for x in range(10, 50):
        for y in range(5, 20):
            image.putpixel((x, y), (30, 30, 220))
    drawing = DrawingImage(image)
    assert sample_drawing_parametric(drawing, 6.0, 12)
    assert slice_drawing_adaptive(drawing)
    assert _image_to_base64(drawing) == drawing.to_png_base64()
