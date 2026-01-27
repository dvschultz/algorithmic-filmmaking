"""Custom styled slider that properly handles background transparency on macOS."""

from PySide6.QtWidgets import QSlider, QStyleFactory
from PySide6.QtCore import Qt


class StyledSlider(QSlider):
    """A QSlider that uses Fusion style to ensure stylesheet backgrounds work properly.

    On macOS, the native Cocoa slider style paints behind Qt stylesheets,
    causing unwanted gray backgrounds. Using Fusion style fixes this.

    Trade-off: This may cause minor visual inconsistency with native widgets on
    Windows, as Fusion has a distinct look. However, this ensures consistent
    theming behavior across all platforms.
    """

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        # Use Fusion style which respects stylesheets properly
        fusion_style = QStyleFactory.create("Fusion")
        if fusion_style:
            self.setStyle(fusion_style)
        # Ensure styled background is painted
        self.setAttribute(Qt.WA_StyledBackground, True)
