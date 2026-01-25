#!/usr/bin/env python3
"""
Algorithmic Filmmaking - Scene Ripper MVP

A desktop application for video artists to automatically detect and extract
scenes from video files for use in collage filmmaking.
"""

import logging
import sys

# Set up logging early
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.theme import theme


def main():
    logger.info("=== MAIN() CALLED ===")
    logger.info(f"sys.argv: {sys.argv}")

    app = QApplication(sys.argv)
    app.setApplicationName("Scene Ripper")
    app.setOrganizationName("Algorithmic Filmmaking")

    logger.info("Creating MainWindow...")
    window = MainWindow()

    # Apply theme (uses saved preference from settings loaded in MainWindow)
    logger.info("Applying initial theme...")
    theme().apply_to_app()

    logger.info("Showing MainWindow...")
    window.show()

    logger.info("Starting event loop...")
    sys.exit(app.exec())


if __name__ == "__main__":
    logger.info("=== SCRIPT START ===")
    main()
