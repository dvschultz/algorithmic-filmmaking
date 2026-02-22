#!/usr/bin/env python3
"""
Algorithmic Filmmaking - Scene Ripper MVP

A desktop application for video artists to automatically detect and extract
scenes from video files for use in collage filmmaking.
"""

import logging
import os
import sys

from core.paths import is_frozen, get_managed_packages_dir, get_log_dir, ensure_app_dirs


def _setup_frozen_environment():
    """Set up environment for frozen (PyInstaller) app.

    - Creates application directories (bin, packages, logs)
    - Adds managed packages dir to sys.path for on-demand packages
    - Configures file-based logging to ~/Library/Logs/Scene Ripper/
    """
    ensure_app_dirs()

    # Add managed packages dir to sys.path so on-demand packages are importable
    packages_dir = get_managed_packages_dir()
    if packages_dir.exists():
        packages_str = str(packages_dir)
        if packages_str not in sys.path:
            sys.path.insert(0, packages_str)

    # Set up file logging for frozen app (users can't see console output)
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scene-ripper.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


# Set up logging early
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Frozen app setup (before importing heavy dependencies)
if is_frozen():
    _setup_frozen_environment()

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.theme import theme


def main():
    logger.info("=== MAIN() CALLED ===")
    logger.info(f"sys.argv: {sys.argv}")
    logger.info(f"Frozen: {is_frozen()}")

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
