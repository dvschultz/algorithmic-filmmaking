#!/usr/bin/env python3
"""
Algorithmic Filmmaking - Scene Ripper MVP

A desktop application for video artists to automatically detect and extract
scenes from video files for use in collage filmmaking.
"""

import locale
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
            # Append (not insert) so bundled/stdlib packages take priority
            # over on-demand packages, preventing module shadowing attacks
            sys.path.append(packages_str)

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

import platform
if platform.system() == "Darwin":
    # Homebrew on Apple Silicon / Intel puts libs in different locations.
    # python-mpv needs libmpv.dylib on the loader path.
    for lib_dir in ("/opt/homebrew/lib", "/usr/local/lib"):
        if os.path.isdir(lib_dir):
            existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            if lib_dir not in existing:
                os.environ["DYLD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{existing}" if existing else lib_dir
                )
            break

from PySide6.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow
from ui.theme import theme

# Fix LC_NUMERIC before any MPV usage — PySide6 may override this on import.
# MPV requires 'C' locale for numeric parsing (decimal points vs commas).
locale.setlocale(locale.LC_NUMERIC, 'C')


def _check_mpv_available() -> bool:
    """Check if libmpv is available. Returns True if OK, False if missing."""
    try:
        import mpv  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _show_mpv_missing_dialog(app: QApplication):
    """Show a user-friendly dialog when libmpv is missing."""
    import platform
    system = platform.system()
    if system == "Darwin":
        install_cmd = "brew install mpv"
    elif system == "Windows":
        install_cmd = "choco install mpv\n\nOr download from https://mpv.io/installation/"
    else:
        install_cmd = "sudo apt install libmpv-dev\n\nOr: sudo dnf install mpv-libs-devel"

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Missing Dependency: libmpv")
    msg.setText("Scene Ripper requires libmpv for video playback.")
    msg.setInformativeText(f"Install it with:\n\n{install_cmd}")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def main():
    logger.info("=== MAIN() CALLED ===")
    logger.info(f"sys.argv: {sys.argv}")
    logger.info(f"Frozen: {is_frozen()}")

    app = QApplication(sys.argv)
    app.setApplicationName("Scene Ripper")
    app.setOrganizationName("Algorithmic Filmmaking")

    # Check for libmpv before creating the main window
    if not _check_mpv_available():
        logger.error("libmpv not found — showing install dialog")
        _show_mpv_missing_dialog(app)
        sys.exit(1)

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
