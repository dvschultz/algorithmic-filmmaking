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

from core.app_version import get_app_version
from core.paths import (
    is_frozen,
    get_managed_package_search_paths,
    get_log_dir,
    ensure_app_dirs,
)
from core.runtime_smoke import (
    RUNTIME_SMOKE_TARGET_ENV,
    run_runtime_smoke_target,
)
from core.single_instance import acquire_single_instance_lock, release_single_instance_lock


def _setup_frozen_environment():
    """Set up environment for frozen (PyInstaller) app.

    - Creates application directories (bin, packages, logs)
    - Adds managed packages dir to sys.path for on-demand packages
    - Configures file-based logging to ~/Library/Logs/Scene Ripper/
    """
    ensure_app_dirs()

    # Add managed packages dir to sys.path so on-demand packages are importable
    for packages_dir in get_managed_package_search_paths():
        if not packages_dir.exists():
            continue

        packages_str = str(packages_dir)
        if packages_str in sys.path:
            continue

        # Append (not insert) so bundled/stdlib packages take priority over
        # on-demand packages, preventing module shadowing attacks.
        # Overlay dirs are yielded before the base dir, so newer repairs win.
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
STARTUP_SMOKE_TEST_ENV = "SCENE_RIPPER_STARTUP_SMOKE_TEST"

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

# Pre-import torch BEFORE PySide6 to prevent "function '_has_torch_function'
# already has a docstring" RuntimeError. This conflict occurs when torch's C
# extension init runs after PySide6's Shiboken import hooks are installed.
# Importing torch first avoids the hook interference entirely.
try:
    import torch  # noqa: F401
except (ImportError, RuntimeError):
    pass

from PySide6.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow
from ui.theme import theme

# Fix LC_NUMERIC before any MPV usage — PySide6 may override this on import.
# MPV requires 'C' locale for numeric parsing (decimal points vs commas).
locale.setlocale(locale.LC_NUMERIC, 'C')

# Pre-import MLX on the main thread to avoid a crash when worker threads
# first-import it. MLX initializes Metal resources on first import, and
# PySide6's Shiboken import hook can enter infinite recursion if MLX is
# first imported from a QThread background worker.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pass


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
    logger.info("PID: %s", os.getpid())
    logger.info(f"sys.argv: {sys.argv}")
    logger.info(f"Frozen: {is_frozen()}")

    startup_smoke_test = os.environ.get(STARTUP_SMOKE_TEST_ENV) == "1"
    runtime_smoke_target = os.environ.get(RUNTIME_SMOKE_TARGET_ENV, "").strip()

    if runtime_smoke_target:
        completed_target = run_runtime_smoke_target(runtime_smoke_target)
        logger.info("Runtime smoke test '%s' completed successfully", completed_target)
        return 0

    if is_frozen() and not acquire_single_instance_lock():
        logger.info("Duplicate frozen app launch detected for pid=%s; exiting early", os.getpid())
        return 0

    # Enable GL context sharing so QOpenGLWidget gets a shared context on macOS.
    # Without this, mpv's render API sees stale compositor data in the FBO.
    from PySide6.QtCore import Qt
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    # On macOS, set a global default GL format before QApplication construction.
    # Qt initializes internal shared contexts at app startup; format mismatch can
    # break sharing with QOpenGLWidget and cause mirrored/stale video frames.
    if platform.system() == "Darwin":
        from PySide6.QtGui import QSurfaceFormat

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 2)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)
        logger.info(
            "Configured macOS default OpenGL format: %d.%d CoreProfile depth=%d",
            fmt.majorVersion(),
            fmt.minorVersion(),
            fmt.depthBufferSize(),
        )

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(release_single_instance_lock)
    app.setApplicationName("Scene Ripper")
    app.setApplicationVersion(get_app_version())
    app.setOrganizationName("Algorithmic Filmmaking")

    # Check for libmpv before creating the main window
    if not _check_mpv_available():
        logger.error("libmpv not found — showing install dialog")
        if startup_smoke_test:
            return 1
        _show_mpv_missing_dialog(app)
        return 1

    logger.info("Creating MainWindow...")
    window = MainWindow()

    if startup_smoke_test:
        from PySide6.QtCore import QTimer

        logger.info("Startup smoke test completed successfully")
        # Keep the process alive after successful startup so CI can validate
        # launch completion without exercising native teardown paths.
        QTimer.singleShot(300000, app.quit)
        return app.exec()

    # Apply theme (uses saved preference from settings loaded in MainWindow)
    logger.info("Applying initial theme...")
    theme().apply_to_app()

    logger.info("Showing MainWindow...")
    window.show()

    logger.info("Starting event loop...")
    return app.exec()


if __name__ == "__main__":
    logger.info("=== SCRIPT START ===")
    sys.exit(main())
