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
    - On macOS: disables posix_spawn for subprocess so background-thread
      subprocess.run() calls don't randomly fail with FileNotFoundError
      on absolute paths to bundled binaries.
    """
    # macOS PyInstaller bundles + thread-pool subprocess.run() = sporadic
    # FileNotFoundError even with an existing, accessible absolute path.
    # Force fork+exec (instead of posix_spawn) globally before any thread
    # is created. Slightly slower per spawn; reliable across threading.
    if sys.platform == "darwin":
        import subprocess
        subprocess._USE_POSIX_SPAWN = False

    # Inject the bundled bin directory into the process-global PATH so
    # libraries that shell out by bare binary name (e.g.
    # ``lightning-whisper-mlx`` calling ``subprocess.run(["ffmpeg", ...])``
    # via PATH lookup) can find the bundled tools. Our own subprocess
    # calls already use ``get_subprocess_env()`` which augments PATH for
    # one-shot calls, but third-party libraries don't go through that
    # helper — they use ``os.environ`` directly. Without this, the
    # bundled app can't transcribe because MLX whisper's internal ffmpeg
    # call falls back to system PATH and fails with
    # ``FileNotFoundError: 'ffmpeg'``.
    from core.paths import get_bundled_bin_dir
    bundled_bin = str(get_bundled_bin_dir())
    if bundled_bin:
        existing_path = os.environ.get("PATH", "")
        path_parts = existing_path.split(os.pathsep) if existing_path else []
        if bundled_bin not in path_parts:
            path_parts.insert(0, bundled_bin)
            os.environ["PATH"] = os.pathsep.join(path_parts)

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

    # Patch litellm's tiktoken encoding before any LLM calls. In frozen builds
    # tiktoken's native extension may be incomplete; the patch injects a
    # lightweight fallback so litellm.completion() works without tiktoken.
    from core.llm_client import patch_litellm_encoding
    patch_litellm_encoding()

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
    """Check if libmpv is discoverable without loading it into the process."""
    from ui.video_player import _find_mpv_library

    return _find_mpv_library() is not None


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
        # Smoke targets may bare-`import mpv`; the real app defers that import
        # until playback starts and patches ctypes.util.find_library first, so
        # mirror the prep here before dispatching.
        if is_frozen():
            from ui.video_player import _prepare_frozen_mpv_import
            _prepare_frozen_mpv_import()
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
        exit_code = app.exec()
        release_single_instance_lock()
        return exit_code

    # Apply theme (uses saved preference from settings loaded in MainWindow)
    logger.info("Applying initial theme...")
    theme().apply_to_app()

    logger.info("Showing MainWindow...")
    window.show()

    logger.info("Starting event loop...")
    exit_code = app.exec()
    # Release the single-instance lock AFTER the event loop exits, not during
    # aboutToQuit. Releasing early creates a race: macOS can relaunch the app
    # (Dock click, Finder double-click) before the process fully terminates,
    # and the new instance passes the lock check because the file is already gone.
    release_single_instance_lock()
    return exit_code


if __name__ == "__main__":
    logger.info("=== SCRIPT START ===")
    sys.exit(main())
