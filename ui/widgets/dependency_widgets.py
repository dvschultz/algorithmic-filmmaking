"""Widgets for dependency management: banners, download dialogs, progress."""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QProgressBar,
    QDialogButtonBox,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QThread

from ui.theme import theme, Spacing, TypeScale

logger = logging.getLogger(__name__)


class DependencyBanner(QWidget):
    """Non-modal banner shown at the top of the main window when a dependency is missing.

    Example: "Scene Ripper needs FFmpeg to process videos. [Download FFmpeg] (~150 MB)"
    """

    download_requested = Signal(str)  # binary name (e.g., "ffmpeg")
    dismissed = Signal()

    def __init__(self, message: str, button_text: str, dep_name: str, parent=None):
        super().__init__(parent)
        self._dep_name = dep_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)

        # Icon + message
        icon_label = QLabel("\u26a0")  # Warning triangle
        icon_label.setStyleSheet(f"font-size: {TypeScale.LG}px;")
        layout.addWidget(icon_label)

        msg_label = QLabel(message)
        msg_label.setStyleSheet(f"font-size: {TypeScale.BASE}px;")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label, stretch=1)

        # Download button
        download_btn = QPushButton(button_text)
        download_btn.setCursor(Qt.PointingHandCursor)
        download_btn.clicked.connect(lambda: self.download_requested.emit(self._dep_name))
        layout.addWidget(download_btn)

        # Dismiss button
        dismiss_btn = QPushButton("\u2715")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setCursor(Qt.PointingHandCursor)
        dismiss_btn.setToolTip("Dismiss")
        dismiss_btn.clicked.connect(self._dismiss)
        layout.addWidget(dismiss_btn)

        self._apply_style()

    def _apply_style(self):
        t = theme()
        self.setStyleSheet(
            f"DependencyBanner {{"
            f"  background-color: {t.accent_orange}22;"
            f"  border: 1px solid {t.accent_orange}44;"
            f"  border-radius: 6px;"
            f"}}"
        )

    def _dismiss(self):
        self.setVisible(False)
        self.dismissed.emit()


class UpdateBanner(QWidget):
    """Dismissible banner for app version updates.

    Example: "Scene Ripper v0.3.0 is available. [Download]"
    """

    download_clicked = Signal(str)  # release URL
    dismissed = Signal()

    def __init__(self, version: str, release_url: str, parent=None):
        super().__init__(parent)
        self._release_url = release_url

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)

        msg_label = QLabel(f"Scene Ripper {version} is available.")
        msg_label.setStyleSheet(f"font-size: {TypeScale.BASE}px;")
        layout.addWidget(msg_label, stretch=1)

        download_btn = QPushButton("Download")
        download_btn.setCursor(Qt.PointingHandCursor)
        download_btn.clicked.connect(lambda: self.download_clicked.emit(self._release_url))
        layout.addWidget(download_btn)

        dismiss_btn = QPushButton("\u2715")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setCursor(Qt.PointingHandCursor)
        dismiss_btn.clicked.connect(self._dismiss)
        layout.addWidget(dismiss_btn)

        self._apply_style()

    def _apply_style(self):
        t = theme()
        self.setStyleSheet(
            f"UpdateBanner {{"
            f"  background-color: {t.accent_blue}22;"
            f"  border: 1px solid {t.accent_blue}44;"
            f"  border-radius: 6px;"
            f"}}"
        )

    def _dismiss(self):
        self.setVisible(False)
        self.dismissed.emit()


class _DownloadWorker(QThread):
    """Background worker for downloading a dependency."""

    progress = Signal(float, str)  # 0.0-1.0, message
    finished_ok = Signal()
    failed = Signal(str)  # error message

    def __init__(self, install_func, progress_callback_adapter):
        super().__init__()
        self._install_func = install_func
        self._progress_adapter = progress_callback_adapter
        self._cancelled = False

    def cancel(self):
        """Request cooperative cancellation."""
        self._cancelled = True

    def run(self):
        try:
            result = self._install_func(self._progress_adapter)
            if not self._cancelled:
                if result is False:
                    self.failed.emit("Install completed but dependency verification failed.")
                else:
                    self.finished_ok.emit()
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(str(e))


class DependencyDownloadDialog(QDialog):
    """Modal dialog for downloading/installing a dependency with progress.

    Shows a progress bar, status message, and cancel button.
    On failure, offers retry.
    """

    download_completed = Signal()

    def __init__(
        self,
        title: str,
        message: str,
        install_func,
        parent=None,
    ):
        """
        Args:
            title: Dialog title (e.g., "Download FFmpeg").
            message: Description (e.g., "FFmpeg is required for video processing (~150 MB)").
            install_func: Callable that takes a progress_callback(float, str) and performs the install.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setModal(True)

        self._install_func = install_func
        self._worker: Optional[_DownloadWorker] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)

        # Message
        self._message_label = QLabel(message)
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Ready to download")
        self._status_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        layout.addWidget(self._status_label)

        # Buttons
        self._button_box = QDialogButtonBox()
        self._download_btn = self._button_box.addButton("Download", QDialogButtonBox.AcceptRole)
        self._cancel_btn = self._button_box.addButton(QDialogButtonBox.Cancel)
        self._download_btn.clicked.connect(self._start_download)
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self._button_box)

    def _start_download(self):
        self._download_btn.setEnabled(False)
        self._cancel_btn.setText("Cancel Download")
        self._status_label.setText("Starting download...")
        self._progress_bar.setValue(0)

        def progress_callback(progress: float, message: str):
            # This is called from the worker thread — emit signal to update UI
            if self._worker:
                self._worker.progress.emit(progress, message)

        self._worker = _DownloadWorker(self._install_func, progress_callback)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.start()

    def _on_progress(self, progress: float, message: str):
        self._progress_bar.setValue(int(progress * 1000))
        self._status_label.setText(message)

    def _on_success(self):
        self._progress_bar.setValue(1000)
        self._status_label.setText("Download complete!")
        self._worker = None
        self.download_completed.emit()
        self.accept()

    def _on_failure(self, error_msg: str):
        self._worker = None
        self._status_label.setText(f"Failed: {error_msg}")
        self._cancel_btn.setText("Close")

        # Offer retry
        self._download_btn.setText("Retry")
        self._download_btn.setEnabled(True)

    def reject(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
            self._worker = None
        super().reject()


def _is_compiler_available() -> bool:
    """Check if a C/C++ compiler is available on this system."""
    import shutil
    import sys
    if sys.platform == "darwin":
        # On macOS, clang++ from Xcode CLT is required for native extensions
        if shutil.which("clang++") is None:
            return False
        # clang++ may exist but Xcode CLT may not be installed (just the shim)
        import subprocess
        try:
            result = subprocess.run(
                ["xcode-select", "-p"],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
    # On Windows/Linux, assume compiler is available (MSVC/gcc usually present)
    return True


def prompt_feature_download(
    feature_name: str,
    parent_widget=None,
) -> bool:
    """Show a download prompt for a feature's missing dependencies.

    Checks if the feature has missing deps, shows a confirmation dialog,
    then runs the download with progress.

    Args:
        feature_name: Feature name from FEATURE_DEPS.
        parent_widget: Parent widget for dialogs.

    Returns:
        True if all deps are now available, False if user cancelled or download failed.
    """
    from core.feature_registry import (
        FEATURE_DEPS,
        check_feature_ready,
        get_feature_size_estimate,
        install_for_feature,
        requires_full_package_repair,
    )

    available, missing = check_feature_ready(feature_name)
    if available:
        return True

    # Check if this feature needs a C/C++ compiler and warn early
    deps = FEATURE_DEPS.get(feature_name)
    if deps and deps.needs_compiler and not _is_compiler_available():
        QMessageBox.warning(
            parent_widget,
            "Developer Tools Required",
            "This feature requires a C/C++ compiler to install.\n\n"
            "On macOS, open Terminal and run:\n"
            "    xcode-select --install\n\n"
            "After the install completes (~1.5 GB), restart Scene Ripper "
            "and try again.",
        )
        return False

    size_mb = get_feature_size_estimate(feature_name)
    if requires_full_package_repair(feature_name, missing):
        repair_packages = FEATURE_DEPS[feature_name].repair_packages or FEATURE_DEPS[feature_name].packages
        missing_str = ", ".join(repair_packages)
    else:
        missing_str = ", ".join(dep.split(":", 1)[1] for dep in missing)

    reply = QMessageBox.question(
        parent_widget,
        "Download Required",
        f"This feature requires: {missing_str}\n"
        f"Estimated download size: ~{size_mb} MB\n\n"
        f"Download now?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )

    if reply != QMessageBox.Yes:
        return False

    dialog = DependencyDownloadDialog(
        title=f"Installing dependencies",
        message=f"Downloading {missing_str} (~{size_mb} MB)...",
        install_func=lambda cb: install_for_feature(feature_name, cb),
        parent=parent_widget,
    )

    result = dialog.exec()
    if result != QDialog.Accepted:
        return False

    # Re-check full runtime readiness, not just package presence.
    available, _ = check_feature_ready(feature_name)
    return available
