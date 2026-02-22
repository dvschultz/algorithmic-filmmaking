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

    def run(self):
        try:
            self._install_func(self._progress_adapter)
            self.finished_ok.emit()
        except Exception as e:
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
            self._worker.terminate()
            self._worker.wait(3000)
            self._worker = None
        super().reject()


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
    from core.feature_registry import check_feature, get_feature_size_estimate, install_for_feature

    available, missing = check_feature(feature_name)
    if available:
        return True

    size_mb = get_feature_size_estimate(feature_name)
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

    # Re-check availability
    available, _ = check_feature(feature_name)
    return available
