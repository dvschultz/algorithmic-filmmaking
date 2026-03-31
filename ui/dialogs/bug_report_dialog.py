"""Bug report dialog for submitting issues via GitHub, Gmail, or email."""

import platform
import sys
import urllib.parse
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from core.app_version import get_app_version
from core.paths import get_log_dir
from core.update_service import GITHUB_OWNER, GITHUB_REPO
from ui.theme import theme, Spacing, TypeScale

_BUG_EMAIL = "dvsmethid+algofilm@gmail.com"
_MAX_LOG_LINES_GITHUB = 50  # GitHub URL body limit ~8KB
_MAX_LOG_LINES_EMAIL = 30


def _get_system_info() -> str:
    """Collect system info for bug reports."""
    lines = [
        f"App Version: {get_app_version()}",
        f"OS: {platform.system()} {platform.version()} ({platform.machine()})",
        f"Python: {sys.version.split()[0]}",
    ]
    return "\n".join(lines)


def _get_log_tail(max_lines: int = 50) -> str:
    """Read the last N lines of the log file."""
    log_file = get_log_dir() / "scene-ripper.log"
    if not log_file.exists():
        return "(No log file found)"
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        return "\n".join(tail)
    except Exception as e:
        return f"(Could not read log: {e})"


def _get_log_path() -> Path:
    """Return the log file path."""
    return get_log_dir() / "scene-ripper.log"


class BugReportDialog(QDialog):
    """Dialog for submitting bug reports via GitHub, Gmail, or email."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report a Bug")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.LG)

        # Header
        header = QLabel("Describe the bug")
        header.setStyleSheet(f"font-size: {TypeScale.LG}px; font-weight: bold;")
        layout.addWidget(header)

        hint = QLabel(
            "What happened? What did you expect? Include steps to reproduce if possible."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(hint)

        # Description input
        self._description = QTextEdit()
        self._description.setPlaceholderText(
            "e.g., The app crashes when I try to run Staccato with a long audio file..."
        )
        self._description.setMaximumHeight(100)
        layout.addWidget(self._description)

        # System info preview
        info_group = QGroupBox("System info (auto-included)")
        info_layout = QVBoxLayout(info_group)
        info_label = QLabel(_get_system_info())
        info_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-family: monospace;"
        )
        info_layout.addWidget(info_label)

        log_path = _get_log_path()
        log_label = QLabel(f"Log: {log_path}")
        log_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: 11px;")
        info_layout.addWidget(log_label)

        layout.addWidget(info_group)

        # Submit buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(Spacing.MD)

        github_btn = QPushButton("Submit to GitHub")
        github_btn.setToolTip("Opens a pre-filled GitHub issue with system info and logs")
        github_btn.clicked.connect(self._submit_github)
        btn_layout.addWidget(github_btn)

        gmail_btn = QPushButton("Send via Gmail")
        gmail_btn.setToolTip("Opens Gmail compose with system info; shows log in Finder to attach")
        gmail_btn.clicked.connect(self._submit_gmail)
        btn_layout.addWidget(gmail_btn)

        email_btn = QPushButton("Send via Email")
        email_btn.setToolTip("Opens your email app with system info; shows log in Finder to attach")
        email_btn.clicked.connect(self._submit_email)
        btn_layout.addWidget(email_btn)

        layout.addLayout(btn_layout)

        # Cancel
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn, alignment=Qt.AlignRight)

    def _get_description(self) -> str:
        return self._description.toPlainText().strip() or "Bug report"

    def _build_body(self, include_logs: bool = True, max_lines: int = 50) -> str:
        """Build the bug report body text."""
        parts = [
            "## Description",
            "",
            self._get_description(),
            "",
            "## System Info",
            "",
            _get_system_info(),
        ]
        if include_logs:
            parts.extend([
                "",
                "## Recent Logs",
                "",
                "```",
                _get_log_tail(max_lines),
                "```",
            ])
        return "\n".join(parts)

    def _submit_github(self):
        """Open a pre-filled GitHub issue."""
        title = self._get_description()[:100]
        body = self._build_body(include_logs=True, max_lines=_MAX_LOG_LINES_GITHUB)

        params = urllib.parse.urlencode({
            "title": title,
            "body": body,
            "labels": "bug",
        })

        url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/issues/new?{params}"

        # GitHub URLs have a ~8KB limit. If too long, truncate logs.
        if len(url) > 8000:
            body = self._build_body(include_logs=True, max_lines=20)
            params = urllib.parse.urlencode({
                "title": title,
                "body": body,
                "labels": "bug",
            })
            url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/issues/new?{params}"

        QDesktopServices.openUrl(url)
        self.accept()

    def _submit_gmail(self):
        """Open Gmail compose and show log file in Finder."""
        subject = f"Scene Ripper Bug Report - {self._get_description()[:80]}"
        body = self._build_body(include_logs=False)
        body += (
            "\n\n---\n"
            "Please attach the log file from the Finder window that just opened.\n"
            f"Log location: {_get_log_path()}"
        )

        params = urllib.parse.urlencode({
            "view": "cm",
            "fs": "1",
            "to": _BUG_EMAIL,
            "su": subject,
            "body": body,
        })

        gmail_url = f"https://mail.google.com/mail/?{params}"
        QDesktopServices.openUrl(gmail_url)

        # Open log directory in Finder so user can drag the file
        self._open_log_in_finder()
        self.accept()

    def _submit_email(self):
        """Open default email client and show log file in Finder."""
        subject = f"Scene Ripper Bug Report - {self._get_description()[:80]}"
        body = self._build_body(include_logs=False)
        body += (
            "\n\n---\n"
            "Please attach the log file from the Finder window that just opened.\n"
            f"Log location: {_get_log_path()}"
        )

        mailto_url = (
            f"mailto:{_BUG_EMAIL}"
            f"?subject={urllib.parse.quote(subject)}"
            f"&body={urllib.parse.quote(body)}"
        )
        QDesktopServices.openUrl(mailto_url)

        # Open log directory in Finder
        self._open_log_in_finder()
        self.accept()

    def _open_log_in_finder(self):
        """Open the log file's directory in the system file manager."""
        log_path = _get_log_path()
        if log_path.exists():
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_path.parent)))
