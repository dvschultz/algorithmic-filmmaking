"""Dialog for importing a remote video with a selected download resolution."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QLabel,
)

from core.downloader import (
    DEFAULT_DOWNLOAD_RESOLUTION,
    DOWNLOAD_RESOLUTION_OPTIONS,
)
from ui.theme import TypeScale


class URLImportDialog(QDialog):
    """Prompt for a source URL and preferred download resolution."""

    def __init__(self, parent=None, initial_url: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import from URL")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        title = QLabel("Import from URL")
        title.setStyleSheet(f"font-size: {TypeScale.LG}px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://youtube.com/watch?v=...")
        self.url_edit.setText(initial_url)
        form.addRow("URL:", self.url_edit)

        self.resolution_combo = QComboBox()
        for value, label in DOWNLOAD_RESOLUTION_OPTIONS:
            self.resolution_combo.addItem(label, value)
        default_index = self.resolution_combo.findData(DEFAULT_DOWNLOAD_RESOLUTION)
        if default_index >= 0:
            self.resolution_combo.setCurrentIndex(default_index)
        form.addRow("Resolution:", self.resolution_combo)

        layout.addLayout(form)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        button_box.button(QDialogButtonBox.Ok).setText("Import")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._import_btn = button_box.button(QDialogButtonBox.Ok)
        self.url_edit.textChanged.connect(self._update_accept_enabled)
        self._update_accept_enabled()
        self.url_edit.setFocus()

    def _update_accept_enabled(self):
        """Enable import only when the URL field is not blank."""
        self._import_btn.setEnabled(bool(self.url_edit.text().strip()))

    def selected_url(self) -> str:
        """Return the trimmed URL."""
        return self.url_edit.text().strip()

    def selected_resolution(self) -> str:
        """Return the normalized resolution tier value."""
        return str(self.resolution_combo.currentData())

    @classmethod
    def get_import_request(cls, parent=None, initial_url: str = "") -> tuple[str | None, str | None]:
        """Show the dialog and return a URL/resolution pair when accepted."""
        dialog = cls(parent=parent, initial_url=initial_url)
        if dialog.exec() != QDialog.Accepted:
            return None, None
        return dialog.selected_url(), dialog.selected_resolution()
