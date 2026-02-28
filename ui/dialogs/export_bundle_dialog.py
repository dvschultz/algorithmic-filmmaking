"""Export Project Bundle dialog.

Allows the user to choose a destination folder, toggle video inclusion,
and see a summary of what will be exported.
"""

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
    QFileDialog,
)
from PySide6.QtCore import Qt

from core.project import Project
from core.settings import format_size
from ui.theme import theme, TypeScale, Spacing, UISizes

logger = logging.getLogger(__name__)


class ExportBundleDialog(QDialog):
    """Modal dialog for exporting a project as a self-contained bundle."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        self.setWindowTitle("Export Project Bundle")
        self.setMinimumWidth(450)

        self._setup_ui()
        self._update_summary()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(Spacing.MD)

        # Title
        title = QLabel("Export Project Bundle")
        title.setStyleSheet(f"font-size: {TypeScale.LG}px; font-weight: bold;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Create a self-contained folder with the project file "
            "and all referenced assets."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(Spacing.SM)

        # Destination folder picker
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Destination:")
        folder_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        folder_layout.addWidget(folder_label)

        self._dest_edit = QLineEdit()
        self._dest_edit.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self._dest_edit.setPlaceholderText("Choose export folder...")
        self._dest_edit.setReadOnly(True)
        folder_layout.addWidget(self._dest_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        browse_btn.clicked.connect(self._on_browse)
        folder_layout.addWidget(browse_btn)

        layout.addLayout(folder_layout)

        # Include videos checkbox
        self._include_videos_cb = QCheckBox("Include source videos")
        self._include_videos_cb.setChecked(True)
        self._include_videos_cb.toggled.connect(self._update_summary)
        layout.addWidget(self._include_videos_cb)

        # Summary label
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        layout.addStretch()

        # Buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._button_box.button(QDialogButtonBox.Ok).setText("Export")
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _on_browse(self):
        project_name = self._project.metadata.name or "Untitled Project"
        default_name = f"{project_name}-export"

        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Export Destination",
            str(Path.home()),
        )
        if not folder:
            return

        dest = Path(folder) / default_name
        self._dest_edit.setText(str(dest))
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def _update_summary(self):
        sources = self._project.sources
        frames = self._project.frames
        include_videos = self._include_videos_cb.isChecked()

        # Calculate video sizes
        video_size = 0
        video_missing = 0
        for s in sources:
            if s.file_path.exists():
                try:
                    video_size += s.file_path.stat().st_size
                except OSError:
                    video_missing += 1
            else:
                video_missing += 1

        # Calculate frame sizes
        frame_size = 0
        frame_missing = 0
        for f in frames:
            if f.file_path.exists():
                try:
                    frame_size += f.file_path.stat().st_size
                except OSError:
                    frame_missing += 1
            else:
                frame_missing += 1

        parts = []
        if sources:
            size_str = format_size(video_size) if include_videos else "excluded"
            parts.append(f"{len(sources)} source(s) ({size_str})")
        if frames:
            parts.append(f"{len(frames)} frame(s) ({format_size(frame_size)})")

        if not parts:
            parts.append("Empty project (project file only)")

        summary = ", ".join(parts)

        # Estimated total
        total = frame_size + (video_size if include_videos else 0)
        if total > 0:
            summary += f"\nEstimated bundle size: {format_size(total)}"

        # Warnings for missing files
        warnings = []
        if video_missing and include_videos:
            warnings.append(f"{video_missing} source(s) missing on disk")
        if frame_missing:
            warnings.append(f"{frame_missing} frame(s) missing on disk")
        if warnings:
            summary += "\n" + "; ".join(warnings)

        self._summary_label.setText(summary)

    def _apply_theme(self):
        self.update()

    @property
    def dest_dir(self) -> Path:
        return Path(self._dest_edit.text())

    @property
    def include_videos(self) -> bool:
        return self._include_videos_cb.isChecked()
