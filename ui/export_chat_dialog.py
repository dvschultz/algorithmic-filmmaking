"""Export chat dialog for configuring chat history export."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from core.chat_export import ChatExportConfig


class ExportChatDialog(QDialog):
    """Dialog for configuring chat export options."""

    def __init__(self, message_count: int, parent=None):
        """Initialize the export chat dialog.

        Args:
            message_count: Number of messages in chat history
            parent: Parent widget
        """
        super().__init__(parent)
        self._message_count = message_count
        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Export Chat")
        self.setMinimumWidth(350)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)

        self._format_group = QButtonGroup(self)

        self._md_radio = QRadioButton("Markdown (.md)")
        self._md_radio.setToolTip("Human-readable format, good for reviewing")
        self._format_group.addButton(self._md_radio)
        format_layout.addWidget(self._md_radio)

        self._json_radio = QRadioButton("JSON (.json)")
        self._json_radio.setToolTip("Machine-parseable format, good for analysis")
        self._format_group.addButton(self._json_radio)
        format_layout.addWidget(self._json_radio)

        self._both_radio = QRadioButton("Both formats")
        self._both_radio.setToolTip("Export both Markdown and JSON files")
        self._both_radio.setChecked(True)
        self._format_group.addButton(self._both_radio)
        format_layout.addWidget(self._both_radio)

        layout.addWidget(format_group)

        # Content selection
        content_group = QGroupBox("Include Content")
        content_layout = QVBoxLayout(content_group)

        self._user_check = QCheckBox("User messages")
        self._user_check.setChecked(True)
        self._user_check.stateChanged.connect(self._update_preview)
        content_layout.addWidget(self._user_check)

        self._assistant_check = QCheckBox("Assistant responses")
        self._assistant_check.setChecked(True)
        self._assistant_check.stateChanged.connect(self._update_preview)
        content_layout.addWidget(self._assistant_check)

        self._tools_check = QCheckBox("Tool calls && results")
        self._tools_check.setChecked(True)
        self._tools_check.stateChanged.connect(self._update_preview)
        content_layout.addWidget(self._tools_check)

        self._tool_args_check = QCheckBox("Tool arguments (verbose)")
        self._tool_args_check.setChecked(False)
        self._tool_args_check.setToolTip("Include full tool arguments and results")
        content_layout.addWidget(self._tool_args_check)

        layout.addWidget(content_group)

        # Preview count
        self._preview_label = QLabel()
        self._preview_label.setStyleSheet("color: #666; font-style: italic;")
        self._update_preview()
        layout.addWidget(self._preview_label)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        button_box.button(QDialogButtonBox.Ok).setText("Export")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _update_preview(self):
        """Update the preview message count."""
        # Simple estimate - actual filtering happens during export
        count = self._message_count
        self._preview_label.setText(f"Approximately {count} messages will be exported")

    def get_format(self) -> str:
        """Get the selected export format.

        Returns:
            "markdown", "json", or "both"
        """
        if self._md_radio.isChecked():
            return "markdown"
        elif self._json_radio.isChecked():
            return "json"
        else:
            return "both"

    def get_config(self, output_dir: Path, project_name: str = "") -> ChatExportConfig:
        """Get the export configuration based on dialog selections.

        Args:
            output_dir: Directory to export files to
            project_name: Name of the current project

        Returns:
            ChatExportConfig with selected options
        """
        return ChatExportConfig(
            output_dir=output_dir,
            format=self.get_format(),
            include_user=self._user_check.isChecked(),
            include_assistant=self._assistant_check.isChecked(),
            include_tools=self._tools_check.isChecked(),
            include_tool_args=self._tool_args_check.isChecked(),
            project_name=project_name
        )
