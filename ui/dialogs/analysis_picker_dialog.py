"""Analysis Picker dialog for selecting which analysis operations to run."""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
    QFrame,
)
from PySide6.QtCore import Qt

from ui.theme import theme
from core.analysis_operations import (
    ANALYSIS_OPERATIONS,
    OPERATIONS_BY_KEY,
    LOCAL_OPS,
    SEQUENTIAL_OPS,
    CLOUD_OPS,
    PHASE_ORDER,
)


class AnalysisPickerDialog(QDialog):
    """Modal dialog for selecting analysis operations to run.

    Shows all 8 operations grouped by execution phase (local, sequential, cloud)
    with checkboxes. Remembers last selection via settings.
    """

    # Phase display metadata
    _PHASE_LABELS = {
        "local": "Local Analysis (parallel)",
        "sequential": "Sequential",
        "cloud": "Cloud API (parallel)",
    }

    _PHASE_KEYS = {
        "local": LOCAL_OPS,
        "sequential": SEQUENTIAL_OPS,
        "cloud": CLOUD_OPS,
    }

    def __init__(self, clip_count: int, scope_label: str, settings, parent=None):
        """Initialize the analysis picker dialog.

        Args:
            clip_count: Number of clips to analyze
            scope_label: Description of clip scope (e.g. "All 42 clips")
            settings: Settings instance for persisting selection
            parent: Parent widget
        """
        super().__init__(parent)
        self._clip_count = clip_count
        self._scope_label = scope_label
        self._settings = settings
        self._checkboxes: dict[str, QCheckBox] = {}

        self.setWindowTitle("Analyze Clips")
        self.setMinimumWidth(350)

        self._setup_ui()
        self._load_selection()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        """Build the dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        # Header
        title = QLabel("Analyze Clips")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        scope = QLabel(f"{self._clip_count} {self._scope_label}")
        scope.setStyleSheet("color: gray; margin-bottom: 8px;")
        layout.addWidget(scope)

        # Phase groups
        for phase in PHASE_ORDER:
            keys = self._PHASE_KEYS.get(phase, [])
            if not keys:
                continue

            # Separator line
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)

            # Phase label
            phase_label = QLabel(self._PHASE_LABELS.get(phase, phase))
            phase_label.setStyleSheet("font-weight: bold; color: gray; font-size: 11px;")
            layout.addWidget(phase_label)

            # Checkboxes for operations in this phase
            for key in keys:
                op = OPERATIONS_BY_KEY[key]
                cb = QCheckBox(op.label)
                cb.setToolTip(op.tooltip)
                self._checkboxes[key] = cb
                layout.addWidget(cb)

        layout.addSpacing(8)

        # Select All / Clear row
        bulk_row = QHBoxLayout()
        bulk_row.addStretch()

        select_all_btn = QPushButton("Select All")
        select_all_btn.setFlat(True)
        select_all_btn.clicked.connect(self._select_all)
        bulk_row.addWidget(select_all_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFlat(True)
        clear_btn.clicked.connect(self._clear_all)
        bulk_row.addWidget(clear_btn)

        layout.addLayout(bulk_row)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        button_box.button(QDialogButtonBox.Ok).setText("Run")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._run_btn = button_box.button(QDialogButtonBox.Ok)
        self._update_run_button()

        # Connect checkbox changes to update run button
        for cb in self._checkboxes.values():
            cb.stateChanged.connect(self._update_run_button)

    def _load_selection(self):
        """Load saved selection from settings."""
        saved = self._settings.analysis_selected_operations
        for key, cb in self._checkboxes.items():
            cb.setChecked(key in saved)

    def _save_selection(self):
        """Save current selection to settings."""
        self._settings.analysis_selected_operations = self.selected_operations()

    def _select_all(self):
        """Check all operation checkboxes."""
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _clear_all(self):
        """Uncheck all operation checkboxes."""
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def _update_run_button(self):
        """Enable/disable Run button based on selection."""
        has_selection = any(cb.isChecked() for cb in self._checkboxes.values())
        self._run_btn.setEnabled(has_selection)

    def _on_accept(self):
        """Save selection and accept dialog."""
        self._save_selection()
        self.accept()

    def _apply_theme(self):
        """Update styling when theme changes."""
        pass  # Theme is applied via parent stylesheet cascade

    def selected_operations(self) -> list[str]:
        """Get list of selected operation keys.

        Returns:
            List of operation key strings (e.g. ["colors", "shots", "transcribe"])
        """
        return [key for key, cb in self._checkboxes.items() if cb.isChecked()]
