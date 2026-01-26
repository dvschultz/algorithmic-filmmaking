"""Parameter controls for sorting algorithms in the Sequence tab."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ui.theme import theme


# Algorithm parameter definitions
ALGORITHM_PARAMS = {
    "color": {
        "title": "Color Sort",
        "icon": "🎨",
        "params": [
            {
                "key": "direction",
                "label": "Direction",
                "type": "combo",
                "options": [
                    ("rainbow", "Rainbow (Hue Order)"),
                    ("warm_to_cool", "Warm to Cool"),
                    ("cool_to_warm", "Cool to Warm"),
                ],
                "default": "rainbow",
            }
        ],
    },
    "duration": {
        "title": "Duration Sort",
        "icon": "⏱️",
        "params": [
            {
                "key": "direction",
                "label": "Direction",
                "type": "combo",
                "options": [
                    ("short_first", "Shortest First"),
                    ("long_first", "Longest First"),
                ],
                "default": "short_first",
            }
        ],
    },
    "shuffle": {
        "title": "Shuffle",
        "icon": "🎲",
        "params": [
            {
                "key": "seed",
                "label": "Random Seed (optional)",
                "type": "spinbox",
                "min": 0,
                "max": 999999,
                "default": 0,
                "special_value": "Random",
            }
        ],
    },
    "sequential": {
        "title": "Sequential",
        "icon": "📋",
        "params": [],
    },
}


class SortingParameterPanel(QWidget):
    """Parameter controls for the selected sorting algorithm.

    Displays algorithm-specific controls and common controls (clip count).
    Emits signals when parameters change or when apply/back is clicked.

    Signals:
        parameters_changed: Emitted with current parameter dict when any value changes
        apply_clicked: Emitted when Apply button is clicked
        back_clicked: Emitted when Back button is clicked
    """

    parameters_changed = Signal(dict)  # current parameter values
    apply_clicked = Signal()
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._algorithm: str | None = None
        self._param_widgets: dict = {}
        self._clip_count_spin: QSpinBox | None = None
        self._setup_ui()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header row with back button and title
        header_layout = QHBoxLayout()

        self.back_btn = QPushButton("← Back")
        self.back_btn.setFixedWidth(80)
        self.back_btn.clicked.connect(self.back_clicked.emit)
        header_layout.addWidget(self.back_btn)

        header_layout.addStretch()

        self.title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Spacer to balance the back button
        spacer = QWidget()
        spacer.setFixedWidth(80)
        header_layout.addWidget(spacer)

        main_layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {theme().border_secondary};")
        main_layout.addWidget(separator)
        self._separator = separator

        # Parameters container
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(12)
        main_layout.addWidget(self.params_container)

        main_layout.addStretch()

        # Apply button at bottom
        self.apply_btn = QPushButton("Apply to Timeline")
        self.apply_btn.setMinimumHeight(40)
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme().accent_blue};
                color: {theme().text_inverted};
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {theme().accent_blue_hover};
            }}
        """)
        self.apply_btn.clicked.connect(self.apply_clicked.emit)
        main_layout.addWidget(self.apply_btn)

    def set_algorithm(self, algorithm: str, available_clips: int = 100):
        """Show parameters for the specified algorithm.

        Args:
            algorithm: Algorithm key (color, duration, shuffle, sequential)
            available_clips: Maximum number of clips available for selection
        """
        self._algorithm = algorithm
        self._param_widgets.clear()

        # Clear existing parameter widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get algorithm config
        config = ALGORITHM_PARAMS.get(algorithm, {"title": algorithm, "icon": "", "params": []})

        # Update title
        self.title_label.setText(f"{config['icon']} {config['title']}")

        # Add clip count control (common to all algorithms)
        clip_count_row = self._create_param_row(
            "Clip Count",
            self._create_spinbox(1, min(available_clips, 100), min(10, available_clips), "clip_count")
        )
        self.params_layout.addLayout(clip_count_row)

        # Add algorithm-specific parameters
        for param in config["params"]:
            if param["type"] == "combo":
                widget = self._create_combobox(param["options"], param["default"], param["key"])
            elif param["type"] == "spinbox":
                widget = self._create_spinbox(
                    param["min"],
                    param["max"],
                    param["default"],
                    param["key"],
                    param.get("special_value")
                )
            else:
                continue

            row = self._create_param_row(param["label"], widget)
            self.params_layout.addLayout(row)

    def _create_param_row(self, label: str, widget: QWidget) -> QHBoxLayout:
        """Create a labeled parameter row."""
        row = QHBoxLayout()

        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {theme().text_secondary};")
        label_widget.setMinimumWidth(150)
        row.addWidget(label_widget)

        row.addWidget(widget)
        row.addStretch()

        return row

    def _create_spinbox(
        self,
        min_val: int,
        max_val: int,
        default: int,
        key: str,
        special_value: str | None = None
    ) -> QSpinBox:
        """Create a spinbox parameter control."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setMinimumWidth(100)

        if special_value:
            spin.setSpecialValueText(special_value)

        spin.valueChanged.connect(self._on_param_changed)
        self._param_widgets[key] = spin

        if key == "clip_count":
            self._clip_count_spin = spin

        return spin

    def _create_combobox(
        self,
        options: list[tuple[str, str]],
        default: str,
        key: str
    ) -> QComboBox:
        """Create a combobox parameter control."""
        combo = QComboBox()
        combo.setMinimumWidth(180)

        default_index = 0
        for i, (value, label) in enumerate(options):
            combo.addItem(label, value)
            if value == default:
                default_index = i

        combo.setCurrentIndex(default_index)
        combo.currentIndexChanged.connect(self._on_param_changed)
        self._param_widgets[key] = combo

        return combo

    def _on_param_changed(self):
        """Handle parameter value change."""
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict:
        """Return current parameter values."""
        params = {"algorithm": self._algorithm}

        for key, widget in self._param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentData()

        return params

    def set_clip_count_max(self, max_clips: int):
        """Update the maximum clip count based on available clips."""
        if self._clip_count_spin:
            current = self._clip_count_spin.value()
            self._clip_count_spin.setMaximum(min(max_clips, 100))
            if current > max_clips:
                self._clip_count_spin.setValue(min(max_clips, 100))

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        self._separator.setStyleSheet(f"background-color: {theme().border_secondary};")
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme().accent_blue};
                color: {theme().text_inverted};
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {theme().accent_blue_hover};
            }}
        """)
