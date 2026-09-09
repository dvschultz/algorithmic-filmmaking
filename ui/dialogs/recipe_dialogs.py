"""Recipe inspection and regeneration dialogs for the Sequence workspace (U16)."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from core.spine.sequences import get_sequence_recipe
from ui.theme import UISizes, theme
from ui.widgets.cost_estimate_panel import CostEstimatePanel


class RecipeInspectDialog(QDialog):
    """Read-only view of a sequence's stored recipe and its reconstructability."""

    def __init__(self, project, sequence_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        result = get_sequence_recipe(project, sequence_id)
        self.setWindowTitle(f"Recipe: {result.get('name', sequence_id)}")
        self.resize(640, 520)
        layout = QVBoxLayout(self)
        status = QLabel()
        status.setWordWrap(True)
        if result.get("success"):
            bits = [
                f"Algorithm {result['recipe']['algorithm']} v{result['recipe']['algorithm_version']}",
                "matches timeline" if result["matches_timeline"] else "timeline edited since generation",
                "reconstructable" if result["reconstructable"] else "inputs changed: " + "; ".join(result["problems"]),
            ]
            if result.get("uses_provider"):
                bits.append("provider-assisted (regenerate makes new provider calls)")
            status.setText(" | ".join(bits))
            body = json.dumps(result["recipe"], indent=2, sort_keys=True)
        else:
            status.setText(result.get("error", "No recipe"))
            status.setStyleSheet(f"color: {theme().text_secondary};")
            body = ""
        layout.addWidget(status)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(body)
        layout.addWidget(self.text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class RegenerateDialog(QDialog):
    """Edit recipe parameters and seed before running a variation.

    Values are entered as JSON so every parameter type round-trips exactly;
    the registry validates them again before the run.
    """

    def __init__(
        self, definition, recipe, default_name: str, parent: Optional[QWidget] = None, *,
        estimates: Optional[list] = None, dependency_warning: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Regenerate {definition.key}")
        self.setMinimumWidth(520)
        self._definition = definition
        self._fields: dict[str, QLineEdit] = {}
        self._defaults: dict[str, Any] = {}
        layout = QVBoxLayout(self)
        intro = QLabel("Change any parameter, then generate a new variation. The original sequence stays as it is.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        # Same cost/prerequisite summary the confirm step shows before a first run.
        self.cost_panel = CostEstimatePanel()
        self.cost_panel.set_estimates(list(estimates or []), estimated=True)
        self.cost_panel.set_warning(dependency_warning)
        self.cost_panel.setVisible(bool(estimates) or bool(dependency_warning))
        layout.addWidget(self.cost_panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        form_host = QWidget()
        form = QFormLayout(form_host)
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        form.addRow("Name", self.name_edit)
        merged = dict(recipe.parameters)
        for key in definition.variation_resets:
            merged.pop(key, None)
        for spec in definition.parameters:
            edit = QLineEdit(json.dumps(merged.get(spec.name, spec.default)))
            edit.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
            hint = spec.description or spec.type
            if spec.choices:
                hint += " (" + ", ".join(json.dumps(c) for c in spec.choices) + ")"
            edit.setToolTip(hint)
            edit.setPlaceholderText("default: " + json.dumps(spec.default))
            form.addRow(spec.name, edit)
            self._fields[spec.name] = edit
            self._defaults[spec.name] = spec.default
        if definition.seeded:
            self.keep_seed = QCheckBox("Keep the same seed")
            self.keep_seed.setChecked(False)
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 2**31 - 1)
            self.seed_spin.setValue(recipe.seed or 0)
            self.seed_spin.setEnabled(False)
            self.explicit_seed = QCheckBox("Use this seed")
            self.explicit_seed.toggled.connect(self.seed_spin.setEnabled)
            self.explicit_seed.toggled.connect(lambda on: self.keep_seed.setEnabled(not on))
            self.keep_seed.toggled.connect(lambda on: self.explicit_seed.setEnabled(not on))
            form.addRow("Seed", self.keep_seed)
            form.addRow("", self.explicit_seed)
            form.addRow("", self.seed_spin)
        else:
            self.keep_seed = None
            self.seed_spin = None
            self.explicit_seed = None
        scroll.setWidget(form_host)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Generate variation")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        try:
            self.parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameter", str(exc))
            return
        self.accept()

    def parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, edit in self._fields.items():
            raw = edit.text().strip()
            if not raw:
                values[name] = deepcopy(self._defaults[name])  # a cleared field means "back to default"
                continue
            try:
                values[name] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name}: enter a JSON value ({exc.msg})") from exc
        return values

    def seed(self) -> Optional[int]:
        if self.explicit_seed is not None and self.explicit_seed.isChecked():
            return int(self.seed_spin.value())
        return None

    def keeps_seed(self) -> bool:
        return bool(self.keep_seed is not None and self.keep_seed.isChecked())

    def sequence_name(self) -> str:
        return self.name_edit.text().strip()
