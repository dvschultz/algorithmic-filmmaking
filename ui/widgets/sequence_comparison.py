"""A/B sequence comparison panel for the Sequence workspace (plan U16).

Two named selectors pick sequences A and B from the project. The panel shows
duration, clip count, seed and changed recipe parameters, offers recipe
inspection, duplicate and regenerate actions, and switches the timeline
between A and B at the same elapsed time (clamped to the shorter one).

The panel never mutates the project itself: every action is a signal the
Sequence tab handles through the shared spine functions, so agent tools
(``compare_sequences``, ``duplicate_sequence``, ``regenerate_sequence``)
stay at parity with the buttons.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from core.spine.sequences import compare_sequences
from ui.theme import Spacing, TypeScale, UISizes, theme

if TYPE_CHECKING:
    from core.project import Project
    from models.sequence import Sequence

logger = logging.getLogger(__name__)

SLOT_A = "a"
SLOT_B = "b"
_PLACEHOLDER = "Choose a sequence..."


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


class _Side(QWidget):
    """One column: selector, summary rows, and per-sequence actions."""

    def __init__(self, slot: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.slot = slot
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)

        self.selector = QComboBox()
        self.selector.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.selector.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self.selector.setToolTip(f"Sequence {slot.upper()}")
        self.selector.setAccessibleName(f"Sequence {slot.upper()} selector")
        layout.addWidget(self.selector)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(Spacing.SM)
        grid.setVerticalSpacing(2)
        self.values: dict[str, QLabel] = {}
        for row, key in enumerate(("algorithm", "clips", "duration", "seed", "preview")):
            title = QLabel(key.capitalize())
            title.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
            value = QLabel("--")
            value.setStyleSheet(f"font-size: {TypeScale.SM}px;")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(title, row, 0)
            grid.addWidget(value, row, 1)
            self.values[key] = value
        layout.addLayout(grid)

        buttons = QHBoxLayout()
        buttons.setSpacing(Spacing.XS)
        self.show_btn = QPushButton(f"Show {slot.upper()}")
        self.show_btn.setToolTip(f"Load {slot.upper()} into the timeline at the current elapsed time ({slot.upper()} key)")
        self.inspect_btn = QPushButton("Recipe")
        self.inspect_btn.setToolTip("Inspect the stored recipe")
        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.setToolTip("Copy this sequence and its recipe as a new sequence")
        self.regenerate_btn = QPushButton("Regenerate...")
        self.regenerate_btn.setToolTip("Run the recipe again as a new variation")
        self.render_btn = QPushButton("Render preview")
        self.render_btn.setToolTip("Render a cached proxy so switching plays smoothly")
        for btn in (self.show_btn, self.inspect_btn, self.duplicate_btn, self.regenerate_btn, self.render_btn):
            btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
            btn.setEnabled(False)
            buttons.addWidget(btn)
        layout.addLayout(buttons)

    def selected_id(self) -> Optional[str]:
        data = self.selector.currentData()
        return data if isinstance(data, str) and data else None

    def clear_values(self) -> None:
        for value in self.values.values():
            value.setText("--")


class SequenceComparisonPanel(QWidget):
    """Compare two project sequences and act on either one."""

    switch_requested = Signal(str)          # sequence_id -> load it at the same elapsed time
    inspect_requested = Signal(str)         # sequence_id
    duplicate_requested = Signal(str)       # sequence_id
    regenerate_requested = Signal(str)      # sequence_id
    render_preview_requested = Signal(str)  # sequence_id
    selection_changed = Signal()            # A or B changed

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project: Optional["Project"] = None
        self._preview_ready: Callable[["Sequence"], bool] = lambda _sequence: False
        self._busy: dict[str, str] = {}  # sequence_id -> progress text
        self.setFocusPolicy(Qt.StrongFocus)
        self._setup_ui()

    # -- UI ---------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.SM)

        header = QLabel("Compare A / B")
        header.setStyleSheet(f"font-size: {TypeScale.MD}px; font-weight: 600;")
        layout.addWidget(header)

        self.empty_label = QLabel("Generate or duplicate a second sequence to compare variations.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self.empty_label)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(Spacing.SM)
        sides = QHBoxLayout()
        sides.setSpacing(Spacing.MD)
        self.side_a = _Side(SLOT_A)
        self.side_b = _Side(SLOT_B)
        sides.addWidget(self.side_a, 1)
        sides.addWidget(self.side_b, 1)
        body_layout.addLayout(sides)

        self.difference_label = QLabel("")
        self.difference_label.setWordWrap(True)
        self.difference_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body_layout.addWidget(self.difference_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        self.status_label.setWordWrap(True)
        body_layout.addWidget(self.status_label)
        layout.addWidget(self.body)

        for side in (self.side_a, self.side_b):
            side.selector.currentIndexChanged.connect(lambda _i, s=side: self._on_selector_changed(s))
            side.show_btn.clicked.connect(lambda _c=False, s=side: self._emit_for(s, self.switch_requested))
            side.inspect_btn.clicked.connect(lambda _c=False, s=side: self._emit_for(s, self.inspect_requested))
            side.duplicate_btn.clicked.connect(lambda _c=False, s=side: self._emit_for(s, self.duplicate_requested))
            side.regenerate_btn.clicked.connect(lambda _c=False, s=side: self._emit_for(s, self.regenerate_requested))
            side.render_btn.clicked.connect(lambda _c=False, s=side: self._emit_for(s, self.render_preview_requested))

    # -- wiring -----------------------------------------------------------

    def set_project(self, project: Optional["Project"]) -> None:
        self._project = project
        self._busy.clear()
        self.refresh()

    def set_preview_probe(self, probe: Callable[["Sequence"], bool]) -> None:
        """Supply the cached-preview check (owned by MainWindow)."""
        self._preview_ready = probe
        self._update_summaries()

    def selected_ids(self) -> tuple[Optional[str], Optional[str]]:
        return self.side_a.selected_id(), self.side_b.selected_id()

    def select(self, slot: str, sequence_id: Optional[str]) -> None:
        """Programmatically choose a sequence for a slot (None clears it)."""
        side = self.side_a if slot == SLOT_A else self.side_b
        index = side.selector.findData(sequence_id) if sequence_id else 0
        side.selector.setCurrentIndex(max(index, 0))

    def set_generation_state(self, sequence_id: Optional[str], message: str = "", running: bool = False) -> None:
        """Reflect shared generation progress for a sequence in this panel."""
        if sequence_id:
            if running:
                self._busy[sequence_id] = message or "Working..."
            else:
                self._busy.pop(sequence_id, None)
        self.status_label.setText(message if running else "")
        self._update_actions()

    def refresh(self) -> None:
        """Rebuild both selectors from the project, keeping choices by id."""
        project = self._project
        sequences = list(project.sequences) if project is not None else []
        for side in (self.side_a, self.side_b):
            previous = side.selected_id()
            side.selector.blockSignals(True)
            side.selector.clear()
            side.selector.addItem(_PLACEHOLDER, "")
            for sequence in sequences:
                side.selector.addItem(sequence.name, sequence.id)
            index = side.selector.findData(previous) if previous else -1
            # A deleted sequence clears its selector; the other side is untouched.
            side.selector.setCurrentIndex(index if index > 0 else 0)
            side.selector.blockSignals(False)
        self.empty_label.setVisible(len(sequences) < 2)
        self.body.setVisible(len(sequences) >= 2)
        self._update_summaries()

    # -- internals --------------------------------------------------------

    def _sequence(self, sequence_id: Optional[str]) -> Optional["Sequence"]:
        if self._project is None or not sequence_id:
            return None
        return next((s for s in self._project.sequences if s.id == sequence_id), None)

    def _on_selector_changed(self, _side: _Side) -> None:
        self._update_summaries()
        self.selection_changed.emit()

    def _emit_for(self, side: _Side, signal) -> None:
        sequence_id = side.selected_id()
        if sequence_id:
            signal.emit(sequence_id)

    def _update_summaries(self) -> None:
        for side in (self.side_a, self.side_b):
            sequence = self._sequence(side.selected_id())
            if sequence is None:
                side.clear_values()
                continue
            recipe = sequence.readable_recipe
            side.values["algorithm"].setText(sequence.algorithm or "manual")
            side.values["clips"].setText(str(len(sequence.get_all_clips())))
            side.values["duration"].setText(format_seconds(sequence.duration_seconds))
            if recipe is None:
                side.values["seed"].setText("no recipe" if sequence.recipe is None else "unreadable recipe")
            else:
                side.values["seed"].setText("--" if recipe.seed is None else str(recipe.seed))
            side.values["preview"].setText(self._preview_text(sequence))
        self._update_differences()
        self._update_actions()

    def _preview_text(self, sequence: "Sequence") -> str:
        if not sequence.get_all_clips():
            return "empty"
        try:
            ready = bool(self._preview_ready(sequence))
        except Exception:  # a probe failure must not break the panel
            logger.debug("Preview probe failed for %s", sequence.id, exc_info=True)
            ready = False
        return "ready" if ready else "not rendered"

    def _update_differences(self) -> None:
        a_id, b_id = self.selected_ids()
        if self._project is None or not a_id or not b_id:
            self.difference_label.setText("Pick A and B to see what changed.")
            return
        if a_id == b_id:
            self.difference_label.setText("A and B are the same sequence; pick two different ones.")
            return
        result = compare_sequences(self._project, a_id, b_id)
        if not result.get("success"):
            self.difference_label.setText(result.get("error", "Cannot compare these sequences."))
            return
        lines: list[str] = []
        if not result["same_algorithm"]:
            lines.append(f"Different algorithms: {result['a']['algorithm']} vs {result['b']['algorithm']}.")
        if result["seed_changed"]:
            lines.append(f"Seed: {result['a']['seed']} -> {result['b']['seed']}.")
        for item in result["parameter_differences"]:
            lines.append(f"{item['key']}: {item['a']!r} -> {item['b']!r}")
        if result["a"]["has_recipe"] and result["b"]["has_recipe"] and not result["parameter_differences"] \
                and not result["seed_changed"] and result["same_algorithm"]:
            lines.append("Same recipe parameters and seed.")
        if not (result["a"]["has_recipe"] and result["b"]["has_recipe"]):
            lines.append("Recipe differences need a recipe on both sides; timelines are compared instead.")
        lines.append(
            "Identical timelines." if result["timelines_identical"] else
            f"Clips {result['clip_count_delta']:+d}, duration {result['duration_delta_seconds']:+.1f}s "
            f"(switch compares the first {format_seconds(result['comparable_seconds'])})."
        )
        self.difference_label.setText("\n".join(lines))

    def _update_actions(self) -> None:
        read_only = bool(self._project is not None and self._project.is_read_only)
        for side in (self.side_a, self.side_b):
            sequence = self._sequence(side.selected_id())
            present = sequence is not None
            busy = present and sequence.id in self._busy
            has_clips = present and bool(sequence.get_all_clips())
            has_recipe = present and sequence.readable_recipe is not None
            side.show_btn.setEnabled(present and not busy)
            side.inspect_btn.setEnabled(present and sequence.recipe is not None)
            side.duplicate_btn.setEnabled(present and not read_only and not busy)
            side.regenerate_btn.setEnabled(has_recipe and not read_only and not busy)
            preview_ready = has_clips and side.values["preview"].text() == "ready"
            side.render_btn.setEnabled(has_clips and not preview_ready and not busy)
            side.render_btn.setText("Preview ready" if preview_ready else "Render preview")

    # -- keyboard ---------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_Left):
            self._emit_for(self.side_a, self.switch_requested)
            event.accept()
            return
        if key in (Qt.Key_B, Qt.Key_Right):
            self._emit_for(self.side_b, self.switch_requested)
            event.accept()
            return
        super().keyPressEvent(event)
