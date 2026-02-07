"""Cost estimate panel showing analysis time and dollar cost for a sequence."""

from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QFrame,
    QGridLayout,
)
from PySide6.QtCore import Signal, Qt

from core.cost_estimates import OperationEstimate, TIERED_OPERATIONS
from ui.theme import theme, UISizes, TypeScale, Spacing


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 1:
        return "< 1s"
    if seconds < 60:
        return f"~{int(math.ceil(seconds))}s"
    minutes = int(math.ceil(seconds / 60))
    return f"~{minutes} min"


def _format_cost(dollars: float) -> str:
    """Format dollar cost into a human-readable string."""
    if dollars == 0:
        return "Free"
    if dollars < 0.01:
        return "< $0.01"
    return f"${dollars:.2f}"


class CostEstimatePanel(QWidget):
    """Inline panel showing cost/time estimates for sequence analysis.

    Shows per-operation rows with clip counts, tier dropdowns, and
    time/cost estimates. Supports collapsed (summary) and expanded
    (detail) states.

    Signals:
        tier_changed(str, str): Emitted when user changes a tier dropdown.
            Args are (operation_key, new_tier).
    """

    tier_changed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._estimates: list[OperationEstimate] = []
        self._tier_combos: dict[str, QComboBox] = {}
        self._collapsed = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main frame with border
        self._frame = QFrame()
        self._frame.setObjectName("costEstimateFrame")
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        frame_layout.setSpacing(Spacing.SM)

        # Header row: title + collapse toggle
        header_row = QHBoxLayout()
        header_row.setSpacing(Spacing.SM)

        self._title_label = QLabel("Analysis Estimate")
        self._title_label.setStyleSheet(
            f"font-size: {TypeScale.MD}px; font-weight: bold;"
        )
        header_row.addWidget(self._title_label)
        header_row.addStretch()

        self._toggle_btn = QLabel()
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.mousePressEvent = lambda _: self.set_collapsed(not self._collapsed)
        header_row.addWidget(self._toggle_btn)

        frame_layout.addLayout(header_row)

        # Summary line (always visible)
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet(f"font-size: {TypeScale.SM}px;")
        frame_layout.addWidget(self._summary_label)

        # Detail area (hidden when collapsed)
        self._detail_widget = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_widget)
        self._detail_layout.setContentsMargins(0, Spacing.SM, 0, 0)
        self._detail_layout.setSpacing(0)

        # Grid for operation rows
        self._grid = QGridLayout()
        self._grid.setSpacing(Spacing.XS)
        self._grid.setColumnStretch(0, 2)  # Operation name
        self._grid.setColumnStretch(1, 1)  # Clips
        self._grid.setColumnStretch(2, 2)  # Tier
        self._grid.setColumnStretch(3, 1)  # Estimate
        self._detail_layout.addLayout(self._grid)

        frame_layout.addWidget(self._detail_widget)

        layout.addWidget(self._frame)
        self._apply_theme()
        self.setVisible(False)

    def _apply_theme(self):
        t = theme()
        self._frame.setStyleSheet(f"""
            QFrame#costEstimateFrame {{
                background-color: {t.background_secondary};
                border: 1px solid {t.border_primary};
                border-radius: 8px;
            }}
        """)
        self._title_label.setStyleSheet(
            f"font-size: {TypeScale.MD}px; font-weight: bold; "
            f"color: {t.text_primary}; border: none;"
        )
        self._summary_label.setStyleSheet(
            f"font-size: {TypeScale.SM}px; color: {t.text_secondary}; border: none;"
        )
        self._update_toggle_text()

    def _update_toggle_text(self):
        t = theme()
        text = "Show details" if self._collapsed else "Hide details"
        self._toggle_btn.setText(text)
        self._toggle_btn.setStyleSheet(
            f"font-size: {TypeScale.SM}px; color: {t.accent_blue}; border: none;"
        )

    def set_estimates(self, estimates: list[OperationEstimate]):
        """Update the panel with new estimates."""
        self._estimates = estimates
        self._rebuild_grid()
        self._update_summary()
        self.setVisible(bool(estimates))

    def get_tier_overrides(self) -> dict[str, str]:
        """Return current tier selections as {operation: tier} dict."""
        overrides = {}
        for op_key, combo in self._tier_combos.items():
            tier = "cloud" if combo.currentIndex() == 1 else "local"
            overrides[op_key] = tier
        return overrides

    def set_collapsed(self, collapsed: bool):
        """Collapse or expand the detail area."""
        self._collapsed = collapsed
        self._detail_widget.setVisible(not collapsed)
        self._update_toggle_text()

    def _rebuild_grid(self):
        """Clear and rebuild the operation grid from current estimates."""
        # Clear existing grid items
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tier_combos.clear()

        if not self._estimates:
            return

        t = theme()
        header_style = (
            f"font-size: {TypeScale.XS}px; font-weight: bold; "
            f"color: {t.text_muted}; border: none;"
        )

        # Column headers
        for col, text in enumerate(["Operation", "Clips", "Tier", "Estimate"]):
            label = QLabel(text)
            label.setStyleSheet(header_style)
            self._grid.addWidget(label, 0, col)

        row_style = f"font-size: {TypeScale.SM}px; color: {t.text_primary}; border: none;"

        for row_idx, est in enumerate(self._estimates, start=1):
            # Operation name
            name_label = QLabel(est.label)
            name_label.setStyleSheet(row_style)
            self._grid.addWidget(name_label, row_idx, 0)

            # Clips needing / total
            clips_label = QLabel(f"{est.clips_needing}/{est.clips_total}")
            clips_label.setStyleSheet(row_style)
            self._grid.addWidget(clips_label, row_idx, 1)

            # Tier dropdown or static label
            if est.operation in TIERED_OPERATIONS:
                combo = QComboBox()
                combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
                combo.addItems(["Local (Free)", "Cloud (Paid)"])
                combo.setCurrentIndex(0 if est.tier == "local" else 1)
                op_key = est.operation
                combo.currentIndexChanged.connect(
                    lambda idx, key=op_key: self._on_tier_changed(key, idx)
                )
                self._tier_combos[op_key] = combo
                self._grid.addWidget(combo, row_idx, 2)
            else:
                tier_label = QLabel("Local" if est.tier == "local" else "Cloud")
                tier_label.setStyleSheet(row_style)
                self._grid.addWidget(tier_label, row_idx, 2)

            # Cost + time estimate
            cost_str = _format_cost(est.cost_dollars)
            time_str = _format_time(est.time_seconds)
            est_label = QLabel(f"{cost_str}  {time_str}")
            est_label.setStyleSheet(row_style)
            self._grid.addWidget(est_label, row_idx, 3)

    def _update_summary(self):
        """Update the summary line from current estimates."""
        if not self._estimates:
            self._summary_label.setText("")
            return

        total_needing = sum(e.clips_needing for e in self._estimates)
        total_cost = sum(e.cost_dollars for e in self._estimates)
        total_time = sum(e.time_seconds for e in self._estimates)

        parts = [f"{total_needing} clips need analysis"]
        parts.append(_format_cost(total_cost))
        parts.append(_format_time(total_time))
        self._summary_label.setText("  |  ".join(parts))

    def _on_tier_changed(self, operation: str, index: int):
        """Handle tier combo box change."""
        tier = "cloud" if index == 1 else "local"
        self.tier_changed.emit(operation, tier)
