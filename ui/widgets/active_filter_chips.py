"""Read-only chip bar summarizing the active filters.

Displays one chip per non-default filter dimension so the user can see
what's applied even when the sidebar is hidden. Each chip has a ``×``
that clears its specific filter. Subscribes to ``FilterState.changed``
and rebuilds on every change.
"""

from typing import Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from core.filter_state import FilterState
from ui.theme import Radii, Spacing, TypeScale, theme


# Map of display title → list of FilterState attributes that should reset
# to a default value when the chip is cleared.
_FIELD_DEFAULTS = {
    "shot_type": set(),
    "color_palette": set(),
    "aspect_ratio": set(),
    "gaze_filter": set(),
    "search_query": "",
    "object_search": "",
    "description_search": "",
    "on_screen_text_search": "",
    "tag_note_search": "",
    "selected_custom_queries": set(),
    "has_analysis_ops": set(),
    "imagenet_labels": set(),
    "yolo_labels": set(),
    "yolo_per_label_rules": [],
    "person_count": None,
    "yolo_total_count": None,
    "has_audio": None,
    "has_transcript": None,
    "has_on_screen_text": None,
    "enabled_filter": None,
    "similarity_anchor_id": None,
    "min_duration": None,
    "max_duration": None,
    "min_brightness": None,
    "max_brightness": None,
    "min_volume": None,
    "max_volume": None,
}


class ActiveFilterChips(QWidget):
    """Compact read-only bar that mirrors the active FilterState fields."""

    def __init__(self, filter_state: FilterState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._filter_state = filter_state
        self._chips: list[QPushButton] = []

        # Size to chip content; never expand vertically beyond what one
        # chip-row needs. Embedded inline in the controls row, so no
        # trailing stretch — chips flow alongside the surrounding buttons.
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(Spacing.XS)

        self._rebuild()
        self._filter_state.changed.connect(self._rebuild)
        self._refresh_theme()
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    @Slot()
    def _rebuild(self) -> None:
        # Clear existing chips
        for btn in self._chips:
            self._layout.removeWidget(btn)
            btn.setParent(None)
        self._chips.clear()

        entries = self._compute_entries()
        for label, clear_fields in entries:
            btn = QPushButton(f"{label}  ×")
            btn.setObjectName("ActiveFilterChip")
            btn.clicked.connect(lambda _checked=False, fields=clear_fields: self._clear_fields(fields))
            self._chips.append(btn)
            self._layout.addWidget(btn)

        self.setVisible(bool(entries))

    def _clear_fields(self, fields: tuple[str, ...]) -> None:
        for field in fields:
            default = _FIELD_DEFAULTS.get(field)
            setattr(self._filter_state, field, default)

    def _compute_entries(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return (display_label, clear_fields) tuples for active filters."""
        fs = self._filter_state
        out: list[tuple[str, tuple[str, ...]]] = []

        def _add_set_field(value: set, title: str, field: str):
            if not value:
                return
            if len(value) <= 2:
                out.append((f"{title}: {', '.join(sorted(value))}", (field,)))
            else:
                out.append((f"{title}: {len(value)} selected", (field,)))

        _add_set_field(fs.shot_type, "Shot", "shot_type")
        _add_set_field(fs.color_palette, "Color", "color_palette")
        _add_set_field(fs.aspect_ratio, "Aspect", "aspect_ratio")
        _add_set_field(fs.gaze_filter, "Gaze", "gaze_filter")
        _add_set_field(fs.selected_custom_queries, "Query", "selected_custom_queries")
        _add_set_field(fs.has_analysis_ops, "Has analysis", "has_analysis_ops")
        _add_set_field(fs.imagenet_labels, "ImageNet", "imagenet_labels")
        _add_set_field(fs.yolo_labels, "YOLO", "yolo_labels")

        for field, title in [
            ("search_query", "Search"),
            ("object_search", "Object"),
            ("description_search", "Description"),
            ("on_screen_text_search", "Text"),
            ("tag_note_search", "Tags/notes"),
        ]:
            val = getattr(fs, field)
            if val:
                out.append((f"{title}: {val}", (field,)))

        if fs.person_count is not None:
            op, n = fs.person_count
            out.append((f"People {op} {n}", ("person_count",)))

        if fs.yolo_total_count is not None:
            op, n = fs.yolo_total_count
            out.append((f"Objects {op} {n}", ("yolo_total_count",)))

        for field, title in [
            ("has_audio", "Audio"),
            ("has_transcript", "Transcript"),
            ("has_on_screen_text", "On-screen text"),
            ("enabled_filter", "Enabled"),
        ]:
            val = getattr(fs, field)
            if val is True:
                out.append((f"{title}: Yes", (field,)))
            elif val is False:
                out.append((f"{title}: No", (field,)))

        # Ranges
        for (min_key, max_key, title) in [
            ("min_duration", "max_duration", "Duration"),
            ("min_brightness", "max_brightness", "Brightness"),
            ("min_volume", "max_volume", "Volume"),
        ]:
            lo = getattr(fs, min_key)
            hi = getattr(fs, max_key)
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None:
                text = f"{title}: {lo}–{hi}"
            elif lo is not None:
                text = f"{title}: ≥ {lo}"
            else:
                text = f"{title}: ≤ {hi}"
            out.append((text, (min_key, max_key)))

        if fs.yolo_per_label_rules:
            out.append((f"Label rules: {len(fs.yolo_per_label_rules)}", ("yolo_per_label_rules",)))

        if fs.similarity_anchor_id is not None:
            out.append(("Similarity mode", ("similarity_anchor_id",)))

        return out

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self.setStyleSheet(
            f"""
            ActiveFilterChips {{
                background: transparent;
            }}
            ActiveFilterChips #ActiveFilterChip {{
                background: {c.badge_analyzed};
                color: {c.badge_analyzed_text};
                border: none;
                border-radius: {Radii.FULL}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-size: {TypeScale.XS}px;
            }}
            ActiveFilterChips #ActiveFilterChip:hover {{
                background: {c.accent_blue};
                color: {c.text_inverted};
            }}
            """
        )
