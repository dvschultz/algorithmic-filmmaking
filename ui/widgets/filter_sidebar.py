"""Toggleable filter sidebar for the Cut and Analyze tabs.

Houses nine collapsible sections that expose every filterable analysis
dimension on a clip. In Unit 3 of the clip-filter sidebar plan the sections
are scaffolded with empty bodies; Units 4–6 populate them with controls
wired to the shared ``FilterState``.

``FilterSidebar`` is a plain ``QWidget`` (not ``QDockWidget``) so it can be
embedded inside each tab's content area via ``QSplitter``. Each tab owns its
own sidebar instance; both point at the same ``FilterState`` so filter
values are shared across tabs.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.analysis.color import COLOR_PALETTES, get_palette_display_name
from core.analysis.gaze import GAZE_CATEGORY_DISPLAY
from core.analysis.shots import SHOT_TYPES, get_display_name
from core.filter_state import FilterState
from ui.theme import Spacing, TypeScale, UISizes, theme
from ui.widgets.chip_group import ChipGroup
from ui.widgets.collapsible_section import CollapsibleSection


SECTION_SHOT = "shot"
SECTION_VISUAL = "visual"
SECTION_PEOPLE = "people"
SECTION_IMAGENET = "imagenet"
SECTION_YOLO = "yolo"
SECTION_TEXT = "text"
SECTION_AUDIO = "audio"
SECTION_CUSTOM_QUERY = "custom_query"
SECTION_META = "meta"


# Ordered list of (key, display title). Drives both scaffold order and
# persistence of expand/collapse state.
SECTIONS: tuple[tuple[str, str], ...] = (
    (SECTION_SHOT, "Shot"),
    (SECTION_VISUAL, "Visual"),
    (SECTION_PEOPLE, "People"),
    (SECTION_IMAGENET, "ImageNet"),
    (SECTION_YOLO, "Objects (YOLO)"),
    (SECTION_TEXT, "Text & Transcript"),
    (SECTION_AUDIO, "Audio"),
    (SECTION_CUSTOM_QUERY, "Custom Queries"),
    (SECTION_META, "Meta"),
)


class FilterSidebar(QWidget):
    """Vertical stack of collapsible filter sections bound to a ``FilterState``.

    Signals:
        visibility_requested(bool): emitted when user toggles the header
            hide button; the hosting tab is responsible for hiding/showing
            the widget and persisting the state.
        section_expanded_changed(str, bool): emitted when any section's
            collapse state changes (``section_key``, ``expanded``).
    """

    visibility_requested = Signal(bool)
    section_expanded_changed = Signal(str, bool)

    def __init__(
        self,
        filter_state: FilterState,
        section_expanded_state: Optional[dict[str, bool]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._filter_state = filter_state
        self._sections: dict[str, CollapsibleSection] = {}
        self._chip_groups: dict[str, ChipGroup] = {}
        self._updating_from_state = False
        self._setup_ui(section_expanded_state or {})
        self._populate_existing_filters()
        self._sync_from_state()
        self._filter_state.changed.connect(self._sync_from_state)
        self._refresh_theme()

        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self, expanded_state: dict[str, bool]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header row: title + hide button
        header = QWidget()
        header.setObjectName("FilterSidebarHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.SM, Spacing.SM)
        header_layout.setSpacing(Spacing.SM)

        title = QLabel("Filters")
        title.setObjectName("FilterSidebarTitle")
        header_layout.addWidget(title)
        header_layout.addStretch()

        hide_btn = QPushButton("Hide")
        hide_btn.setObjectName("FilterSidebarHideBtn")
        hide_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT - 8)
        hide_btn.clicked.connect(lambda: self.visibility_requested.emit(False))
        header_layout.addWidget(hide_btn)

        root.addWidget(header)

        # Scrollable section stack
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        section_layout = QVBoxLayout(container)
        section_layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.LG)
        section_layout.setSpacing(Spacing.MD)

        for key, title_text in SECTIONS:
            section = CollapsibleSection(title_text)
            placeholder = QLabel("(no filters yet)")
            placeholder.setObjectName("FilterSidebarPlaceholder")
            section.setContentWidget(placeholder)

            expanded = expanded_state.get(key, True)
            section.set_expanded(expanded)
            section.expanded_changed.connect(
                lambda exp, k=key: self.section_expanded_changed.emit(k, exp)
            )
            self._sections[key] = section
            section_layout.addWidget(section)

        section_layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        self.setMinimumWidth(280)

    # ── Filter population ────────────────────────────────────────────

    def _populate_existing_filters(self) -> None:
        """Install chip controls for the four multi-select enum filters.

        Unit 4 coverage: shot type, color palette, aspect ratio (inside
        Shot section), gaze direction. Duration / brightness / other
        controls arrive in Units 5–6.
        """
        # Shot type chips (Visual section)
        shot_chips = self._build_chip_group(
            [(st, get_display_name(st)) for st in SHOT_TYPES],
            lambda values: self._set_state("shot_type", values),
        )
        self._chip_groups["shot_type"] = shot_chips
        shot_label = self._labelled("Shot type", shot_chips)
        self.set_section_content(SECTION_VISUAL, shot_label)

        # Color palette chips (Visual section is already occupied — put into
        # the Visual section's layout via replaceWidget? Simpler: nest both
        # chip groups inside a single container widget.)

        color_chips = self._build_chip_group(
            [(cp, get_palette_display_name(cp)) for cp in COLOR_PALETTES],
            lambda values: self._set_state("color_palette", values),
        )
        self._chip_groups["color_palette"] = color_chips

        # Rebuild Visual section content to hold both shot + palette chips
        visual_container = QWidget()
        v_layout = QVBoxLayout(visual_container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(Spacing.SM)
        v_layout.addWidget(self._labelled("Shot type", shot_chips))
        v_layout.addWidget(self._labelled("Color palette", color_chips))
        self.set_section_content(SECTION_VISUAL, visual_container)

        # Aspect ratio chips (Shot section)
        aspect_chips = self._build_chip_group(
            [("16:9", "16:9"), ("4:3", "4:3"), ("9:16", "9:16 (vertical)")],
            lambda values: self._set_state("aspect_ratio", values),
        )
        self._chip_groups["aspect_ratio"] = aspect_chips
        self.set_section_content(SECTION_SHOT, self._labelled("Aspect ratio", aspect_chips))

        # Gaze direction chips (People section)
        gaze_options = [
            (key, display) for key, display in GAZE_CATEGORY_DISPLAY.items()
        ]
        gaze_chips = self._build_chip_group(
            gaze_options,
            lambda values: self._set_state("gaze_filter", values),
        )
        self._chip_groups["gaze_filter"] = gaze_chips
        self.set_section_content(SECTION_PEOPLE, self._labelled("Gaze direction", gaze_chips))

    def _build_chip_group(self, options, on_change) -> ChipGroup:
        group = ChipGroup()
        group.set_options(options)
        group.selection_changed.connect(on_change)
        return group

    def _labelled(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XXS)
        label = QLabel(label_text)
        label.setProperty("class", "FilterSidebarFieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return container

    def _set_state(self, field_name: str, values: set) -> None:
        """UI → state. Guarded against the round-trip from state → UI."""
        if self._updating_from_state:
            return
        setattr(self._filter_state, field_name, values)

    @Slot()
    def _sync_from_state(self) -> None:
        """State → UI. Push current FilterState values back onto the chip groups."""
        self._updating_from_state = True
        try:
            for field_name, group in self._chip_groups.items():
                current = getattr(self._filter_state, field_name)
                if isinstance(current, set):
                    group.set_selected(current)
        finally:
            self._updating_from_state = False

    # ── Public API ───────────────────────────────────────────────────

    @property
    def filter_state(self) -> FilterState:
        return self._filter_state

    def section(self, key: str) -> Optional[CollapsibleSection]:
        """Return the collapsible section for a given key, or None."""
        return self._sections.get(key)

    def section_expanded_state(self) -> dict[str, bool]:
        """Return the current expanded state for every section — for persistence."""
        return {key: section.expanded for key, section in self._sections.items()}

    def set_section_content(self, key: str, widget: QWidget) -> None:
        """Install a content widget inside a section."""
        section = self._sections.get(key)
        if section is None:
            raise KeyError(f"Unknown filter section: {key!r}")
        section.setContentWidget(widget)

    # ── Theming ──────────────────────────────────────────────────────

    @Slot()
    def _refresh_theme(self) -> None:
        c = theme()
        self.setStyleSheet(
            f"""
            FilterSidebar {{
                background: {c.background_secondary};
                border-left: 1px solid {c.border_secondary};
            }}
            #FilterSidebarHeader {{
                background: {c.background_secondary};
                border-bottom: 1px solid {c.border_secondary};
            }}
            #FilterSidebarTitle {{
                color: {c.text_primary};
                font-size: {TypeScale.LG}px;
                font-weight: 600;
            }}
            #FilterSidebarHideBtn {{
                background: {c.background_tertiary};
                color: {c.text_secondary};
                border: 1px solid {c.border_secondary};
                border-radius: 4px;
                padding: 2px {Spacing.SM}px;
                font-size: {TypeScale.SM}px;
            }}
            #FilterSidebarHideBtn:hover {{
                background: {c.accent_blue};
                color: {c.text_inverted};
                border: 1px solid {c.accent_blue};
            }}
            #FilterSidebarPlaceholder {{
                color: {c.text_muted};
                font-size: {TypeScale.SM}px;
                font-style: italic;
                padding: {Spacing.SM}px 0;
            }}
            """
        )
