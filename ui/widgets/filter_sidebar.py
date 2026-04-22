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
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.analysis.classification import load_imagenet_class_list
from core.analysis.color import COLOR_PALETTES, get_palette_display_name
from core.analysis.gaze import GAZE_CATEGORY_DISPLAY
from core.analysis.shots import SHOT_TYPES, get_display_name
from core.analysis_operations import ANALYSIS_OPERATIONS
from core.filter_state import FilterState
from ui.theme import Spacing, TypeScale, UISizes, theme
from ui.widgets.chip_group import ChipGroup
from ui.widgets.collapsible_section import CollapsibleSection
from ui.widgets.count_operator import CountOperator
from ui.widgets.typeahead_input import TypeaheadInput


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
        # Unit 5 controls
        self._person_count_control: Optional[CountOperator] = None
        self._tribool_groups: dict[str, tuple[QRadioButton, QRadioButton, QRadioButton]] = {}
        self._updating_from_state = False
        # Unit 6 state — dynamic YOLO vocabulary
        self._yolo_label_vocabulary: set[str] = set()
        self._imagenet_typeahead: Optional[TypeaheadInput] = None
        self._imagenet_chip_list: Optional[QWidget] = None
        self._imagenet_chip_buttons: dict[str, QPushButton] = {}
        self._imagenet_mode_any: Optional[QRadioButton] = None
        self._imagenet_mode_all: Optional[QRadioButton] = None
        self._yolo_total_control: Optional[CountOperator] = None
        self._setup_ui(section_expanded_state or {})
        self._populate_existing_filters()
        self._populate_unit5_filters()
        self._populate_unit6_filters()
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

    def _populate_unit5_filters(self) -> None:
        """Wire controls for person count, tribool booleans, has-analysis chips."""

        # People section — extend with person count operator
        existing_people = self._sections[SECTION_PEOPLE].findChild(QWidget)
        self._person_count_control = CountOperator()
        self._person_count_control.value_changed.connect(self._on_person_count_changed)

        people_container = QWidget()
        pc_layout = QVBoxLayout(people_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(Spacing.SM)
        if existing_people is not None:
            # Re-parent existing gaze control
            pc_layout.addWidget(self._chip_groups["gaze_filter"].parent())
        pc_layout.addWidget(
            self._labelled("Person count", self._person_count_control)
        )
        self.set_section_content(SECTION_PEOPLE, people_container)

        # Text section — has_transcript + has_on_screen_text tribools
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(Spacing.SM)
        text_layout.addWidget(
            self._tribool_row("Has transcript", "has_transcript")
        )
        text_layout.addWidget(
            self._tribool_row("Has on-screen text", "has_on_screen_text")
        )
        self.set_section_content(SECTION_TEXT, text_container)

        # Audio section — has_audio tribool only (volume slider deferred)
        audio_container = QWidget()
        audio_layout = QVBoxLayout(audio_container)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(Spacing.SM)
        audio_layout.addWidget(self._tribool_row("Has audio", "has_audio"))
        self.set_section_content(SECTION_AUDIO, audio_container)

        # Meta section — has_analysis_ops chips + enabled tribool
        meta_container = QWidget()
        meta_layout = QVBoxLayout(meta_container)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(Spacing.SM)

        op_options = [(op.key, op.label) for op in ANALYSIS_OPERATIONS]
        has_ops_chips = self._build_chip_group(
            op_options,
            lambda values: self._set_state("has_analysis_ops", values),
        )
        self._chip_groups["has_analysis_ops"] = has_ops_chips
        meta_layout.addWidget(self._labelled("Has analysis", has_ops_chips))
        meta_layout.addWidget(self._tribool_row("Enabled", "enabled_filter"))
        self.set_section_content(SECTION_META, meta_container)

    def _tribool_row(self, label_text: str, field_name: str) -> QWidget:
        """Yes / No / Any radio triplet bound to a tribool FilterState field."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        label = QLabel(label_text)
        label.setMinimumWidth(140)
        layout.addWidget(label)

        group = QButtonGroup(container)
        yes_btn = QRadioButton("Yes")
        no_btn = QRadioButton("No")
        any_btn = QRadioButton("Any")
        any_btn.setChecked(True)
        for btn in (yes_btn, no_btn, any_btn):
            group.addButton(btn)
            layout.addWidget(btn)
        layout.addStretch()

        def on_toggled(_checked):
            if self._updating_from_state:
                return
            if yes_btn.isChecked():
                setattr(self._filter_state, field_name, True)
            elif no_btn.isChecked():
                setattr(self._filter_state, field_name, False)
            else:
                setattr(self._filter_state, field_name, None)

        yes_btn.toggled.connect(on_toggled)
        no_btn.toggled.connect(on_toggled)
        any_btn.toggled.connect(on_toggled)

        self._tribool_groups[field_name] = (yes_btn, no_btn, any_btn)
        return container

    def _on_person_count_changed(self, op, n):
        if self._updating_from_state:
            return
        if op is None or n is None:
            self._filter_state.person_count = None
        else:
            self._filter_state.person_count = (op, n)

    # ── Unit 6 (ImageNet + YOLO) ─────────────────────────────────────

    def _populate_unit6_filters(self) -> None:
        """ImageNet typeahead (1000 classes) + YOLO filters."""
        self._populate_imagenet_section()
        self._populate_yolo_section()

    def _populate_imagenet_section(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # Mode switch (Any / All)
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(Spacing.SM)
        mode_layout.addWidget(QLabel("Match:"))
        self._imagenet_mode_any = QRadioButton("Any selected")
        self._imagenet_mode_all = QRadioButton("All selected")
        self._imagenet_mode_any.setChecked(True)
        mode_group = QButtonGroup(container)
        mode_group.addButton(self._imagenet_mode_any)
        mode_group.addButton(self._imagenet_mode_all)
        mode_layout.addWidget(self._imagenet_mode_any)
        mode_layout.addWidget(self._imagenet_mode_all)
        mode_layout.addStretch()

        def on_mode_toggled(_checked):
            if self._updating_from_state:
                return
            self._filter_state.imagenet_mode = (
                "all" if self._imagenet_mode_all.isChecked() else "any"
            )

        self._imagenet_mode_any.toggled.connect(on_mode_toggled)
        self._imagenet_mode_all.toggled.connect(on_mode_toggled)
        layout.addWidget(mode_row)

        # Typeahead input
        self._imagenet_typeahead = TypeaheadInput()
        self._imagenet_typeahead.setPlaceholderText("Search ImageNet class…")
        vocab = load_imagenet_class_list()
        if not vocab:
            self._imagenet_typeahead._line.setPlaceholderText(
                "Run Classify to build vocabulary"
            )
            self._imagenet_typeahead.setEnabled(False)
        else:
            self._imagenet_typeahead.set_vocabulary(vocab)
            self._imagenet_typeahead.value_selected.connect(
                self._on_imagenet_value_selected
            )
        layout.addWidget(self._imagenet_typeahead)

        # Selected-chips display
        self._imagenet_chip_list = QWidget()
        self._imagenet_chip_list_layout = QHBoxLayout(self._imagenet_chip_list)
        self._imagenet_chip_list_layout.setContentsMargins(0, 0, 0, 0)
        self._imagenet_chip_list_layout.setSpacing(Spacing.XS)
        self._imagenet_chip_list_layout.addStretch()
        layout.addWidget(self._imagenet_chip_list)

        self.set_section_content(SECTION_IMAGENET, container)

    def _on_imagenet_value_selected(self, value: str) -> None:
        if self._updating_from_state:
            return
        current = set(self._filter_state.imagenet_labels)
        if value in current:
            return
        current.add(value)
        self._filter_state.imagenet_labels = current

    def _rebuild_imagenet_chip_list(self) -> None:
        """Sync the visual chip list with the current imagenet_labels set."""
        if self._imagenet_chip_list is None:
            return
        layout = self._imagenet_chip_list_layout
        # Remove existing chip buttons
        for value, btn in list(self._imagenet_chip_buttons.items()):
            layout.removeWidget(btn)
            btn.setParent(None)
        self._imagenet_chip_buttons.clear()

        # Insert before trailing stretch
        for value in sorted(self._filter_state.imagenet_labels):
            btn = QPushButton(f"{value}  ×")
            btn.setProperty("chip_value", value)

            def on_click(_checked=False, v=value):
                if self._updating_from_state:
                    return
                current = set(self._filter_state.imagenet_labels)
                current.discard(v)
                self._filter_state.imagenet_labels = current

            btn.clicked.connect(on_click)
            btn.setObjectName("ImageNetChipBadge")
            self._imagenet_chip_buttons[value] = btn
            layout.insertWidget(layout.count() - 1, btn)

    def _populate_yolo_section(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # YOLO label chips (dynamic vocabulary)
        yolo_chips = ChipGroup()
        yolo_chips.selection_changed.connect(
            lambda values: self._set_state("yolo_labels", values)
        )
        self._chip_groups["yolo_labels"] = yolo_chips
        layout.addWidget(self._labelled("Detected labels", yolo_chips))

        # Total count operator
        self._yolo_total_control = CountOperator()
        self._yolo_total_control.value_changed.connect(
            self._on_yolo_total_count_changed
        )
        layout.addWidget(
            self._labelled("Total object count", self._yolo_total_control)
        )

        # Note about per-label count rules (full UI deferred)
        note = QLabel(
            "Per-label count rules available via apply_filters "
            "({'yolo_per_label_rules': [('person','=',1)]})"
        )
        note.setWordWrap(True)
        note.setObjectName("FilterSidebarPlaceholder")
        layout.addWidget(note)

        self.set_section_content(SECTION_YOLO, container)

    def _on_yolo_total_count_changed(self, op, n):
        if self._updating_from_state:
            return
        if op is None or n is None:
            self._filter_state.yolo_total_count = None
        else:
            self._filter_state.yolo_total_count = (op, n)

    def refresh_yolo_vocabulary(self, labels: set[str]) -> None:
        """Update the YOLO label chip vocabulary from the union of detected labels."""
        if labels == self._yolo_label_vocabulary:
            return
        self._yolo_label_vocabulary = set(labels)
        options = [(label, label) for label in sorted(labels)]
        self._chip_groups["yolo_labels"].set_options(options)
        # Restore selection after options rebuild
        self._chip_groups["yolo_labels"].set_selected(self._filter_state.yolo_labels)

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
        """State → UI. Push current FilterState values back onto controls."""
        self._updating_from_state = True
        try:
            for field_name, group in self._chip_groups.items():
                current = getattr(self._filter_state, field_name)
                if isinstance(current, set):
                    group.set_selected(current)

            # Person count operator
            if self._person_count_control is not None:
                pc = self._filter_state.person_count
                if pc is None:
                    self._person_count_control.clear()
                else:
                    self._person_count_control.set_value(pc[0], pc[1])

            # Triboolean radio groups
            for field_name, (yes_btn, no_btn, any_btn) in self._tribool_groups.items():
                current = getattr(self._filter_state, field_name)
                if current is True:
                    yes_btn.setChecked(True)
                elif current is False:
                    no_btn.setChecked(True)
                else:
                    any_btn.setChecked(True)

            # Unit 6 — ImageNet
            if self._imagenet_mode_any is not None:
                mode = self._filter_state.imagenet_mode
                if mode == "all":
                    self._imagenet_mode_all.setChecked(True)
                else:
                    self._imagenet_mode_any.setChecked(True)
            self._rebuild_imagenet_chip_list()

            # Unit 6 — YOLO total count
            if self._yolo_total_control is not None:
                yc = self._filter_state.yolo_total_count
                if yc is None:
                    self._yolo_total_control.clear()
                else:
                    self._yolo_total_control.set_value(yc[0], yc[1])
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
