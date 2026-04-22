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

from core.filter_state import FilterState
from ui.theme import Spacing, TypeScale, UISizes, theme
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
        self._setup_ui(section_expanded_state or {})
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
