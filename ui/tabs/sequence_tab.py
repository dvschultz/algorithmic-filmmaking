"""Sequence tab for timeline editing and playback with card-based sorting."""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QWidget,
    QStackedWidget,
    QComboBox,
    QPushButton,
    QLabel,
    QCheckBox,
    QMessageBox,
)
from PySide6.QtCore import Signal, Qt, Slot

from .base_tab import BaseTab
from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.widgets import SortingCardGrid, TimelinePreview, CostEstimatePanel
from ui.dialogs import ExquisiteCorpusDialog, StorytellerDialog, MissingDescriptionsDialog, ReferenceGuideDialog, SignatureStyleDialog, RoseHobartDialog, DiceRollDialog, FreeAssociationDialog
from ui.theme import theme, Spacing, TypeScale, UISizes
from ui.workers.sequence_worker import SequenceWorker
from core.remix import generate_sequence
from core.cost_estimates import estimate_sequence_cost
from core.analysis_dependencies import get_operation_feature_candidates
from core.feature_registry import check_feature_ready
from core.settings import get_llm_api_key, get_replicate_api_key, load_settings, save_settings

logger = logging.getLogger(__name__)

from ui.algorithm_config import ALGORITHM_CONFIG, get_algorithm_config, get_algorithm_label

# Reverse lookup: display label -> algorithm key
_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in ALGORITHM_CONFIG.items()}

# Gaze category filter options: (display_label, internal_key_or_None)
from core.analysis.gaze import GAZE_CATEGORY_DISPLAY

GAZE_FILTER_OPTIONS: list[tuple[str, str | None]] = [
    ("All Gaze", None),
] + [(display, key) for key, display in GAZE_CATEGORY_DISPLAY.items()]


def get_algorithm_key(label: str) -> str:
    """Get the algorithm key from a display label.

    Args:
        label: Display label (e.g., 'Chromatic Flow')

    Returns:
        Algorithm key (e.g., 'color')
    """
    return _LABEL_TO_KEY.get(label, label.lower().replace(" ", "_"))


class SequenceTab(BaseTab):
    """Tab for arranging clips on the timeline and previewing.

    Uses a two-state UI model:
    - STATE_CARDS: Shows grid of sorting algorithm cards (empty state)
    - STATE_TIMELINE: Shows header + video player + timeline preview + timeline

    Cards apply directly using selected clips from Analyze or Cut tabs.
    Header provides algorithm dropdown for "redo" functionality.

    Signals:
        playback_requested: Emitted when playback is requested (start_frame: int)
        stop_requested: Emitted when stop is requested
        export_requested: Emitted when export is requested
        clip_added: Emitted when a clip is added to timeline (clip: Clip, source: Source)
    """

    playback_requested = Signal(int)  # start_frame
    stop_requested = Signal()
    export_requested = Signal()
    clip_added = Signal(object, object)  # Clip, Source
    clips_data_changed = Signal(list)  # Emitted when auto-compute mutates clip metadata
    chromatic_bar_setting_changed = Signal(bool)  # True when bottom color bar should be visible

    # Intention-first workflow signal: emitted when card is clicked with no clips
    # Parameters: algorithm (str), direction (str or None)
    intention_import_requested = Signal(str, object)

    # Request description analysis on specific clips (for Storyteller)
    # Parameters: list of clip IDs that need descriptions
    description_analysis_requested = Signal(list)

    # State constants
    STATE_CARDS = 0      # Show card grid only
    STATE_TIMELINE = 1   # Show header + timeline + preview
    STATE_CONFIRM = 2    # Show cost estimate + generate button

    def __init__(self, parent=None):
        self._current_source = None
        self._clips = []  # List of Clip objects
        self._sources = {}  # source_id -> Source
        self._available_clips = []  # List of (Clip, Source) tuples
        self._current_algorithm = None
        self._current_state = self.STATE_CARDS
        self._gui_state = None  # Set by MainWindow
        self._project = None  # Set by MainWindow via set_project()

        # Guard flags
        self._apply_in_progress = False
        self._sequence_worker: Optional[SequenceWorker] = None
        self._algorithm_running = False  # Prevents dirty flag during algo runs
        self._sequence_dirty = False  # Set on manual user edits (drag, remove)

        # Confirm state: pending algorithm and clips for generation
        self._pending_algorithm: Optional[str] = None
        self._pending_clips: list = []
        self._show_chromatic_color_bar = False

        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Sequence tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # State stack for cards vs timeline
        self.state_stack = QStackedWidget()

        # STATE_CARDS (index 0): Guidance + card grid
        self.cards_view = self._create_cards_view()
        self.state_stack.addWidget(self.cards_view)

        # STATE_TIMELINE (index 1): Header + content
        self.timeline_view = self._create_timeline_view()
        self.state_stack.addWidget(self.timeline_view)

        # STATE_CONFIRM (index 2): Cost estimate + generate/back buttons
        self.confirm_view = self._create_confirm_view()
        self.state_stack.addWidget(self.confirm_view)

        layout.addWidget(self.state_stack)

        # Start in cards state
        self._set_state(self.STATE_CARDS)

    def _create_cards_view(self) -> QWidget:
        """Create the cards view with empty-sequence guidance."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.MD)

        self.cards_hint_label = QLabel(
            "Timeline is empty. Select clips in Analyze or Cut, or choose a sequencer below to build one."
        )
        self.cards_hint_label.setWordWrap(True)
        self.cards_hint_label.setStyleSheet(
            f"color: {theme().text_muted}; font-size: {TypeScale.SM}px;"
        )
        layout.addWidget(self.cards_hint_label)

        self.card_grid = SortingCardGrid()
        self.card_grid.algorithm_selected.connect(self._on_card_clicked)
        self.card_grid.category_changed.connect(self._on_category_changed)
        layout.addWidget(self.card_grid)

        # Restore persisted category (set_category does not emit, so no save loop)
        settings = load_settings()
        self.card_grid.set_category(settings.sequence_selected_category)

        return container

    def _create_timeline_view(self) -> QWidget:
        """Create the timeline view with header, player, preview, and timeline."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # New header row
        self.header_widget = self._create_header()
        layout.addWidget(self.header_widget)

        # Main content splitter
        self.timeline_splitter = QSplitter(Qt.Vertical)
        self.timeline_splitter.setChildrenCollapsible(False)

        # Video player
        self.video_player = VideoPlayer()
        self.video_player.setMinimumHeight(180)
        self.video_player.show_ab_loop_controls(True)
        self.video_player.set_sequence_mode(True)
        self.video_player.play_requested.connect(self._on_playback_requested)
        self.video_player.stop_requested.connect(self._on_stop_requested)
        self.timeline_splitter.addWidget(self.video_player)

        # Timeline preview strip (moved from parameter view)
        self.timeline_preview = TimelinePreview()
        self.timeline_preview.setMinimumHeight(90)
        self.timeline_preview.setMaximumHeight(100)
        self.timeline_splitter.addWidget(self.timeline_preview)

        # Timeline widget
        self.timeline = TimelineWidget()
        self.timeline.setMinimumHeight(180)
        self.timeline.playhead_changed.connect(self._on_playhead_changed)
        self.timeline.export_requested.connect(self._on_export_requested)
        self.timeline.sequence_changed.connect(self._on_timeline_user_edit)
        self.timeline_splitter.addWidget(self.timeline)

        # Set splitter sizes
        self.timeline_splitter.setSizes([300, 90, 220])
        for index in range(self.timeline_splitter.count()):
            self.timeline_splitter.setCollapsible(index, False)

        layout.addWidget(self.timeline_splitter)

        return container

    def _create_header(self) -> QWidget:
        """Create the header row with sequence dropdown, algorithm dropdown, and buttons."""
        header = QWidget()
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {theme().background_secondary};
                border-bottom: 1px solid {theme().border_primary};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)

        # Sequence dropdown (leftmost — higher-level context)
        seq_label = QLabel("Sequence:")
        seq_label.setStyleSheet(f"color: {theme().text_secondary}; border: none;")
        layout.addWidget(seq_label)

        self.sequence_dropdown = QComboBox()
        self.sequence_dropdown.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self.sequence_dropdown.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.sequence_dropdown.currentIndexChanged.connect(self._on_sequence_switched)
        self.sequence_dropdown.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sequence_dropdown.customContextMenuRequested.connect(self._on_sequence_context_menu)
        layout.addWidget(self.sequence_dropdown)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {theme().text_secondary}; border: none; margin: 0 4px;")
        layout.addWidget(sep)

        label = QLabel("Algorithm:")
        label.setStyleSheet(f"color: {theme().text_secondary}; border: none;")
        layout.addWidget(label)

        self.algorithm_dropdown = QComboBox()
        # Populate with labels from non-dialog algorithms (exclude exquisite_corpus, storyteller)
        _dropdown_keys = [
            "shuffle", "sequential", "duration", "color",
            "brightness", "volume", "shot_type", "proximity",
            "similarity_chain", "match_cut",
            "gaze_sort", "gaze_consistency",
        ]
        self.algorithm_dropdown.addItems([get_algorithm_label(k) for k in _dropdown_keys])
        self.algorithm_dropdown.setMinimumWidth(140)
        self.algorithm_dropdown.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_dropdown)

        # Direction dropdown (visible for Duration and Color algorithms)
        self.direction_label = QLabel("Direction:")
        self.direction_label.setStyleSheet(f"color: {theme().text_secondary}; border: none; margin-left: {Spacing.LG}px;")
        layout.addWidget(self.direction_label)

        self.direction_dropdown = QComboBox()
        self.direction_dropdown.setMinimumWidth(140)
        self.direction_dropdown.currentTextChanged.connect(self._on_direction_changed)
        layout.addWidget(self.direction_dropdown)

        self.chromatic_bar_checkbox = QCheckBox("Show Color Bar")
        self.chromatic_bar_checkbox.setToolTip(
            "Show a full-width bar at the bottom of playback/export that "
            "uses each clip's dominant color."
        )
        self.chromatic_bar_checkbox.toggled.connect(self._on_chromatic_bar_toggled)
        layout.addWidget(self.chromatic_bar_checkbox)

        # Gaze filter dropdown
        self.gaze_filter_label = QLabel("Gaze:")
        self.gaze_filter_label.setStyleSheet(f"color: {theme().text_secondary}; border: none; margin-left: {Spacing.LG}px;")
        layout.addWidget(self.gaze_filter_label)

        self.gaze_filter_dropdown = QComboBox()
        self.gaze_filter_dropdown.setMinimumWidth(140)
        for display_label, _internal_key in GAZE_FILTER_OPTIONS:
            self.gaze_filter_dropdown.addItem(display_label)
        self.gaze_filter_dropdown.currentIndexChanged.connect(self._on_gaze_filter_changed)
        layout.addWidget(self.gaze_filter_dropdown)

        # Initially hide direction controls and gaze filter
        self.direction_label.hide()
        self.direction_dropdown.hide()
        self.chromatic_bar_checkbox.hide()
        self.gaze_filter_label.hide()
        self.gaze_filter_dropdown.hide()

        layout.addStretch()

        self.clear_btn = QPushButton("Clear Sequence")
        self.clear_btn.setToolTip("Clear timeline and return to card selection")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self.clear_btn)

        return header

    def _create_confirm_view(self) -> QWidget:
        """Create the confirmation view with cost estimate panel and generate button."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(Spacing.LG)

        # Algorithm title
        self._confirm_algo_label = QLabel()
        self._confirm_algo_label.setStyleSheet(
            f"font-size: {TypeScale.XL}px; font-weight: bold;"
        )
        layout.addWidget(self._confirm_algo_label)

        # Clip count summary
        self._confirm_clips_label = QLabel()
        self._confirm_clips_label.setStyleSheet(f"font-size: {TypeScale.BASE}px;")
        layout.addWidget(self._confirm_clips_label)

        # Cost estimate panel
        self._confirm_cost_panel = CostEstimatePanel()
        self._confirm_cost_panel.tier_changed.connect(self._on_confirm_tier_changed)
        layout.addWidget(self._confirm_cost_panel)

        self._confirm_chromatic_bar_checkbox = QCheckBox(
            "Show bottom color bar (preview + export)"
        )
        self._confirm_chromatic_bar_checkbox.setVisible(False)
        layout.addWidget(self._confirm_chromatic_bar_checkbox)

        # No-color-data handling dropdown (visible only for color algorithm)
        self._confirm_no_color_layout = QHBoxLayout()
        no_color_label = QLabel("Clips without color data:")
        no_color_label.setStyleSheet(f"color: {theme().text_secondary}; border: none;")
        self._confirm_no_color_layout.addWidget(no_color_label)
        self._confirm_no_color_dropdown = QComboBox()
        self._confirm_no_color_dropdown.addItems([
            "Append at End",
            "Exclude",
            "Sort Inline",
        ])
        self._confirm_no_color_dropdown.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._confirm_no_color_layout.addWidget(self._confirm_no_color_dropdown)
        self._confirm_no_color_layout.addStretch()
        self._confirm_no_color_container = QWidget()
        self._confirm_no_color_container.setLayout(self._confirm_no_color_layout)
        self._confirm_no_color_container.setVisible(False)
        layout.addWidget(self._confirm_no_color_container)

        layout.addStretch()

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._confirm_back_btn = QPushButton("Back")
        self._confirm_back_btn.clicked.connect(lambda: self._set_state(self.STATE_CARDS))
        btn_layout.addWidget(self._confirm_back_btn)

        self._confirm_generate_btn = QPushButton("Generate Sequence")
        self._confirm_generate_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-size: {TypeScale.MD}px;
                font-weight: bold;
            }}
        """)
        self._confirm_generate_btn.clicked.connect(self._on_confirm_generate)
        btn_layout.addWidget(self._confirm_generate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return container

    def _show_confirm_view(self, algorithm: str, clips: list, estimates: list):
        """Show the confirmation view with cost estimates.

        Args:
            algorithm: Algorithm key
            clips: List of (Clip, Source) tuples
            estimates: List of OperationEstimate from cost engine
        """
        config = get_algorithm_config(algorithm)
        self._confirm_algo_label.setText(f"{config['icon']}  {config['label']}")
        self._confirm_clips_label.setText(f"{len(clips)} clips selected")
        self._confirm_cost_panel.set_estimates(estimates)
        self._update_confirm_warnings(estimates)
        is_color = algorithm.lower() == "color"
        self._confirm_chromatic_bar_checkbox.setVisible(is_color)
        if is_color:
            self._confirm_chromatic_bar_checkbox.setChecked(self._show_chromatic_color_bar)
        else:
            self._confirm_chromatic_bar_checkbox.setChecked(False)
        self._confirm_no_color_container.setVisible(is_color)
        if is_color:
            self._confirm_no_color_dropdown.setCurrentIndex(0)  # "Append at End"
        self._set_state(self.STATE_CONFIRM)

    def _refresh_confirm_estimates(self):
        """Recalculate cost estimates for the confirm view using current state."""
        if not self._pending_algorithm or not self._pending_clips:
            return
        overrides = self._confirm_cost_panel.get_tier_overrides()
        clip_objects = [clip for clip, source in self._pending_clips]
        estimates = estimate_sequence_cost(
            self._pending_algorithm, clip_objects, tier_overrides=overrides
        )
        self._confirm_cost_panel.set_estimates(estimates)

        # If all analysis is now satisfied, update the clips label
        if not estimates:
            self._confirm_clips_label.setText(
                f"{len(self._pending_clips)} clips selected — all analysis complete"
            )

        self._update_confirm_warnings(estimates)

    def _get_cloud_api_warning(self, estimates: list) -> str | None:
        """Return warning text when cloud tiers are selected without API keys."""
        if not estimates:
            return None

        cloud_ops = [e for e in estimates if e.tier == "cloud"]
        if not cloud_ops:
            return None

        missing = []
        needs_llm = any(e.operation in ("describe", "extract_text", "cinematography") for e in cloud_ops)
        needs_replicate = any(e.operation == "shots" for e in cloud_ops)

        if needs_llm and not get_llm_api_key():
            missing.append("LLM")
        if needs_replicate and not get_replicate_api_key():
            missing.append("Replicate")

        if missing:
            keys = " and ".join(missing)
            return (
                f"Cloud tier selected but no {keys} API key configured. "
                f"Set keys in Settings or switch to Local tier."
            )
        return None

    def _get_missing_local_dependency_warning(self, estimates: list) -> str | None:
        """Return warning text when local sequence dependencies are unavailable."""
        if not estimates:
            return None

        settings = load_settings()
        blocked_labels: list[str] = []

        for estimate in estimates:
            feature_candidates = get_operation_feature_candidates(
                estimate.operation,
                settings,
            )
            if not feature_candidates:
                continue
            if any(check_feature_ready(name)[0] for name in feature_candidates):
                continue
            blocked_labels.append(estimate.label)

        if not blocked_labels:
            return None

        unique_labels = list(dict.fromkeys(blocked_labels))
        labels = ", ".join(unique_labels)
        return (
            f"{labels} require local dependencies that are not installed or are broken. "
            "Install them in Settings > Dependencies before generating this sequence."
        )

    def _update_confirm_warnings(self, estimates: list) -> None:
        """Refresh confirm-view warnings and generate-button enabled state."""
        warnings: list[str] = []

        cloud_warning = self._get_cloud_api_warning(estimates)
        if cloud_warning:
            warnings.append(cloud_warning)

        dependency_warning = self._get_missing_local_dependency_warning(estimates)
        if dependency_warning:
            warnings.append(dependency_warning)

        self._confirm_cost_panel.set_warning("\n\n".join(warnings) if warnings else None)
        self._confirm_generate_btn.setEnabled(dependency_warning is None)

    def _on_confirm_tier_changed(self, operation: str, tier: str):
        """Recalculate estimates when user changes a tier dropdown in confirm view."""
        self._refresh_confirm_estimates()

    def _on_confirm_generate(self):
        """Handle Generate button click in confirm view."""
        if not self._pending_algorithm or not self._pending_clips:
            return

        algorithm = self._pending_algorithm
        clips = self._pending_clips

        # Clear pending state
        self._pending_algorithm = None
        self._pending_clips = []

        no_color_handling = None
        if algorithm.lower() == "color":
            self.set_chromatic_color_bar_enabled(
                self._confirm_chromatic_bar_checkbox.isChecked(),
                emit_signal=False,
            )
            no_color_handling = self._get_confirm_no_color_handling()

        # Dialog-based algorithms still use their dialogs for the actual generation
        if algorithm == "shuffle":
            self._apply_chromatic_bar_to_sequence("shuffle")
            self._show_dice_roll_dialog(clips)
            return
        if algorithm == "exquisite_corpus":
            self._apply_chromatic_bar_to_sequence("exquisite_corpus")
            self._show_exquisite_corpus_dialog(clips)
            return
        if algorithm == "storyteller":
            self._apply_chromatic_bar_to_sequence("storyteller")
            self._show_storyteller_dialog(clips)
            return
        if algorithm == "signature_style":
            self._apply_chromatic_bar_to_sequence("signature_style")
            self._show_signature_style_dialog(clips)
            return
        if algorithm == "rose_hobart":
            self._apply_chromatic_bar_to_sequence("rose_hobart")
            self._show_rose_hobart_dialog(clips)
            return
        if algorithm == "staccato":
            self._apply_chromatic_bar_to_sequence("staccato")
            self._show_staccato_dialog(clips)
            return

        self._apply_algorithm(algorithm, clips, no_color_handling=no_color_handling)

    def set_gui_state(self, gui_state):
        """Set the GUI state reference (called by MainWindow)."""
        self._gui_state = gui_state

    def set_available_clips(self, clips_with_sources: list, all_clips: list = None, sources_by_id: dict = None):
        """Set the pool of all clips available for sequencing.

        Args:
            clips_with_sources: List of (Clip, Source) tuples
            all_clips: Optional list of all Clip objects (derived from clips_with_sources if not provided)
            sources_by_id: Optional dict mapping source_id -> Source (derived from clips_with_sources if not provided)
        """
        self._available_clips = clips_with_sources
        self._clips = all_clips if all_clips is not None else [clip for clip, source in clips_with_sources]
        if sources_by_id is not None:
            self._sources.update(sources_by_id)
        else:
            self._sources.update({source.id: source for clip, source in clips_with_sources if source})
        logger.debug(f"Set available clips in Sequence tab: {len(self._available_clips)}")

        # Refresh cost estimates if the confirm view is active
        if self.state_stack.currentIndex() == self.STATE_CONFIRM:
            self._refresh_confirm_estimates()

    def _resolve_selected_clips(self, selected_ids: list[str]) -> list[tuple]:
        """Resolve selected clip IDs to ``(clip, source)`` tuples.

        Falls back to the tab's authoritative clip/source registries when
        ``_available_clips`` is empty or only contains a filtered subset.
        """
        if not selected_ids:
            return []

        clip_lookup: dict[str, tuple] = {}

        for clip, source in self._available_clips:
            if source is not None:
                clip_lookup[clip.id] = (clip, source)

        for clip in self._clips:
            source = self._sources.get(clip.source_id)
            if source is not None:
                clip_lookup.setdefault(clip.id, (clip, source))

        return [
            clip_lookup[clip_id]
            for clip_id in selected_ids
            if clip_id in clip_lookup
        ]

    # --- Signal handlers ---

    @Slot(str)
    def _on_category_changed(self, category: str):
        """Persist selected category to settings."""
        settings = load_settings()
        settings.sequence_selected_category = category
        save_settings(settings)

    def _on_card_clicked(self, algorithm: str):
        """Handle card click - generate sequence from selected clips.

        If no clips exist in the project, triggers the intention-first workflow
        by emitting intention_import_requested signal.

        Special handling for "exquisite_corpus" - opens a dialog for the
        text-to-poem workflow.
        """
        logger.debug(f"Card clicked: {algorithm}")

        # Check if we have any clips in the project at all
        # (not just selected - the intention workflow is for empty projects)
        if not self._available_clips and not self._clips:
            logger.info(f"No clips available, triggering intention workflow for {algorithm}")
            # Emit signal with algorithm and direction (None for now, MainWindow handles)
            self.intention_import_requested.emit(algorithm, None)
            return

        # Get selected clips from GUI state (prefer Analyze, fallback to Cut)
        selected_ids = []
        if self._gui_state:
            selected_ids = (
                self._gui_state.analyze_selected_ids
                or self._gui_state.cut_selected_ids
                or []
            )

        if not selected_ids:
            QMessageBox.warning(
                self,
                "No Clips Selected",
                "Select clips in the Analyze or Cut tab first."
            )
            return

        clips = self._resolve_selected_clips(selected_ids)

        if not clips:
            QMessageBox.warning(
                self,
                "Clips Not Found",
                "Selected clips are not available in the Sequence tab. "
                "The clips may be from a source that hasn't been loaded."
            )
            return

        # Dialog-based algorithms handle their own analysis prereqs
        # Route them directly to avoid the cost confirmation gate
        cfg = ALGORITHM_CONFIG.get(algorithm, {})
        if cfg.get("is_dialog"):
            if algorithm == "shuffle":
                self._show_dice_roll_dialog(clips)
                return
            if algorithm == "exquisite_corpus":
                self._show_exquisite_corpus_dialog(clips)
                return
            if algorithm == "storyteller":
                self._show_storyteller_dialog(clips)
                return
            if algorithm == "reference_guided":
                self._show_reference_guide_dialog(clips)
                return
            if algorithm == "signature_style":
                self._show_signature_style_dialog(clips)
                return
            if algorithm == "rose_hobart":
                self._show_rose_hobart_dialog(clips)
                return
            if algorithm == "staccato":
                self._show_staccato_dialog(clips)
                return
            if algorithm == "eyes_without_a_face":
                self._show_eyes_without_a_face_dialog(clips)
                return
            if algorithm == "free_association":
                self._show_free_association_dialog(clips)
                return

        # Compute cost estimates for this algorithm
        clip_objects = [clip for clip, source in clips]
        estimates = estimate_sequence_cost(algorithm, clip_objects)

        if estimates or algorithm == "color":
            # Show gatekeeper with cost panel. Chromatics always uses this
            # step so users can choose color-bar and no-color-data settings.
            self._pending_algorithm = algorithm
            self._pending_clips = clips
            self._show_confirm_view(algorithm, clips, estimates)
        else:
            self._apply_algorithm(algorithm, clips)

    _NO_COLOR_HANDLING_MAP = {
        "Append at End": "append_end",
        "Exclude": "exclude",
        "Sort Inline": "sort_inline",
    }

    def _get_confirm_no_color_handling(self) -> str:
        """Get the no-color-data handling option from the confirm view dropdown."""
        text = self._confirm_no_color_dropdown.currentText()
        return self._NO_COLOR_HANDLING_MAP.get(text, "append_end")

    def _apply_algorithm(self, algorithm: str, clips: list, direction: str = None, no_color_handling: str = None):
        """Generate sequence in a background worker and transition to timeline.

        Heavy auto-compute operations (brightness, volume, CLIP embeddings)
        run off the main thread so the UI stays responsive.

        Args:
            algorithm: Algorithm name
            clips: List of (Clip, Source) tuples
            direction: Sort direction (e.g., "short_first", "long_first" for duration)
            no_color_handling: For color algorithm — "append_end", "exclude", or "sort_inline"
        """
        if self._apply_in_progress:
            logger.warning("Apply already in progress, ignoring")
            return

        algo_lower = algorithm.lower()
        estimates = estimate_sequence_cost(
            algo_lower,
            [clip for clip, _source in clips],
        )
        dependency_warning = self._get_missing_local_dependency_warning(estimates)
        if dependency_warning:
            QMessageBox.warning(self, "Missing Dependencies", dependency_warning)
            return

        self._apply_in_progress = True

        # Cancel any previous worker
        if self._sequence_worker is not None:
            self._sequence_worker.cancel()
            self._sequence_worker.wait(2000)
            self._sequence_worker = None

        self.status_message.emit(f"Generating {algo_lower} sequence...")

        worker = SequenceWorker(
            algorithm=algo_lower,
            clips=clips,
            direction=direction,
            no_color_handling=no_color_handling,
            parent=self,
        )
        # Store algorithm for the completion slot
        worker._pending_algorithm = algorithm
        worker._pending_direction = direction
        worker.sequence_ready.connect(self._on_sequence_ready)
        worker.error.connect(self._on_sequence_error)
        self._sequence_worker = worker
        worker.start()

    def _on_sequence_ready(self, sorted_clips: list):
        """Handle completed sequence generation (runs on main thread)."""
        worker = self._sequence_worker
        algorithm = getattr(worker, "_pending_algorithm", "") if worker else ""
        direction = getattr(worker, "_pending_direction", None) if worker else None
        algo_lower = algorithm.lower()

        try:
            # Create a new sequence for this algorithm run
            self._create_and_activate_sequence(algo_lower)
            self.timeline.clear_timeline()

            current_frame = 0
            for clip, source in sorted_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Update preview
            self.timeline_preview.set_clips(sorted_clips, self._sources)

            # Zoom to fit
            self.timeline._on_zoom_fit()

            # Update dropdowns to show current algorithm (block signals to avoid recursion)
            self.algorithm_dropdown.blockSignals(True)
            self.algorithm_dropdown.setCurrentText(get_algorithm_label(algorithm))
            self.algorithm_dropdown.blockSignals(False)

            # Update direction dropdown
            self._update_direction_dropdown(algorithm, direction)

            self._current_algorithm = algo_lower

            # Persist algorithm on the sequence for SRT export
            sequence = self.timeline.get_sequence()
            sequence.algorithm = algo_lower
            self._apply_chromatic_bar_to_sequence(algo_lower)
            self._update_chromatic_bar_controls(algo_lower)
            self._emit_chromatic_bar_setting_changed()

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sorted_clips)} clips with {algorithm} algorithm")

            # Notify that clip metadata may have been mutated by auto-compute
            self.clips_data_changed.emit([clip for clip, _ in sorted_clips])

        except Exception as e:
            logger.error(f"Error populating timeline: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate sequence: {e}")

        finally:
            self._algorithm_running = False
            self._apply_in_progress = False
            self._sequence_worker = None

    def _on_sequence_error(self, error_msg: str):
        """Handle sequence generation failure."""
        logger.error(f"Sequence worker error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Failed to generate sequence: {error_msg}")
        self._apply_in_progress = False
        self._sequence_worker = None

    def _show_exquisite_corpus_dialog(self, clips: list):
        """Show the Exquisite Corpus dialog for text extraction and poem generation.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        # Build sources_by_id from the clips
        sources_by_id = {source.id: source for clip, source in clips}

        # Extract just the clip objects
        clip_objects = [clip for clip, source in clips]

        dialog = ExquisiteCorpusDialog(
            clips=clip_objects,
            sources_by_id=sources_by_id,
            parent=self,
        )

        # Connect to sequence_ready signal
        dialog.sequence_ready.connect(self._apply_exquisite_corpus_sequence)

        dialog.exec()

    def _apply_dialog_sequence(
        self,
        sequence_clips: list,
        algorithm_key: str,
        display_label: str,
        sequence_metadata: Optional[dict] = None,
    ):
        """Apply a sequence from any dialog-based algorithm.

        Shared implementation for Exquisite Corpus, Storyteller, and Reference Guide.

        Args:
            sequence_clips: List of (Clip, Source) tuples in order
            algorithm_key: Internal algorithm key (e.g. "exquisite_corpus")
            display_label: Display name for dropdown (e.g. "Exquisite Corpus")
            sequence_metadata: Optional dict of extra fields to set on the Sequence
                (e.g. reference_source_id, dimension_weights for reference_guided)
        """
        if not sequence_clips:
            logger.warning(f"No clips in {display_label} sequence")
            return

        try:
            # Create a new sequence for this dialog algorithm
            self._create_and_activate_sequence(algorithm_key)
            self.timeline.clear_timeline()

            first_clip, first_source = sequence_clips[0]
            self.timeline.set_fps(first_source.fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source in sequence_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            self.timeline_preview.set_clips(sequence_clips, self._sources)
            self.timeline._on_zoom_fit()

            self.algorithm_dropdown.blockSignals(True)
            if self.algorithm_dropdown.findText(display_label) == -1:
                self.algorithm_dropdown.addItem(display_label)
            self.algorithm_dropdown.setCurrentText(display_label)
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = algorithm_key

            sequence = self.timeline.get_sequence()
            sequence.algorithm = algorithm_key
            self._apply_chromatic_bar_to_sequence(algorithm_key)
            self._update_chromatic_bar_controls(algorithm_key)
            self._emit_chromatic_bar_setting_changed()

            if sequence_metadata:
                for key, value in sequence_metadata.items():
                    if hasattr(sequence, key):
                        setattr(sequence, key, value)

            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_clips)} clips from {display_label}")

        except Exception as e:
            logger.error(f"Error applying {display_label} sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

        finally:
            self._algorithm_running = False

    @Slot(list)
    def _apply_exquisite_corpus_sequence(self, sequence_clips: list):
        """Apply the sequence from Exquisite Corpus dialog."""
        self._apply_dialog_sequence(sequence_clips, "exquisite_corpus", "Exquisite Corpus")

    def _show_storyteller_dialog(self, clips: list):
        """Show the Storyteller dialog for narrative sequence generation.

        Handles missing descriptions by prompting user to either exclude
        clips without descriptions or navigate to Analyze tab.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        # Build sources_by_id from the clips
        sources_by_id = {source.id: source for clip, source in clips}

        # Extract just the clip objects
        clip_objects = [clip for clip, source in clips]

        # Check for clips with descriptions
        clips_with_desc = [c for c in clip_objects if c.description]
        clips_without_desc = [c for c in clip_objects if not c.description]

        if not clips_with_desc:
            # No clips have descriptions
            QMessageBox.warning(
                self,
                "No Descriptions",
                "None of the selected clips have descriptions.\n\n"
                "Run description analysis in the Analyze tab first, "
                "then return to create a narrative sequence."
            )
            return

        # If some clips are missing descriptions, show prompt
        if clips_without_desc:
            missing_dialog = MissingDescriptionsDialog(
                clips_without_descriptions=clips_without_desc,
                total_clips=len(clip_objects),
                parent=self,
            )

            # Connect signals - pass clips without descriptions for analysis
            clips_needing_analysis = clips_without_desc
            missing_dialog.analyze_selected.connect(
                lambda: self._on_storyteller_analyze_requested(clips_needing_analysis)
            )

            result = missing_dialog.exec()
            if result == 0:  # Rejected/cancelled
                return

            # If analyze was selected, the signal handler navigated away
            # Only continue if exclude was selected (result == 1 means accepted)
            if not missing_dialog.result():
                return

            # User chose to exclude - filter to only clips with descriptions
            clip_objects = clips_with_desc
            # Update sources_by_id to match
            sources_by_id = {
                source.id: source for clip, source in clips
                if clip.description
            }

        # Show the Storyteller dialog
        dialog = StorytellerDialog(
            clips=clip_objects,
            sources_by_id=sources_by_id,
            project=None,  # Not needed for basic operation
            parent=self,
        )

        # Connect to sequence_ready signal
        dialog.sequence_ready.connect(self._apply_storyteller_sequence)

        dialog.exec()

    def _on_storyteller_analyze_requested(self, clips_without_descriptions: list):
        """Handle user choosing to run analysis before Storyteller.

        Emits signal to MainWindow to run description analysis on the specified clips.

        Args:
            clips_without_descriptions: List of Clip objects that need descriptions
        """
        clip_ids = [clip.id for clip in clips_without_descriptions]
        logger.info(f"Requesting description analysis for {len(clip_ids)} clips")
        self.description_analysis_requested.emit(clip_ids)

    @Slot(list)
    def _apply_storyteller_sequence(self, sequence_clips: list):
        """Apply the sequence from Storyteller dialog."""
        self._apply_dialog_sequence(sequence_clips, "storyteller", "Storyteller")

    def _show_free_association_dialog(self, clips: list):
        """Show the Free Association dialog for step-by-step LLM sequencing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        sources_by_id = {source.id: source for clip, source in clips}
        clip_objects = [clip for clip, source in clips]

        clips_with_desc = [c for c in clip_objects if c.description]
        if not clips_with_desc:
            QMessageBox.warning(
                self,
                "No Descriptions",
                "None of the selected clips have descriptions.\n\n"
                "Run description analysis in the Analyze tab first, "
                "then return to build a Free Association sequence.",
            )
            return

        dialog = FreeAssociationDialog(
            clips=clips_with_desc,
            sources_by_id=sources_by_id,
            project=None,
            parent=self,
        )
        dialog.sequence_ready.connect(self._apply_free_association_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_free_association_sequence(self, payload: list):
        """Apply the sequence from the Free Association dialog.

        The payload is a list of (Clip, Source, Optional[str]) triples — the
        third element is the LLM-generated rationale for that transition
        (None for the user-selected first clip). This differs from the
        generic _apply_dialog_sequence path because that path's timeline
        pipeline reconstructs SequenceClip internally with no mechanism
        to thread a rationale parameter through.

        Args:
            payload: List of (Clip, Source, Optional[str]) tuples in order.
        """
        if not payload:
            logger.warning("No clips in Free Association sequence")
            return

        # Strip rationales to feed the standard apply path that already
        # handles timeline clearing, clip addition, fps/video setup, etc.
        sequence_clips = [(clip, source) for clip, source, _ in payload]
        try:
            self._apply_dialog_sequence(
                sequence_clips, "free_association", "Free Association"
            )
        except Exception:
            logger.exception("Failed to apply Free Association sequence")
            return

        # Now thread rationales onto the SequenceClips the timeline just
        # created. Match by (source_clip_id, start_frame) which is unique
        # for the clips we just placed sequentially in this apply call.
        sequence = self.timeline.get_sequence()
        existing_clips = {
            (sc.source_clip_id, sc.start_frame): sc for sc in sequence.get_all_clips()
        }

        current_frame = 0
        attached = 0
        for clip, _source, rationale in payload:
            if rationale is not None:
                key = (clip.id, current_frame)
                seq_clip = existing_clips.get(key)
                if seq_clip is not None:
                    seq_clip.rationale = rationale
                    attached += 1
                else:
                    logger.warning(
                        "Could not find SequenceClip for (%s, %s) to attach rationale",
                        clip.id,
                        current_frame,
                    )
            current_frame += clip.duration_frames

        logger.info(
            "Attached %d rationale(s) to Free Association sequence",
            attached,
        )

    def _show_reference_guide_dialog(self, clips: list):
        """Show the Reference Guide dialog for reference-guided remixing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        sources_by_id = {source.id: source for clip, source in clips}

        dialog = ReferenceGuideDialog(
            clips=clips,
            sources_by_id=sources_by_id,
            parent=self,
        )

        dialog.sequence_ready.connect(self._apply_reference_guide_sequence)
        dialog.exec()

    @Slot(list, dict)
    def _apply_reference_guide_sequence(self, sequence_clips: list, metadata: dict = None):
        """Apply the sequence from Reference Guide dialog."""
        self._apply_dialog_sequence(
            sequence_clips, "reference_guided", "Reference Guide",
            sequence_metadata=metadata,
        )

    def _show_signature_style_dialog(self, clips: list):
        """Show the Signature Style dialog for drawing-based sequencing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        clip_objects = [clip for clip, source in clips]
        sources_by_id = {source.id: source for clip, source in clips}

        dialog = SignatureStyleDialog(
            clips=clip_objects,
            sources_by_id=sources_by_id,
            parent=self,
        )

        dialog.sequence_ready.connect(self._apply_signature_style_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_signature_style_sequence(self, sequence_data: list):
        """Apply the sequence from Signature Style dialog.

        The dialog emits (Clip, Source, in_point, out_point) tuples
        to support clip trimming based on drawing segment durations.
        """
        if not sequence_data:
            logger.warning("No clips in Signature Style sequence")
            return

        try:
            from models.sequence import SequenceClip

            self._create_and_activate_sequence("signature_style")
            self.timeline.clear_timeline()

            first_clip, first_source, _, _ = sequence_data[0]
            fps = first_source.fps
            self.timeline.set_fps(fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source, in_point, out_point in sequence_data:
                self.timeline.scene.add_clip_to_track(
                    track_index=0,
                    source_clip_id=clip.id,
                    source_id=source.id,
                    start_frame=current_frame,
                    in_point=clip.start_frame + in_point,
                    out_point=clip.start_frame + out_point,
                    thumbnail_path=str(clip.thumbnail_path) if clip.thumbnail_path else None,
                )
                duration_frames = out_point - in_point
                current_frame += duration_frames
                self.clip_added.emit(clip, source)

            # Build (Clip, Source) list for timeline preview
            preview_clips = [(clip, source) for clip, source, _, _ in sequence_data]
            self.timeline_preview.set_clips(preview_clips, self._sources)
            self.timeline._on_zoom_fit()

            self.algorithm_dropdown.blockSignals(True)
            display_label = "Signature Style"
            if self.algorithm_dropdown.findText(display_label) == -1:
                self.algorithm_dropdown.addItem(display_label)
            self.algorithm_dropdown.setCurrentText(display_label)
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "signature_style"

            sequence = self.timeline.get_sequence()
            sequence.algorithm = "signature_style"
            sequence.allow_repeats = True
            self._apply_chromatic_bar_to_sequence("signature_style")
            self._update_chromatic_bar_controls("signature_style")
            self._emit_chromatic_bar_setting_changed()

            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_data)} clips from Signature Style")

        except Exception as e:
            logger.error(f"Error applying Signature Style sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

        finally:
            self._algorithm_running = False

    def _show_rose_hobart_dialog(self, clips: list):
        """Show the Rose Hobart dialog for face-filter sequencing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        clip_objects = [clip for clip, source in clips]
        sources_by_id = {source.id: source for clip, source in clips}

        dialog = RoseHobartDialog(
            clips=clip_objects,
            sources_by_id=sources_by_id,
            parent=self,
        )

        dialog.sequence_ready.connect(self._apply_rose_hobart_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_rose_hobart_sequence(self, sequence_clips: list):
        """Apply the sequence from Rose Hobart dialog."""
        self._apply_dialog_sequence(
            sequence_clips, "rose_hobart", "Rose Hobart",
        )

    def _show_staccato_dialog(self, clips: list):
        """Show the Staccato dialog for beat-driven sequencing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        from ui.dialogs.staccato_dialog import StaccatoDialog

        dialog = StaccatoDialog(clips=clips, parent=self)
        dialog.sequence_ready.connect(
            lambda seq_clips: self._apply_staccato_sequence(seq_clips, dialog.music_path)
        )
        dialog.exec()

    def _apply_staccato_sequence(self, sequence_clips: list, music_path=None):
        """Apply the sequence from Staccato dialog.

        Staccato emits (Clip, Source, slot_duration_seconds) tuples.
        Each clip is trimmed to fit its beat slot duration so the
        total sequence matches the music track length.
        """
        if not sequence_clips:
            return

        try:
            self._create_and_activate_sequence("staccato")
            self.timeline.clear_timeline()

            first_clip, first_source = sequence_clips[0][0], sequence_clips[0][1]
            fps = first_source.fps
            self.timeline.set_fps(fps)
            self.video_player.load_video(first_source.file_path)

            # Accumulate timeline position in seconds to avoid per-slot
            # rounding error (int truncation over 50+ slots adds up).
            current_time = 0.0
            logger.info(
                "Staccato: placing %d clips at timeline fps=%.3f",
                len(sequence_clips), fps,
            )
            for entry in sequence_clips:
                clip, source = entry[0], entry[1]
                slot_duration = entry[2] if len(entry) > 2 else None

                if slot_duration is not None:
                    # Trim clip to fit the beat slot duration
                    slot_frames_src = round(slot_duration * source.fps)
                    clip_frames = clip.end_frame - clip.start_frame
                    trimmed_frames = min(slot_frames_src, clip_frames)
                    in_point = clip.start_frame
                    out_point = clip.start_frame + trimmed_frames
                else:
                    in_point = clip.start_frame
                    out_point = clip.end_frame
                    slot_duration = (out_point - in_point) / source.fps

                current_frame = round(current_time * fps)
                self.timeline.add_clip(
                    clip, source, track_index=0, start_frame=current_frame,
                    in_point=in_point, out_point=out_point,
                )
                self.clip_added.emit(clip, source)
                current_time += slot_duration

            logger.info(
                "Staccato: total timeline duration=%.2fs (%d clips placed)",
                current_time, len(sequence_clips),
            )

            # Convert (clip, source, duration) back to (clip, source) for preview
            preview_clips = [(entry[0], entry[1]) for entry in sequence_clips]
            self.timeline_preview.set_clips(preview_clips, self._sources)
            self.timeline._on_zoom_fit()

            self.algorithm_dropdown.blockSignals(True)
            if self.algorithm_dropdown.findText("Staccato") == -1:
                self.algorithm_dropdown.addItem("Staccato")
            self.algorithm_dropdown.setCurrentText("Staccato")
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "staccato"

            sequence = self.timeline.get_sequence()
            sequence.algorithm = "staccato"
            if music_path:
                sequence.music_path = music_path
            self._apply_chromatic_bar_to_sequence("staccato")
            self._update_chromatic_bar_controls("staccato")

            self._set_state(self.STATE_TIMELINE)
            self._emit_chromatic_bar_setting_changed()

        except Exception as e:
            logger.error(f"Failed to apply Staccato sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence:\n{e}")

        finally:
            self._algorithm_running = False

    def _show_eyes_without_a_face_dialog(self, clips: list):
        """Show the Eyes Without a Face dialog for gaze-based sequencing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        from ui.dialogs.eyes_without_a_face_dialog import EyesWithoutAFaceDialog

        dialog = EyesWithoutAFaceDialog(clips=clips, parent=self)
        dialog.sequence_ready.connect(self._apply_eyes_without_a_face_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_eyes_without_a_face_sequence(self, sequence_clips: list):
        """Apply the sequence from Eyes Without a Face dialog."""
        self._apply_dialog_sequence(
            sequence_clips, "eyes_without_a_face", "Eyes Without a Face",
        )

    def _show_dice_roll_dialog(self, clips: list):
        """Show the Dice Roll dialog for shuffle + optional transforms.

        Args:
            clips: List of (Clip, Source) tuples to shuffle
        """
        dialog = DiceRollDialog(clips=clips, parent=self)
        dialog.sequence_ready.connect(self._apply_dice_roll_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_dice_roll_sequence(self, sequence_data: list):
        """Apply the sequence from Dice Roll dialog.

        Args:
            sequence_data: List of (Clip, Source, dict) where dict has
                hflip, vflip, reverse, prerendered_path keys.
        """
        if not sequence_data:
            logger.warning("No clips in Dice Roll sequence")
            return

        try:
            self._create_and_activate_sequence("shuffle")
            self.timeline.clear_timeline()

            first_clip, first_source, _ = sequence_data[0]
            self.timeline.set_fps(first_source.fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source, transform_info in sequence_data:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Set transform flags and prerendered_path on sequence clips
            sequence = self.timeline.get_sequence()
            if sequence.tracks and sequence.tracks[0].clips:
                for seq_clip, (_, _, transform_info) in zip(
                    sequence.tracks[0].clips, sequence_data
                ):
                    seq_clip.hflip = transform_info.get("hflip", False)
                    seq_clip.vflip = transform_info.get("vflip", False)
                    seq_clip.reverse = transform_info.get("reverse", False)
                    seq_clip.prerendered_path = transform_info.get("prerendered_path")

            # Build (Clip, Source) list for timeline preview
            preview_clips = [(clip, source) for clip, source, _ in sequence_data]
            self.timeline_preview.set_clips(preview_clips, self._sources)
            self.timeline._on_zoom_fit()

            self.algorithm_dropdown.blockSignals(True)
            self.algorithm_dropdown.setCurrentText(get_algorithm_label("shuffle"))
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "shuffle"

            sequence.algorithm = "shuffle"
            self._apply_chromatic_bar_to_sequence("shuffle")
            self._update_chromatic_bar_controls("shuffle")
            self._emit_chromatic_bar_setting_changed()

            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_data)} clips from Dice Roll")

            # Notify that clip metadata may have been mutated
            self.clips_data_changed.emit([clip for clip, _, _ in sequence_data])

        except Exception as e:
            logger.error(f"Error applying Dice Roll sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

        finally:
            self._algorithm_running = False

    # Direction options per algorithm: list of (display_label, internal_key).
    # First entry is the default. Algorithms not listed have no direction.
    _DIRECTION_OPTIONS: dict[str, list[tuple[str, str]]] = {
        "duration": [("Shortest First", "short_first"), ("Longest First", "long_first")],
        "color": [("Rainbow", "rainbow"), ("Warm to Cool", "warm_to_cool"), ("Cool to Warm", "cool_to_warm"), ("Complementary", "complementary")],
        "brightness": [("Bright to Dark", "bright_to_dark"), ("Dark to Bright", "dark_to_bright")],
        "volume": [("Quiet to Loud", "quiet_to_loud"), ("Loud to Quiet", "loud_to_quiet")],
        "proximity": [("Far to Close", "far_to_close"), ("Close to Far", "close_to_far")],
        "gaze_sort": [
            ("Left to Right", "left_to_right"),
            ("Right to Left", "right_to_left"),
            ("Up to Down", "up_to_down"),
            ("Down to Up", "down_to_up"),
        ],
    }

    def _update_direction_dropdown(self, algorithm: str, selected_direction: str | None = None):
        """Update direction dropdown options based on selected algorithm.

        Args:
            algorithm: Algorithm key/name
            selected_direction: Optional internal direction key to select
                (e.g., "warm_to_cool"). If omitted or unknown, defaults to first option.
        """
        options = self._DIRECTION_OPTIONS.get(algorithm.lower())

        self.direction_dropdown.blockSignals(True)
        self.direction_dropdown.clear()

        if options:
            labels = [label for label, _ in options]
            self.direction_dropdown.addItems(labels)

            # Keep dropdown UI in sync with active direction used for generation.
            selected_index = 0
            if selected_direction:
                for i, (_label, key) in enumerate(options):
                    if key == selected_direction:
                        selected_index = i
                        break
            self.direction_dropdown.setCurrentIndex(selected_index)

            self.direction_label.show()
            self.direction_dropdown.show()
        else:
            self.direction_label.hide()
            self.direction_dropdown.hide()

        self.direction_dropdown.blockSignals(False)

    def _get_current_direction(self) -> str | None:
        """Get the current direction based on dropdown selection."""
        algo_key = get_algorithm_key(self.algorithm_dropdown.currentText())
        options = self._DIRECTION_OPTIONS.get(algo_key)
        if not options:
            return None

        direction_text = self.direction_dropdown.currentText()
        for label, key in options:
            if label == direction_text:
                return key
        return options[0][1]  # Default to first option

    @Slot(str)
    def _on_direction_changed(self, direction_text: str):
        """Handle direction dropdown change - regenerate with new direction."""
        if self._current_state != self.STATE_TIMELINE:
            return

        # Regenerate with current algorithm and new direction
        algo_key = get_algorithm_key(self.algorithm_dropdown.currentText())
        self._regenerate_sequence(algo_key)

    @Slot(str)
    def _on_algorithm_changed(self, label: str):
        """Handle algorithm dropdown change - regenerate in place."""
        if self._current_state != self.STATE_TIMELINE:
            return

        algo_key = get_algorithm_key(label)

        # Update direction dropdown for this algorithm
        self._update_direction_dropdown(algo_key)
        self._update_chromatic_bar_controls(algo_key)
        self._emit_chromatic_bar_setting_changed()

        # Regenerate sequence
        self._regenerate_sequence(algo_key)

    @Slot(bool)
    def _on_chromatic_bar_toggled(self, checked: bool):
        """Handle in-timeline chromatic bar toggle changes."""
        self.set_chromatic_color_bar_enabled(checked, emit_signal=True)
        self._apply_chromatic_bar_to_sequence(get_algorithm_key(self.algorithm_dropdown.currentText()))
        self.timeline.sequence_changed.emit()

    def _regenerate_sequence(self, algorithm: str):
        """Regenerate the sequence with current timeline clips."""
        # Use clips currently on timeline
        sequence = self.timeline.get_sequence()
        if not sequence.tracks or not sequence.tracks[0].clips:
            return

        # Gather clips from timeline
        clips = []
        for seq_clip in sequence.tracks[0].clips:
            source_clip_id = seq_clip.source_clip_id
            source_id = seq_clip.source_id

            # Look up in available clips
            for clip, source in self._available_clips:
                if clip.id == source_clip_id:
                    clips.append((clip, source))
                    break

        if clips:
            direction = self._get_current_direction()
            self._apply_algorithm(algorithm, clips, direction=direction)

    @Slot()
    def _on_clear_clicked(self):
        """Clear sequence and return to cards."""
        self.timeline.clear_timeline()
        self.timeline_preview.clear()
        self._current_algorithm = None
        self.set_chromatic_color_bar_enabled(False, emit_signal=False)
        self._update_chromatic_bar_controls(None)
        self._emit_chromatic_bar_setting_changed()
        self._set_state(self.STATE_CARDS)

    def clear(self):
        """Clear all state including available clips (called on new project)."""
        self._clips = []
        self._available_clips = []
        self._sources = {}
        self._current_source = None
        self._current_algorithm = None
        self.set_chromatic_color_bar_enabled(False, emit_signal=False)
        self._update_chromatic_bar_controls(None)
        self._emit_chromatic_bar_setting_changed()
        self.timeline.clear_timeline()
        self.timeline_preview.clear()
        self._set_state(self.STATE_CARDS)
        # Reset all cards to enabled for intention flow (fresh project)
        self._reset_card_availability()

    # --- Multi-sequence management ---

    def set_project(self, project):
        """Set project reference and populate sequence dropdown.

        Called by MainWindow after project load/create.
        """
        self._project = project
        self._sequence_dirty = False
        self._sync_sequence_dropdown()

    def _generate_sequence_name(self, algorithm_key: str) -> str:
        """Generate a monotonic auto-name for a new sequence.

        First run of an algorithm uses the display label alone (e.g., "Chromatics").
        Subsequent runs use "{Label} #{N}" where N is max_existing + 1.
        """
        display_label = get_algorithm_label(algorithm_key)
        if not self._project:
            return display_label

        # Scan existing names for the highest N matching "{Label} #{N}"
        max_n = 0
        bare_exists = False
        for seq in self._project.sequences:
            if seq.name == display_label:
                bare_exists = True
            elif seq.name.startswith(f"{display_label} #"):
                try:
                    n = int(seq.name[len(display_label) + 2:])
                    max_n = max(max_n, n)
                except ValueError:
                    pass

        if not bare_exists and max_n == 0:
            return display_label
        return f"{display_label} #{max(max_n + 1, 2)}"

    def _create_and_activate_sequence(
        self, algorithm_key: str, display_label: Optional[str] = None
    ) -> "Sequence":
        """Create a new sequence, append it to the project, activate it, and update the dropdown.

        This is the shared entry point for all apply handlers. After this call,
        the timeline is cleared and ready for the handler to populate with clips.

        Args:
            algorithm_key: Internal algorithm key (e.g., "color", "storyteller")
            display_label: Optional custom name. If None, auto-generated.

        Returns:
            The new Sequence (already set as active on the project).
        """
        from models.sequence import Sequence as SeqModel

        if not self._project:
            return SeqModel()

        # Persist the departing sequence (no dirty prompt — callers handle that)
        self._persist_current_sequence()

        # Generate name
        if display_label is None:
            name = self._generate_sequence_name(algorithm_key)
        else:
            name = display_label

        # Create and add
        new_seq = SeqModel(name=name, algorithm=algorithm_key)
        self._project.add_sequence(new_seq)
        self._project.set_active_sequence(len(self._project.sequences) - 1)

        # Update dropdown
        self._sync_sequence_dropdown()

        # Set guard flag so timeline changes don't trigger dirty
        self._algorithm_running = True
        self._sequence_dirty = False

        return new_seq

    def _sync_sequence_dropdown(self):
        """Rebuild dropdown items from project.sequences. Blocks signals."""
        if not self._project:
            return
        self.sequence_dropdown.blockSignals(True)
        self.sequence_dropdown.clear()
        for seq in self._project.sequences:
            self.sequence_dropdown.addItem(seq.name)
        self.sequence_dropdown.setCurrentIndex(self._project.active_sequence_index)
        self.sequence_dropdown.blockSignals(False)

    def _on_sequence_switched(self, new_index: int):
        """Handle user selecting a different sequence in the dropdown."""
        if not self._project or new_index < 0 or new_index >= len(self._project.sequences):
            return
        if new_index == self._project.active_sequence_index:
            return

        # Dirty check: prompt if user manually edited the timeline
        should_persist = True
        if self._sequence_dirty:
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "The current sequence has unsaved changes.\nSave before switching?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if result == QMessageBox.Cancel:
                # Revert dropdown to current index
                self.sequence_dropdown.blockSignals(True)
                self.sequence_dropdown.setCurrentIndex(self._project.active_sequence_index)
                self.sequence_dropdown.blockSignals(False)
                return
            elif result == QMessageBox.Discard:
                should_persist = False
            # Save: persist below

        if should_persist:
            self._persist_current_sequence()

        # Switch active sequence
        self._project.set_active_sequence(new_index)
        self._sequence_dirty = False

        # Load arriving sequence into timeline
        self._load_active_sequence()

    def _persist_current_sequence(self):
        """Sync timeline state back to the project's active sequence."""
        if not self._project:
            return
        timeline_seq = self.timeline.get_sequence()
        self._project.sequences[self._project.active_sequence_index] = timeline_seq

    def _load_active_sequence(self):
        """Load the project's active sequence into the timeline and update UI controls."""
        if not self._project:
            return
        sequence = self._project.sequence
        if sequence and sequence.get_all_clips():
            sources = self._sources
            self.timeline.load_sequence(
                sequence,
                {s_id: src for s_id, src in sources.items()},
                {c.id: c for c in self._clips},
            )
            self.timeline_preview.set_clips(
                [(c, sources.get(c.source_id)) for track in sequence.tracks for c in track.clips if sources.get(c.source_id)],
                sources,
            )
            self._set_state(self.STATE_TIMELINE)
        else:
            self.timeline.clear_timeline()
            self.timeline_preview.clear()
            self._set_state(self.STATE_CARDS)

        # Sync algorithm controls to the arriving sequence
        self.sync_sequence_metadata(sequence)

    def _on_timeline_user_edit(self):
        """Called when user manually edits the timeline (not from algorithm run)."""
        if not self._algorithm_running:
            self._sequence_dirty = True

    def _on_sequence_context_menu(self, pos):
        """Show context menu on sequence dropdown (Rename, Delete)."""
        # Will be fully implemented in Units 5 and 6
        pass

    def _is_chromatic_flow_algorithm(self, algorithm: Optional[str]) -> bool:
        """Whether the algorithm uses Chromatics (color-based sorting)."""
        return bool(algorithm and algorithm.lower() == "color")

    def set_chromatic_color_bar_enabled(self, enabled: bool, emit_signal: bool = True):
        """Set per-sequence Chromatic Flow color-bar preference."""
        self._show_chromatic_color_bar = bool(enabled)

        self.chromatic_bar_checkbox.blockSignals(True)
        self.chromatic_bar_checkbox.setChecked(self._show_chromatic_color_bar)
        self.chromatic_bar_checkbox.blockSignals(False)

        self._confirm_chromatic_bar_checkbox.blockSignals(True)
        self._confirm_chromatic_bar_checkbox.setChecked(self._show_chromatic_color_bar)
        self._confirm_chromatic_bar_checkbox.blockSignals(False)

        if emit_signal:
            self._emit_chromatic_bar_setting_changed()

    def should_show_chromatic_color_bar(self) -> bool:
        """Whether the preview should currently render the chromatic color bar."""
        if self._current_state != self.STATE_TIMELINE:
            return False
        algorithm = get_algorithm_key(self.algorithm_dropdown.currentText())
        return self._show_chromatic_color_bar and self._is_chromatic_flow_algorithm(algorithm)

    def _update_chromatic_bar_controls(self, algorithm: Optional[str]):
        """Show/hide chromatic bar toggle depending on active algorithm and state."""
        if algorithm is None and self._current_state == self.STATE_TIMELINE:
            algorithm = get_algorithm_key(self.algorithm_dropdown.currentText())
        show_toggle = self._current_state == self.STATE_TIMELINE and self._is_chromatic_flow_algorithm(algorithm)
        self.chromatic_bar_checkbox.setVisible(show_toggle)
        self.chromatic_bar_checkbox.blockSignals(True)
        self.chromatic_bar_checkbox.setChecked(self._show_chromatic_color_bar)
        self.chromatic_bar_checkbox.blockSignals(False)

    def _apply_chromatic_bar_to_sequence(self, algorithm: Optional[str]):
        """Persist chromatic bar state on the current sequence model."""
        sequence = self.timeline.get_sequence()
        show = self._show_chromatic_color_bar and self._is_chromatic_flow_algorithm(algorithm)
        sequence.show_chromatic_color_bar = show

    def _emit_chromatic_bar_setting_changed(self):
        """Emit normalized chromatic bar state for preview listeners."""
        self.chromatic_bar_setting_changed.emit(self.should_show_chromatic_color_bar())

    def sync_sequence_metadata(self, sequence):
        """Sync header controls from a loaded/restored sequence model."""
        algorithm = (sequence.algorithm or "").lower()
        self._current_algorithm = algorithm or None

        if algorithm:
            label = get_algorithm_label(algorithm)
            index = self.algorithm_dropdown.findText(label)
            if index >= 0:
                self.algorithm_dropdown.blockSignals(True)
                self.algorithm_dropdown.setCurrentIndex(index)
                self.algorithm_dropdown.blockSignals(False)

        active_algorithm = algorithm or get_algorithm_key(self.algorithm_dropdown.currentText())
        self._update_direction_dropdown(active_algorithm)

        self.set_chromatic_color_bar_enabled(
            getattr(sequence, "show_chromatic_color_bar", False),
            emit_signal=False,
        )
        self._update_chromatic_bar_controls(active_algorithm)
        self._emit_chromatic_bar_setting_changed()

    def _on_playhead_changed(self, time_seconds: float):
        """Handle playhead position change."""
        pass  # Handled by MainWindow for cross-component coordination

    def _on_playback_requested(self):
        """Handle playback request from VideoPlayer."""
        frame = self.timeline.get_playhead_frame()
        self.playback_requested.emit(frame)

    def _on_stop_requested(self):
        """Handle stop request."""
        self.stop_requested.emit()

    def _on_export_requested(self):
        """Handle export request."""
        self.export_requested.emit()

    def _set_state(self, state: int):
        """Unified state setter - ALWAYS use this, never set index directly."""
        self._current_state = state
        self.state_stack.setCurrentIndex(state)
        state_names = {self.STATE_CARDS: "CARDS", self.STATE_TIMELINE: "TIMELINE", self.STATE_CONFIRM: "CONFIRM"}
        logger.debug(f"Sequence tab state changed to: {state_names.get(state, state)}")

    # --- Public methods for MainWindow to call ---

    def set_source(self, source):
        """Set the current video source."""
        self._current_source = source
        if source:
            self._sources[source.id] = source
            self.video_player.load_video(source.file_path)


    def _has_clips_on_timeline(self) -> bool:
        """Check if there are clips on the timeline."""
        sequence = self.timeline.get_sequence()
        return any(track.clips for track in sequence.tracks)

    def _update_card_availability(self):
        """Update which algorithm cards are available based on clip analysis."""
        if not self._clips:
            return

        # Check if any clips have dominant colors
        has_colors = any(clip.dominant_colors for clip in self._clips)

        # Check if any clips have gaze data
        has_gaze = any(clip.gaze_category is not None for clip in self._clips)

        # Check if any clips have shot types or cinematography
        has_shot_types = any(
            clip.shot_type or clip.cinematography
            for clip in self._clips
        )
        has_embeddings = all(clip.embedding is not None for clip in self._clips)
        has_boundary_embeddings = all(
            clip.first_frame_embedding is not None and clip.last_frame_embedding is not None
            for clip in self._clips
        )
        embeddings_available = any(
            check_feature_ready(name)[0]
            for name in get_operation_feature_candidates("embeddings", load_settings())
        )

        availability = {
            "color": (has_colors, "Run color analysis first" if not has_colors else ""),
            "duration": True,
            "brightness": True,  # Auto-computed on demand
            "volume": True,  # Auto-computed on demand
            "shuffle": True,
            "sequential": True,
            "shot_type": (has_shot_types, "Run shot type analysis first" if not has_shot_types else ""),
            "proximity": (has_shot_types, "Run shot type or cinematography analysis first" if not has_shot_types else ""),
            "similarity_chain": (
                has_embeddings or embeddings_available,
                "Install embeddings dependencies in Settings > Dependencies or run embedding analysis first"
                if not (has_embeddings or embeddings_available)
                else "",
            ),
            "match_cut": (
                has_boundary_embeddings or embeddings_available,
                "Install embeddings dependencies in Settings > Dependencies or run embedding analysis first"
                if not (has_boundary_embeddings or embeddings_available)
                else "",
            ),
            "exquisite_corpus": True,  # Always available - dialog handles text extraction
            "storyteller": True,
            "free_association": True,  # Dialog handles its own prereqs
            "reference_guided": True,  # Dialog handles its own prereqs
            "signature_style": True,  # Dialog handles its own prereqs
            "gaze_sort": (has_gaze, "Run gaze analysis first" if not has_gaze else ""),
            "gaze_consistency": (has_gaze, "Run gaze analysis first" if not has_gaze else ""),
            "eyes_without_a_face": (has_gaze, "Run gaze analysis first" if not has_gaze else ""),
        }

        self.card_grid.set_algorithm_availability(availability)

        # Show/hide gaze filter dropdown based on whether any clips have gaze data
        if has_gaze:
            self.gaze_filter_label.show()
            self.gaze_filter_dropdown.show()
        else:
            self.gaze_filter_label.hide()
            self.gaze_filter_dropdown.hide()

    def _reset_card_availability(self):
        """Reset all cards to enabled state (for fresh project/intention flow)."""
        availability = {key: True for key in ALGORITHM_CONFIG}
        self.card_grid.set_algorithm_availability(availability)

    def add_clip_to_timeline(self, clip, source):
        """Add a clip to the timeline."""
        self.timeline.set_fps(source.fps)
        self.timeline.add_clip(clip, source)
        self.clip_added.emit(clip, source)

        # Ensure we're in timeline state
        self._set_state(self.STATE_TIMELINE)
        self._update_chromatic_bar_controls(self._current_algorithm)
        self._emit_chromatic_bar_setting_changed()

    def get_sequence(self):
        """Get the current sequence from timeline."""
        return self.timeline.get_sequence()

    def get_clip_at_playhead(self):
        """Get clip at current playhead position."""
        return self.timeline.get_clip_at_playhead()

    def set_playhead_time(self, time_seconds: float):
        """Set the playhead position."""
        self.timeline.set_playhead_time(time_seconds)

    def get_playhead_time(self) -> float:
        """Get current playhead time in seconds."""
        return self.timeline.get_playhead_time()

    def seek_video_to(self, time_seconds: float):
        """Seek the video player to a position."""
        self.video_player.seek_to(time_seconds)

    def play_video_range(self, start_seconds: float, end_seconds: float):
        """Play a range in the video player."""
        self.video_player.play_range(start_seconds, end_seconds)

    def load_video(self, file_path):
        """Load a video file into the player."""
        self.video_player.load_video(file_path)

    # --- Agent API methods ---

    def get_sorting_state(self) -> dict:
        """Get current sorting state for agent tools."""
        return {
            "current_state": "cards" if self._current_state == self.STATE_CARDS else "timeline",
            "current_algorithm": self._current_algorithm,
            "available_clip_count": len(self._available_clips),
            "timeline_clip_count": len([
                c for track in self.timeline.sequence.tracks
                for c in track.clips
            ]),
        }

    def set_sorting_algorithm(self, algorithm: str):
        """Set the sorting algorithm (for agent tools)."""
        valid_algorithms = list(ALGORITHM_CONFIG.keys())
        if algorithm.lower() in valid_algorithms:
            # If in timeline state, use the dropdown to regenerate (except dialog-based)
            dialog_algorithms = tuple(k for k, v in ALGORITHM_CONFIG.items() if v.get("is_dialog"))
            if self._current_state == self.STATE_TIMELINE and algorithm.lower() not in dialog_algorithms:
                self.algorithm_dropdown.setCurrentText(get_algorithm_label(algorithm))
            else:
                # If in cards state or dialog algorithm, simulate card click
                self._on_card_clicked(algorithm)

    def apply_shot_type_filter(self, shot_type: str | None) -> int:
        """Filter the current sequence/clips by shot type.

        Args:
            shot_type: Shot type to filter by (e.g. 'close-up'), or None for all.

        Returns:
            Number of clips remaining after filtering.
        """
        if self._current_state == self.STATE_TIMELINE:
            # Filter timeline clips
            seq = self.timeline.sequence
            if not seq or not seq.tracks:
                return 0
            all_seq_clips = seq.get_all_clips()
            if not shot_type:
                return len(all_seq_clips)
            # Count clips matching the filter (timeline clips reference source clips)
            matching = 0
            for sc in all_seq_clips:
                for clip, source in self._available_clips:
                    if clip.id == sc.source_clip_id and clip.shot_type == shot_type:
                        matching += 1
                        break
            return matching
        else:
            # In cards state, filter available clips display
            if not shot_type:
                return len(self._available_clips)
            return sum(
                1 for clip, source in self._available_clips
                if clip.shot_type == shot_type
            )

    def apply_gaze_filter(self, gaze_category: str | None) -> int:
        """Filter the current sequence/clips by gaze category.

        Args:
            gaze_category: Gaze category to filter by (e.g. 'at_camera'),
                or None to show all clips (no filter).

        Returns:
            Number of clips matching the filter.
        """
        if self._current_state == self.STATE_TIMELINE:
            seq = self.timeline.sequence
            if not seq or not seq.tracks:
                return 0
            all_seq_clips = seq.get_all_clips()
            if not gaze_category:
                return len(all_seq_clips)
            matching = 0
            for sc in all_seq_clips:
                for clip, source in self._available_clips:
                    if clip.id == sc.source_clip_id and clip.gaze_category == gaze_category:
                        matching += 1
                        break
            return matching
        else:
            if not gaze_category:
                return len(self._available_clips)
            return sum(
                1 for clip, source in self._available_clips
                if clip.gaze_category == gaze_category
            )

    @Slot(int)
    def _on_gaze_filter_changed(self, index: int):
        """Handle gaze filter dropdown selection change.

        Updates card visibility in the sorting grid to show only clips
        matching the selected gaze category.
        """
        if index < 0 or index >= len(GAZE_FILTER_OPTIONS):
            return
        _display_label, gaze_category = GAZE_FILTER_OPTIONS[index]
        count = self.apply_gaze_filter(gaze_category)
        logger.debug("Gaze filter %r: %d clips match", gaze_category, count)

        # Update card grid visibility when in cards state
        if self._current_state == self.STATE_CARDS and hasattr(self, '_sorting_card_grid'):
            total = len(self._available_clips)
            if gaze_category is None:
                self._sorting_card_grid.setToolTip("")
            else:
                self._sorting_card_grid.setToolTip(
                    f"Gaze filter active: {count}/{total} clips match '{_display_label}'"
                )

    def generate_and_apply(
        self,
        algorithm: str,
        clip_count: int = None,
        direction: str = None,
        seed: int = None,
        no_color_handling: str = None,
        transform_options: dict = None,
    ) -> dict:
        """Generate and apply a sequence (for agent tools).

        Args:
            algorithm: Sorting algorithm to use
            clip_count: Number of clips (unused - uses all selected)
            direction: Sort direction (passed to generate_sequence)
            seed: Random seed (passed to generate_sequence)
            no_color_handling: For color algorithm — "append_end", "exclude", or "sort_inline"
            transform_options: Dict of transform flags, e.g. {"hflip": True}.
                When provided for shuffle, clips are pre-rendered with baked transforms.

        Returns:
            Dict with success status and applied clip info
        """
        # Get selected clips from GUI state
        selected_ids = []
        if self._gui_state:
            selected_ids = (
                self._gui_state.analyze_selected_ids
                or self._gui_state.cut_selected_ids
                or []
            )

        if not selected_ids:
            return {"success": False, "error": "No clips selected in Analyze or Cut tab"}

        clips = self._resolve_selected_clips(selected_ids)

        if not clips:
            return {"success": False, "error": "Selected clips not available for sequencing"}

        try:
            # Generate sequence
            sorted_clips = generate_sequence(
                algorithm=algorithm.lower(),
                clips=clips,
                clip_count=len(clips),
                direction=direction,
                seed=seed,
                no_color_handling=no_color_handling,
            )

            # Create new sequence and apply to timeline
            self._create_and_activate_sequence(algorithm.lower())
            self.timeline.clear_timeline()

            current_frame = 0
            for clip, source in sorted_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames

            # Update preview and dropdown
            self.timeline_preview.set_clips(sorted_clips, self._sources)
            self.algorithm_dropdown.blockSignals(True)
            self.algorithm_dropdown.setCurrentText(get_algorithm_label(algorithm))
            self.algorithm_dropdown.blockSignals(False)
            self._update_direction_dropdown(algorithm, direction)
            self._current_algorithm = algorithm.lower()
            self._apply_chromatic_bar_to_sequence(self._current_algorithm)
            self._update_chromatic_bar_controls(self._current_algorithm)
            self._emit_chromatic_bar_setting_changed()

            # Apply random transforms with pre-rendering if requested
            if transform_options and any(transform_options.values()):
                from core.remix import assign_random_transforms
                from core.remix.prerender import prerender_batch
                sequence = self.timeline.get_sequence()
                if sequence.tracks and sequence.tracks[0].clips:
                    assign_random_transforms(sequence.tracks[0].clips, transform_options, seed=seed)

                    # Pre-render clips with assigned transforms
                    clips_with_transforms = []
                    for seq_clip, (clip, source) in zip(sequence.tracks[0].clips, sorted_clips):
                        clips_with_transforms.append((clip, source, {
                            "hflip": seq_clip.hflip,
                            "vflip": seq_clip.vflip,
                            "reverse": seq_clip.reverse,
                        }))

                    from core.remix.prerender import get_transform_cache_dir
                    output_dir = get_transform_cache_dir()

                    # Keep the UI responsive during pre-rendering.
                    # This runs on the main thread (via gui_tool_requested),
                    # but prerender_batch uses a ThreadPoolExecutor internally.
                    # Process Qt events on progress updates so the UI doesn't freeze.
                    from PySide6.QtWidgets import QApplication
                    def _progress_keep_alive(current, total):
                        app = QApplication.instance()
                        if app:
                            app.processEvents()

                    rendered = prerender_batch(
                        clips_with_transforms=clips_with_transforms,
                        output_dir=output_dir,
                        progress_cb=_progress_keep_alive,
                    )

                    # Set prerendered_path on sequence clips
                    for seq_clip, (_, _, prerendered_path) in zip(
                        sequence.tracks[0].clips, rendered
                    ):
                        seq_clip.prerendered_path = str(prerendered_path) if prerendered_path else None

            self.timeline._on_zoom_fit()

            # Ensure timeline state
            self._set_state(self.STATE_TIMELINE)

            # Notify that clip metadata may have been mutated by auto-compute
            self.clips_data_changed.emit([clip for clip, _ in sorted_clips])

            return {
                "success": True,
                "algorithm": algorithm,
                "clip_count": len(sorted_clips),
                "clips": [
                    {
                        "id": clip.id,
                        "source_id": source.id,
                        "duration": clip.duration_seconds(source.fps),
                    }
                    for clip, source in sorted_clips
                ],
            }

        except Exception as e:
            logger.error(f"Error in generate_and_apply: {e}")
            return {"success": False, "error": str(e)}

        finally:
            self._algorithm_running = False

    def generate_reference_guided(
        self,
        reference_source_id: str,
        weights: dict[str, float],
        allow_repeats: bool = False,
    ) -> dict:
        """Generate and apply a reference-guided sequence (for agent tools).

        Args:
            reference_source_id: Source ID to use as reference
            weights: Dimension weights (e.g. {"brightness": 0.8, "color": 0.5})
            allow_repeats: Allow same clip in multiple positions

        Returns:
            Dict with success status and matched clip info
        """
        from core.remix.reference_match import reference_guided_match

        if not self._available_clips:
            return {"success": False, "error": "No clips available for sequencing"}

        # Split into reference and user pools
        reference_clips = [
            (clip, source) for clip, source in self._available_clips
            if clip.source_id == reference_source_id
        ]
        user_clips = [
            (clip, source) for clip, source in self._available_clips
            if clip.source_id != reference_source_id
        ]

        if not reference_clips:
            return {
                "success": False,
                "error": f"No clips found for reference source '{reference_source_id}'"
            }
        if not user_clips:
            return {
                "success": False,
                "error": "No user clips available (all clips belong to reference source)"
            }

        try:
            matched = reference_guided_match(
                reference_clips=reference_clips,
                user_clips=user_clips,
                weights=weights,
                allow_repeats=allow_repeats,
            )

            if not matched:
                return {
                    "success": False,
                    "error": "No clips could be matched. Try different weights or enable allow_repeats."
                }

            # Create new sequence and apply to timeline
            self._create_and_activate_sequence("reference_guided")
            self.timeline.clear_timeline()

            first_clip, first_source = matched[0]
            self.timeline.set_fps(first_source.fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source in matched:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            self.timeline_preview.set_clips(matched, self._sources)
            self.timeline._on_zoom_fit()

            self.algorithm_dropdown.blockSignals(True)
            label = "Reference Guide"
            if self.algorithm_dropdown.findText(label) == -1:
                self.algorithm_dropdown.addItem(label)
            self.algorithm_dropdown.setCurrentText(label)
            self.algorithm_dropdown.blockSignals(False)
            self._current_algorithm = "reference_guided"

            # Persist metadata on sequence
            sequence = self.timeline.get_sequence()
            sequence.algorithm = "reference_guided"
            sequence.reference_source_id = reference_source_id
            sequence.dimension_weights = weights
            sequence.allow_repeats = allow_repeats
            self._apply_chromatic_bar_to_sequence("reference_guided")
            self._update_chromatic_bar_controls("reference_guided")
            self._emit_chromatic_bar_setting_changed()

            self._set_state(self.STATE_TIMELINE)

            return {
                "success": True,
                "algorithm": "reference_guided",
                "reference_source_id": reference_source_id,
                "clip_count": len(matched),
                "unmatched": len(reference_clips) - len(matched),
                "clips": [
                    {
                        "id": clip.id,
                        "source_id": source.id,
                        "duration": clip.duration_seconds(source.fps),
                    }
                    for clip, source in matched
                ],
            }

        except Exception as e:
            logger.error(f"Error in generate_reference_guided: {e}")
            return {"success": False, "error": str(e)}

        finally:
            self._algorithm_running = False

    def clear_sequence(self) -> dict:
        """Clear the sequence (for agent tools).

        Returns:
            Dict with success status
        """
        self._on_clear_clicked()
        return {"success": True, "message": "Sequence cleared"}

    def apply_intention_workflow_result(
        self,
        algorithm: str,
        clips_with_sources: list,
        direction: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Apply sequence from intention workflow completion.

        Called by MainWindow when the intention workflow finishes processing
        all sources and clips.

        Args:
            algorithm: The sorting algorithm to apply
            clips_with_sources: List of (Clip, Source) tuples from the workflow
            direction: Optional sort direction (e.g., "rainbow" for color)
            seed: Optional random seed for shuffle

        Returns:
            Dict with success status and clip info
        """
        if not clips_with_sources:
            return {"success": False, "error": "No clips from workflow"}

        # Update our available clips
        self._available_clips = clips_with_sources
        self._clips = [clip for clip, source in clips_with_sources]
        for clip, source in clips_with_sources:
            if source:
                self._sources[source.id] = source

        try:
            # Generate sorted sequence
            sorted_clips = generate_sequence(
                algorithm=algorithm.lower(),
                clips=clips_with_sources,
                clip_count=len(clips_with_sources),
                direction=direction,
                seed=seed,
            )

            # Create new sequence and populate timeline
            self._create_and_activate_sequence(algorithm.lower())
            self.timeline.clear_timeline()

            # Set FPS from first source
            if sorted_clips:
                first_clip, first_source = sorted_clips[0]
                self.timeline.set_fps(first_source.fps)
                self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source in sorted_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Update preview
            self.timeline_preview.set_clips(sorted_clips, self._sources)

            # Zoom to fit
            self.timeline._on_zoom_fit()

            # Update dropdowns
            self.algorithm_dropdown.blockSignals(True)
            self.algorithm_dropdown.setCurrentText(get_algorithm_label(algorithm))
            self.algorithm_dropdown.blockSignals(False)

            # Update direction dropdown
            self._update_direction_dropdown(algorithm, direction)

            self._current_algorithm = algorithm.lower()
            self._apply_chromatic_bar_to_sequence(self._current_algorithm)
            self._update_chromatic_bar_controls(self._current_algorithm)
            self._emit_chromatic_bar_setting_changed()

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            # Update card availability for future use
            self._update_card_availability()

            logger.info(f"Applied {len(sorted_clips)} clips from intention workflow with {algorithm}")

            # Notify that clip metadata may have been mutated by auto-compute
            self.clips_data_changed.emit([clip for clip, _ in sorted_clips])

            return {
                "success": True,
                "algorithm": algorithm,
                "clip_count": len(sorted_clips),
            }

        except Exception as e:
            logger.error(f"Error applying intention workflow result: {e}")
            return {"success": False, "error": str(e)}

        finally:
            self._algorithm_running = False

    def on_tab_activated(self):
        """Called when this tab becomes visible."""
        # Update card availability when tab is activated
        if self._clips:
            self._update_card_availability()

        # If we're in the confirm view, refresh estimates (clips may have been
        # analyzed on another tab) but stay in confirm state
        if self.state_stack.currentIndex() == self.STATE_CONFIRM and self._pending_algorithm:
            self._refresh_confirm_estimates()
            return

        # Determine correct state based on timeline content
        if self._has_clips_on_timeline():
            self._set_state(self.STATE_TIMELINE)
        else:
            self._set_state(self.STATE_CARDS)
        self._update_chromatic_bar_controls(self._current_algorithm)
        self._emit_chromatic_bar_setting_changed()

    def on_tab_deactivated(self):
        """Called when switching away from this tab."""
        pass
