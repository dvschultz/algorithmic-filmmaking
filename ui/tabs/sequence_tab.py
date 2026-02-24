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
    QMessageBox,
)
from PySide6.QtCore import Signal, Qt, Slot

from .base_tab import BaseTab
from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.widgets import SortingCardGrid, TimelinePreview, CostEstimatePanel
from ui.dialogs import ExquisiteCorpusDialog, StorytellerDialog, MissingDescriptionsDialog, ReferenceGuideDialog
from ui.theme import theme, Spacing, TypeScale
from ui.workers.sequence_worker import SequenceWorker
from core.remix import generate_sequence
from core.cost_estimates import estimate_sequence_cost
from core.settings import get_llm_api_key, get_replicate_api_key

logger = logging.getLogger(__name__)

from ui.algorithm_config import ALGORITHM_CONFIG, get_algorithm_config, get_algorithm_label

# Reverse lookup: display label -> algorithm key
_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in ALGORITHM_CONFIG.items()}


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

        # Guard flags
        self._apply_in_progress = False
        self._sequence_worker: Optional[SequenceWorker] = None

        # Confirm state: pending algorithm and clips for generation
        self._pending_algorithm: Optional[str] = None
        self._pending_clips: list = []

        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Sequence tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # State stack for cards vs timeline
        self.state_stack = QStackedWidget()

        # STATE_CARDS (index 0): Just the card grid
        self.card_grid = SortingCardGrid()
        self.card_grid.algorithm_selected.connect(self._on_card_clicked)
        self.state_stack.addWidget(self.card_grid)

        # STATE_TIMELINE (index 1): Header + content
        self.timeline_view = self._create_timeline_view()
        self.state_stack.addWidget(self.timeline_view)

        # STATE_CONFIRM (index 2): Cost estimate + generate/back buttons
        self.confirm_view = self._create_confirm_view()
        self.state_stack.addWidget(self.confirm_view)

        layout.addWidget(self.state_stack)

        # Start in cards state
        self._set_state(self.STATE_CARDS)

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
        splitter = QSplitter(Qt.Vertical)

        # Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        # Timeline preview strip (moved from parameter view)
        self.timeline_preview = TimelinePreview()
        self.timeline_preview.setMaximumHeight(100)
        splitter.addWidget(self.timeline_preview)

        # Timeline widget
        self.timeline = TimelineWidget()
        self.timeline.playhead_changed.connect(self._on_playhead_changed)
        self.timeline.playback_requested.connect(self._on_playback_requested)
        self.timeline.stop_requested.connect(self._on_stop_requested)
        self.timeline.export_requested.connect(self._on_export_requested)
        splitter.addWidget(self.timeline)

        # Set splitter sizes
        splitter.setSizes([300, 80, 200])

        layout.addWidget(splitter)

        return container

    def _create_header(self) -> QWidget:
        """Create the header row with algorithm dropdown and clear button."""
        header = QWidget()
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {theme().background_secondary};
                border-bottom: 1px solid {theme().border_primary};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)

        label = QLabel("Algorithm:")
        label.setStyleSheet(f"color: {theme().text_secondary}; border: none;")
        layout.addWidget(label)

        self.algorithm_dropdown = QComboBox()
        # Populate with labels from non-dialog algorithms (exclude exquisite_corpus, storyteller)
        _dropdown_keys = [
            "shuffle", "sequential", "duration", "color", "color_cycle",
            "brightness", "volume", "shot_type", "proximity",
            "similarity_chain", "match_cut",
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

        # Initially hide direction controls
        self.direction_label.hide()
        self.direction_dropdown.hide()

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
        self._update_cloud_api_warning(estimates)
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

        # Check for missing API keys when cloud tier is selected
        self._update_cloud_api_warning(estimates)

    def _update_cloud_api_warning(self, estimates: list):
        """Show a warning if cloud tier is selected but API key is missing."""
        if not estimates:
            self._confirm_cost_panel.set_warning(None)
            return

        cloud_ops = [e for e in estimates if e.tier == "cloud"]
        if not cloud_ops:
            self._confirm_cost_panel.set_warning(None)
            return

        missing = []
        needs_llm = any(e.operation in ("describe", "extract_text", "cinematography") for e in cloud_ops)
        needs_replicate = any(e.operation == "shots" for e in cloud_ops)

        if needs_llm and not get_llm_api_key():
            missing.append("LLM")
        if needs_replicate and not get_replicate_api_key():
            missing.append("Replicate")

        if missing:
            keys = " and ".join(missing)
            self._confirm_cost_panel.set_warning(
                f"Cloud tier selected but no {keys} API key configured. "
                f"Set keys in Settings or switch to Local tier."
            )
        else:
            self._confirm_cost_panel.set_warning(None)

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

        # Dialog-based algorithms still use their dialogs for the actual generation
        if algorithm == "exquisite_corpus":
            self._show_exquisite_corpus_dialog(clips)
            return
        if algorithm == "storyteller":
            self._show_storyteller_dialog(clips)
            return

        self._apply_algorithm(algorithm, clips)

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

    # --- Signal handlers ---

    @Slot(str)
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

        # Get clip objects from our clips_by_id lookup
        clips = []
        for cid in selected_ids:
            # Look up in available clips
            for clip, source in self._available_clips:
                if clip.id == cid:
                    clips.append((clip, source))
                    break

        if not clips:
            QMessageBox.warning(
                self,
                "Clips Not Found",
                "Selected clips are not available in the Sequence tab. "
                "The clips may be from a source that hasn't been loaded."
            )
            return

        # Compute cost estimates for this algorithm
        clip_objects = [clip for clip, source in clips]
        estimates = estimate_sequence_cost(algorithm, clip_objects)

        if estimates:
            # Show gatekeeper with cost panel
            self._pending_algorithm = algorithm
            self._pending_clips = clips
            self._show_confirm_view(algorithm, clips, estimates)
        else:
            # No analysis needed — skip directly to generation
            # Special handling for dialog-based algorithms
            if algorithm == "exquisite_corpus":
                self._show_exquisite_corpus_dialog(clips)
                return
            if algorithm == "storyteller":
                self._show_storyteller_dialog(clips)
                return
            if algorithm == "reference_guided":
                self._show_reference_guide_dialog(clips)
                return
            self._apply_algorithm(algorithm, clips)

    def _apply_algorithm(self, algorithm: str, clips: list, direction: str = None):
        """Generate sequence in a background worker and transition to timeline.

        Heavy auto-compute operations (brightness, volume, CLIP embeddings)
        run off the main thread so the UI stays responsive.

        Args:
            algorithm: Algorithm name
            clips: List of (Clip, Source) tuples
            direction: Sort direction (e.g., "short_first", "long_first" for duration)
        """
        if self._apply_in_progress:
            logger.warning("Apply already in progress, ignoring")
            return
        self._apply_in_progress = True

        algo_lower = algorithm.lower()

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
            parent=self,
        )
        # Store algorithm for the completion slot
        worker._pending_algorithm = algorithm
        worker.sequence_ready.connect(self._on_sequence_ready)
        worker.error.connect(self._on_sequence_error)
        self._sequence_worker = worker
        worker.start()

    def _on_sequence_ready(self, sorted_clips: list):
        """Handle completed sequence generation (runs on main thread)."""
        worker = self._sequence_worker
        algorithm = getattr(worker, "_pending_algorithm", "") if worker else ""
        algo_lower = algorithm.lower()

        try:
            # Clear and populate timeline
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
            self._update_direction_dropdown(algorithm)

            self._current_algorithm = algo_lower

            # Persist algorithm on the sequence for SRT export
            sequence = self.timeline.get_sequence()
            sequence.algorithm = algo_lower

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sorted_clips)} clips with {algorithm} algorithm")

            # Notify that clip metadata may have been mutated by auto-compute
            self.clips_data_changed.emit([clip for clip, _ in sorted_clips])

        except Exception as e:
            logger.error(f"Error populating timeline: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate sequence: {e}")

        finally:
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

    @Slot(list)
    def _apply_exquisite_corpus_sequence(self, sequence_clips: list):
        """Apply the sequence from Exquisite Corpus dialog.

        Args:
            sequence_clips: List of (Clip, Source) tuples in poem order
        """
        if not sequence_clips:
            logger.warning("No clips in Exquisite Corpus sequence")
            return

        try:
            # Clear and populate timeline
            self.timeline.clear_timeline()

            # Set FPS from first source
            first_clip, first_source = sequence_clips[0]
            self.timeline.set_fps(first_source.fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source in sequence_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Update preview
            self.timeline_preview.set_clips(sequence_clips, self._sources)

            # Zoom to fit
            self.timeline._on_zoom_fit()

            # Update dropdown - Exquisite Corpus isn't in dropdown, so set to empty/custom
            self.algorithm_dropdown.blockSignals(True)
            # Add Exquisite Corpus to dropdown if not present
            if self.algorithm_dropdown.findText("Exquisite Corpus") == -1:
                self.algorithm_dropdown.addItem("Exquisite Corpus")
            self.algorithm_dropdown.setCurrentText("Exquisite Corpus")
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "exquisite_corpus"

            # Persist algorithm on the sequence for SRT export
            sequence = self.timeline.get_sequence()
            sequence.algorithm = "exquisite_corpus"

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_clips)} clips from Exquisite Corpus")

        except Exception as e:
            logger.error(f"Error applying Exquisite Corpus sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

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
        """Apply the sequence from Storyteller dialog.

        Args:
            sequence_clips: List of (Clip, Source) tuples in narrative order
        """
        if not sequence_clips:
            logger.warning("No clips in Storyteller sequence")
            return

        try:
            # Clear and populate timeline
            self.timeline.clear_timeline()

            # Set FPS from first source
            first_clip, first_source = sequence_clips[0]
            self.timeline.set_fps(first_source.fps)
            self.video_player.load_video(first_source.file_path)

            current_frame = 0
            for clip, source in sequence_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Update preview
            self.timeline_preview.set_clips(sequence_clips, self._sources)

            # Zoom to fit
            self.timeline._on_zoom_fit()

            # Update dropdown
            self.algorithm_dropdown.blockSignals(True)
            if self.algorithm_dropdown.findText("Storyteller") == -1:
                self.algorithm_dropdown.addItem("Storyteller")
            self.algorithm_dropdown.setCurrentText("Storyteller")
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "storyteller"

            # Persist algorithm on the sequence for SRT export
            sequence = self.timeline.get_sequence()
            sequence.algorithm = "storyteller"

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_clips)} clips from Storyteller")

        except Exception as e:
            logger.error(f"Error applying Storyteller sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

    def _show_reference_guide_dialog(self, clips: list):
        """Show the Reference Guide dialog for reference-guided remixing.

        Args:
            clips: List of (Clip, Source) tuples to process
        """
        sources_by_id = {source.id: source for clip, source in clips}

        dialog = ReferenceGuideDialog(
            clips=clips,
            sources_by_id=sources_by_id,
            project=None,
            parent=self,
        )

        dialog.sequence_ready.connect(self._apply_reference_guide_sequence)
        dialog.exec()

    @Slot(list)
    def _apply_reference_guide_sequence(self, sequence_clips: list):
        """Apply the sequence from Reference Guide dialog.

        Args:
            sequence_clips: List of (Clip, Source) tuples in matched order
        """
        if not sequence_clips:
            logger.warning("No clips in Reference Guide sequence")
            return

        try:
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
            label = "Reference Guide"
            if self.algorithm_dropdown.findText(label) == -1:
                self.algorithm_dropdown.addItem(label)
            self.algorithm_dropdown.setCurrentText(label)
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = "reference_guided"

            # Persist algorithm and reference metadata on the sequence
            sequence = self.timeline.get_sequence()
            sequence.algorithm = "reference_guided"

            # Store dialog config if available (for save/load round-trip)
            dialog = self.sender()
            if dialog and hasattr(dialog, '_last_ref_source_id'):
                sequence.reference_source_id = dialog._last_ref_source_id
                sequence.dimension_weights = dialog._last_weights
                sequence.allow_repeats = dialog._last_allow_repeats
                sequence.match_reference_timing = dialog._last_match_timing

            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sequence_clips)} clips from Reference Guide")

        except Exception as e:
            logger.error(f"Error applying Reference Guide sequence: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply sequence: {e}")

    # Direction options per algorithm: list of (display_label, internal_key).
    # First entry is the default. Algorithms not listed have no direction.
    _DIRECTION_OPTIONS: dict[str, list[tuple[str, str]]] = {
        "duration": [("Shortest First", "short_first"), ("Longest First", "long_first")],
        "color": [("Rainbow", "rainbow"), ("Warm to Cool", "warm_to_cool"), ("Cool to Warm", "cool_to_warm")],
        "brightness": [("Bright to Dark", "bright_to_dark"), ("Dark to Bright", "dark_to_bright")],
        "volume": [("Quiet to Loud", "quiet_to_loud"), ("Loud to Quiet", "loud_to_quiet")],
        "proximity": [("Far to Close", "far_to_close"), ("Close to Far", "close_to_far")],
        "color_cycle": [("Spectrum", "spectrum"), ("Complementary", "complementary")],
    }

    def _update_direction_dropdown(self, algorithm: str):
        """Update direction dropdown options based on selected algorithm."""
        options = self._DIRECTION_OPTIONS.get(algorithm.lower())

        self.direction_dropdown.blockSignals(True)
        self.direction_dropdown.clear()

        if options:
            self.direction_dropdown.addItems([label for label, _ in options])
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

        # Regenerate sequence
        self._regenerate_sequence(algo_key)

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
        self._set_state(self.STATE_CARDS)

    def clear(self):
        """Clear all state including available clips (called on new project)."""
        self._clips = []
        self._available_clips = []
        self._sources = {}
        self._current_source = None
        self._current_algorithm = None
        self.timeline.clear_timeline()
        self.timeline_preview.clear()
        self._set_state(self.STATE_CARDS)
        # Reset all cards to enabled for intention flow (fresh project)
        self._reset_card_availability()

    def _on_playhead_changed(self, time_seconds: float):
        """Handle playhead position change."""
        pass  # Handled by MainWindow for cross-component coordination

    def _on_playback_requested(self, start_frame: int):
        """Handle playback request."""
        self.playback_requested.emit(start_frame)

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

    def set_clips_available(self, clips, source):
        """Set available clips for the timeline."""
        self._clips = clips
        self._current_source = source

        if source:
            self._sources[source.id] = source

        if clips:
            # Build (Clip, Source) tuples
            self._available_clips = [(clip, source) for clip in clips]

            self.timeline.set_fps(source.fps)
            self.timeline.set_available_clips(clips, source)

            # Update card availability based on clip analysis
            self._update_card_availability()

            # Ensure video player has the source loaded
            self.video_player.load_video(source.file_path)

            # Determine state based on timeline content
            if self._has_clips_on_timeline():
                self._set_state(self.STATE_TIMELINE)
            else:
                self._set_state(self.STATE_CARDS)
        else:
            self._available_clips = []
            self._set_state(self.STATE_CARDS)

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

        # Check if any clips have shot types or cinematography
        has_shot_types = any(
            clip.shot_type or clip.cinematography
            for clip in self._clips
        )

        availability = {
            "color": (has_colors, "Run color analysis first" if not has_colors else ""),
            "color_cycle": (has_colors, "Run color analysis first" if not has_colors else ""),
            "duration": True,
            "brightness": True,  # Auto-computed on demand
            "volume": True,  # Auto-computed on demand
            "shuffle": True,
            "sequential": True,
            "shot_type": (has_shot_types, "Run shot type analysis first" if not has_shot_types else ""),
            "proximity": (has_shot_types, "Run shot type or cinematography analysis first" if not has_shot_types else ""),
            "similarity_chain": True,  # Auto-computed on demand
            "match_cut": True,  # Auto-computed on demand
            "exquisite_corpus": True,  # Always available - dialog handles text extraction
            "storyteller": True,
        }

        self.card_grid.set_algorithm_availability(availability)

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

    def set_playing(self, is_playing: bool):
        """Update UI for playing state."""
        self.timeline.set_playing(is_playing)

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
            dialog_algorithms = ("exquisite_corpus", "storyteller")
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

    def generate_and_apply(
        self,
        algorithm: str,
        clip_count: int = None,
        direction: str = None,
        seed: int = None
    ) -> dict:
        """Generate and apply a sequence (for agent tools).

        Args:
            algorithm: Sorting algorithm to use
            clip_count: Number of clips (unused - uses all selected)
            direction: Sort direction (passed to generate_sequence)
            seed: Random seed (passed to generate_sequence)

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

        # Get clip objects
        clips = []
        for cid in selected_ids:
            for clip, source in self._available_clips:
                if clip.id == cid:
                    clips.append((clip, source))
                    break

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
            )

            # Clear and apply to timeline
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
            self._current_algorithm = algorithm.lower()

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

            # Clear and populate timeline
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
            self._update_direction_dropdown(algorithm)

            self._current_algorithm = algorithm.lower()

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

    def on_tab_deactivated(self):
        """Called when switching away from this tab."""
        pass
