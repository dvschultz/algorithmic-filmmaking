"""Sequence tab for timeline editing and playback with card-based sorting."""

import logging
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
from ui.widgets import SortingCardGrid, TimelinePreview
from ui.theme import theme
from core.remix import generate_sequence

logger = logging.getLogger(__name__)


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

    # State constants (2 states instead of 3)
    STATE_CARDS = 0      # Show card grid only
    STATE_TIMELINE = 1   # Show header + timeline + preview

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
        self.algorithm_dropdown.addItems(["Color", "Duration", "Shuffle", "Sequential"])
        self.algorithm_dropdown.setMinimumWidth(120)
        self.algorithm_dropdown.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_dropdown)

        layout.addStretch()

        self.clear_btn = QPushButton("Clear Sequence")
        self.clear_btn.setToolTip("Clear timeline and return to card selection")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self.clear_btn)

        return header

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

    # --- Signal handlers ---

    @Slot(str)
    def _on_card_clicked(self, algorithm: str):
        """Handle card click - generate sequence from selected clips."""
        logger.debug(f"Card clicked: {algorithm}")

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

        # Generate and apply
        self._apply_algorithm(algorithm, clips)

    def _apply_algorithm(self, algorithm: str, clips: list):
        """Generate sequence and transition to timeline state.

        Args:
            algorithm: Algorithm name (Color, Duration, Shuffle, Sequential)
            clips: List of (Clip, Source) tuples
        """
        if self._apply_in_progress:
            logger.warning("Apply already in progress, ignoring")
            return
        self._apply_in_progress = True

        try:
            # Normalize algorithm name
            algo_lower = algorithm.lower()

            # Generate sorted clips
            sorted_clips = generate_sequence(
                algorithm=algo_lower,
                clips=clips,
                clip_count=len(clips),
            )

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

            # Update dropdown to show current algorithm (block signals to avoid recursion)
            self.algorithm_dropdown.blockSignals(True)
            self.algorithm_dropdown.setCurrentText(algorithm.capitalize())
            self.algorithm_dropdown.blockSignals(False)

            self._current_algorithm = algo_lower

            # Transition to timeline state
            self._set_state(self.STATE_TIMELINE)

            logger.info(f"Applied {len(sorted_clips)} clips with {algorithm} algorithm")

        except Exception as e:
            logger.error(f"Error applying algorithm: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate sequence: {e}")

        finally:
            self._apply_in_progress = False

    @Slot(str)
    def _on_algorithm_changed(self, algorithm: str):
        """Handle algorithm dropdown change - regenerate in place."""
        if self._current_state != self.STATE_TIMELINE:
            return

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
            self._apply_algorithm(algorithm, clips)

    @Slot()
    def _on_clear_clicked(self):
        """Clear sequence and return to cards."""
        self.timeline.clear_timeline()
        self.timeline_preview.clear()
        self._current_algorithm = None
        self._set_state(self.STATE_CARDS)

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
        logger.debug(f"Sequence tab state changed to: {'CARDS' if state == self.STATE_CARDS else 'TIMELINE'}")

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

        availability = {
            "color": (has_colors, "Run color analysis first" if not has_colors else ""),
            "duration": True,
            "shuffle": True,
            "sequential": True,
        }

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
        if algorithm.lower() in ["color", "duration", "shuffle", "sequential"]:
            # If in timeline state, use the dropdown to regenerate
            if self._current_state == self.STATE_TIMELINE:
                self.algorithm_dropdown.setCurrentText(algorithm.capitalize())
            else:
                # If in cards state, simulate card click
                self._on_card_clicked(algorithm)

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
            self.algorithm_dropdown.setCurrentText(algorithm.capitalize())
            self.algorithm_dropdown.blockSignals(False)
            self._current_algorithm = algorithm.lower()

            self.timeline._on_zoom_fit()

            # Ensure timeline state
            self._set_state(self.STATE_TIMELINE)

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

    def on_tab_activated(self):
        """Called when this tab becomes visible."""
        # Update card availability when tab is activated
        if self._clips:
            self._update_card_availability()

        # Determine correct state based on timeline content
        if self._has_clips_on_timeline():
            self._set_state(self.STATE_TIMELINE)
        else:
            self._set_state(self.STATE_CARDS)

    def on_tab_deactivated(self):
        """Called when switching away from this tab."""
        pass
