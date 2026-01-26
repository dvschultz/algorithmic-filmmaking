"""Sequence tab for timeline editing and playback with card-based sorting."""

import logging
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer, Slot

from .base_tab import BaseTab
from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget
from ui.widgets import (
    EmptyStateWidget,
    SortingCardGrid,
    SortingParameterPanel,
    TimelinePreview,
)
from core.remix import generate_sequence

logger = logging.getLogger(__name__)


class SequenceTab(BaseTab):
    """Tab for arranging clips on the timeline and previewing.

    Uses a card-based UI for selecting sorting algorithms:
    - STATE_NO_CLIPS: Shows empty state when no clips available
    - STATE_CARD_SELECTION: Shows grid of sorting algorithm cards
    - STATE_PARAMETER_VIEW: Shows parameter panel with live preview

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

    # State constants
    STATE_NO_CLIPS = 0
    STATE_CARD_SELECTION = 1
    STATE_PARAMETER_VIEW = 2

    # Preview debounce delay in milliseconds
    PREVIEW_DEBOUNCE_MS = 300

    def __init__(self, parent=None):
        self._current_source = None
        self._clips = []  # List of Clip objects
        self._sources = {}  # source_id -> Source
        self._available_clips = []  # List of (Clip, Source) tuples
        self._preview_clips = []  # Current preview sequence
        self._current_algorithm = None
        self._current_state = self.STATE_NO_CLIPS

        # Guard flags to prevent duplicate signal handling
        self._apply_in_progress = False
        self._preview_update_pending = False

        super().__init__(parent)

        # Debounce timer for preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview_debounced)

    def _setup_ui(self):
        """Set up the Sequence tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main content splitter (top: sorting UI, bottom: timeline/player)
        self.main_splitter = QSplitter(Qt.Vertical)

        # Top section: Stacked widget for different states
        self.state_stack = QStackedWidget()

        # State 0: No clips available
        self.no_clips_widget = EmptyStateWidget(
            "No Clips Available",
            "Detect scenes in the Analyze tab first, then return here to create sequences"
        )
        self.state_stack.addWidget(self.no_clips_widget)

        # State 1: Card selection grid
        self.card_grid = SortingCardGrid()
        self.card_grid.algorithm_selected.connect(self._on_algorithm_selected)
        self.state_stack.addWidget(self.card_grid)

        # State 2: Parameter view with preview
        self.parameter_view = self._create_parameter_view()
        self.state_stack.addWidget(self.parameter_view)

        self.main_splitter.addWidget(self.state_stack)

        # Bottom section: Timeline content (video player + timeline)
        self.content_widget = self._create_content_area()
        self.main_splitter.addWidget(self.content_widget)

        # Set initial splitter sizes
        self.main_splitter.setSizes([400, 300])

        layout.addWidget(self.main_splitter)

        # Start with no clips state
        self._set_state(self.STATE_NO_CLIPS)

    def _create_parameter_view(self) -> QWidget:
        """Create the parameter view with controls and preview."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Left side: Parameter panel
        self.param_panel = SortingParameterPanel()
        self.param_panel.parameters_changed.connect(self._on_parameters_changed)
        self.param_panel.back_clicked.connect(self._on_back_clicked)
        self.param_panel.apply_clicked.connect(self._on_apply_clicked)
        self.param_panel.setMinimumWidth(300)
        self.param_panel.setMaximumWidth(400)
        layout.addWidget(self.param_panel)

        # Right side: Timeline preview
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 20, 20, 20)

        self.timeline_preview = TimelinePreview()
        preview_layout.addWidget(self.timeline_preview)
        preview_layout.addStretch()

        layout.addWidget(preview_container, 1)  # Stretch factor 1

        return container

    def _create_content_area(self) -> QWidget:
        """Create the main content area with video player and timeline."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter for player and timeline
        splitter = QSplitter(Qt.Vertical)

        # Top: Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        # Bottom: Timeline
        self.timeline = TimelineWidget()
        self.timeline.playhead_changed.connect(self._on_playhead_changed)
        self.timeline.playback_requested.connect(self._on_playback_requested)
        self.timeline.stop_requested.connect(self._on_stop_requested)
        self.timeline.export_requested.connect(self._on_export_requested)
        splitter.addWidget(self.timeline)

        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

        return content

    def _set_state(self, state: int):
        """Set the current UI state."""
        self._current_state = state
        self.state_stack.setCurrentIndex(state)

        # Update splitter behavior based on state
        if state == self.STATE_NO_CLIPS:
            # Hide content area when no clips
            self.content_widget.setVisible(False)
        else:
            self.content_widget.setVisible(True)

    # --- Signal handlers ---

    @Slot(str)
    def _on_algorithm_selected(self, algorithm: str):
        """Handle algorithm card selection."""
        logger.debug(f"Algorithm selected: {algorithm}")
        self._current_algorithm = algorithm

        # Configure parameter panel for this algorithm
        available_count = len(self._available_clips)
        self.param_panel.set_algorithm(algorithm, available_count)

        # Clear and switch to parameter view
        self.timeline_preview.clear()
        self._set_state(self.STATE_PARAMETER_VIEW)

        # Trigger initial preview generation
        self._schedule_preview_update()

    @Slot(dict)
    def _on_parameters_changed(self, params: dict):
        """Handle parameter value changes - debounced."""
        self._schedule_preview_update()

    @Slot()
    def _on_back_clicked(self):
        """Handle back button click."""
        self._current_algorithm = None
        self._preview_clips = []
        self.timeline_preview.clear()
        self._set_state(self.STATE_CARD_SELECTION)

    @Slot()
    def _on_apply_clicked(self):
        """Handle apply button click - commit preview to timeline."""
        # Guard against duplicate calls
        if self._apply_in_progress:
            logger.warning("Apply already in progress, ignoring duplicate call")
            return
        self._apply_in_progress = True

        try:
            if not self._preview_clips:
                logger.warning("No preview clips to apply")
                return

            # Clear existing timeline
            self.timeline.clear_timeline()

            # Add clips to timeline
            current_frame = 0
            for clip, source in self._preview_clips:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                self.clip_added.emit(clip, source)

            # Zoom to fit
            self.timeline._on_zoom_fit()

            logger.info(f"Applied {len(self._preview_clips)} clips to timeline")

        finally:
            self._apply_in_progress = False

    def _on_playhead_changed(self, time_seconds: float):
        """Handle playhead position change."""
        # This is handled by MainWindow for cross-component coordination
        pass

    def _on_playback_requested(self, start_frame: int):
        """Handle playback request."""
        self.playback_requested.emit(start_frame)

    def _on_stop_requested(self):
        """Handle stop request."""
        self.stop_requested.emit()

    def _on_export_requested(self):
        """Handle export request."""
        self.export_requested.emit()

    # --- Preview update logic ---

    def _schedule_preview_update(self):
        """Schedule a debounced preview update."""
        self._preview_timer.stop()
        self._preview_timer.start(self.PREVIEW_DEBOUNCE_MS)

    @Slot()
    def _update_preview_debounced(self):
        """Actually update the preview (called after debounce)."""
        if not self._current_algorithm or not self._available_clips:
            return

        # Show loading state
        self.timeline_preview.set_loading(True)

        # Get current parameters
        params = self.param_panel.get_parameters()
        algorithm = params.get("algorithm", self._current_algorithm)
        clip_count = params.get("clip_count", 10)

        # Get algorithm-specific parameters
        direction = params.get("direction")
        seed = params.get("seed", 0)

        # Handle duration algorithm mapping
        if algorithm == "duration":
            direction = params.get("direction", "short_first")
        elif algorithm == "color":
            direction = params.get("direction", "rainbow")

        logger.debug(f"Generating preview: algorithm={algorithm}, count={clip_count}, direction={direction}")

        try:
            # Generate sequence
            self._preview_clips = generate_sequence(
                algorithm=algorithm,
                clips=self._available_clips,
                clip_count=clip_count,
                direction=direction,
                seed=seed if seed > 0 else None,
            )

            # Update preview
            self.timeline_preview.set_clips(self._preview_clips, self._sources)

        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            self.timeline_preview.clear()
            self._preview_clips = []

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

            # Show card selection state
            self._set_state(self.STATE_CARD_SELECTION)
        else:
            self._available_clips = []
            self._set_state(self.STATE_NO_CLIPS)

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
        params = self.param_panel.get_parameters() if self._current_algorithm else {}
        return {
            "current_state": ["no_clips", "card_selection", "parameter_view"][self._current_state],
            "current_algorithm": self._current_algorithm,
            "parameters": params,
            "preview_clip_count": len(self._preview_clips),
            "available_clip_count": len(self._available_clips),
            "timeline_clip_count": len([
                c for track in self.timeline.sequence.tracks
                for c in track.clips
            ]),
        }

    def set_sorting_algorithm(self, algorithm: str):
        """Set the sorting algorithm (for agent tools)."""
        if algorithm in ["color", "duration", "shuffle", "sequential"]:
            self._on_algorithm_selected(algorithm)

    def generate_and_apply(
        self,
        algorithm: str,
        clip_count: int,
        direction: str = None,
        seed: int = None
    ) -> dict:
        """Generate and apply a sequence (for agent tools).

        Returns:
            Dict with success status and applied clip info
        """
        if not self._available_clips:
            return {"success": False, "error": "No clips available"}

        try:
            # Generate sequence
            sequenced = generate_sequence(
                algorithm=algorithm,
                clips=self._available_clips,
                clip_count=clip_count,
                direction=direction,
                seed=seed,
            )

            # Clear and apply to timeline
            self.timeline.clear_timeline()

            current_frame = 0
            for clip, source in sequenced:
                self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames

            self.timeline._on_zoom_fit()

            return {
                "success": True,
                "algorithm": algorithm,
                "clip_count": len(sequenced),
                "clips": [
                    {
                        "id": clip.id,
                        "source_id": source.id,
                        "duration": clip.duration_seconds(source.fps),
                    }
                    for clip, source in sequenced
                ],
            }

        except Exception as e:
            logger.error(f"Error in generate_and_apply: {e}")
            return {"success": False, "error": str(e)}

    def on_tab_activated(self):
        """Called when this tab becomes visible."""
        # Update card availability when tab is activated
        if self._clips:
            self._update_card_availability()

        # Refresh preview if we're in parameter view (may have been interrupted)
        if self._current_state == self.STATE_PARAMETER_VIEW and self._current_algorithm:
            self._schedule_preview_update()

    def on_tab_deactivated(self):
        """Called when switching away from this tab."""
        # Cancel any pending preview updates
        self._preview_timer.stop()
