"""Main timeline widget container."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QStyle,
)
from PySide6.QtCore import Qt, Signal

from models.sequence import Sequence, SequenceClip
from models.clip import Clip, Source
from ui.timeline.timeline_scene import TimelineScene
from ui.timeline.timeline_view import TimelineView
from ui.timeline.playhead import Playhead
from ui.theme import theme
from core.remix import generate_sequence


class TimelineWidget(QWidget):
    """Main timeline container with toolbar and view."""

    # Signals
    playhead_changed = Signal(float)  # time in seconds
    clip_selected = Signal(str)  # clip_id
    sequence_changed = Signal()  # sequence was modified
    export_requested = Signal()  # request to export sequence
    playback_requested = Signal(int)  # start_frame
    stop_requested = Signal()  # request to stop playback

    def __init__(self, parent=None):
        super().__init__(parent)

        self._source_lookup = {}  # source_id -> Source
        self._clip_lookup = {}  # clip_id -> (Clip, Source)
        self._available_clips = []  # (Clip, Source) tuples for remixing
        self._playhead = None

        self._setup_ui()
        self._connect_signals()

    @property
    def sequence(self) -> Sequence:
        """Get the sequence from the scene."""
        return self.scene.sequence

    def _setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addLayout(toolbar)

        # Timeline view
        self.scene = TimelineScene()
        self.view = TimelineView()
        self.view.setScene(self.scene)

        layout.addWidget(self.view)

        # Create playhead
        self._create_playhead()

    def _create_toolbar(self) -> QHBoxLayout:
        """Create the timeline toolbar."""
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)

        # Timeline label
        label = QLabel("Timeline")
        label.setStyleSheet(f"font-weight: bold; color: {theme().text_muted};")
        toolbar.addWidget(label)

        toolbar.addSpacing(16)

        # Play button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setToolTip("Play sequence")
        self.play_btn.setFixedSize(32, 28)
        toolbar.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setToolTip("Stop playback")
        self.stop_btn.setFixedSize(32, 28)
        toolbar.addWidget(self.stop_btn)

        toolbar.addStretch()

        # Remix algorithm selector
        toolbar.addWidget(QLabel("Remix:"))
        self.remix_combo = QComboBox()
        self.remix_combo.addItems([
            "Shuffle",
            "Sequential",
            "Color (HSV)",
            "Shot Type",
            "Duration (Long)",
            "Duration (Short)",
        ])
        self.remix_combo.setToolTip("Algorithm for generating sequences")
        self.remix_combo.setMinimumWidth(120)
        toolbar.addWidget(self.remix_combo)

        # Clip count spinner
        toolbar.addWidget(QLabel("Clips:"))
        self.clip_count_spin = QSpinBox()
        self.clip_count_spin.setRange(1, 100)
        self.clip_count_spin.setValue(10)
        self.clip_count_spin.setToolTip("Number of clips to include")
        toolbar.addWidget(self.clip_count_spin)

        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setToolTip("Generate a remix sequence")
        toolbar.addWidget(self.generate_btn)

        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear all clips from timeline")
        toolbar.addWidget(self.clear_btn)

        toolbar.addStretch()

        # Add track button
        self.add_track_btn = QPushButton("+ Track")
        self.add_track_btn.setToolTip("Add a new video track")
        toolbar.addWidget(self.add_track_btn)

        # Zoom controls
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Zoom to fit sequence")
        toolbar.addWidget(self.zoom_fit_btn)

        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_reset_btn.setToolTip("Reset zoom to 100%")
        toolbar.addWidget(self.zoom_reset_btn)

        # Export sequence button
        self.export_btn = QPushButton("Export Sequence")
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)

        return toolbar

    def _create_playhead(self):
        """Create the playhead after scene is set up."""
        height = self.scene.RULER_HEIGHT + len(self.sequence.tracks) * (
            self.scene.TRACK_HEIGHT + self.scene.TRACK_SPACING
        )
        self._playhead = Playhead(self.scene, height, self.sequence.fps)

    def _connect_signals(self):
        """Connect internal signals."""
        # Scene signals
        self.scene.playhead_moved.connect(self._on_playhead_moved)
        self.scene.clip_selected.connect(self._on_clip_selected)

        # View signals
        self.view.position_clicked.connect(self._on_ruler_clicked)

        # Button signals
        self.clear_btn.clicked.connect(self.clear_timeline)
        self.add_track_btn.clicked.connect(self._on_add_track)
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self.zoom_reset_btn.clicked.connect(self.view.reset_zoom)
        self.generate_btn.clicked.connect(self._on_generate)
        self.export_btn.clicked.connect(lambda: self.export_requested.emit())
        self.play_btn.clicked.connect(self._on_play_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

    def _on_playhead_moved(self, time_seconds: float):
        """Handle playhead position change."""
        self.playhead_changed.emit(time_seconds)

    def _on_clip_selected(self, clip_id: str):
        """Handle clip selection."""
        self.clip_selected.emit(clip_id)

    def _on_ruler_clicked(self, time_seconds: float):
        """Handle click on ruler to move playhead."""
        if self._playhead:
            self._playhead.set_time(time_seconds)

    def _on_play_clicked(self):
        """Request playback from current playhead position."""
        frame = self._playhead.get_frame() if self._playhead else 0
        self.playback_requested.emit(frame)

    def _on_stop_clicked(self):
        """Request playback stop."""
        self.stop_requested.emit()

    def set_playing(self, playing: bool):
        """Update UI to reflect playback state."""
        if playing:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_btn.setToolTip("Pause sequence")
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_btn.setToolTip("Play sequence")

    def _on_add_track(self):
        """Add a new track."""
        self.sequence.add_track()
        self._rebuild_scene()

    def _on_zoom_fit(self):
        """Zoom to fit sequence duration."""
        duration = self.sequence.duration_seconds
        if duration > 0:
            self.view.set_zoom_to_fit(duration)
        else:
            self.view.reset_zoom()

    # Map display names to algorithm keys
    ALGORITHM_MAP = {
        "Shuffle": "shuffle",
        "Sequential": "sequential",
        "Color (HSV)": "color",
        "Shot Type": "shot_type",
        "Duration (Long)": "duration_long",
        "Duration (Short)": "duration_short",
    }

    def _on_generate(self):
        """Handle Generate button click."""
        if not self._available_clips:
            return

        display_name = self.remix_combo.currentText()
        algorithm = self.ALGORITHM_MAP.get(display_name, "sequential")
        clip_count = self.clip_count_spin.value()

        # Clear existing timeline
        self.clear_timeline()

        # Generate sequence using core algorithm
        sequenced = generate_sequence(algorithm, self._available_clips, clip_count)

        # Add to timeline sequentially
        current_frame = 0
        for clip, source in sequenced:
            self.add_clip(clip, source, track_index=0, start_frame=current_frame)
            current_frame += clip.duration_frames

        # Zoom to fit
        self._on_zoom_fit()
        self.sequence_changed.emit()

    def _rebuild_scene(self):
        """Rebuild scene from sequence data."""
        self.scene.set_sequence(self.sequence)
        self._update_playhead_height()
        self._update_export_button()

    def _update_playhead_height(self):
        """Update playhead line height based on track count."""
        if self._playhead:
            height = self.scene.RULER_HEIGHT + len(self.sequence.tracks) * (
                self.scene.TRACK_HEIGHT + self.scene.TRACK_SPACING
            )
            self._playhead.set_height(height)

    def _update_export_button(self):
        """Enable/disable export button based on content."""
        has_clips = any(track.clips for track in self.sequence.tracks)
        self.export_btn.setEnabled(has_clips)

    # --- Public API ---

    def set_fps(self, fps: float):
        """Set the timeline frame rate."""
        self.sequence.fps = fps
        if self._playhead:
            self._playhead.set_fps(fps)

    def add_clip(
        self,
        clip: Clip,
        source: Source,
        track_index: int = 0,
        start_frame: int = None,
    ):
        """
        Add a clip to the timeline.

        Args:
            clip: The source Clip object
            source: The Source video containing the clip
            track_index: Which track to add to (default 0)
            start_frame: Where to place on timeline (default: end of track)
        """
        # Store reference
        self._source_lookup[source.id] = source
        self._clip_lookup[clip.id] = (clip, source)

        # Calculate start position if not specified
        if start_frame is None:
            # Place at end of existing clips on this track
            if track_index < len(self.sequence.tracks):
                track = self.sequence.tracks[track_index]
                if track.clips:
                    start_frame = max(c.end_frame() for c in track.clips)
                else:
                    start_frame = 0
            else:
                start_frame = 0

        # Get thumbnail path
        thumb_path = str(clip.thumbnail_path) if clip.thumbnail_path else None

        # Add to scene
        self.scene.add_clip_to_track(
            track_index=track_index,
            source_clip_id=clip.id,
            source_id=source.id,
            start_frame=start_frame,
            in_point=clip.start_frame,
            out_point=clip.end_frame,
            thumbnail_path=thumb_path,
        )

        self._update_export_button()
        self.sequence_changed.emit()

    def add_clip_at_position(
        self,
        clip: Clip,
        source: Source,
        x_position: float,
        track_index: int = 0,
    ):
        """Add a clip at a specific x position (from drag-drop)."""
        start_frame = self.scene.get_frame_at_x(x_position)
        self.add_clip(clip, source, track_index, start_frame)

    def clear_timeline(self):
        """Remove all clips from the timeline."""
        self.scene.clear_all_clips()
        self._update_export_button()
        self.sequence_changed.emit()

    def get_sequence(self) -> Sequence:
        """Get the current sequence."""
        return self.sequence

    def clear(self):
        """Clear all clips and reset lookups."""
        self.scene.clear_all_clips()
        self._source_lookup.clear()
        self._clip_lookup.clear()
        self._available_clips.clear()
        self._update_export_button()
        self.sequence_changed.emit()

    def load_sequence(self, sequence: Sequence, sources: dict, clips: list = None):
        """Load a saved sequence.

        Args:
            sequence: The sequence to load
            sources: Dict mapping source_id to Source objects
            clips: List of Clip objects to populate _clip_lookup for playback
        """
        # Register all sources
        for source_id, source in sources.items():
            self._source_lookup[source_id] = source

        # Populate _clip_lookup from provided clips with their correct sources
        if clips:
            for clip in clips:
                clip_source = sources.get(clip.source_id)
                if clip_source:
                    self._clip_lookup[clip.id] = (clip, clip_source)

        # Set the sequence on the scene (this rebuilds all visuals)
        self.scene.set_sequence(sequence)

        self._update_export_button()
        self.sequence_changed.emit()

    def set_playhead_time(self, time_seconds: float):
        """Set playhead position."""
        if self._playhead:
            self._playhead.set_time(time_seconds)

    def get_playhead_time(self) -> float:
        """Get current playhead time."""
        if self._playhead:
            return self._playhead.get_time()
        return 0.0

    def get_clip_at_playhead(self) -> tuple:
        """
        Get the clip under the playhead.

        Returns:
            Tuple of (SequenceClip, Clip, Source) or (None, None, None)
        """
        frame = self._playhead.get_frame() if self._playhead else 0

        # Check each track
        for track in self.sequence.tracks:
            seq_clip = track.get_clip_at_frame(frame)
            if seq_clip:
                clip_data = self._clip_lookup.get(seq_clip.source_clip_id)
                if clip_data:
                    clip, source = clip_data
                    return (seq_clip, clip, source)

        return (None, None, None)

    def set_available_clips(self, clips: list[Clip], source: Source):
        """Set the available clips for remix generation."""
        self._available_clips = [(clip, source) for clip in clips]
        # Update clip count max
        self.clip_count_spin.setMaximum(len(clips))
        if self.clip_count_spin.value() > len(clips):
            self.clip_count_spin.setValue(len(clips))
