"""Analyze tab for clip analysis features."""

import logging

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QWidget,
    QStackedWidget,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser
from ui.video_player import VideoPlayer
from ui.widgets import EmptyStateWidget

logger = logging.getLogger(__name__)


class AnalyzeTab(BaseTab):
    """Tab for analyzing clips sent from the Cut tab.

    Signals:
        analyze_colors_requested: Emitted when color extraction is requested
        analyze_shots_requested: Emitted when shot type classification is requested
        transcribe_requested: Emitted when transcription is requested
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
        clip_dragged_to_timeline: Emitted when a clip is dragged to timeline (clip: Clip)
        clips_cleared: Emitted when all clips are cleared
    """

    analyze_colors_requested = Signal()
    analyze_shots_requested = Signal()
    transcribe_requested = Signal()
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip
    clips_cleared = Signal()

    # State constants for stacked widget
    STATE_NO_CLIPS = 0
    STATE_CLIPS = 1

    def __init__(self, parent=None):
        # Clip ID references (not copies) - resolved via MainWindow.clips_by_id
        self._clip_ids: set[str] = set()
        # Source lookup for video preview (set by MainWindow)
        self._sources_by_id: dict = {}
        self._clips_by_id: dict = {}
        super().__init__(parent)

    def _setup_ui(self):
        """Set up the Analyze tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top controls bar
        controls = self._create_controls()
        layout.addLayout(controls)

        # Stacked widget for different states
        self.state_stack = QStackedWidget()

        # State 0: No clips sent yet
        self.no_clips_widget = EmptyStateWidget(
            "No Clips to Analyze",
            "Select clips in the Cut tab and click 'Analyze Selected'"
        )
        self.state_stack.addWidget(self.no_clips_widget)

        # State 1: Clips present - main content
        self.content_widget = self._create_content_area()
        self.state_stack.addWidget(self.content_widget)

        layout.addWidget(self.state_stack)

        # Start with no clips state
        self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)

    def _create_controls(self) -> QHBoxLayout:
        """Create the top controls bar."""
        controls = QHBoxLayout()
        controls.setContentsMargins(10, 10, 10, 10)

        # Extract Colors button
        self.colors_btn = QPushButton("Extract Colors")
        self.colors_btn.setToolTip("Extract dominant colors from clips")
        self.colors_btn.setEnabled(False)
        self.colors_btn.clicked.connect(self._on_colors_click)
        controls.addWidget(self.colors_btn)

        # Classify Shots button
        self.shots_btn = QPushButton("Classify Shots")
        self.shots_btn.setToolTip("Classify shot types (close-up, wide, etc.)")
        self.shots_btn.setEnabled(False)
        self.shots_btn.clicked.connect(self._on_shots_click)
        controls.addWidget(self.shots_btn)

        # Transcribe button
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setToolTip("Transcribe speech in clips")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self._on_transcribe_click)
        controls.addWidget(self.transcribe_btn)

        controls.addSpacing(20)

        # Clear All button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setToolTip("Remove all clips from analysis")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._on_clear_click)
        controls.addWidget(self.clear_btn)

        controls.addStretch()

        # Clip count label
        self.clip_count_label = QLabel("")
        controls.addWidget(self.clip_count_label)

        return controls

    def _create_content_area(self) -> QWidget:
        """Create the main content area with clip browser and video player."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal splitter for browser and player
        splitter = QSplitter(Qt.Horizontal)

        # Left: Clip browser
        self.clip_browser = ClipBrowser()
        self.clip_browser.set_drag_enabled(True)
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged)
        splitter.addWidget(self.clip_browser)

        # Right: Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        return content

    def _on_colors_click(self):
        """Handle extract colors button click."""
        self.analyze_colors_requested.emit()

    def _on_shots_click(self):
        """Handle classify shots button click."""
        self.analyze_shots_requested.emit()

    def _on_transcribe_click(self):
        """Handle transcribe button click."""
        self.transcribe_requested.emit()

    def _on_clear_click(self):
        """Handle clear all button click."""
        self.clear_clips()
        self.clips_cleared.emit()

    def _on_clip_selected(self, clip):
        """Handle clip selection."""
        self.clip_selected.emit(clip)
        # Load the source video for this clip
        source = self._sources_by_id.get(clip.source_id)
        if source:
            self.video_player.load_video(source.file_path)
            start_time = clip.start_time(source.fps)
            self.video_player.seek_to(start_time)

    def _on_clip_double_clicked(self, clip):
        """Handle clip double-click."""
        self.clip_double_clicked.emit(clip)
        source = self._sources_by_id.get(clip.source_id)
        if source:
            self.video_player.load_video(source.file_path)
            start_time = clip.start_time(source.fps)
            end_time = clip.end_time(source.fps)
            self.video_player.play_range(start_time, end_time)

    def _on_clip_dragged(self, clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    def _update_ui_state(self):
        """Update UI based on current clip count."""
        count = len(self._clip_ids)
        has_clips = count > 0

        self.colors_btn.setEnabled(has_clips)
        self.shots_btn.setEnabled(has_clips)
        self.transcribe_btn.setEnabled(has_clips)
        self.clear_btn.setEnabled(has_clips)

        if has_clips:
            self.clip_count_label.setText(f"{count} clips")
            self.state_stack.setCurrentIndex(self.STATE_CLIPS)
        else:
            self.clip_count_label.setText("")
            self.state_stack.setCurrentIndex(self.STATE_NO_CLIPS)

    # Public methods for MainWindow to call

    def set_lookups(self, clips_by_id: dict, sources_by_id: dict):
        """Set the lookup dictionaries for resolving clip/source references.

        Args:
            clips_by_id: dict mapping clip_id to Clip object
            sources_by_id: dict mapping source_id to Source object
        """
        self._clips_by_id = clips_by_id
        self._sources_by_id = sources_by_id

    def add_clips(self, clip_ids: list[str]):
        """Add clips to the analysis tab (merge with deduplication).

        Args:
            clip_ids: List of clip IDs to add
        """
        added_count = 0
        for clip_id in clip_ids:
            if clip_id not in self._clip_ids:
                self._clip_ids.add(clip_id)
                # Add to ClipBrowser if clip exists
                clip = self._clips_by_id.get(clip_id)
                source = self._sources_by_id.get(clip.source_id) if clip else None
                if clip and source:
                    self.clip_browser.add_clip(clip, source)
                    added_count += 1
                else:
                    logger.warning(f"Could not add clip {clip_id}: clip or source not found")
                    self._clip_ids.discard(clip_id)

        if added_count > 0:
            logger.info(f"Added {added_count} clips to Analyze tab")
        self._update_ui_state()

    def remove_clip(self, clip_id: str):
        """Remove a single clip from the analysis tab.

        Args:
            clip_id: ID of clip to remove
        """
        if clip_id in self._clip_ids:
            self._clip_ids.discard(clip_id)
            # Remove from ClipBrowser
            clip = self._clips_by_id.get(clip_id)
            if clip:
                self.clip_browser.remove_clips_for_source(clip.source_id)
                # Re-add remaining clips for that source
                for cid in self._clip_ids:
                    c = self._clips_by_id.get(cid)
                    if c and c.source_id == clip.source_id:
                        source = self._sources_by_id.get(c.source_id)
                        if source:
                            self.clip_browser.add_clip(c, source)
        self._update_ui_state()

    def clear_clips(self):
        """Remove all clips from the analysis tab."""
        self._clip_ids.clear()
        self.clip_browser.clear()
        self._update_ui_state()

    def remove_orphaned_clips(self, valid_clip_ids: set[str]) -> int:
        """Remove clips that no longer exist in the main clip collection.

        Args:
            valid_clip_ids: Set of clip IDs that still exist

        Returns:
            Number of clips removed
        """
        orphaned = self._clip_ids - valid_clip_ids
        if orphaned:
            logger.warning(f"Removing {len(orphaned)} orphaned clips from Analyze tab")
            for clip_id in orphaned:
                self._clip_ids.discard(clip_id)
            # Rebuild ClipBrowser
            self.clip_browser.clear()
            for clip_id in self._clip_ids:
                clip = self._clips_by_id.get(clip_id)
                source = self._sources_by_id.get(clip.source_id) if clip else None
                if clip and source:
                    self.clip_browser.add_clip(clip, source)
            self._update_ui_state()
            return len(orphaned)
        return 0

    def get_clip_ids(self) -> list[str]:
        """Get list of clip IDs in the analysis tab.

        Returns:
            List of clip IDs
        """
        return list(self._clip_ids)

    def get_clips(self) -> list:
        """Get list of Clip objects in the analysis tab.

        Returns:
            List of Clip objects (resolved from IDs)
        """
        return [self._clips_by_id[cid] for cid in self._clip_ids
                if cid in self._clips_by_id]

    def update_clip_colors(self, clip_id: str, colors: list):
        """Update colors for a clip."""
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_colors(clip_id, colors)

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update shot type for a clip."""
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_shot_type(clip_id, shot_type)

    def update_clip_thumbnail(self, clip_id: str, thumb_path):
        """Update thumbnail for a clip."""
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_thumbnail(clip_id, thumb_path)

    def update_clip_transcript(self, clip_id: str, segments: list):
        """Update transcript for a clip."""
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_transcript(clip_id, segments)

    def set_analyzing(self, is_analyzing: bool, operation: str = ""):
        """Update UI state during analysis operations.

        Args:
            is_analyzing: Whether an analysis operation is running
            operation: Which operation is running ("colors", "shots", "transcribe")
        """
        # Disable all buttons during any analysis
        self.colors_btn.setEnabled(not is_analyzing and len(self._clip_ids) > 0)
        self.shots_btn.setEnabled(not is_analyzing and len(self._clip_ids) > 0)
        self.transcribe_btn.setEnabled(not is_analyzing and len(self._clip_ids) > 0)
        self.clear_btn.setEnabled(not is_analyzing and len(self._clip_ids) > 0)

        # Update button text based on operation
        if is_analyzing:
            if operation == "colors":
                self.colors_btn.setText("Extracting...")
            elif operation == "shots":
                self.shots_btn.setText("Classifying...")
            elif operation == "transcribe":
                self.transcribe_btn.setText("Transcribing...")
        else:
            self.colors_btn.setText("Extract Colors")
            self.shots_btn.setText("Classify Shots")
            self.transcribe_btn.setText("Transcribe")
