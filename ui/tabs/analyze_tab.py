"""Analyze tab for clip analysis features."""

import logging

from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QStackedWidget,
    QComboBox,
)
from PySide6.QtCore import Signal, Qt

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser
from ui.widgets import EmptyStateWidget
from ui.dialogs import GlossaryDialog
from ui.theme import UISizes
from core.analysis_operations import ANALYSIS_OPERATIONS

logger = logging.getLogger(__name__)


class AnalyzeTab(BaseTab):
    """Tab for analyzing clips sent from the Cut tab.

    Signals:
        quick_run_requested: Emitted for single-operation immediate run (op_key: str)
        analyze_picker_requested: Emitted to open the analysis picker modal
        clip_selected: Emitted when a clip is selected (clip: Clip)
        clip_double_clicked: Emitted when a clip is double-clicked (clip: Clip)
        clip_dragged_to_timeline: Emitted when a clip is dragged to timeline (clip: Clip)
        clips_cleared: Emitted when all clips are cleared
    """

    quick_run_requested = Signal(str)  # operation key
    analyze_picker_requested = Signal()
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip
    clips_cleared = Signal()
    selection_changed = Signal(list)  # list[str] selected clip IDs
    clips_changed = Signal(list)  # list[str] current clip IDs in tab

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

        # Quick Run dropdown + button
        self.quick_run_combo = QComboBox()
        self.quick_run_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.quick_run_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        for op in ANALYSIS_OPERATIONS:
            self.quick_run_combo.addItem(op.label, op.key)
            # Set tooltip on each item via the combo's model
            idx = self.quick_run_combo.count() - 1
            self.quick_run_combo.setItemData(idx, op.tooltip, Qt.ToolTipRole)
        self.quick_run_combo.setEnabled(False)
        controls.addWidget(self.quick_run_combo)

        self.quick_run_btn = QPushButton("Run")
        self.quick_run_btn.setToolTip("Run the selected operation immediately")
        self.quick_run_btn.setEnabled(False)
        self.quick_run_btn.clicked.connect(self._on_quick_run_click)
        controls.addWidget(self.quick_run_btn)

        controls.addSpacing(10)

        # Analyze... button (opens picker modal)
        self.analyze_btn = QPushButton("Analyze...")
        self.analyze_btn.setToolTip("Choose multiple operations to run")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze_click)
        controls.addWidget(self.analyze_btn)

        controls.addSpacing(20)

        # Clear All button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setToolTip("Remove all clips from analysis")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._on_clear_click)
        controls.addWidget(self.clear_btn)

        controls.addStretch()

        # Glossary button
        self.glossary_btn = QPushButton("?")
        self.glossary_btn.setToolTip("Open Film Language Glossary")
        self.glossary_btn.setFixedWidth(32)
        self.glossary_btn.setAccessibleName("Film Glossary")
        self.glossary_btn.clicked.connect(self._on_glossary_click)
        controls.addWidget(self.glossary_btn)

        controls.addSpacing(8)

        # Clip count label
        self.clip_count_label = QLabel("")
        controls.addWidget(self.clip_count_label)

        return controls

    def _create_content_area(self) -> QWidget:
        """Create the main content area with clip browser and video player."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Clip browser (full width - video preview is in clip details sidebar)
        self.clip_browser = ClipBrowser()
        self.clip_browser.set_drag_enabled(True)
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged)
        self.clip_browser.filters_changed.connect(self._on_filters_changed)
        layout.addWidget(self.clip_browser)

        return content

    def _on_quick_run_click(self):
        """Handle quick run button click - immediate single operation."""
        op_key = self.quick_run_combo.currentData()
        if op_key:
            self.quick_run_requested.emit(op_key)

    def _on_analyze_click(self):
        """Handle analyze button click - open picker modal."""
        self.analyze_picker_requested.emit()

    def _on_glossary_click(self):
        """Handle glossary button click - open the film language glossary dialog."""
        dialog = GlossaryDialog(self)
        dialog.exec()

    def _on_clear_click(self):
        """Handle clear all button click."""
        self.clear_clips()
        self.clips_cleared.emit()

    def _on_clip_selected(self, clip):
        """Handle clip selection."""
        self.clip_selected.emit(clip)
        self._update_selection_ui()

    def _update_selection_ui(self):
        """Update selection state and notify parent."""
        selected = self.clip_browser.get_selected_clips()
        self.selection_changed.emit([c.id for c in selected])

    def _on_clip_double_clicked(self, clip):
        """Handle clip double-click."""
        self.clip_double_clicked.emit(clip)

    def _on_clip_dragged(self, clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    def _on_filters_changed(self):
        """Handle filter changes - clear selection and update clip count."""
        # Clear selection
        self.clip_browser.selected_clips.clear()
        for thumb in self.clip_browser.thumbnails:
            thumb.set_selected(False)
        self._update_selection_ui()

        # Update clip count label
        visible = self.clip_browser.get_visible_clip_count()
        total = len(self.clip_browser.thumbnails)
        if self.clip_browser.has_active_filters():
            self.clip_count_label.setText(f"{visible}/{total} clips")
        else:
            self.clip_count_label.setText(f"{total} clips")

    def _update_ui_state(self):
        """Update UI based on current clip count."""
        count = len(self._clip_ids)
        has_clips = count > 0

        self.quick_run_combo.setEnabled(has_clips)
        self.quick_run_btn.setEnabled(has_clips)
        self.analyze_btn.setEnabled(has_clips)
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
        self.clips_changed.emit(self.get_clip_ids())

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
        self.clips_changed.emit(self.get_clip_ids())

    def clear_clips(self):
        """Remove all clips from the analysis tab."""
        self._clip_ids.clear()
        self.clip_browser.clear()
        # Clear lookup dictionaries (will be re-set on next project load)
        self._clips_by_id = {}
        self._sources_by_id = {}
        self._update_ui_state()
        self.clips_changed.emit([])

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
            self.clips_changed.emit(self.get_clip_ids())
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

    def update_clip_extracted_text(self, clip_id: str, texts: list):
        """Update extracted text for a clip."""
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_extracted_text(clip_id, texts)

    def update_clip_cinematography(self, clip_id: str, cinematography):
        """Update cinematography analysis for a clip.

        Args:
            clip_id: ID of the clip to update
            cinematography: CinematographyAnalysis object
        """
        if clip_id in self._clip_ids:
            self.clip_browser.update_clip_cinematography(clip_id, cinematography)

    def set_analyzing(self, is_analyzing: bool, operation: str = ""):
        """Update UI state during analysis operations.

        Args:
            is_analyzing: Whether an analysis operation is running
            operation: Which operation is running (operation key or "pipeline")
        """
        has_clips = len(self._clip_ids) > 0
        self.quick_run_combo.setEnabled(not is_analyzing and has_clips)
        self.quick_run_btn.setEnabled(not is_analyzing and has_clips)
        self.analyze_btn.setEnabled(not is_analyzing and has_clips)
        self.clear_btn.setEnabled(not is_analyzing and has_clips)

        if is_analyzing:
            self.analyze_btn.setText("Analyzing...")
        else:
            self.analyze_btn.setText("Analyze...")

    def get_active_filters(self) -> dict:
        """Get the current filter state from the clip browser.

        Returns:
            Dict with filter names and their current values
        """
        return self.clip_browser.get_active_filters()

    def has_active_filters(self) -> bool:
        """Check if any filters are currently active.

        Returns:
            True if at least one filter is set
        """
        return self.clip_browser.has_active_filters()
