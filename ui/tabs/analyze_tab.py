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
    QSplitter,
)
from PySide6.QtCore import Signal, Qt, QTimer

from .base_tab import BaseTab
from ui.clip_browser import ClipBrowser, VIRTUALIZATION_THRESHOLD
from ui.widgets import EmptyStateWidget
from ui.widgets.active_filter_chips import ActiveFilterChips
from ui.widgets.filter_sidebar import FilterSidebar
from ui.dialogs import GlossaryDialog
from ui.theme import UISizes
from core.analysis_availability import compute_disabled_operations
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

    def __init__(self, parent=None, filter_state=None):
        # Clip ID references (not copies) - resolved via MainWindow.clips_by_id
        self._clip_ids: set[str] = set()
        # Source lookup for video preview (set by MainWindow)
        self._sources_by_id: dict = {}
        self._clips_by_id: dict = {}
        self._pending_browser_clip_ids: list[str] = []
        self._is_analyzing = False
        self._disabled_quick_ops: set[str] = set()
        if filter_state is None:
            from core.filter_state import FilterState
            filter_state = FilterState()
        self._filter_state = filter_state
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
            "Select clips in the Cut tab and click 'Analyze Selected', or ask the Agent"
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

        # Filter sidebar toggle
        self.filter_toggle_btn = QPushButton("Filters")
        self.filter_toggle_btn.setCheckable(True)
        self.filter_toggle_btn.setChecked(True)
        self.filter_toggle_btn.setToolTip("Show/hide filter sidebar")
        self.filter_toggle_btn.clicked.connect(lambda checked: self.set_filter_sidebar_visible(checked))
        controls.addWidget(self.filter_toggle_btn)

        # Reset filters
        self.reset_filters_btn = QPushButton("Reset filters")
        self.reset_filters_btn.setToolTip("Clear all filter values")
        self.reset_filters_btn.clicked.connect(self._filter_state.clear_all)
        controls.addWidget(self.reset_filters_btn)

        # Active filter chips appear inline next to the buttons so they
        # don't claim a full row of vertical space when filters are on.
        self.active_filter_chips = ActiveFilterChips(self._filter_state)
        controls.addWidget(self.active_filter_chips)

        controls.addSpacing(20)

        controls.addStretch()

        # Selection count label
        self.selection_label = QLabel("")
        controls.addWidget(self.selection_label)

        controls.addSpacing(10)

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
        """Create the main content area with clip browser + filter sidebar."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self.clip_browser = ClipBrowser(filter_state=self._filter_state)
        self.clip_browser.set_drag_enabled(True)
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged)
        self.clip_browser.selection_changed.connect(self._on_browser_selection_changed)
        self.clip_browser.filters_changed.connect(self._on_filters_changed)

        self.filter_sidebar = FilterSidebar(self._filter_state)
        self.filter_sidebar.visibility_requested.connect(self._on_sidebar_visibility_request)

        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.addWidget(self.clip_browser)
        self.content_splitter.addWidget(self.filter_sidebar)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([800, 320])

        layout.addWidget(self.content_splitter)

        return content

    def set_filter_sidebar_visible(self, visible: bool) -> None:
        self.filter_sidebar.setVisible(visible)
        if hasattr(self, "filter_toggle_btn") and self.filter_toggle_btn.isChecked() != visible:
            self.filter_toggle_btn.setChecked(visible)
        self._persist_sidebar_visibility(visible)

    def _on_sidebar_visibility_request(self, visible: bool) -> None:
        self.set_filter_sidebar_visible(visible)

    def _persist_sidebar_visibility(self, visible: bool) -> None:
        """Save Analyze tab sidebar visibility to settings."""
        try:
            from core.settings import load_settings, save_settings
            s = load_settings()
            if s.analyze_filter_sidebar_visible != visible:
                s.analyze_filter_sidebar_visible = visible
                save_settings(s)
        except Exception:
            pass

    def refresh_filter_vocabularies(self) -> None:
        """Recompute YOLO label vocabulary from current clips and push to sidebar."""
        labels: set[str] = set()
        for thumb in self.clip_browser.thumbnails:
            for det in (thumb.clip.detected_objects or []):
                label = det.get("label", "")
                if label:
                    labels.add(label)
        self.filter_sidebar.refresh_yolo_vocabulary(labels)

    def _on_quick_run_click(self):
        """Handle quick run button click - immediate single operation."""
        op_key = self.quick_run_combo.currentData()
        if op_key and op_key not in self._disabled_quick_ops:
            self.quick_run_requested.emit(op_key)

    def _refresh_quick_run_availability(self):
        """Enable/disable quick-run operations based on current clip metadata."""
        clips = self.get_clips()
        op_keys = [op.key for op in ANALYSIS_OPERATIONS]
        self._disabled_quick_ops = compute_disabled_operations(clips, op_keys)

        # Disable individual combo rows for operations that are already complete.
        model = self.quick_run_combo.model()
        for idx, op in enumerate(ANALYSIS_OPERATIONS):
            item = model.item(idx)
            if item is None:
                continue
            disabled = op.key in self._disabled_quick_ops
            item.setEnabled(not disabled)
            tooltip = op.tooltip
            if disabled:
                tooltip = f"{tooltip}\n\nAlready analyzed for all clips in this scope."
            self.quick_run_combo.setItemData(idx, tooltip, Qt.ToolTipRole)

        has_clips = len(self._clip_ids) > 0
        enabled_indices = [
            idx for idx, op in enumerate(ANALYSIS_OPERATIONS)
            if op.key not in self._disabled_quick_ops
        ]
        has_enabled_ops = bool(enabled_indices)

        # Ensure current selection is valid.
        current_idx = self.quick_run_combo.currentIndex()
        if has_enabled_ops:
            current_item = model.item(current_idx) if current_idx >= 0 else None
            if current_item is None or not current_item.isEnabled():
                self.quick_run_combo.setCurrentIndex(enabled_indices[0])

        controls_enabled = has_clips and has_enabled_ops and not self._is_analyzing
        self.quick_run_combo.setEnabled(controls_enabled)
        self.quick_run_btn.setEnabled(controls_enabled)

    def _on_analyze_click(self):
        """Handle analyze button click - open picker modal."""
        self.analyze_picker_requested.emit()

    def _on_glossary_click(self):
        """Handle glossary button click - open the film language glossary dialog."""
        dialog = GlossaryDialog(self)
        dialog.exec()

    def _on_clip_selected(self, clip):
        """Handle clip selection."""
        self.clip_selected.emit(clip)

    def _on_browser_selection_changed(self, _clip_ids: list[str]):
        """Handle selection changes from the clip browser."""
        self._update_selection_ui()

    def _update_selection_ui(self):
        """Update selection count label and notify parent."""
        selected = self.clip_browser.get_selected_clips()
        count = len(selected)
        if count > 0:
            self.selection_label.setText(f"{count} selected")
        else:
            self.selection_label.setText("")
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
        self.clip_browser.clear_selection()

        # Update clip count label
        visible = self.clip_browser.get_visible_clip_count()
        total = self.clip_browser.get_total_clip_count()
        if self.clip_browser.has_active_filters():
            self.clip_count_label.setText(f"{visible}/{total} clips")
        else:
            self.clip_count_label.setText(f"{total} clips")

    def _update_ui_state(self):
        """Update UI based on current clip count."""
        count = len(self._clip_ids)
        has_clips = count > 0
        controls_enabled = has_clips and not self._is_analyzing

        self._refresh_quick_run_availability()
        self.analyze_btn.setEnabled(controls_enabled)

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

    def add_clips(self, clip_ids: list[str], populate_browser: bool = True):
        """Add clips to the analysis tab (merge with deduplication).

        Args:
            clip_ids: List of clip IDs to add
            populate_browser: When False, restore only the clip ID state and
                defer ClipBrowser widget creation until the Analyze tab is shown.
        """
        clip_source_pairs = []
        for clip_id in clip_ids:
            if clip_id not in self._clip_ids:
                self._clip_ids.add(clip_id)
                # Add to ClipBrowser if clip exists
                clip = self._clips_by_id.get(clip_id)
                source = self._sources_by_id.get(clip.source_id) if clip else None
                if clip and source:
                    clip_source_pairs.append((clip, source))
                else:
                    logger.warning(f"Could not add clip {clip_id}: clip or source not found")
                    self._clip_ids.discard(clip_id)

        added_count = len(clip_source_pairs)
        if clip_source_pairs:
            if populate_browser:
                self._add_pairs_to_browser(clip_source_pairs)
            else:
                self._pending_browser_clip_ids.extend(
                    clip.id for clip, _source in clip_source_pairs
                )

        if added_count > 0:
            logger.info(f"Added {added_count} clips to Analyze tab")
        self._update_ui_state()
        self.clips_changed.emit(self.get_clip_ids())

    def _add_pairs_to_browser(self, clip_source_pairs: list[tuple]) -> None:
        """Populate the clip browser with one filter sync and one layout rebuild."""
        if len(clip_source_pairs) >= VIRTUALIZATION_THRESHOLD:
            self.clip_browser.set_virtual_clips(clip_source_pairs)
            return

        self.clip_browser.add_clips(
            clip_source_pairs,
            defer_rebuild=True,
            defer_filter_sync=True,
        )
        self.clip_browser.finalize_batch_load()

    def _populate_pending_browser_clips(self) -> None:
        """Create deferred Analyze clip widgets when the tab is first shown."""
        if not self._pending_browser_clip_ids:
            return

        pending_ids = self._pending_browser_clip_ids
        self._pending_browser_clip_ids = []
        clip_source_pairs = []
        for clip_id in pending_ids:
            if clip_id not in self._clip_ids:
                continue
            if clip_id in self.clip_browser._thumbnail_by_id:
                continue
            clip = self._clips_by_id.get(clip_id)
            source = self._sources_by_id.get(clip.source_id) if clip else None
            if clip and source:
                clip_source_pairs.append((clip, source))
            else:
                logger.warning(f"Could not restore deferred clip {clip_id}: clip or source not found")
                self._clip_ids.discard(clip_id)

        if clip_source_pairs:
            self._add_pairs_to_browser(clip_source_pairs)
            self._log_deferred_browser_population(len(clip_source_pairs))

    def _log_deferred_browser_population(self, clip_count: int) -> None:
        """Log deferred Analyze restore with virtual/realized counts."""
        if not self.clip_browser.is_virtualized():
            logger.info(f"Populated {clip_count} deferred Analyze clips")
            return

        QTimer.singleShot(
            0,
            lambda: logger.info(
                "Virtualized Analyze browser for %s deferred clips; realized %s widgets",
                clip_count,
                self.clip_browser.get_realized_clip_count(),
            ),
        )

    def remove_clip(self, clip_id: str):
        """Remove a single clip from the analysis tab.

        Args:
            clip_id: ID of clip to remove
        """
        if clip_id in self._clip_ids:
            self._clip_ids.discard(clip_id)
            self._pending_browser_clip_ids = [
                pending_id
                for pending_id in self._pending_browser_clip_ids
                if pending_id != clip_id
            ]
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
        self._pending_browser_clip_ids.clear()
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
            self._pending_browser_clip_ids = [
                clip_id
                for clip_id in self._pending_browser_clip_ids
                if clip_id in self._clip_ids
            ]
            # Rebuild ClipBrowser — also discard IDs that can't be resolved
            # (_clips_by_id may be stale if set_lookups hasn't been called yet)
            self.clip_browser.clear()
            unresolvable = []
            for clip_id in self._clip_ids:
                clip = self._clips_by_id.get(clip_id)
                source = self._sources_by_id.get(clip.source_id) if clip else None
                if clip and source:
                    self.clip_browser.add_clip(clip, source)
                else:
                    unresolvable.append(clip_id)
            for clip_id in unresolvable:
                self._clip_ids.discard(clip_id)
            if unresolvable:
                logger.warning(
                    f"Discarded {len(unresolvable)} unresolvable clip IDs "
                    f"during orphan rebuild"
                )
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
        clips = [self._clips_by_id[cid] for cid in self._clip_ids
                 if cid in self._clips_by_id]
        if len(clips) != len(self._clip_ids):
            unresolved = len(self._clip_ids) - len(clips)
            logger.warning(
                f"get_clips: {unresolved} of {len(self._clip_ids)} clip IDs "
                f"could not be resolved (stale _clips_by_id reference?)"
            )
        return clips

    def update_clip_colors(self, clip_id: str, colors: list):
        """Update colors for a clip."""
        if clip_id in self._clip_ids:
            if clip_id in self.clip_browser._thumbnail_by_id:
                self.clip_browser.update_clip_colors(clip_id, colors)
            self._refresh_quick_run_availability()

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update shot type for a clip."""
        if clip_id in self._clip_ids:
            if clip_id in self.clip_browser._thumbnail_by_id:
                self.clip_browser.update_clip_shot_type(clip_id, shot_type)
            self._refresh_quick_run_availability()

    def update_clip_thumbnail(self, clip_id: str, thumb_path):
        """Update thumbnail for a clip."""
        if (
            clip_id in self._clip_ids
            and clip_id in self.clip_browser._thumbnail_by_id
        ):
            self.clip_browser.update_clip_thumbnail(clip_id, thumb_path)

    def update_clip_transcript(self, clip_id: str, segments: list):
        """Update transcript for a clip."""
        if clip_id in self._clip_ids:
            if clip_id in self.clip_browser._thumbnail_by_id:
                self.clip_browser.update_clip_transcript(clip_id, segments)
            self._refresh_quick_run_availability()

    def update_clip_extracted_text(self, clip_id: str, texts: list):
        """Update extracted text for a clip."""
        if clip_id in self._clip_ids:
            if clip_id in self.clip_browser._thumbnail_by_id:
                self.clip_browser.update_clip_extracted_text(clip_id, texts)
            self._refresh_quick_run_availability()

    def update_clip_custom_queries(self, clip_id: str, custom_queries: list[dict] | None):
        """Update custom query results for a clip."""
        if (
            clip_id in self._clip_ids
            and clip_id in self.clip_browser._thumbnail_by_id
        ):
            self.clip_browser.update_clip_custom_queries(clip_id, custom_queries)

    def update_clip_cinematography(self, clip_id: str, cinematography):
        """Update cinematography analysis for a clip.

        Args:
            clip_id: ID of the clip to update
            cinematography: CinematographyAnalysis object
        """
        if clip_id in self._clip_ids:
            if clip_id in self.clip_browser._thumbnail_by_id:
                self.clip_browser.update_clip_cinematography(clip_id, cinematography)
            self._refresh_quick_run_availability()

    def set_analyzing(self, is_analyzing: bool, operation: str = ""):
        """Update UI state during analysis operations.

        Args:
            is_analyzing: Whether an analysis operation is running
            operation: Which operation is running (operation key or "pipeline")
        """
        self._is_analyzing = is_analyzing

        if is_analyzing:
            self.analyze_btn.setText("Analyzing...")
        else:
            self.analyze_btn.setText("Analyze...")

        self._update_ui_state()

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

    def on_tab_activated(self):
        """Refresh clip browser layout when tab becomes visible."""
        self._populate_pending_browser_clips()
        if (
            self.state_stack.currentIndex() == self.STATE_CLIPS
            and self.clip_browser.thumbnails
        ):
            self.clip_browser.refresh_layout()
