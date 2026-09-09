"""Qt signal adapter for the Project class.

This module bridges the callback-based observer pattern used by Project
with Qt's signal/slot mechanism used by the GUI.
"""

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core.project import Project
from ui.models import ClipLibraryModel, FrameLibraryModel

logger = logging.getLogger(__name__)


class ProjectSignalAdapter(QObject):
    """Bridges Project callbacks to Qt signals for UI updates.

    The adapter also owns the shared item models (KTD13): ``clip_model`` and
    ``frame_model`` are updated from project events *before* the matching Qt
    signal fires, so slots always observe models that already reflect the
    change. Workspaces attach to these models for clip/source/frame lookups
    and keep only their own membership, selection, and filters.

    Usage:
        project = Project.new()
        adapter = ProjectSignalAdapter(project, parent=self)
        adapter.source_added.connect(self._on_source_added)
        adapter.clips_added.connect(self._on_clips_added)
        cut_tab.clip_browser.attach_model(adapter.clip_model)
    """

    # Signals emitted when project state changes
    source_added = Signal(object)       # Source
    source_removed = Signal(object)     # Source
    source_updated = Signal(object)     # Source
    sources_changed = Signal(list)     # atomic source removal/restoration
    clips_added = Signal(list)          # list[Clip]
    clips_updated = Signal(list)        # list[Clip]
    clips_removed = Signal(list)        # list[Clip]
    frames_added = Signal(list)         # list[Frame]
    frames_removed = Signal(list)       # list[Frame]
    frames_updated = Signal(list)
    project_metadata_changed = Signal()
    sequences_changed = Signal(list)
    active_sequence_changed = Signal(int)
    sequence_changed = Signal(list)     # list[str] clip_ids
    audio_sources_changed = Signal(list)  # list[AudioSource]
    project_saved = Signal(object)      # Path
    project_loaded = Signal()           # (no data)
    project_cleared = Signal()          # (no data)

    def __init__(self, project: Project, parent: Optional[QObject] = None):
        """Create adapter for a Project instance.

        Args:
            project: The Project to adapt
            parent: Qt parent object
        """
        super().__init__(parent)
        self._project = project
        self.clip_model = ClipLibraryModel(self)
        self.frame_model = FrameLibraryModel(self)
        self._reset_models()
        project.add_observer(self._on_project_event)

    @property
    def project(self) -> Project:
        """The underlying Project instance."""
        return self._project

    def set_project(self, project: Project) -> None:
        """Switch to a different Project instance.

        Removes observer from old project and adds to new one.

        Args:
            project: New Project to adapt
        """
        if self._project is not None:
            self._project.remove_observer(self._on_project_event)
        self._project = project
        project.add_observer(self._on_project_event)
        self._reset_models()
        self.project_loaded.emit()

    def _reset_models(self) -> None:
        self.clip_model.set_project(self._project)
        self.frame_model.set_project(self._project)

    def _project_models(self, event: str, data: Any) -> None:
        """Mirror a project event into the shared item models."""
        project = self._project
        if project is None:
            return
        if event in ("source_added", "source_updated"):
            self.clip_model.set_sources(project.sources)
        elif event == "source_removed":
            self.clip_model.remove_source(data.id)
        elif event == "sources_changed":
            # Atomic removal/restoration (undo history): reconcile without a
            # reset so surviving rows keep their identity.
            self.clip_model.sync_project(project)
            self.frame_model.set_frames(project.frames)
        elif event == "clips_added":
            sources = project.sources_by_id
            self.clip_model.upsert(
                (clip, sources[clip.source_id]) for clip in data if clip.source_id in sources
            )
        elif event == "clips_updated":
            self.clip_model.refresh(data)
        elif event == "clips_removed":
            self.clip_model.remove([clip.id for clip in data])
        elif event == "frames_added":
            self.frame_model.append(data)
        elif event == "frames_updated":
            self.frame_model.refresh(data)
        elif event == "frames_removed":
            self.frame_model.remove([frame.id for frame in data])
        elif event in ("project_loaded", "project_cleared"):
            self._reset_models()

    def _on_project_event(self, event: str, data: Any) -> None:
        """Convert callback events to Qt signals.

        Args:
            event: Event name from Project
            data: Event-specific data
        """
        self._project_models(event, data)
        if event == "source_added":
            self.source_added.emit(data)
        elif event == "source_removed":
            self.source_removed.emit(data)
        elif event == "source_updated":
            self.source_updated.emit(data)
        elif event == "sources_changed":
            self.sources_changed.emit(data)
        elif event == "clips_added":
            self.clips_added.emit(data)
        elif event == "clips_updated":
            self.clips_updated.emit(data)
        elif event == "clips_removed":
            self.clips_removed.emit(data)
        elif event == "frames_added":
            self.frames_added.emit(data)
        elif event == "frames_removed":
            self.frames_removed.emit(data)
        elif event == "frames_updated":
            self.frames_updated.emit(data)
        elif event == "project_metadata_changed":
            self.project_metadata_changed.emit()
        elif event == "sequences_changed":
            self.sequences_changed.emit(data)
        elif event == "active_sequence_changed":
            self.active_sequence_changed.emit(data)
        elif event == "sequence_changed":
            self.sequence_changed.emit(data)
        elif event == "audio_sources_changed":
            self.audio_sources_changed.emit(data)
        elif event == "project_saved":
            self.project_saved.emit(data)
        elif event == "project_loaded":
            self.project_loaded.emit()
        elif event == "project_cleared":
            self.project_cleared.emit()
        else:
            logger.debug(f"Unknown project event: {event}")

    def disconnect_from_project(self) -> None:
        """Remove observer from the project.

        Call this before discarding the adapter.
        """
        if self._project is not None:
            self._project.remove_observer(self._on_project_event)
            self._project = None
            self._reset_models()
