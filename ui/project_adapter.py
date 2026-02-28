"""Qt signal adapter for the Project class.

This module bridges the callback-based observer pattern used by Project
with Qt's signal/slot mechanism used by the GUI.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core.project import Project
from models.clip import Source, Clip

logger = logging.getLogger(__name__)


class ProjectSignalAdapter(QObject):
    """Bridges Project callbacks to Qt signals for UI updates.

    Usage:
        project = Project.new()
        adapter = ProjectSignalAdapter(project, parent=self)
        adapter.source_added.connect(self._on_source_added)
        adapter.clips_added.connect(self._on_clips_added)
    """

    # Signals emitted when project state changes
    source_added = Signal(object)       # Source
    source_removed = Signal(object)     # Source
    source_updated = Signal(object)     # Source
    clips_added = Signal(list)          # list[Clip]
    clips_updated = Signal(list)        # list[Clip]
    sequence_changed = Signal(list)     # list[str] clip_ids
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
        self.project_loaded.emit()

    def _on_project_event(self, event: str, data: Any) -> None:
        """Convert callback events to Qt signals.

        Args:
            event: Event name from Project
            data: Event-specific data
        """
        if event == "source_added":
            self.source_added.emit(data)
        elif event == "source_removed":
            self.source_removed.emit(data)
        elif event == "source_updated":
            self.source_updated.emit(data)
        elif event == "clips_added":
            self.clips_added.emit(data)
        elif event == "clips_updated":
            self.clips_updated.emit(data)
        elif event == "sequence_changed":
            self.sequence_changed.emit(data)
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
