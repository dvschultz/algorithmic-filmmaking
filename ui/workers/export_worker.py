"""Background worker for project bundle export.

Runs export_project_bundle() in a background thread so large file copies
don't block the UI.
"""

import logging
from pathlib import Path

from PySide6.QtCore import Signal

from core.project import Project
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class ExportBundleWorker(CancellableWorker):
    """Background worker that exports a project bundle.

    Signals:
        progress: Emitted with (current_file, total_files, filename)
        export_completed: Emitted with ExportResult on success
        error: Emitted with error string on failure (inherited)
    """

    progress = Signal(int, int, str)  # current, total, filename
    export_completed = Signal(object)  # ExportResult

    def __init__(
        self,
        project: Project,
        dest_dir: Path,
        include_videos: bool = True,
        include_clips: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._snapshot = project.snapshot_for_save()
        from core.artifacts import ArtifactLease

        self._artifact_lease = ArtifactLease.for_snapshot(self._snapshot)
        self._project_path = project.path
        self._dest_dir = dest_dir
        self._include_videos = include_videos
        self._include_clips = include_clips

    def run(self):
        self._log_start()
        project = None
        try:
            from core.project_export import export_project_bundle

            snapshot = dict(self._snapshot)
            extra = snapshot.pop("extra_data")
            project = Project(
                path=self._project_path,
                **snapshot,
                sequences=extra["_all_sequences"],
                active_sequence_index=extra["active_sequence_index"],
            )
            result = export_project_bundle(
                project=project,
                dest_dir=self._dest_dir,
                include_videos=self._include_videos,
                include_clips=self._include_clips,
                progress_callback=lambda c, t, f: self.progress.emit(c, t, f),
                cancel_check=self.is_cancelled,
            )

            if not self.is_cancelled():
                self.export_completed.emit(result)
        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Bundle export failed: {e}")
                self.error.emit(str(e))
        finally:
            if project is not None:
                project.session.close()
            self._artifact_lease.close()
        self._log_complete()
