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
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._dest_dir = dest_dir
        self._include_videos = include_videos

    def run(self):
        self._log_start()
        try:
            from core.project_export import export_project_bundle

            result = export_project_bundle(
                project=self._project,
                dest_dir=self._dest_dir,
                include_videos=self._include_videos,
                progress_callback=lambda c, t, f: self.progress.emit(c, t, f),
                cancel_check=self.is_cancelled,
            )

            if not self.is_cancelled():
                self.export_completed.emit(result)
        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Bundle export failed: {e}")
                self.error.emit(str(e))
        self._log_complete()
