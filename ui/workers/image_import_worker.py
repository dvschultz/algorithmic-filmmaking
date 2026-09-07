"""Qt adapter for shared detached still-image import."""

from pathlib import Path
from PySide6.QtCore import Signal

from core.operations.image_import import (
    ImageImportTask,
    ImageImportOutcome,
    run_image_import,
)
from ui.workers.base import CancellableWorker


class ImageImportWorker(CancellableWorker):
    progress = Signal(int, int)
    outcome_ready = Signal(object)

    def __init__(
        self,
        paths: list[Path],
        output_dir: Path,
        parent=None,
        *,
        copy_files: bool = False,
        validate_paths: bool = False,
    ) -> None:
        super().__init__(parent)
        self.task = ImageImportTask.from_paths(
            paths, output_dir, copy_files=copy_files, validate_paths=validate_paths
        )
        self.result: ImageImportOutcome | None = None

    def run(self) -> None:
        self.result = run_image_import(
            self.task, cancel_event=self._cancel_event, progress=self.progress.emit
        )
        self.outcome_ready.emit(self.result)
