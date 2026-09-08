"""Complete agent imports after queued metadata reaches the project owner."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Slot

from core.spine.sources import find_source_by_path
from ui.workers.qt_lifetime import RetiringQObject


class AgentSourceImport(RetiringQObject):
    def __init__(self, window: Any) -> None:
        super().__init__(window)
        self.window = window
        self.reply = window._dispatch_gui_reply
        self.queue = window._source_import_queue
        self.paths: list[Path] = []
        self.errors: dict[Path, str] = {}
        self.cancelled = False
        self.existing_ids = set(window.project.sources_by_id)
        if not hasattr(window, "_active_source_imports"):
            window._active_source_imports = set()
        window._active_source_imports.add(self)
        self.queue.failed.connect(self.failed)
        self.queue.drained.connect(self.completed)

    def start(self, paths: list[Path]) -> None:
        self.paths = [path.expanduser().absolute() for path in paths]
        if not self.reply.is_current(self.window):
            self.cancel()
            return
        for path in self.paths:
            if find_source_by_path(self.window.project, path) is None:
                self.window._queue_source_import(path, reply=self.reply)
        if not self.queue.pending:
            self.completed()

    def cancel(self) -> None:
        self.cancelled = True
        self.queue.cancel_reply(self.reply)
        if not self.queue.pending:
            self.completed()

    @Slot(object, str)
    def failed(self, request: Any, message: str) -> None:
        if request.reply is self.reply:
            self.errors[request.path] = message

    @Slot()
    def completed(self) -> None:
        self.queue.failed.disconnect(self.failed)
        self.queue.drained.disconnect(self.completed)
        self.window._active_source_imports.discard(self)
        try:
            if self.cancelled or not self.reply.is_current(self.window):
                return
            imported, skipped, failed = [], [], []
            seen = set(self.existing_ids)
            for path in self.paths:
                source = find_source_by_path(self.window.project, path)
                if source is None:
                    failed.append({"filename": path.name, "error": self.errors.get(path, "Import did not complete")})
                elif source.id in seen:
                    skipped.append(path.name)
                else:
                    imported.append(path.name)
                    seen.add(source.id)
            result = {
                "success": not failed,
                "imported_count": len(imported), "imported_files": imported,
                "skipped_count": len(skipped), "skipped_files": skipped,
                "failed_count": len(failed), "failed_files": failed,
                "total_sources": len(self.window.project.sources),
            }
            if self.reply.name == "import_video" and not failed:
                source = find_source_by_path(self.window.project, self.paths[0])
                assert source is not None
                result.update(source_id=source.id, filename=source.filename, already_imported=bool(skipped))
            self.reply.send(self.window, {"success": result["success"], "result": result})
        finally:
            self.retire()
