"""Serial background metadata preparation with owner-thread result delivery."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.spine.sources import prepare_source_import
from ui.workers.base import CancellableWorker
from ui.workers.gui_tool_reply import GuiToolReply


def _stamp(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    except OSError:
        return None


@dataclass(frozen=True)
class SourceImportRequest:
    path: Path
    context: tuple[str, int | None]
    generation: int
    media_stamp: tuple[int, int, int, int, int] | None
    reply: GuiToolReply | None = None


class SourceImportWorker(CancellableWorker):
    result_ready = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, request: SourceImportRequest, parent: QObject) -> None:
        super().__init__(parent)
        self.request = request

    def run(self) -> None:
        if self.is_cancelled():
            return
        try:
            if _stamp(self.request.path) != self.request.media_stamp:
                raise RuntimeError("Media changed before import")
            source = prepare_source_import(self.request.path, self._cancel_event)
        except Exception as exc:
            if not self.is_cancelled():
                self.failed.emit(self.request, str(exc))
            return
        if source is not None and not self.is_cancelled():
            self.result_ready.emit(self.request, source)


class SourceImportQueue(QObject):
    """Keep one probe running, retaining its thread until Qt reports completion.

    Cancellation drops queued requests immediately. An active native probe may
    finish later; its result is rejected by generation before forwarding. No live
    project is passed to the worker. Call close() before destroying the owner.
    """

    result_ready = Signal(object, object)
    failed = Signal(object, str)
    drained = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[SourceImportRequest] = deque()
        self._worker: SourceImportWorker | None = None
        self._generation = 0
        self._closed = False

    @property
    def pending(self) -> bool:
        return self._worker is not None or bool(self._queue)

    def submit(
        self,
        path: Path,
        context: tuple[str, int | None],
        *,
        reply: GuiToolReply | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Source import queue is closed")
        path = Path(path).expanduser().absolute()
        self._queue.append(
            SourceImportRequest(path, context, self._generation, _stamp(path), reply)
        )
        self._start_next()

    def _start_next(self) -> None:
        if self._worker is not None or not self._queue:
            return
        self._worker = SourceImportWorker(self._queue.popleft(), self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(object, object)
    def _on_result(self, request: SourceImportRequest, source: object) -> None:
        if request.generation != self._generation or self._closed:
            return
        if request.media_stamp is None or _stamp(request.path) != request.media_stamp:
            self.failed.emit(
                request, "Media changed or became unavailable during import"
            )
            return
        self.result_ready.emit(request, source)

    @Slot(object, str)
    def _on_failed(self, request: SourceImportRequest, error: str) -> None:
        if request.generation == self._generation and not self._closed:
            self.failed.emit(request, error)

    @Slot()
    def _on_finished(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        worker.deleteLater()
        self._start_next()
        if not self.pending:
            self.drained.emit()

    def cancel_reply(self, reply: GuiToolReply) -> None:
        """Cancel one agent request without dropping unrelated UI imports."""
        self._queue = deque(request for request in self._queue if request.reply is not reply)
        if self._worker is not None and self._worker.request.reply is reply:
            self._worker.cancel()

    def cancel_pending(self) -> None:
        self._generation += 1
        self._queue.clear()
        if self._worker is not None:
            self._worker.cancel()

    def close(self) -> None:
        self._closed = True
        self.cancel_pending()
        if self._worker is not None:
            # Media subprocesses have timeouts; never terminate a native thread.
            self._worker.wait()
