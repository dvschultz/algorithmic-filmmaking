"""Run-bound download publication and native-completion phase advancement."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from PySide6.QtCore import Slot

from core.downloader import DownloadResult
from core.intention_workflow import WorkflowState
from core.operations.downloads import DownloadApplication, DownloadItem
from core.jobs.media import media_stamp
from core.spine.sources import same_source_path
from ui.workers.download_delivery import DownloadDelivery
from ui.workers.download_workers import URLBulkDownloadWorker
from ui.workers.intention_run import IntentionRun

if TYPE_CHECKING:
    from models.clip import Source


class IntentionDownloadDelivery(DownloadDelivery):
    """Reuse the download channel's retention while binding it to one plan."""

    def __init__(
        self, window: Any, worker: URLBulkDownloadWorker, run: IntentionRun
    ) -> None:
        self.run_identity = run
        self.workflow = run.workflow
        self.application = DownloadApplication(run.project)
        self.accepted: dict[int, DownloadResult] = {}
        self.sources: dict[int, Source] = {}
        self.cancelled = False
        self.done = False
        self.run_error: str | None = None
        super().__init__(
            window,
            "url_bulk_download_worker",
            worker,
            {},
            finished=self._native_finished,
        )
        if not hasattr(window, "_active_intention_downloads"):
            window._active_intention_downloads = set()
        window._active_intention_downloads.add(self)
        worker.item_ready.connect(self._item)
        worker.progress.connect(self._on_progress)
        worker.error.connect(self._on_error)
        self.workflow.workflow_cancelled.connect(self.cancel)
        self.workflow.workflow_completed.connect(self._workflow_completed)

    def _current(self) -> bool:
        return bool(
            not self.cancelled
            and not self.done
            and not self.worker.is_cancelled()
            and self.window._download_deliveries.get(self.attribute) is self
            and getattr(self.window, self.attribute, None) is self.worker
            and self.run_identity.is_current(self.window, WorkflowState.DOWNLOADING)
        )

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, message: str) -> None:
        if self.sender() is self.worker and self._current():
            self.workflow.on_download_progress(current, total, message)

    @Slot(str)
    def _on_error(self, error: str) -> None:
        if self.sender() is self.worker and self._current():
            self.run_error = error

    @Slot(object)
    def _item(self, item: DownloadItem) -> None:
        if self.sender() is self.worker:
            self._accept(item)

    def _accept(self, item: DownloadItem) -> None:
        if (
            not self._current()
            or item.index in self.accepted
            or not 0 <= item.index < len(self.worker.requests)
            or self.worker.requests[item.index].url != item.url
            or self.worker.items.get(item.index) != item
        ):
            return
        result = item.as_result()
        self.accepted[item.index] = result
        if result.success:
            try:
                publication = self.application.apply(item, still_current=self._current)
            except Exception as exc:
                result = DownloadResult(success=False, error=str(exc))
                self.accepted[item.index] = result
            else:
                if publication is None or not self._current():
                    return
                source, added = publication
                self.sources[item.index] = source
                try:
                    if added:
                        self.window.collect_tab.add_source(source)
                except Exception as exc:
                    if self._current():
                        self.workflow.fail(str(exc))
                    return
                if self.run_identity.project.sources_by_id.get(source.id) is not source:
                    result = DownloadResult(
                        success=False, error="Download source was replaced"
                    )
                    self.accepted[item.index] = result

    @Slot()
    def _native_finished(self) -> None:
        if self.done or self.sender() is not self.worker or self.worker.isRunning():
            return
        try:
            if not self._current():
                self.cancel()
                return
            for index, request in enumerate(self.worker.requests):
                if not self._current():
                    return
                item = self.worker.items.get(index)
                if item is not None:
                    self._accept(item)
                if index not in self.accepted:
                    result = DownloadResult(
                        success=False,
                        error=self.run_error or "Download completed without a result",
                    )
                    self.accepted[index] = result
                result = self.accepted[index]
                if result.success:
                    source = self.sources.get(index)
                    if (
                        source is None
                        or item is None
                        or item.path is None
                        or self.run_identity.project.sources_by_id.get(source.id)
                        is not source
                        or not same_source_path(source.file_path, item.path)
                        or media_stamp(item.path) != item.stamp
                    ):
                        self.accepted[index] = DownloadResult(
                            success=False,
                            error="Downloaded source changed before detection",
                        )
            if self._current():
                self.workflow.on_download_all_finished(
                    [
                        {
                            "url": request.url,
                            "success": self.accepted[index].success,
                            "file_path": self.accepted[index].file_path,
                            "error": self.accepted[index].error,
                        }
                        for index, request in enumerate(self.worker.requests)
                    ]
                )
        except Exception as exc:
            if self._current():
                self.workflow.fail(str(exc))
        finally:
            self.done = True
            self.window._active_intention_downloads.discard(self)
            super()._finished()

    def abort_start(self) -> None:
        """Release a worker that failed to start; retain it if it did start."""
        self.cancel()
        if not self.worker.isRunning() and not self.done:
            self.done = True
            self.window._active_intention_downloads.discard(self)
            super()._finished()

    @Slot(object)
    def _workflow_completed(self, _result: Any) -> None:
        self.cancel()

    @Slot()
    def cancel(self) -> None:
        if self.cancelled or self.done:
            return
        self.cancelled = True
        self.worker.cancel()
        if getattr(self.window, self.attribute, None) is self.worker:
            setattr(self.window, self.attribute, None)
        if self.window._download_deliveries.get(self.attribute) is self:
            del self.window._download_deliveries[self.attribute]
        if (
            self.window.intention_workflow is self.workflow
            and self.workflow.plan is self.run_identity.plan
            and self.workflow.is_running
        ):
            self.workflow.cancel()
