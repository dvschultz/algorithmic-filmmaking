"""Base classes for background workers.

Provides common functionality for cancellable QThread workers.
"""

import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class CancellableWorker(QThread):
    """Base class for workers that support cancellation.

    Provides:
    - _cancelled flag for cooperative cancellation
    - cancel() method with logging
    - Common logging patterns for worker lifecycle

    Subclasses should:
    - Call super().__init__() in their __init__
    - Check self._cancelled in their run() loop
    - Call _log_start() and _log_complete() in run()
    - Override run() to implement the worker's task
    """

    # Common signals - subclasses can add more
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    @property
    def worker_name(self) -> str:
        """Return the worker class name for logging."""
        return self.__class__.__name__

    def cancel(self):
        """Request cancellation of the worker.

        Sets the _cancelled flag. The worker's run() method should
        check this flag periodically and exit gracefully when True.
        """
        logger.info(f"{self.worker_name}.cancel() called")
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def _log_start(self):
        """Log that the worker is starting."""
        logger.info(f"{self.worker_name}.run() STARTING")

    def _log_cancelled(self):
        """Log that the worker was cancelled."""
        logger.info(f"{self.worker_name} cancelled")

    def _log_complete(self):
        """Log that the worker completed."""
        logger.info(f"{self.worker_name}.run() COMPLETED")

    def _log_error(self, error: str, clip_id: str = None):
        """Log an error during processing."""
        if clip_id:
            logger.warning(f"{self.worker_name} failed for {clip_id}: {error}")
        else:
            logger.error(f"{self.worker_name} error: {error}")


class BatchProcessingWorker(CancellableWorker):
    """Base class for workers that process multiple items (clips, files, etc.).

    Provides:
    - Standard progress signal (current, total)
    - Common iteration pattern with cancellation checks
    - Consistent logging

    Subclasses should implement:
    - process_item(item, index): Process a single item
    - Optionally override get_items() to customize what's processed
    """

    progress = Signal(int, int)  # current, total
    completed = Signal()  # Emitted when all items processed

    def __init__(self, items: list = None, parent=None):
        super().__init__(parent)
        self._items = items or []

    def get_items(self) -> list:
        """Return the items to process. Can be overridden by subclasses."""
        return self._items

    def process_item(self, item, index: int):
        """Process a single item. Override in subclasses.

        Args:
            item: The item to process
            index: Zero-based index of the item

        Should emit any item-specific signals (e.g., result_ready).
        Exceptions are caught by run() and logged.
        """
        raise NotImplementedError("Subclasses must implement process_item()")

    def run(self):
        """Process all items with cancellation support."""
        self._log_start()
        items = self.get_items()
        total = len(items)

        for i, item in enumerate(items):
            if self._cancelled:
                self._log_cancelled()
                break

            try:
                self.process_item(item, i)
            except Exception as e:
                item_id = getattr(item, "id", None) or str(i)
                self._log_error(str(e), item_id)

            self.progress.emit(i + 1, total)

        self.completed.emit()
        self._log_complete()
