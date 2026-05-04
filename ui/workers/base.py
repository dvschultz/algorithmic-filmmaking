"""Base classes for background workers.

Provides common functionality for cancellable QThread workers.
"""

import logging
import threading
from typing import Iterable

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


def is_transient_provider_error(message: str) -> bool:
    """Return True for transient cloud/provider failures worth retrying.

    Covers rate limits (429), gateway/server errors (500/502/503/504), and
    network/timeout failures. Used by VLM/LLM workers (description, custom
    query) that retry transient failures with exponential backoff.
    """
    normalized = message.lower()
    return (
        "429" in normalized
        or "rate limit" in normalized
        or "too many requests" in normalized
        or "500" in normalized
        or "502" in normalized
        or "503" in normalized
        or "504" in normalized
        or "internalservererror" in normalized
        or "internal error" in normalized
        or "temporarily unavailable" in normalized
        or "service unavailable" in normalized
        or "timeout" in normalized
        or "timed out" in normalized
        or "connection" in normalized
        or "network" in normalized
    )


def summarize_clip_errors(
    errors: list[tuple[str, str]],
    *,
    operation_label: str,
    preview_count: int = 3,
) -> str:
    """Return a compact user-facing summary for a batch of per-clip failures.

    Used by analysis workers that collect (clip_id, message) pairs and need a
    single string to surface to the user. When only one error is present, the
    raw message is returned verbatim so callers don't see a redundant header.
    """
    if len(errors) == 1:
        return errors[0][1]

    preview = "\n".join(
        f"- {clip_id}: {message}" for clip_id, message in errors[:preview_count]
    )
    remaining = len(errors) - preview_count
    summary = f"{operation_label} failed for {len(errors)} clips:\n\n{preview}"
    if remaining > 0:
        summary += f"\n\n... and {remaining} more"
    return summary


def summarize_messages(
    messages: Iterable[str],
    *,
    header: str,
    preview_count: int = 3,
) -> str:
    """Return a compact summary for a list of pre-formatted error messages.

    The single-message case returns the message verbatim. ``header`` is used
    only when more than one message needs to be aggregated.
    """
    items = list(messages)
    if len(items) == 1:
        return items[0]

    preview = "\n".join(f"- {message}" for message in items[:preview_count])
    remaining = len(items) - preview_count
    summary = f"{header}\n\n{preview}"
    if remaining > 0:
        summary += f"\n\n... and {remaining} more"
    return summary


class CancellableWorker(QThread):
    """Base class for workers that support cancellation.

    Uses threading.Event for thread-safe cancellation, which is essential
    when workers use ThreadPoolExecutor or other multi-threaded patterns.

    Provides:
    - Thread-safe cancellation via threading.Event
    - cancel() method with logging
    - Common logging patterns for worker lifecycle

    Subclasses should:
    - Call super().__init__() in their __init__
    - Check self.is_cancelled() in their run() loop
    - Call _log_start() and _log_complete() in run()
    - Override run() to implement the worker's task
    """

    # Common signals - subclasses can add more
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_event = threading.Event()

    @property
    def worker_name(self) -> str:
        """Return the worker class name for logging."""
        return self.__class__.__name__

    def cancel(self):
        """Request cancellation of the worker (thread-safe).

        Sets the cancellation event. The worker's run() method should
        check is_cancelled() periodically and exit gracefully when True.
        """
        logger.info(f"{self.worker_name}.cancel() called")
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Thread-safe check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def _log_start(self):
        """Log that the worker is starting."""
        logger.info(f"{self.worker_name}.run() STARTING")

    def _log_cancelled(self):
        """Log that the worker was cancelled."""
        logger.info(f"{self.worker_name} cancelled")

    def _log_complete(self):
        """Log that the worker completed."""
        logger.info(f"{self.worker_name}.run() COMPLETED")

    def _log_error(self, error: str, item_id: str = None):
        """Log an error during processing."""
        if item_id:
            logger.warning(f"{self.worker_name} failed for {item_id}: {error}")
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
            if self.is_cancelled():
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
