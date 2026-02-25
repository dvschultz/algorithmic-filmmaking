"""Signal handling for graceful CLI shutdown with partial progress saving.

This module provides:
- GracefulExit exception for clean shutdown
- ProgressCheckpoint for saving partial progress
- Signal handlers for SIGINT/SIGTERM
"""

import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import click

logger = logging.getLogger(__name__)


class GracefulExit(Exception):
    """Raised when graceful shutdown is requested via signal."""
    pass


class ProgressCheckpoint:
    """Manages partial progress saving for interruptible operations.

    Usage:
        checkpoint = ProgressCheckpoint(Path("checkpoint.json"))
        checkpoint.set_total(100)

        for i, item in enumerate(items):
            if checkpoint.should_stop:
                break
            process(item)
            checkpoint.add_result(item_result)
            checkpoint.update_progress(i + 1)

        if checkpoint.was_interrupted:
            checkpoint.save()
            print(f"Progress saved. Resume with --resume {checkpoint.path}")
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to save checkpoint file (None = auto-generate)
        """
        self._checkpoint_path = checkpoint_path
        self._current_item: int = 0
        self._total_items: int = 0
        self._partial_results: list[Any] = []
        self._metadata: dict[str, Any] = {}
        self._should_stop: bool = False
        self._was_interrupted: bool = False

    @property
    def path(self) -> Optional[Path]:
        """Get checkpoint file path."""
        return self._checkpoint_path

    @property
    def should_stop(self) -> bool:
        """Whether processing should stop (interrupt received)."""
        return self._should_stop

    @property
    def was_interrupted(self) -> bool:
        """Whether an interrupt signal was received."""
        return self._was_interrupted

    @property
    def progress_fraction(self) -> float:
        """Get progress as fraction 0.0-1.0."""
        if self._total_items == 0:
            return 0.0
        return self._current_item / self._total_items

    def set_total(self, total: int) -> None:
        """Set total number of items to process."""
        self._total_items = total

    def update_progress(self, current: int) -> None:
        """Update current progress."""
        self._current_item = current

    def add_result(self, result: Any) -> None:
        """Add a partial result."""
        self._partial_results.append(result)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata to save with checkpoint."""
        self._metadata[key] = value

    def request_stop(self) -> None:
        """Request processing to stop (called by signal handler)."""
        self._should_stop = True
        self._was_interrupted = True

    def save(self) -> bool:
        """Save current progress to checkpoint file.

        Returns:
            True if save succeeded, False otherwise
        """
        if not self._checkpoint_path:
            return False

        try:
            checkpoint_data = {
                "version": "1.0",
                "current_item": self._current_item,
                "total_items": self._total_items,
                "partial_results_count": len(self._partial_results),
                "partial_results": self._partial_results,
                "metadata": self._metadata,
            }

            # Atomic write
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            temp_path.rename(self._checkpoint_path)

            logger.info(f"Checkpoint saved: {self._checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load(self) -> bool:
        """Load checkpoint from file.

        Returns:
            True if load succeeded, False otherwise
        """
        if not self._checkpoint_path or not self._checkpoint_path.exists():
            return False

        try:
            with open(self._checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)

            self._current_item = data.get("current_item", 0)
            self._total_items = data.get("total_items", 0)
            self._partial_results = data.get("partial_results", [])
            self._metadata = data.get("metadata", {})

            logger.info(f"Checkpoint loaded: {self._current_item}/{self._total_items} items")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def clear(self) -> None:
        """Clear checkpoint file if it exists."""
        if self._checkpoint_path and self._checkpoint_path.exists():
            try:
                self._checkpoint_path.unlink()
                logger.debug(f"Checkpoint cleared: {self._checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")


# Global checkpoint reference for signal handlers
_active_checkpoint: Optional[ProgressCheckpoint] = None


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    logger.info(f"Received {signal_name}, requesting graceful shutdown...")

    if _active_checkpoint:
        _active_checkpoint.request_stop()
        click.echo(f"\n{signal_name} received. Saving progress...", err=True)
    else:
        click.echo(f"\n{signal_name} received. Shutting down...", err=True)
        raise GracefulExit()


def setup_signal_handlers(checkpoint: Optional[ProgressCheckpoint] = None) -> None:
    """Install signal handlers for graceful shutdown.

    Args:
        checkpoint: Optional ProgressCheckpoint to save on interrupt
    """
    global _active_checkpoint
    _active_checkpoint = checkpoint

    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)


def restore_default_handlers() -> None:
    """Restore default signal handlers."""
    global _active_checkpoint
    _active_checkpoint = None

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def interruptible_loop(
    items: list[Any],
    process_func: Callable[[Any], Any],
    checkpoint_path: Optional[Path] = None,
    label: str = "Processing",
) -> tuple[list[Any], bool]:
    """Process items with interrupt support and checkpoint saving.

    Args:
        items: Items to process
        process_func: Function to process each item
        checkpoint_path: Path to save checkpoint on interrupt
        label: Label for progress display

    Returns:
        Tuple of (results list, was_completed bool)
    """
    checkpoint = ProgressCheckpoint(checkpoint_path)
    checkpoint.set_total(len(items))

    setup_signal_handlers(checkpoint)

    results = []
    try:
        for i, item in enumerate(items):
            if checkpoint.should_stop:
                break

            result = process_func(item)
            results.append(result)
            checkpoint.add_result(result)
            checkpoint.update_progress(i + 1)

        completed = not checkpoint.was_interrupted

    except GracefulExit:
        completed = False
    finally:
        restore_default_handlers()

    # Save checkpoint if interrupted
    if checkpoint.was_interrupted and checkpoint_path:
        checkpoint.save()
        click.echo(
            f"Progress saved to {checkpoint_path}. "
            f"Processed {len(results)}/{len(items)} items.",
            err=True
        )

    return results, completed
