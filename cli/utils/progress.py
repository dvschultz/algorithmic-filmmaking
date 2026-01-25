"""Progress display utilities for the CLI."""

import sys
from typing import Callable, Optional

import click


def create_progress_callback(
    label: str,
    show_status: bool = True,
) -> Callable[[float, str], None]:
    """Create a progress callback compatible with core/ functions.

    Progress is displayed on stderr to keep stdout clean for piping.
    In non-interactive mode, only status changes are logged.

    Args:
        label: Label to display with the progress bar
        show_status: Whether to show status messages

    Returns:
        Callback function that accepts (progress: float, status: str)
    """
    if not sys.stderr.isatty():
        # Non-interactive: just log status changes
        last_status: list[Optional[str]] = [None]

        def callback(progress: float, status: str) -> None:
            if show_status and status != last_status[0]:
                click.echo(f"{label}: {status}", err=True)
                last_status[0] = status

        return callback
    else:
        # Interactive: use progress bar
        # We need to manage the progress bar state
        state = {"bar": None, "last_pos": 0}

        def callback(progress: float, status: str) -> None:
            # Create bar on first call
            if state["bar"] is None:
                state["bar"] = click.progressbar(
                    length=100,
                    label=label,
                    file=sys.stderr,
                    show_eta=True,
                    show_percent=True,
                )
                state["bar"].__enter__()

            # Update progress
            new_pos = int(progress * 100)
            if new_pos > state["last_pos"]:
                state["bar"].update(new_pos - state["last_pos"])
                state["last_pos"] = new_pos

            # Close bar when complete
            if progress >= 1.0:
                state["bar"].__exit__(None, None, None)
                if show_status and status:
                    click.echo(f"{label}: {status}", err=True)

        return callback


class ProgressContext:
    """Context manager for progress display.

    Automatically closes the progress bar when exiting the context.

    Usage:
        with ProgressContext("Processing") as progress:
            for i, item in enumerate(items):
                progress.update(i / len(items), f"Item {i}")
    """

    def __init__(self, label: str, show_status: bool = True):
        """Initialize progress context.

        Args:
            label: Label for the progress display
            show_status: Whether to show status messages
        """
        self.label = label
        self.show_status = show_status
        self._bar: Optional[click.progressbar] = None
        self._last_pos = 0
        self._interactive = sys.stderr.isatty()

    def __enter__(self) -> "ProgressContext":
        """Enter the context."""
        if self._interactive:
            self._bar = click.progressbar(
                length=100,
                label=self.label,
                file=sys.stderr,
                show_eta=True,
                show_percent=True,
            )
            self._bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and close the progress bar."""
        if self._bar is not None:
            self._bar.__exit__(exc_type, exc_val, exc_tb)

    def update(self, progress: float, status: str = "") -> None:
        """Update progress.

        Args:
            progress: Progress value from 0.0 to 1.0
            status: Optional status message
        """
        if self._interactive and self._bar is not None:
            new_pos = int(progress * 100)
            if new_pos > self._last_pos:
                self._bar.update(new_pos - self._last_pos)
                self._last_pos = new_pos
        elif self.show_status and status:
            click.echo(f"{self.label}: {status}", err=True)

    def get_callback(self) -> Callable[[float, str], None]:
        """Get a callback function for use with core functions.

        Returns:
            Callback that calls update()
        """
        return self.update
