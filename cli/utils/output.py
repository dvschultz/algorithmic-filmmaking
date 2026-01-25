"""Output formatting utilities for the CLI."""

import json
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import click


def _serialize_value(value: Any) -> Any:
    """Serialize a value to JSON-compatible format.

    Args:
        value: Any value to serialize

    Returns:
        JSON-serializable value
    """
    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, timedelta):
        return value.total_seconds()
    elif is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize_value(v) for k, v in asdict(value).items()}
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


def output_result(data: Any, as_json: bool = False) -> None:
    """Output data in requested format.

    Args:
        data: Data to output (dict, list, dataclass, or string)
        as_json: If True, output as JSON; otherwise human-readable text
    """
    if as_json:
        serialized = _serialize_value(data)
        click.echo(json.dumps(serialized, indent=2, default=str))
    else:
        if is_dataclass(data) and not isinstance(data, type):
            data = asdict(data)

        if isinstance(data, dict):
            for key, value in data.items():
                # Format key nicely
                display_key = key.replace("_", " ").title()
                click.echo(f"{display_key}: {_format_value(value)}")
        elif isinstance(data, list):
            for item in data:
                click.echo(_format_value(item))
        else:
            click.echo(str(data))


def _format_value(value: Any) -> str:
    """Format a value for human-readable display.

    Args:
        value: Value to format

    Returns:
        Formatted string
    """
    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, timedelta):
        total_seconds = int(value.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    elif isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, list):
        if len(value) <= 3:
            return ", ".join(str(v) for v in value)
        return f"{len(value)} items"
    elif isinstance(value, dict):
        return f"{len(value)} entries"
    return str(value)


def output_table(
    headers: list[str],
    rows: list[list[Any]],
    as_json: bool = False,
) -> None:
    """Output data as a table.

    Args:
        headers: Column headers
        rows: List of rows, each row is a list of values
        as_json: If True, output as JSON array of objects
    """
    if as_json:
        data = [dict(zip(headers, row)) for row in rows]
        output_result(data, as_json=True)
    else:
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        click.echo(header_line)
        click.echo("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            click.echo(row_line)


def output_success(message: str) -> None:
    """Output a success message in green.

    Args:
        message: Message to display
    """
    click.echo(click.style(message, fg="green"))


def output_warning(message: str) -> None:
    """Output a warning message in yellow.

    Args:
        message: Message to display
    """
    click.echo(click.style(f"Warning: {message}", fg="yellow"), err=True)


def output_info(message: str) -> None:
    """Output an info message.

    Args:
        message: Message to display
    """
    click.echo(message, err=True)
