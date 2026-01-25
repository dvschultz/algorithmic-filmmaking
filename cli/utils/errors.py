"""Exit codes and error handling for the CLI."""

import sys
from enum import IntEnum
from typing import NoReturn, Optional

import click


class ExitCode(IntEnum):
    """Standard exit codes for the CLI."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 2
    FILE_NOT_FOUND = 3
    DEPENDENCY_MISSING = 4
    NETWORK_ERROR = 5
    PERMISSION_ERROR = 6
    VALIDATION_ERROR = 7


def exit_with(code: ExitCode, message: Optional[str] = None) -> NoReturn:
    """Exit with a specific code and optional error message.

    Args:
        code: Exit code from ExitCode enum
        message: Optional error message to display on stderr
    """
    if message:
        click.echo(click.style(f"Error: {message}", fg="red"), err=True)
    sys.exit(code)


def handle_error(error: Exception) -> NoReturn:
    """Handle exceptions and exit with appropriate code.

    Maps common exception types to exit codes.

    Args:
        error: The exception to handle
    """
    if isinstance(error, FileNotFoundError):
        exit_with(ExitCode.FILE_NOT_FOUND, str(error))
    elif isinstance(error, PermissionError):
        exit_with(ExitCode.PERMISSION_ERROR, str(error))
    elif isinstance(error, ValueError):
        exit_with(ExitCode.VALIDATION_ERROR, str(error))
    else:
        exit_with(ExitCode.GENERAL_ERROR, str(error))
