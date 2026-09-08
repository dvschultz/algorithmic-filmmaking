"""Job recovery errors shared by storage and project publication."""


class StaleJobResult(RuntimeError):
    """Stored output cannot safely be reused or applied to the current project."""
