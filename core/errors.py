"""Shared exceptions used across multiple analysis modules."""


class ModelDownloadError(RuntimeError):
    """Raised when a model download or first-load fails.

    Used across analysis modules (embeddings, detection, stem separation, etc.)
    to provide consistent error classification in workers.
    """

    pass
