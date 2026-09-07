"""Provider failure classification without GUI or inference dependencies."""


def is_transient_provider_error(message: str) -> bool:
    """Identify the transient failures retried by visual-analysis workers."""
    normalized = message.lower()
    return any(
        fragment in normalized
        for fragment in (
            "429",
            "rate limit",
            "too many requests",
            "500",
            "502",
            "503",
            "504",
            "internalservererror",
            "internal error",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "connection",
            "network",
        )
    )
