"""Shared ordering, cancellation and progress for multi-analysis requests."""

from collections.abc import Callable, Sequence
from threading import Event

from core.analysis_operations import OPERATIONS_BY_KEY

Progress = Callable[[float, str], None]


def run_analysis_plan(
    operations: Sequence[str] | None,
    execute: Callable[[str, Progress | None], dict],
    *,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> dict:
    steps = tuple(operations or ())
    if not steps:
        return {
            "success": False,
            "error": {"code": "no_operations", "message": "operations is required"},
        }
    invalid = [op for op in steps if op not in OPERATIONS_BY_KEY]
    if invalid:
        return {
            "success": False,
            "error": {"code": "invalid_operations", "operations": invalid},
        }
    results = {}
    for index, op in enumerate(steps):
        if cancel is not None and cancel.is_set():
            break
        if progress is not None:
            progress(index / len(steps), f"Starting {op} ({index + 1}/{len(steps)})")
        if cancel is not None and cancel.is_set():
            break

        def report(value: float, message: str, *, index=index, op=op) -> None:
            if progress is None:
                return
            try:
                normalized = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                normalized = 0.0
            progress(
                (index + normalized) / len(steps), f"{op}: {message}" if message else op
            )

        result = execute(op, report if progress is not None else None)
        results[op] = result
        if result.get("success") is False:
            return {"success": False, "error": result.get("error"), "result": results}
    if progress is not None:
        progress(1.0, f"Done: {len(results)} operation(s)")
    return {"success": True, "result": {"operations": results}}
