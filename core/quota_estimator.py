"""Heuristic ChatGPT Plus quota estimator for pre-flight batch warnings.

R12 requires Scene Ripper to warn a subscription-mode user *before*
starting a batch operation that's likely to exhaust the ChatGPT Plus
quota for the current 3-hour window. The Codex backend doesn't (today)
expose remaining quota in any header, so this is heuristic-only.

R-Q3 from the doc-review: default to **wide tolerance** for v1 so we
don't train users to dismiss the warning before the estimator is
calibrated. Calibration is a follow-up step — once a week of
subscription-mode usage data exists, the per-operation thresholds in
``_OPERATION_HEURISTICS`` get tightened in-place.

Each operation key maps to a small set of numeric heuristics:

  - ``messages_per_unit``  — roughly how many backend round-trips one
    "unit" of work consumes (1 message per clip for describe; 1 per
    clip per query for custom_query)
  - ``tokens_per_unit``    — input + expected output tokens per unit
  - ``warn_threshold_units`` — soft cap, wide for v1, units above this
    trigger a pre-flight warning

The exact warning UX (R-Q1 — "Switch to API key" is current-operation-only,
not a persisted mode change) lives in the worker that consumes this
estimator. The estimator itself returns a small dict; callers branch
on ``should_warn`` to decide whether to show the dialog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Per-operation heuristic constants. Tunable as a single module-level
# dict so calibration changes are a one-line edit. Wide v1 thresholds:
# values intentionally high to avoid false-positive fatigue before real
# Plus-quota data is collected (R-Q3).
_OPERATION_HEURISTICS: dict[str, dict[str, int]] = {
    "describe": {
        "messages_per_unit": 1,
        "tokens_per_unit": 500,
        "warn_threshold_units": 300,
    },
    "classify": {
        "messages_per_unit": 1,
        "tokens_per_unit": 200,
        "warn_threshold_units": 500,
    },
    "cinematography": {
        "messages_per_unit": 1,
        "tokens_per_unit": 800,
        "warn_threshold_units": 200,
    },
    "custom_query": {
        "messages_per_unit": 1,
        "tokens_per_unit": 300,
        "warn_threshold_units": 250,
    },
    "ocr": {
        "messages_per_unit": 1,
        "tokens_per_unit": 200,
        "warn_threshold_units": 500,
    },
}


@dataclass(frozen=True)
class BatchLoadEstimate:
    """Estimate returned by :func:`estimate_batch_load`.

    ``should_warn`` is the simple boolean callers branch on. The
    individual fields are exposed for debugging / log output but are
    not load-bearing — they're heuristic until calibration lands.
    """

    operation: str
    count: int
    estimated_messages: int
    estimated_tokens: int
    warn_threshold_units: int
    should_warn: bool


def estimate_batch_load(operation: str, count: int) -> BatchLoadEstimate:
    """Return a heuristic estimate for an N-unit batch of ``operation``.

    Unknown operations return a zero-everywhere estimate with
    ``should_warn=False`` so the caller's warning path is silent rather
    than firing on every operation it doesn't recognize.
    """
    heuristics = _OPERATION_HEURISTICS.get(operation)
    if heuristics is None or count <= 0:
        return BatchLoadEstimate(
            operation=operation,
            count=max(0, count),
            estimated_messages=0,
            estimated_tokens=0,
            warn_threshold_units=0,
            should_warn=False,
        )

    messages = heuristics["messages_per_unit"] * count
    tokens = heuristics["tokens_per_unit"] * count
    threshold = heuristics["warn_threshold_units"]
    return BatchLoadEstimate(
        operation=operation,
        count=count,
        estimated_messages=messages,
        estimated_tokens=tokens,
        warn_threshold_units=threshold,
        should_warn=count > threshold,
    )


def adjust_threshold(operation: str, new_threshold_units: int) -> None:
    """Tunable threshold update — used by calibration follow-up.

    Mutates the module-level dict in place; subsequent calls to
    :func:`estimate_batch_load` pick up the new value. Pre-existing
    BatchLoadEstimate instances are immutable so they're unaffected.
    """
    if operation in _OPERATION_HEURISTICS:
        _OPERATION_HEURISTICS[operation]["warn_threshold_units"] = int(
            new_threshold_units
        )


__all__ = [
    "BatchLoadEstimate",
    "adjust_threshold",
    "estimate_batch_load",
]
