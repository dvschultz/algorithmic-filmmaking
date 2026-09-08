"""Small immutable contracts for the color-operation pilot.

Keep this module free of model, GUI and media-runtime imports.
"""

from dataclasses import dataclass
from typing import Literal

Palette = tuple[tuple[int, int, int], ...]
OutcomeStatus = Literal["succeeded", "skipped", "failed", "unprocessed"]


@dataclass(frozen=True)
class ColorOutcome:
    target_id: str
    status: OutcomeStatus
    colors: Palette = ()
    code: str | None = None
    message: str | None = None
    record_json: str | None = None


@dataclass(frozen=True)
class ColorResult:
    request_id: str
    outcomes: tuple[ColorOutcome, ...]
