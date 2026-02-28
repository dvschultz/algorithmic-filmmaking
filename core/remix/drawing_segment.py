"""DrawingSegment — shared intermediate data structure for Signature Style.

Both parametric and VLM modes produce DrawingSegment[], which is then
consumed by the clip matching algorithm.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DrawingSegment:
    """A segment of a drawing interpreted as a sequencing guide.

    Produced by either parametric sampling or VLM interpretation.
    Consumed by the clip matcher to select and arrange clips.
    """

    x_start: int  # Pixel position on canvas (left edge)
    x_end: int  # Pixel position on canvas (right edge)
    target_duration_seconds: float  # Derived from output duration + segment proportion
    target_pacing: float  # 0.0 (slow/long holds) to 1.0 (fast/short cuts)
    target_color: Optional[tuple[int, int, int]] = None  # RGB, None if B&W/no color
    is_bw: bool = False  # True if this region is B&W

    # VLM-only fields (None in parametric mode):
    shot_type: Optional[str] = None
    energy: Optional[float] = None  # 0.0-1.0
    brightness: Optional[float] = None  # 0.0-1.0
    color_mood: Optional[str] = None  # "warm", "cool", "neutral", etc.

    @property
    def width(self) -> int:
        """Width of this segment in pixels."""
        return self.x_end - self.x_start

    @property
    def proportion(self) -> float:
        """Proportion of the canvas this segment occupies (requires canvas width context)."""
        return self.width  # Caller divides by canvas_width
