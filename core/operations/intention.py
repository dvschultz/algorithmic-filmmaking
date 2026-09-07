"""Ordered prerequisites for an intention-first import and sequence workflow."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.clip import Clip


ANALYSIS_REQUIREMENTS = {
    "color": ("colors",),
    "shot_type": ("shot_type",),
    "storyteller": ("descriptions",),
}


@dataclass(frozen=True)
class IntentionStep:
    name: str
    prerequisites: tuple[str, ...] = ()


class IntentionPlan:
    """One run's phase order; an unsuccessful prerequisite prevents advancement."""

    def __init__(self, algorithm: str, *, downloads: bool) -> None:
        self.analysis_requirements = ANALYSIS_REQUIREMENTS.get(algorithm, ())
        names = ["downloading"] if downloads else []
        names.extend(("detecting", "thumbnails"))
        if self.analysis_requirements:
            names.append("analyzing")
        names.append("building")
        self.steps = tuple(
            IntentionStep(name, (names[index - 1],) if index else ())
            for index, name in enumerate(names)
        )
        self.completed: list[str] = []
        self.cancelled = False
        self.error: str | None = None

    @property
    def active(self) -> str | None:
        if self.cancelled or self.error or len(self.completed) == len(self.steps):
            return None
        return self.steps[len(self.completed)].name

    def finish(self, step: str) -> bool:
        if self.active != step:
            return False
        if any(
            p not in self.completed
            for p in self.steps[len(self.completed)].prerequisites
        ):
            return False
        self.completed.append(step)
        return True

    def missing_analysis(self, clips: list["Clip"]) -> dict[str, tuple[str, ...]]:
        fields = {
            "colors": "dominant_colors",
            "shot_type": "shot_type",
            "descriptions": "description",
        }
        return {
            clip.id: missing
            for clip in clips
            if (
                missing := tuple(
                    requirement
                    for requirement in self.analysis_requirements
                    if getattr(clip, fields[requirement]) is None
                )
            )
        }
