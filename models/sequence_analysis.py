"""Data models for sequence-level analysis.

These models represent analysis results computed across multiple clips
in a sequence, including pacing metrics, continuity checks, and
visual consistency measures.

Note: SequenceAnalysis is NOT persisted to project files. It's computed
on-demand and cached in memory, invalidated when sequence changes.
"""

from dataclasses import dataclass, field
from typing import Optional


# Genre-specific pacing norms for comparison
GENRE_PACING_NORMS = {
    "action": {
        "avg_shot_duration": 2.5,
        "variance": "high",
        "description": "Fast cuts, high energy"
    },
    "drama": {
        "avg_shot_duration": 5.0,
        "variance": "medium",
        "description": "Measured pacing, emotional beats"
    },
    "documentary": {
        "avg_shot_duration": 8.0,
        "variance": "high",
        "description": "Longer takes, varied pacing"
    },
    "music_video": {
        "avg_shot_duration": 1.5,
        "variance": "very_high",
        "description": "Rapid cuts, rhythm-driven"
    },
    "commercial": {
        "avg_shot_duration": 2.0,
        "variance": "medium",
        "description": "Quick, attention-grabbing"
    },
    "art_film": {
        "avg_shot_duration": 15.0,
        "variance": "high",
        "description": "Long takes, contemplative"
    },
}

# Pacing classification thresholds (in seconds)
PACING_THRESHOLDS = {
    "very_fast": 1.5,   # < 1.5s average
    "fast": 3.0,        # 1.5-3.0s
    "medium": 6.0,      # 3.0-6.0s
    "slow": 10.0,       # 6.0-10.0s
    # > 10s = very_slow
}


@dataclass
class PacingAnalysis:
    """Shot duration and rhythm analysis for a sequence.

    Attributes:
        clip_count: Number of clips analyzed
        total_duration_seconds: Total sequence duration
        average_duration: Mean shot duration in seconds
        min_duration: Shortest shot duration
        max_duration: Longest shot duration
        variance: Statistical variance in durations
        std_deviation: Standard deviation
        classification: Pacing category (very_fast, fast, medium, slow, very_slow)
        duration_curve: List of durations in sequence order (for visualization)
    """
    clip_count: int = 0
    total_duration_seconds: float = 0.0
    average_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    variance: float = 0.0
    std_deviation: float = 0.0
    classification: str = "unknown"
    duration_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clip_count": self.clip_count,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "average_duration": round(self.average_duration, 2),
            "min_duration": round(self.min_duration, 2),
            "max_duration": round(self.max_duration, 2),
            "variance": round(self.variance, 3),
            "std_deviation": round(self.std_deviation, 3),
            "classification": self.classification,
            "duration_curve": [round(d, 2) for d in self.duration_curve],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PacingAnalysis":
        if data is None:
            return cls()
        return cls(
            clip_count=data.get("clip_count", 0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            average_duration=data.get("average_duration", 0.0),
            min_duration=data.get("min_duration", 0.0),
            max_duration=data.get("max_duration", 0.0),
            variance=data.get("variance", 0.0),
            std_deviation=data.get("std_deviation", 0.0),
            classification=data.get("classification", "unknown"),
            duration_curve=data.get("duration_curve", []),
        )


@dataclass
class ContinuityWarning:
    """Advisory warning about a potential continuity issue.

    These are soft warnings - most rule-breaking is intentional and valid.

    Attributes:
        warning_type: Type of issue (jump_cut, similar_consecutive, shot_size_jump)
        clip_pair: Tuple of (clip_id_1, clip_id_2) involved
        severity: low, medium, or high
        explanation: Human-readable description of the issue
        can_be_intentional: Whether this could be a valid artistic choice
    """
    warning_type: str = ""
    clip_pair: tuple[str, str] = field(default_factory=lambda: ("", ""))
    severity: str = "low"
    explanation: str = ""
    can_be_intentional: bool = True

    def to_dict(self) -> dict:
        return {
            "warning_type": self.warning_type,
            "clip_pair": list(self.clip_pair),
            "severity": self.severity,
            "explanation": self.explanation,
            "can_be_intentional": self.can_be_intentional,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContinuityWarning":
        if data is None:
            return cls()
        # Validate and coerce clip_pair to exactly 2 elements
        raw_pair = data.get("clip_pair", ["", ""])
        if len(raw_pair) >= 2:
            clip_pair = (raw_pair[0], raw_pair[1])
        else:
            clip_pair = ("", "")
        return cls(
            warning_type=data.get("warning_type", ""),
            clip_pair=clip_pair,
            severity=data.get("severity", "low"),
            explanation=data.get("explanation", ""),
            can_be_intentional=data.get("can_be_intentional", True),
        )


@dataclass
class VisualConsistency:
    """Visual consistency metrics across sequence clips.

    Attributes:
        color_consistency: 0-1 score of how similar color palettes are
        lighting_consistency: 0-1 score of how similar lighting is
        shot_size_variety: Count of distinct shot sizes used
        dominant_shot_size: Most frequently used shot size
        color_temperature_shifts: Number of warm/cool transitions
    """
    color_consistency: float = 0.0
    lighting_consistency: float = 0.0
    shot_size_variety: int = 0
    dominant_shot_size: str = "unknown"
    color_temperature_shifts: int = 0

    def to_dict(self) -> dict:
        return {
            "color_consistency": round(self.color_consistency, 3),
            "lighting_consistency": round(self.lighting_consistency, 3),
            "shot_size_variety": self.shot_size_variety,
            "dominant_shot_size": self.dominant_shot_size,
            "color_temperature_shifts": self.color_temperature_shifts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VisualConsistency":
        if data is None:
            return cls()
        return cls(
            color_consistency=data.get("color_consistency", 0.0),
            lighting_consistency=data.get("lighting_consistency", 0.0),
            shot_size_variety=data.get("shot_size_variety", 0),
            dominant_shot_size=data.get("dominant_shot_size", "unknown"),
            color_temperature_shifts=data.get("color_temperature_shifts", 0),
        )


@dataclass
class GenreComparison:
    """Comparison of sequence pacing to genre norms.

    Attributes:
        genre: The genre being compared to
        genre_norm_duration: Expected average shot duration for genre
        actual_duration: Actual average duration in sequence
        difference_percent: How much faster/slower than norm
        assessment: Brief assessment text
    """
    genre: str = ""
    genre_norm_duration: float = 0.0
    actual_duration: float = 0.0
    difference_percent: float = 0.0
    assessment: str = ""

    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "genre_norm_duration": round(self.genre_norm_duration, 2),
            "actual_duration": round(self.actual_duration, 2),
            "difference_percent": round(self.difference_percent, 1),
            "assessment": self.assessment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenreComparison":
        if data is None:
            return cls()
        return cls(
            genre=data.get("genre", ""),
            genre_norm_duration=data.get("genre_norm_duration", 0.0),
            actual_duration=data.get("actual_duration", 0.0),
            difference_percent=data.get("difference_percent", 0.0),
            assessment=data.get("assessment", ""),
        )


@dataclass
class SequenceAnalysis:
    """Complete analysis of a sequence.

    This is the top-level analysis result containing all metrics.
    NOT persisted to project files - computed on demand.

    Attributes:
        pacing: Shot duration and rhythm analysis
        continuity_warnings: List of potential continuity issues
        visual_consistency: Color/lighting/shot variety metrics
        suggestions: Advisory improvement suggestions
    """
    pacing: PacingAnalysis = field(default_factory=PacingAnalysis)
    continuity_warnings: list[ContinuityWarning] = field(default_factory=list)
    visual_consistency: VisualConsistency = field(default_factory=VisualConsistency)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pacing": self.pacing.to_dict(),
            "continuity_warnings": [w.to_dict() for w in self.continuity_warnings],
            "visual_consistency": self.visual_consistency.to_dict(),
            "suggestions": self.suggestions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceAnalysis":
        if data is None:
            return cls()
        return cls(
            pacing=PacingAnalysis.from_dict(data.get("pacing")),
            continuity_warnings=[
                ContinuityWarning.from_dict(w)
                for w in data.get("continuity_warnings", [])
            ],
            visual_consistency=VisualConsistency.from_dict(
                data.get("visual_consistency")
            ),
            suggestions=data.get("suggestions", []),
        )

    def compare_to_genre(self, genre: Optional[str]) -> Optional[GenreComparison]:
        """Compare this sequence's pacing to a genre norm.

        Args:
            genre: Genre name (action, drama, documentary, music_video, etc.)

        Returns:
            GenreComparison or None if genre not found or None
        """
        if not genre:
            return None
        norm = GENRE_PACING_NORMS.get(genre.lower())
        if not norm:
            return None

        norm_duration = norm["avg_shot_duration"]
        actual = self.pacing.average_duration

        if norm_duration > 0:
            diff_percent = ((actual - norm_duration) / norm_duration) * 100
        else:
            diff_percent = 0.0

        # Generate assessment
        if abs(diff_percent) < 15:
            assessment = f"Pacing is typical for {genre}"
        elif diff_percent > 0:
            assessment = f"Pacing is {abs(diff_percent):.0f}% slower than typical {genre}"
        else:
            assessment = f"Pacing is {abs(diff_percent):.0f}% faster than typical {genre}"

        return GenreComparison(
            genre=genre,
            genre_norm_duration=norm_duration,
            actual_duration=actual,
            difference_percent=diff_percent,
            assessment=assessment,
        )
