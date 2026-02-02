"""Data model for rich cinematography analysis.

This module defines the CinematographyAnalysis dataclass which captures
film language metadata for clips: shot size, camera angle, movement,
composition, lighting, and derived emotional properties.
"""

from dataclasses import dataclass
from typing import Optional


# Valid values for each cinematography field
SHOT_SIZES = [
    "ELS",  # Extreme Long Shot
    "VLS",  # Very Long Shot
    "LS",   # Long Shot
    "MLS",  # Medium Long Shot
    "MS",   # Medium Shot
    "MCU",  # Medium Close-Up
    "CU",   # Close-Up
    "BCU",  # Big Close-Up
    "ECU",  # Extreme Close-Up
    "Insert",
]

CAMERA_ANGLES = [
    "low_angle",
    "eye_level",
    "high_angle",
    "dutch_angle",
    "birds_eye",
    "worms_eye",
]

ANGLE_EFFECTS = [
    "power",         # Low angle - subject dominance
    "neutral",       # Eye level - equal footing
    "vulnerability", # High angle - diminishment
    "disorientation", # Dutch angle - unease
    "omniscience",   # Bird's eye - pattern reveal
    "extreme_power", # Worm's eye - maximum dominance
]

CAMERA_MOVEMENTS = [
    "static",
    "pan",
    "tilt",
    "track",
    "handheld",
    "crane",
    "arc",
    "n/a",  # Not available (frame-only analysis)
]

MOVEMENT_DIRECTIONS = [
    "left",
    "right",
    "up",
    "down",
    "forward",
    "backward",
    "clockwise",
    "counterclockwise",
]

SUBJECT_POSITIONS = [
    "left_third",
    "center",
    "right_third",
    "distributed",
]

SPACING_VALUES = [
    "tight",
    "normal",
    "excessive",
    "n/a",
]

BALANCE_VALUES = [
    "balanced",
    "left_heavy",
    "right_heavy",
    "symmetrical",
]

SUBJECT_COUNTS = [
    "empty",
    "single",
    "two_shot",
    "group",
]

SUBJECT_TYPES = [
    "person",
    "object",
    "landscape",
    "text",
    "mixed",
]

FOCUS_TYPES = [
    "deep",
    "shallow",
    "rack_focus",
]

BACKGROUND_TYPES = [
    "blurred",
    "sharp",
    "cluttered",
    "plain",
]

LIGHTING_STYLES = [
    "high_key",
    "low_key",
    "natural",
    "dramatic",
]

LIGHTING_DIRECTIONS = [
    "front",
    "three_quarter",
    "side",
    "back",
    "below",
]

INTENSITY_LEVELS = [
    "low",
    "medium",
    "high",
]

PACING_VALUES = [
    "fast",
    "medium",
    "slow",
]


@dataclass
class CinematographyAnalysis:
    """Rich cinematography analysis capturing film language metadata.

    This dataclass captures multiple dimensions of cinematography from a clip:
    - Shot size (granular classification from ELS to ECU)
    - Camera angle and its narrative effect
    - Camera movement (when video analysis is available)
    - Composition (subject position, balance, headroom/lead room)
    - Subject analysis (count, type)
    - Focus and depth characteristics
    - Lighting style and direction
    - Derived emotional properties

    Attributes:
        shot_size: Granular shot classification (ELS, VLS, LS, MLS, MS, MCU, CU, BCU, ECU, Insert)
        shot_size_confidence: Confidence score for shot size classification (0.0-1.0)
        camera_angle: Camera position relative to subject
        angle_effect: Narrative effect of the camera angle
        camera_movement: Type of camera movement (n/a for frame-only analysis)
        movement_direction: Direction of camera movement if applicable
        subject_position: Where the main subject is positioned in frame
        headroom: Amount of space above subject's head
        lead_room: Amount of space in direction of gaze/movement
        balance: Visual weight distribution in frame
        subject_count: Number of subjects in frame
        subject_type: Primary subject type
        focus_type: Depth of field characteristics
        background_type: Background appearance
        lighting_style: Overall lighting approach
        lighting_direction: Primary light source direction
        emotional_intensity: Derived emotional impact level
        suggested_pacing: Suggested edit duration based on complexity
        analysis_model: Model that generated this analysis
        analysis_mode: Whether analyzed from video or single frame
    """

    # Shot Size
    shot_size: str = "MS"  # Default to medium shot
    shot_size_confidence: float = 0.0

    # Camera Angle
    camera_angle: str = "eye_level"
    angle_effect: str = "neutral"

    # Camera Movement (requires video analysis)
    camera_movement: str = "n/a"
    movement_direction: Optional[str] = None

    # Composition
    subject_position: str = "center"
    headroom: str = "normal"
    lead_room: str = "n/a"
    balance: str = "balanced"

    # Subject Analysis
    subject_count: str = "single"
    subject_type: str = "person"

    # Focus & Depth
    focus_type: str = "deep"
    background_type: str = "sharp"

    # Lighting
    lighting_style: str = "natural"
    lighting_direction: str = "front"

    # Derived Properties
    emotional_intensity: str = "medium"
    suggested_pacing: str = "medium"

    # Metadata
    analysis_model: Optional[str] = None
    analysis_mode: str = "frame"  # "frame" or "video"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export.

        Only includes fields with non-default or meaningful values.
        """
        data = {
            "shot_size": self.shot_size,
            "shot_size_confidence": self.shot_size_confidence,
            "camera_angle": self.camera_angle,
            "angle_effect": self.angle_effect,
            "subject_position": self.subject_position,
            "headroom": self.headroom,
            "lead_room": self.lead_room,
            "balance": self.balance,
            "subject_count": self.subject_count,
            "subject_type": self.subject_type,
            "focus_type": self.focus_type,
            "background_type": self.background_type,
            "lighting_style": self.lighting_style,
            "lighting_direction": self.lighting_direction,
            "emotional_intensity": self.emotional_intensity,
            "suggested_pacing": self.suggested_pacing,
            "analysis_mode": self.analysis_mode,
        }

        # Camera movement (only if not n/a or has direction)
        if self.camera_movement != "n/a":
            data["camera_movement"] = self.camera_movement
        if self.movement_direction:
            data["movement_direction"] = self.movement_direction

        # Analysis model (only if set)
        if self.analysis_model:
            data["analysis_model"] = self.analysis_model

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CinematographyAnalysis":
        """Deserialize from dictionary."""
        return cls(
            shot_size=data.get("shot_size", "MS"),
            shot_size_confidence=data.get("shot_size_confidence", 0.0),
            camera_angle=data.get("camera_angle", "eye_level"),
            angle_effect=data.get("angle_effect", "neutral"),
            camera_movement=data.get("camera_movement", "n/a"),
            movement_direction=data.get("movement_direction"),
            subject_position=data.get("subject_position", "center"),
            headroom=data.get("headroom", "normal"),
            lead_room=data.get("lead_room", "n/a"),
            balance=data.get("balance", "balanced"),
            subject_count=data.get("subject_count", "single"),
            subject_type=data.get("subject_type", "person"),
            focus_type=data.get("focus_type", "deep"),
            background_type=data.get("background_type", "sharp"),
            lighting_style=data.get("lighting_style", "natural"),
            lighting_direction=data.get("lighting_direction", "front"),
            emotional_intensity=data.get("emotional_intensity", "medium"),
            suggested_pacing=data.get("suggested_pacing", "medium"),
            analysis_model=data.get("analysis_model"),
            analysis_mode=data.get("analysis_mode", "frame"),
        )

    def get_display_badges(self) -> list[str]:
        """Get compact badges for UI display.

        Returns a list of the most informative attributes for display
        as compact tags on clip cards. Badge text is formatted to match
        film terminology standards and enable glossary tooltip lookup.
        """
        badges = []

        # Shot size is always informative (use standard abbreviation)
        badges.append(self.shot_size)

        # Camera angle (only if not eye level - that's the default/neutral)
        # Keep underscore format for glossary lookup compatibility
        if self.camera_angle != "eye_level":
            badges.append(self.camera_angle)

        # Lighting (only if notable)
        # Keep underscore format for glossary lookup
        if self.lighting_style in ("low_key", "dramatic"):
            badges.append(self.lighting_style)

        # Focus (only if shallow - that's notable)
        if self.focus_type == "shallow":
            badges.append("shallow")

        # Camera movement (if detected)
        if self.camera_movement not in ("n/a", "static"):
            badges.append(self.camera_movement)

        # Subject count (only if not single)
        # Keep underscore format for glossary lookup
        if self.subject_count in ("two_shot", "group"):
            badges.append(self.subject_count)

        return badges

    def get_display_badges_formatted(self) -> list[tuple[str, str]]:
        """Get badges with both raw key and formatted display text.

        Returns a list of (key, display_text) tuples where key is for
        glossary lookup and display_text is for UI rendering.
        """
        badges = []

        # Shot size
        badges.append((self.shot_size, self.shot_size))

        # Camera angle
        if self.camera_angle != "eye_level":
            display = self.camera_angle.replace("_angle", "").replace("_", " ")
            badges.append((self.camera_angle, display))

        # Lighting
        if self.lighting_style in ("low_key", "dramatic"):
            display = self.lighting_style.replace("_", " ")
            badges.append((self.lighting_style, display))

        # Focus
        if self.focus_type == "shallow":
            badges.append(("shallow", "shallow focus"))

        # Camera movement
        if self.camera_movement not in ("n/a", "static"):
            badges.append((self.camera_movement, self.camera_movement))

        # Subject count
        if self.subject_count in ("two_shot", "group"):
            display = self.subject_count.replace("_", " ")
            badges.append((self.subject_count, display))

        return badges

    def get_simple_shot_type(self) -> str:
        """Map granular shot size to simple 5-category shot type.

        This maintains compatibility with the existing shot_type field
        which uses: wide, full, medium, close-up, ECU

        Returns:
            Simple shot type string
        """
        mapping = {
            "ELS": "wide",
            "VLS": "wide",
            "LS": "full",
            "MLS": "full",
            "MS": "medium",
            "MCU": "medium",
            "CU": "close-up",
            "BCU": "close-up",
            "ECU": "ECU",
            "Insert": "close-up",
        }
        return mapping.get(self.shot_size, "medium")
