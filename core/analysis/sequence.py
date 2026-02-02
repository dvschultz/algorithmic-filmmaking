"""Sequence-level analysis for pacing, continuity, and visual consistency.

This module analyzes relationships between clips in a sequence,
providing metrics that require multi-clip context (unlike single-clip analysis).

Key features:
- Pacing analysis: Shot duration statistics and classification
- Continuity warnings: Detect potential jump cuts and similar consecutive shots
- Visual consistency: Color palette and lighting variance across clips
- Genre comparison: Compare pacing to typical genre norms
"""

import logging
import statistics
from collections import Counter
from typing import Optional, TYPE_CHECKING

from models.sequence_analysis import (
    SequenceAnalysis,
    PacingAnalysis,
    ContinuityWarning,
    VisualConsistency,
    PACING_THRESHOLDS,
)

if TYPE_CHECKING:
    from models.clip import Clip
    from models.sequence import Sequence, SequenceClip
    from core.project import Project

logger = logging.getLogger(__name__)


def analyze_sequence(
    sequence: "Sequence",
    project: "Project",
) -> SequenceAnalysis:
    """Perform complete analysis of a sequence.

    Args:
        sequence: The sequence to analyze
        project: Project containing source clips

    Returns:
        SequenceAnalysis with pacing, continuity, and consistency metrics
    """
    all_clips = sequence.get_all_clips()

    if not all_clips:
        return SequenceAnalysis()

    # Resolve source clips for metadata access
    resolved_clips = _resolve_source_clips(all_clips, project)

    # Analyze pacing
    pacing = analyze_pacing(all_clips, project)

    # Check continuity
    continuity_warnings = check_continuity(resolved_clips)

    # Analyze visual consistency
    visual_consistency = analyze_visual_consistency(resolved_clips)

    # Generate suggestions
    suggestions = generate_suggestions(pacing, continuity_warnings, visual_consistency)

    return SequenceAnalysis(
        pacing=pacing,
        continuity_warnings=continuity_warnings,
        visual_consistency=visual_consistency,
        suggestions=suggestions,
    )


def _resolve_source_clips(
    sequence_clips: list["SequenceClip"],
    project: "Project",
) -> list[tuple["SequenceClip", Optional["Clip"], Optional[float]]]:
    """Resolve sequence clips to their source clip data.

    Args:
        sequence_clips: List of SequenceClip objects
        project: Project containing clips and sources

    Returns:
        List of (sequence_clip, source_clip, fps) tuples
    """
    resolved = []
    for seq_clip in sequence_clips:
        source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
        source = project.sources_by_id.get(seq_clip.source_id)
        fps = source.fps if source else 30.0
        resolved.append((seq_clip, source_clip, fps))
    return resolved


def analyze_pacing(
    sequence_clips: list["SequenceClip"],
    project: "Project",
) -> PacingAnalysis:
    """Analyze shot duration and pacing patterns.

    Args:
        sequence_clips: Clips in sequence order
        project: Project for source FPS lookup

    Returns:
        PacingAnalysis with duration statistics
    """
    if not sequence_clips:
        return PacingAnalysis()

    # Calculate durations
    durations = []
    for seq_clip in sequence_clips:
        source = project.sources_by_id.get(seq_clip.source_id)
        fps = source.fps if source else 30.0
        duration = seq_clip.duration_seconds(fps)
        durations.append(duration)

    if not durations:
        return PacingAnalysis()

    total_duration = sum(durations)
    avg_duration = statistics.mean(durations)
    min_duration = min(durations)
    max_duration = max(durations)

    # Calculate variance (need at least 2 clips)
    variance = 0.0
    std_dev = 0.0
    if len(durations) >= 2:
        variance = statistics.variance(durations)
        std_dev = statistics.stdev(durations)

    # Classify pacing based on average duration
    classification = _classify_pacing(avg_duration)

    return PacingAnalysis(
        clip_count=len(durations),
        total_duration_seconds=total_duration,
        average_duration=avg_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        variance=variance,
        std_deviation=std_dev,
        classification=classification,
        duration_curve=durations,
    )


def _classify_pacing(avg_duration: float) -> str:
    """Classify pacing based on average shot duration.

    Args:
        avg_duration: Average shot duration in seconds

    Returns:
        Classification string
    """
    if avg_duration < PACING_THRESHOLDS["very_fast"]:
        return "very_fast"
    elif avg_duration < PACING_THRESHOLDS["fast"]:
        return "fast"
    elif avg_duration < PACING_THRESHOLDS["medium"]:
        return "medium"
    elif avg_duration < PACING_THRESHOLDS["slow"]:
        return "slow"
    else:
        return "very_slow"


def check_continuity(
    resolved_clips: list[tuple["SequenceClip", Optional["Clip"], Optional[float]]],
) -> list[ContinuityWarning]:
    """Check for potential continuity issues between consecutive clips.

    Uses heuristics based on available metadata:
    - Similar consecutive shots (potential jump cuts)
    - Large shot size jumps (e.g., ELS to ECU)
    - Same shot type in sequence (monotonous)

    Args:
        resolved_clips: List of (sequence_clip, source_clip, fps) tuples

    Returns:
        List of advisory ContinuityWarning objects
    """
    warnings = []

    if len(resolved_clips) < 2:
        return warnings

    # Define shot size order for jump detection
    shot_size_order = {
        "ELS": 0, "VLS": 1, "LS": 2, "MLS": 3, "MS": 4,
        "MCU": 5, "CU": 6, "BCU": 7, "ECU": 8, "Insert": 4,
    }

    for i in range(len(resolved_clips) - 1):
        seq_clip1, clip1, fps1 = resolved_clips[i]
        seq_clip2, clip2, fps2 = resolved_clips[i + 1]

        if not clip1 or not clip2:
            continue

        # Check for similar consecutive shots (potential jump cut)
        if _are_shots_similar(clip1, clip2):
            warnings.append(ContinuityWarning(
                warning_type="similar_consecutive",
                clip_pair=(seq_clip1.id, seq_clip2.id),
                severity="medium",
                explanation="These shots appear similar - may create a jump cut effect",
                can_be_intentional=True,
            ))

        # Check for large shot size jumps
        if clip1.cinematography and clip2.cinematography:
            size1 = clip1.cinematography.shot_size
            size2 = clip2.cinematography.shot_size

            if size1 in shot_size_order and size2 in shot_size_order:
                jump = abs(shot_size_order[size1] - shot_size_order[size2])
                if jump >= 4:  # Large jump (e.g., LS to CU or bigger)
                    warnings.append(ContinuityWarning(
                        warning_type="shot_size_jump",
                        clip_pair=(seq_clip1.id, seq_clip2.id),
                        severity="low",
                        explanation=f"Large shot size jump from {size1} to {size2}",
                        can_be_intentional=True,
                    ))

    return warnings


def _are_shots_similar(clip1: "Clip", clip2: "Clip") -> bool:
    """Determine if two clips are visually similar (potential jump cut).

    Uses available metadata to estimate similarity.
    A more sophisticated approach would use image embeddings.

    Args:
        clip1: First clip
        clip2: Second clip

    Returns:
        True if shots appear similar
    """
    # Same source is a strong indicator of potential jump cut
    if clip1.source_id == clip2.source_id:
        # Check if cinematography data matches
        if clip1.cinematography and clip2.cinematography:
            c1 = clip1.cinematography
            c2 = clip2.cinematography

            # Count matching attributes
            matches = 0
            total = 0

            if c1.shot_size and c2.shot_size:
                total += 1
                if c1.shot_size == c2.shot_size:
                    matches += 1

            if c1.camera_angle and c2.camera_angle:
                total += 1
                if c1.camera_angle == c2.camera_angle:
                    matches += 1

            if c1.subject_position and c2.subject_position:
                total += 1
                if c1.subject_position == c2.subject_position:
                    matches += 1

            # Similar if most attributes match
            if total > 0 and matches / total >= 0.7:
                return True

    return False


def analyze_visual_consistency(
    resolved_clips: list[tuple["SequenceClip", Optional["Clip"], Optional[float]]],
) -> VisualConsistency:
    """Analyze visual consistency across sequence clips.

    Measures:
    - Color palette similarity
    - Lighting consistency
    - Shot size variety

    Args:
        resolved_clips: List of (sequence_clip, source_clip, fps) tuples

    Returns:
        VisualConsistency metrics
    """
    if not resolved_clips:
        return VisualConsistency()

    # Collect shot sizes for variety analysis
    shot_sizes = []
    lighting_styles = []
    color_temperatures = []

    for _, clip, _ in resolved_clips:
        if clip and clip.cinematography:
            cine = clip.cinematography

            if cine.shot_size and cine.shot_size != "unknown":
                shot_sizes.append(cine.shot_size)

            if cine.lighting_style and cine.lighting_style != "unknown":
                lighting_styles.append(cine.lighting_style)

            if cine.color_temperature and cine.color_temperature != "unknown":
                color_temperatures.append(cine.color_temperature)

    # Shot size variety
    unique_shot_sizes = set(shot_sizes)
    shot_size_variety = len(unique_shot_sizes)

    # Most common shot size
    dominant_shot_size = "unknown"
    if shot_sizes:
        counter = Counter(shot_sizes)
        dominant_shot_size = counter.most_common(1)[0][0]

    # Lighting consistency (0-1, higher = more consistent)
    lighting_consistency = 0.0
    if lighting_styles:
        counter = Counter(lighting_styles)
        most_common_count = counter.most_common(1)[0][1]
        lighting_consistency = most_common_count / len(lighting_styles)

    # Color temperature shifts
    color_temp_shifts = 0
    if len(color_temperatures) >= 2:
        for i in range(len(color_temperatures) - 1):
            if color_temperatures[i] != color_temperatures[i + 1]:
                # Check for warm/cool transitions specifically
                warm_to_cool = (
                    color_temperatures[i] == "warm" and
                    color_temperatures[i + 1] == "cool"
                )
                cool_to_warm = (
                    color_temperatures[i] == "cool" and
                    color_temperatures[i + 1] == "warm"
                )
                if warm_to_cool or cool_to_warm:
                    color_temp_shifts += 1

    # Color consistency (simplified - based on dominant colors if available)
    color_consistency = _calculate_color_consistency(resolved_clips)

    return VisualConsistency(
        color_consistency=color_consistency,
        lighting_consistency=lighting_consistency,
        shot_size_variety=shot_size_variety,
        dominant_shot_size=dominant_shot_size,
        color_temperature_shifts=color_temp_shifts,
    )


def _calculate_color_consistency(
    resolved_clips: list[tuple["SequenceClip", Optional["Clip"], Optional[float]]],
) -> float:
    """Calculate color palette consistency across clips.

    Uses dominant colors from each clip to estimate visual coherence.

    Args:
        resolved_clips: List of (sequence_clip, source_clip, fps) tuples

    Returns:
        Consistency score from 0.0 (inconsistent) to 1.0 (consistent)
    """
    # Collect dominant hues
    hues = []
    for _, clip, _ in resolved_clips:
        if clip and clip.dominant_colors:
            # Get primary hue from first dominant color
            if len(clip.dominant_colors) > 0:
                r, g, b = clip.dominant_colors[0]
                hue = _rgb_to_hue(r, g, b)
                hues.append(hue)

    if len(hues) < 2:
        return 1.0  # Not enough data, assume consistent

    # Calculate hue variance (accounting for circular nature)
    # Use circular standard deviation
    mean_hue = _circular_mean(hues)

    # Calculate average angular deviation from mean
    total_deviation = 0.0
    for h in hues:
        deviation = abs(h - mean_hue)
        if deviation > 180:
            deviation = 360 - deviation
        total_deviation += deviation

    avg_deviation = total_deviation / len(hues)

    # Convert to 0-1 consistency score (lower deviation = higher consistency)
    # Max deviation would be 180 (opposite hues)
    consistency = max(0.0, 1.0 - (avg_deviation / 90.0))

    return consistency


def _rgb_to_hue(r: int, g: int, b: int) -> float:
    """Convert RGB to hue (0-360).

    Args:
        r, g, b: RGB values 0-255

    Returns:
        Hue value 0-360
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        return 0.0

    if max_c == r:
        hue = ((g - b) / delta) % 6
    elif max_c == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4

    return hue * 60


def _circular_mean(angles: list[float]) -> float:
    """Calculate circular mean of angles.

    Args:
        angles: List of angles in degrees (0-360)

    Returns:
        Circular mean in degrees
    """
    import math

    if not angles:
        return 0.0

    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)

    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))
    if mean_angle < 0:
        mean_angle += 360

    return mean_angle


def generate_suggestions(
    pacing: PacingAnalysis,
    warnings: list[ContinuityWarning],
    consistency: VisualConsistency,
) -> list[str]:
    """Generate advisory suggestions based on analysis.

    These are soft recommendations - not prescriptive rules.

    Args:
        pacing: Pacing analysis results
        warnings: Continuity warnings
        consistency: Visual consistency metrics

    Returns:
        List of suggestion strings
    """
    suggestions = []

    # Pacing suggestions
    if pacing.clip_count > 0:
        if pacing.average_duration > 12:
            suggestions.append(
                "Consider shorter shots to increase visual energy"
            )

        if pacing.variance < 0.5 and pacing.clip_count > 5:
            suggestions.append(
                "Shot lengths are very uniform - varying durations could add visual interest"
            )

        if pacing.std_deviation > pacing.average_duration:
            suggestions.append(
                "High variation in shot lengths - may feel uneven (or intentionally dynamic)"
            )

    # Continuity suggestions
    jump_cuts = [w for w in warnings if w.warning_type == "similar_consecutive"]
    if len(jump_cuts) > 2:
        suggestions.append(
            f"Found {len(jump_cuts)} potential jump cuts - "
            "consider cutaways or longer transitions if unintentional"
        )

    # Visual consistency suggestions
    if consistency.shot_size_variety == 1 and pacing.clip_count > 3:
        suggestions.append(
            f"All shots are {consistency.dominant_shot_size} - "
            "mixing shot sizes adds visual variety"
        )

    if consistency.lighting_consistency < 0.5 and pacing.clip_count > 3:
        suggestions.append(
            "Lighting varies significantly between shots - "
            "may feel inconsistent (or create intentional contrast)"
        )

    if consistency.color_temperature_shifts > pacing.clip_count // 3:
        suggestions.append(
            "Frequent warm/cool color shifts - consider color grading for cohesion"
        )

    return suggestions


def get_pacing_curve(
    sequence: "Sequence",
    project: "Project",
) -> list[dict]:
    """Get shot duration data for visualization.

    Args:
        sequence: Sequence to analyze
        project: Project for metadata

    Returns:
        List of dicts with position, duration, clip_id for charting
    """
    all_clips = sequence.get_all_clips()
    curve = []

    for i, seq_clip in enumerate(all_clips):
        source = project.sources_by_id.get(seq_clip.source_id)
        fps = source.fps if source else 30.0
        duration = seq_clip.duration_seconds(fps)

        curve.append({
            "position": i,
            "duration": round(duration, 2),
            "clip_id": seq_clip.id,
        })

    return curve
