"""Algorithmic remix algorithms for video clip sequencing."""

import logging
import random
from typing import List, Tuple, Any, Optional
from core.remix.shuffle import constrained_shuffle
from core.analysis.color import get_primary_hue, rgb_to_hsv
from core.analysis.shots import SHOT_TYPES
from core.remix.audio_sync import (
    AlignmentSuggestion,
    suggest_beat_aligned_cuts,
    align_times_to_beats,
    calculate_beat_intervals,
    get_beats_in_range,
    estimate_clip_count_for_duration,
    generate_cut_times_from_beats,
)

__all__ = [
    "constrained_shuffle",
    "generate_sequence",
    "assign_random_transforms",
    # Audio sync
    "AlignmentSuggestion",
    "suggest_beat_aligned_cuts",
    "align_times_to_beats",
    "calculate_beat_intervals",
    "get_beats_in_range",
    "estimate_clip_count_for_duration",
    "generate_cut_times_from_beats",
]

logger = logging.getLogger(__name__)

# Shot type order for sorting (wide to close)
SHOT_TYPE_ORDER = {shot: i for i, shot in enumerate(SHOT_TYPES)}

# 10-class cinematography shot size → proximity score
_SHOT_SIZE_PROXIMITY = {
    "ELS": 1.0,
    "VLS": 2.0,
    "LS": 3.0,
    "MLS": 4.0,
    "MS": 5.0,
    "MCU": 6.0,
    "CU": 7.0,
    "BCU": 8.0,
    "ECU": 9.0,
    "Insert": 10.0,
}

# 5-class shot_type → proximity score (fallback)
_SHOT_TYPE_PROXIMITY = {
    "wide shot": 2.0,
    "full shot": 4.0,
    "medium shot": 5.0,
    "close-up": 7.0,
    "extreme close-up": 9.0,
}


def _get_warmth_score(hue: float) -> float:
    """Calculate a warmth score from hue for warm_to_cool sorting.

    Warm colors (reds, oranges, yellows) get low scores.
    Cool colors (cyans, blues) get high scores.

    Args:
        hue: Hue value 0-360

    Returns:
        Warmth score where 0 = warmest (red), 1 = coolest (cyan)
    """
    # Calculate angular distance from cyan (180°), the coolest point
    # This gives us: red (0/360) = 180, cyan (180) = 0
    distance_from_cyan = abs(180 - hue)
    if distance_from_cyan > 180:
        distance_from_cyan = 360 - distance_from_cyan

    # Normalize to 0-1 where 0 = cool, 1 = warm
    warmth = distance_from_cyan / 180.0

    # Invert so warm colors have LOW scores for ascending sort
    return 1.0 - warmth


def generate_sequence(
    algorithm: str,
    clips: List[Tuple[Any, Any]],  # List of (Clip, Source) tuples
    clip_count: int,
    direction: Optional[str] = None,
    seed: Optional[int] = None,
    no_color_handling: Optional[str] = None,
) -> List[Tuple[Any, Any]]:
    """
    Generate a sequence of clips using the specified algorithm.

    Args:
        algorithm: Algorithm name ("shuffle", "sequential", "color", "shot_type",
                   "duration", "brightness", "volume", etc.)
        clips: List of (Clip, Source) tuples to sequence
        clip_count: Maximum number of clips to include
        direction: For color: "rainbow", "warm_to_cool", "cool_to_warm", "complementary"
                   For duration: "short_first", "long_first"
        seed: Random seed for shuffle reproducibility (0 = random)
        no_color_handling: For color algorithm — how to handle clips without color data.
                   "append_end" (default): append after sorted clips
                   "exclude": drop clips without color data
                   "sort_inline": treat as hue 0 and sort normally

    Returns:
        Ordered list of (Clip, Source) tuples ready for timeline
    """
    clips_to_use = clips[:clip_count]

    if algorithm == "shuffle":
        # Use local Random instance for deterministic shuffling
        rng = random.Random(seed) if seed and seed > 0 else random.Random()

        # Constrained shuffle - no same source back-to-back
        result = constrained_shuffle(
            items=clips_to_use,
            get_category=lambda x: x[1].id,  # x is (Clip, Source), category by source
            max_consecutive=1,
            rng=rng,
        )

        return result

    elif algorithm == "color":
        color_direction = direction or "rainbow"
        color_handling = no_color_handling or "append_end"

        # Separate clips with and without color data
        with_colors = []
        without_colors = []
        for clip, source in clips_to_use:
            if clip.dominant_colors:
                with_colors.append((clip, source))
            else:
                without_colors.append((clip, source))

        if without_colors:
            logger.info(
                f"Chromatics: {len(without_colors)} clips lack color data "
                f"(handling: {color_handling})"
            )

        if not with_colors:
            logger.warning("No clips with color data for Chromatics sort")
            if color_handling == "exclude":
                return []
            return clips_to_use

        if color_direction == "rainbow":
            def get_hue(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                return get_primary_hue(clip.dominant_colors) if clip.dominant_colors else 0.0
            sorted_clips = sorted(with_colors, key=get_hue)

        elif color_direction == "warm_to_cool":
            def get_warmth(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    return _get_warmth_score(get_primary_hue(clip.dominant_colors))
                return 0.5
            sorted_clips = sorted(with_colors, key=get_warmth)

        elif color_direction == "cool_to_warm":
            def get_coolness(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    return 1.0 - _get_warmth_score(get_primary_hue(clip.dominant_colors))
                return 0.5
            sorted_clips = sorted(with_colors, key=get_coolness)

        elif color_direction == "complementary":
            def get_hue(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                return get_primary_hue(clip.dominant_colors) if clip.dominant_colors else 0.0
            sorted_by_hue = sorted(with_colors, key=get_hue)
            # Interleave from opposite ends for maximum contrast
            sorted_clips = []
            lo, hi = 0, len(sorted_by_hue) - 1
            toggle = True
            while lo <= hi:
                if toggle:
                    sorted_clips.append(sorted_by_hue[lo])
                    lo += 1
                else:
                    sorted_clips.append(sorted_by_hue[hi])
                    hi -= 1
                toggle = not toggle
        else:
            sorted_clips = sorted(
                with_colors,
                key=lambda item: get_primary_hue(item[0].dominant_colors) if item[0].dominant_colors else 0.0,
            )

        # Apply no-color-data handling
        if color_handling == "exclude":
            return sorted_clips
        elif color_handling == "sort_inline":
            # Re-sort with colorless clips included (hue 0.0 / warmth 0.5)
            all_clips = sorted_clips + without_colors
            if color_direction == "warm_to_cool":
                all_clips = sorted(
                    with_colors + without_colors,
                    key=lambda item: _get_warmth_score(get_primary_hue(item[0].dominant_colors)) if item[0].dominant_colors else 0.5,
                )
            elif color_direction == "cool_to_warm":
                all_clips = sorted(
                    with_colors + without_colors,
                    key=lambda item: 1.0 - _get_warmth_score(get_primary_hue(item[0].dominant_colors)) if item[0].dominant_colors else 0.5,
                )
            elif color_direction in ("rainbow", "complementary"):
                # For complementary, inline sort falls back to rainbow order
                all_clips = sorted(
                    with_colors + without_colors,
                    key=lambda item: get_primary_hue(item[0].dominant_colors) if item[0].dominant_colors else 0.0,
                )
            return all_clips
        else:
            # append_end (default)
            return sorted_clips + without_colors

    elif algorithm == "shot_type":
        # Sort by shot type (wide -> medium -> close-up -> extreme close-up)
        def get_shot_order(item: Tuple[Any, Any]) -> int:
            clip, _ = item
            if clip.shot_type:
                return SHOT_TYPE_ORDER.get(clip.shot_type, 999)
            return 999  # Unknown shot types at end

        return sorted(clips_to_use, key=get_shot_order)

    elif algorithm == "duration":
        # Unified duration sort with direction parameter
        duration_direction = direction or "short_first"

        if duration_direction == "long_first":
            def get_duration(item: Tuple[Any, Any]) -> float:
                clip, source = item
                return -clip.duration_seconds(source.fps)  # Negative for descending
            return sorted(clips_to_use, key=get_duration)
        else:  # short_first
            def get_duration(item: Tuple[Any, Any]) -> float:
                clip, source = item
                return clip.duration_seconds(source.fps)
            return sorted(clips_to_use, key=get_duration)

    elif algorithm == "brightness":
        # Sort by average brightness (luminance)
        brightness_direction = direction or "bright_to_dark"

        # Auto-compute brightness for clips that don't have it cached
        _auto_compute_brightness(clips_to_use)

        def get_brightness(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            val = clip.average_brightness if clip.average_brightness is not None else 0.5
            return -val if brightness_direction == "bright_to_dark" else val

        return sorted(clips_to_use, key=get_brightness)

    elif algorithm == "volume":
        # Sort by audio volume (RMS level in dB)
        volume_direction = direction or "quiet_to_loud"

        # Auto-compute volume for clips that don't have it cached
        _auto_compute_volume(clips_to_use)

        # Filter out clips without volume data (no audio track)
        clips_with_volume = [
            (clip, source) for clip, source in clips_to_use
            if clip.rms_volume is not None
        ]
        excluded = len(clips_to_use) - len(clips_with_volume)
        if excluded > 0:
            logger.info(f"Volume sort: excluded {excluded} clips (no audio)")

        if not clips_with_volume:
            logger.warning("No clips with audio data for volume sort")
            return clips_to_use

        def get_volume(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            val = clip.rms_volume if clip.rms_volume is not None else -60.0
            return val if volume_direction == "quiet_to_loud" else -val

        return sorted(clips_with_volume, key=get_volume)

    elif algorithm == "proximity":
        # Sort by camera-to-subject distance (proximity score)
        proximity_direction = direction or "far_to_close"

        def get_proximity(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            # Prefer 10-class cinematography shot_size
            if clip.cinematography and clip.cinematography.shot_size:
                score = _SHOT_SIZE_PROXIMITY.get(clip.cinematography.shot_size, 5.0)
            elif clip.shot_type:
                score = _SHOT_TYPE_PROXIMITY.get(clip.shot_type, 5.0)
            else:
                score = 5.0  # Middle default
            return score if proximity_direction == "far_to_close" else -score

        return sorted(clips_to_use, key=get_proximity)

    elif algorithm == "similarity_chain":
        from core.remix.similarity_chain import similarity_chain
        # Auto-compute embeddings for clips that don't have them
        _auto_compute_embeddings(clips_to_use)
        return similarity_chain(clips_to_use, start_clip_id=None)

    elif algorithm == "match_cut":
        from core.remix.match_cut import match_cut_chain
        # Auto-compute boundary embeddings
        _auto_compute_boundary_embeddings(clips_to_use)
        return match_cut_chain(clips_to_use, start_clip_id=None)

    else:
        # Sequential - use original order
        return clips_to_use


def assign_random_transforms(
    sequence_clips: list,
    transform_options: dict[str, bool],
    seed: Optional[int] = None,
) -> None:
    """Assign random transforms to SequenceClip objects in-place.

    Each enabled transform has a 50% chance of being applied per clip.

    Args:
        sequence_clips: List of SequenceClip objects to modify
        transform_options: Dict of transform flags, e.g. {"hflip": True, "vflip": False, "reverse": True}
        seed: Optional random seed for deterministic assignment
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    for seq_clip in sequence_clips:
        seq_clip.hflip = bool(transform_options.get("hflip")) and rng.random() < 0.5
        seq_clip.vflip = bool(transform_options.get("vflip")) and rng.random() < 0.5
        seq_clip.reverse = bool(transform_options.get("reverse")) and rng.random() < 0.5


def _auto_compute_brightness(clips: List[Tuple[Any, Any]]) -> None:
    """Compute brightness for clips that don't have it cached."""
    from core.analysis.color import get_average_brightness

    for clip, source in clips:
        if clip.average_brightness is None:
            try:
                brightness = get_average_brightness(
                    source_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                )
                clip.average_brightness = brightness
            except Exception as e:
                logger.warning(f"Failed to compute brightness for clip {clip.id}: {e}")
                clip.average_brightness = 0.5


def _auto_compute_volume(clips: List[Tuple[Any, Any]]) -> None:
    """Compute volume for clips that don't have it cached."""
    from core.analysis.audio import extract_clip_volume, has_audio_track

    # Cache has_audio_track per source to avoid repeated ffprobe calls
    audio_cache: dict[str, bool] = {}

    for clip, source in clips:
        if clip.rms_volume is None:
            # Check audio track cache before spawning ffmpeg
            if source.id not in audio_cache:
                audio_cache[source.id] = has_audio_track(source.file_path)
            if not audio_cache[source.id]:
                continue

            try:
                start_seconds = clip.start_frame / source.fps
                duration_seconds = clip.duration_seconds(source.fps)
                volume = extract_clip_volume(
                    source_path=source.file_path,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    _has_audio=True,
                )
                clip.rms_volume = volume
            except Exception as e:
                logger.warning(f"Failed to compute volume for clip {clip.id}: {e}")


def _auto_compute_embeddings(clips: List[Tuple[Any, Any]]) -> None:
    """Compute DINOv2 embeddings for clips that don't have them."""
    from core.analysis.embeddings import extract_clip_embeddings_batch, _EMBEDDING_MODEL_TAG

    needs_embedding = [
        (clip, source) for clip, source in clips
        if clip.embedding is None and clip.thumbnail_path
    ]
    if not needs_embedding:
        return

    thumbnail_paths = [clip.thumbnail_path for clip, _ in needs_embedding]
    try:
        embeddings = extract_clip_embeddings_batch(thumbnail_paths)
        for (clip, _), emb in zip(needs_embedding, embeddings):
            clip.embedding = emb
            clip.embedding_model = _EMBEDDING_MODEL_TAG
    except Exception as e:
        logger.warning(f"Failed to compute embeddings: {e}")


def _auto_compute_boundary_embeddings(clips: List[Tuple[Any, Any]]) -> None:
    """Compute first/last frame DINOv2 embeddings for clips that don't have them."""
    from core.analysis.embeddings import extract_boundary_embeddings, _EMBEDDING_MODEL_TAG

    for clip, source in clips:
        if clip.first_frame_embedding is None or clip.last_frame_embedding is None:
            try:
                first_emb, last_emb = extract_boundary_embeddings(
                    source_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                )
                clip.first_frame_embedding = first_emb
                clip.last_frame_embedding = last_emb
                clip.embedding_model = _EMBEDDING_MODEL_TAG
            except Exception as e:
                logger.warning(
                    f"Failed to compute boundary embeddings for clip {clip.id}: {e}"
                )
