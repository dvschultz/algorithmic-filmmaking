"""Rich cinematography analysis using VLM.

Provides comprehensive film language analysis of video clips including:
- Shot size (granular: ELS to ECU)
- Camera angle and effect
- Camera movement (video mode only)
- Composition (subject position, balance, headroom/lead room)
- Subject analysis (count, type)
- Focus/depth characteristics
- Lighting style and direction
- Derived emotional properties

Primary implementation uses Gemini with video support for movement detection,
with frame-only fallback mode for faster/cheaper analysis.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import litellm

from core.settings import load_settings, get_gemini_api_key
from core.analysis.description import extract_clip_segment, encode_image_base64, encode_video_base64
from models.cinematography import CinematographyAnalysis

logger = logging.getLogger(__name__)


# JSON schema for structured output
CINEMATOGRAPHY_SCHEMA = {
    "type": "object",
    "properties": {
        "shot_size": {
            "type": "string",
            "enum": ["ELS", "VLS", "LS", "MLS", "MS", "MCU", "CU", "BCU", "ECU", "Insert"],
            "description": "Shot size classification"
        },
        "shot_size_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in shot size classification (0.0-1.0)"
        },
        "camera_angle": {
            "type": "string",
            "enum": ["low_angle", "eye_level", "high_angle", "dutch_angle", "birds_eye", "worms_eye"],
            "description": "Camera angle relative to subject"
        },
        "angle_effect": {
            "type": "string",
            "enum": ["power", "neutral", "vulnerability", "disorientation", "omniscience", "extreme_power"],
            "description": "Narrative effect of the camera angle"
        },
        "camera_movement": {
            "type": "string",
            "enum": ["static", "pan", "tilt", "track", "handheld", "crane", "arc", "n/a"],
            "description": "Type of camera movement (n/a if analyzing single frame)"
        },
        "movement_direction": {
            "type": ["string", "null"],
            "enum": ["left", "right", "up", "down", "forward", "backward", "clockwise", "counterclockwise", None],
            "description": "Direction of camera movement if applicable"
        },
        "subject_position": {
            "type": "string",
            "enum": ["left_third", "center", "right_third", "distributed"],
            "description": "Main subject position in frame"
        },
        "headroom": {
            "type": "string",
            "enum": ["tight", "normal", "excessive", "n/a"],
            "description": "Amount of space above subject's head"
        },
        "lead_room": {
            "type": "string",
            "enum": ["tight", "normal", "excessive", "n/a"],
            "description": "Amount of space in direction of gaze/movement"
        },
        "balance": {
            "type": "string",
            "enum": ["balanced", "left_heavy", "right_heavy", "symmetrical"],
            "description": "Visual weight distribution in frame"
        },
        "subject_count": {
            "type": "string",
            "enum": ["empty", "single", "two_shot", "group"],
            "description": "Number of main subjects in frame"
        },
        "subject_type": {
            "type": "string",
            "enum": ["person", "object", "landscape", "text", "mixed"],
            "description": "Primary subject type"
        },
        "focus_type": {
            "type": "string",
            "enum": ["deep", "shallow", "rack_focus"],
            "description": "Depth of field characteristics"
        },
        "background_type": {
            "type": "string",
            "enum": ["blurred", "sharp", "cluttered", "plain"],
            "description": "Background appearance"
        },
        "lighting_style": {
            "type": "string",
            "enum": ["high_key", "low_key", "natural", "dramatic"],
            "description": "Overall lighting approach"
        },
        "lighting_direction": {
            "type": "string",
            "enum": ["front", "three_quarter", "side", "back", "below"],
            "description": "Primary light source direction"
        },
        "emotional_intensity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Derived emotional impact level"
        },
        "suggested_pacing": {
            "type": "string",
            "enum": ["fast", "medium", "slow"],
            "description": "Suggested edit duration based on visual complexity"
        }
    },
    "required": [
        "shot_size", "camera_angle", "subject_position", "subject_count",
        "focus_type", "lighting_style", "lighting_direction",
        "emotional_intensity", "suggested_pacing"
    ]
}


# Prompt for cinematography analysis
CINEMATOGRAPHY_PROMPT_FRAME = """Analyze this film frame using professional cinematography terminology.

Evaluate each dimension carefully:

**Shot Size** (distance from subject):
- ELS (Extreme Long Shot): Vast environment, people tiny/absent
- VLS (Very Long Shot): Full environment with visible people
- LS (Long Shot): Head to toe, full body
- MLS (Medium Long Shot): 3/4 body (knees up)
- MS (Medium Shot): Waist to head
- MCU (Medium Close-Up): Head and shoulders
- CU (Close-Up): Face fills frame
- BCU (Big Close-Up): Face, partial features
- ECU (Extreme Close-Up): Single feature (eyes, lips)
- Insert: Object detail shot

**Camera Angle** (position relative to subject):
- low_angle: Below subject looking up (power, heroism)
- eye_level: Subject's eye height (neutral, equal)
- high_angle: Above subject looking down (vulnerability)
- dutch_angle: Tilted horizon (disorientation, unease)
- birds_eye: Directly above (omniscience, pattern)
- worms_eye: Directly below (extreme power)

**Composition**:
- Subject position: left_third, center, right_third, or distributed
- Headroom: tight, normal, excessive, or n/a
- Lead room: tight, normal, excessive, or n/a (space in direction of gaze)
- Balance: balanced, left_heavy, right_heavy, or symmetrical

**Subject Analysis**:
- Count: empty, single, two_shot (2 people), or group (3+)
- Type: person, object, landscape, text, or mixed

**Focus & Depth**:
- Focus type: deep (all sharp), shallow (subject isolated), or rack_focus
- Background: blurred, sharp, cluttered, or plain

**Lighting**:
- Style: high_key (bright, few shadows), low_key (dark, heavy shadows), natural, or dramatic
- Direction: front, three_quarter, side, back, or below

**Derived Properties**:
- Emotional intensity: low, medium, or high (based on shot size + angle + lighting)
- Suggested pacing: fast (simple shots), medium, or slow (complex/wide shots)

Since this is a single frame, set camera_movement to "n/a".

Return your analysis as a JSON object with these exact field names."""


CINEMATOGRAPHY_PROMPT_VIDEO = """Analyze this video clip using professional cinematography terminology.

Evaluate each dimension carefully:

**Shot Size** (distance from subject):
- ELS (Extreme Long Shot): Vast environment, people tiny/absent
- VLS (Very Long Shot): Full environment with visible people
- LS (Long Shot): Head to toe, full body
- MLS (Medium Long Shot): 3/4 body (knees up)
- MS (Medium Shot): Waist to head
- MCU (Medium Close-Up): Head and shoulders
- CU (Close-Up): Face fills frame
- BCU (Big Close-Up): Face, partial features
- ECU (Extreme Close-Up): Single feature (eyes, lips)
- Insert: Object detail shot

**Camera Angle** (position relative to subject):
- low_angle: Below subject looking up (power, heroism)
- eye_level: Subject's eye height (neutral, equal)
- high_angle: Above subject looking down (vulnerability)
- dutch_angle: Tilted horizon (disorientation, unease)
- birds_eye: Directly above (omniscience, pattern)
- worms_eye: Directly below (extreme power)

**Camera Movement** (watch for motion throughout the clip):
- static: No camera movement
- pan: Horizontal rotation (left/right)
- tilt: Vertical rotation (up/down)
- track: Camera moves through space (dolly/traveling)
- handheld: Unstable, organic movement
- crane: Vertical spatial movement
- arc: Circling around subject

If movement is detected, also note the direction:
left, right, up, down, forward, backward, clockwise, counterclockwise

**Composition**:
- Subject position: left_third, center, right_third, or distributed
- Headroom: tight, normal, excessive, or n/a
- Lead room: tight, normal, excessive, or n/a (space in direction of gaze)
- Balance: balanced, left_heavy, right_heavy, or symmetrical

**Subject Analysis**:
- Count: empty, single, two_shot (2 people), or group (3+)
- Type: person, object, landscape, text, or mixed

**Focus & Depth**:
- Focus type: deep (all sharp), shallow (subject isolated), or rack_focus
- Background: blurred, sharp, cluttered, or plain

**Lighting**:
- Style: high_key (bright, few shadows), low_key (dark, heavy shadows), natural, or dramatic
- Direction: front, three_quarter, side, back, or below

**Derived Properties**:
- Emotional intensity: low, medium, or high (based on shot size + angle + lighting)
- Suggested pacing: fast (simple shots), medium, or slow (complex/wide shots)

Return your analysis as a JSON object with these exact field names."""


def _parse_json_response(response_text: str) -> dict:
    """Parse JSON from VLM response, handling markdown code blocks.

    Args:
        response_text: Raw response from VLM

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    text = response_text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object pattern
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def _validate_and_normalize(data: dict) -> dict:
    """Validate and normalize cinematography analysis data.

    Ensures all fields have valid values, applying defaults where needed.

    Args:
        data: Raw parsed JSON data

    Returns:
        Normalized dictionary with valid values
    """
    # Valid value sets for validation
    valid_shot_sizes = {"ELS", "VLS", "LS", "MLS", "MS", "MCU", "CU", "BCU", "ECU", "Insert"}
    valid_angles = {"low_angle", "eye_level", "high_angle", "dutch_angle", "birds_eye", "worms_eye"}
    valid_movements = {"static", "pan", "tilt", "track", "handheld", "crane", "arc", "n/a"}
    valid_directions = {"left", "right", "up", "down", "forward", "backward", "clockwise", "counterclockwise", None}
    valid_positions = {"left_third", "center", "right_third", "distributed"}
    valid_spacing = {"tight", "normal", "excessive", "n/a"}
    valid_balance = {"balanced", "left_heavy", "right_heavy", "symmetrical"}
    valid_counts = {"empty", "single", "two_shot", "group"}
    valid_types = {"person", "object", "landscape", "text", "mixed"}
    valid_focus = {"deep", "shallow", "rack_focus"}
    valid_bg = {"blurred", "sharp", "cluttered", "plain"}
    valid_lighting_style = {"high_key", "low_key", "natural", "dramatic"}
    valid_lighting_dir = {"front", "three_quarter", "side", "back", "below"}
    valid_intensity = {"low", "medium", "high"}
    valid_pacing = {"fast", "medium", "slow"}

    result = {}

    # Shot size
    shot_size = data.get("shot_size", "MS")
    result["shot_size"] = shot_size if shot_size in valid_shot_sizes else "MS"

    # Safe confidence parsing - VLM may return "high" instead of 0.8
    raw_confidence = data.get("shot_size_confidence", 0.8)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.8
    result["shot_size_confidence"] = min(1.0, max(0.0, confidence))

    # Camera angle
    angle = data.get("camera_angle", "eye_level")
    result["camera_angle"] = angle if angle in valid_angles else "eye_level"

    # Angle effect - derive from angle if not provided
    angle_effects = {
        "low_angle": "power",
        "eye_level": "neutral",
        "high_angle": "vulnerability",
        "dutch_angle": "disorientation",
        "birds_eye": "omniscience",
        "worms_eye": "extreme_power",
    }
    result["angle_effect"] = data.get("angle_effect", angle_effects.get(result["camera_angle"], "neutral"))

    # Camera movement
    movement = data.get("camera_movement", "n/a")
    result["camera_movement"] = movement if movement in valid_movements else "n/a"

    # Movement direction
    direction = data.get("movement_direction")
    result["movement_direction"] = direction if direction in valid_directions else None

    # Composition
    position = data.get("subject_position", "center")
    result["subject_position"] = position if position in valid_positions else "center"

    headroom = data.get("headroom", "normal")
    result["headroom"] = headroom if headroom in valid_spacing else "normal"

    lead_room = data.get("lead_room", "n/a")
    result["lead_room"] = lead_room if lead_room in valid_spacing else "n/a"

    balance = data.get("balance", "balanced")
    result["balance"] = balance if balance in valid_balance else "balanced"

    # Subject analysis
    count = data.get("subject_count", "single")
    result["subject_count"] = count if count in valid_counts else "single"

    subj_type = data.get("subject_type", "person")
    result["subject_type"] = subj_type if subj_type in valid_types else "person"

    # Focus and depth
    focus = data.get("focus_type", "deep")
    result["focus_type"] = focus if focus in valid_focus else "deep"

    bg = data.get("background_type", "sharp")
    result["background_type"] = bg if bg in valid_bg else "sharp"

    # Lighting
    style = data.get("lighting_style", "natural")
    result["lighting_style"] = style if style in valid_lighting_style else "natural"

    direction = data.get("lighting_direction", "front")
    result["lighting_direction"] = direction if direction in valid_lighting_dir else "front"

    # Derived properties
    intensity = data.get("emotional_intensity", "medium")
    result["emotional_intensity"] = intensity if intensity in valid_intensity else "medium"

    pacing = data.get("suggested_pacing", "medium")
    result["suggested_pacing"] = pacing if pacing in valid_pacing else "medium"

    return result


def analyze_cinematography_frame(
    image_path: Path,
    model: Optional[str] = None,
) -> CinematographyAnalysis:
    """Analyze cinematography from a single frame.

    Args:
        image_path: Path to the image file
        model: VLM model to use (default: from settings)

    Returns:
        CinematographyAnalysis with all fields populated

    Raises:
        ValueError: If API key is not configured
        RuntimeError: If analysis fails
    """
    settings = load_settings()
    model = model or settings.cinematography_model

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not configured. Please add it in Settings.")

    # Normalize model name for LiteLLM
    llm_model = model
    if "gemini" in model.lower() and not model.startswith(("gemini/", "vertex_ai/")):
        llm_model = f"gemini/{model}"

    logger.info(f"Analyzing cinematography (frame mode) with {model}")

    # Encode image
    base64_image = encode_image_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CINEMATOGRAPHY_PROMPT_FRAME},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    try:
        response = litellm.completion(
            model=llm_model,
            messages=messages,
            api_key=api_key,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content

        # Parse and validate response
        raw_data = _parse_json_response(response_text)
        normalized = _validate_and_normalize(raw_data)

        # Create CinematographyAnalysis object
        analysis = CinematographyAnalysis(
            shot_size=normalized["shot_size"],
            shot_size_confidence=normalized["shot_size_confidence"],
            camera_angle=normalized["camera_angle"],
            angle_effect=normalized["angle_effect"],
            camera_movement="n/a",  # Frame mode cannot detect movement
            movement_direction=None,
            subject_position=normalized["subject_position"],
            headroom=normalized["headroom"],
            lead_room=normalized["lead_room"],
            balance=normalized["balance"],
            subject_count=normalized["subject_count"],
            subject_type=normalized["subject_type"],
            focus_type=normalized["focus_type"],
            background_type=normalized["background_type"],
            lighting_style=normalized["lighting_style"],
            lighting_direction=normalized["lighting_direction"],
            emotional_intensity=normalized["emotional_intensity"],
            suggested_pacing=normalized["suggested_pacing"],
            analysis_model=model,
            analysis_mode="frame",
        )

        logger.info(f"Cinematography analysis complete: {analysis.shot_size}, {analysis.camera_angle}")
        return analysis

    except Exception as e:
        logger.error(f"Cinematography analysis failed: {e}")
        raise RuntimeError(f"Cinematography analysis failed: {e}") from e


def analyze_cinematography_video(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    model: Optional[str] = None,
    max_video_size_mb: float = 20.0,
) -> CinematographyAnalysis:
    """Analyze cinematography from a video clip.

    Extracts the clip segment and sends to Gemini for analysis,
    enabling camera movement detection.

    Args:
        source_path: Path to source video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Video frame rate
        model: VLM model to use (default: from settings)
        max_video_size_mb: Maximum video segment size in MB (default: 20)

    Returns:
        CinematographyAnalysis with all fields populated including movement

    Raises:
        ValueError: If API key is not configured or frame range is invalid
        RuntimeError: If analysis fails
    """
    # Validate frame range
    if start_frame < 0:
        raise ValueError(f"start_frame ({start_frame}) must be non-negative")
    if end_frame <= start_frame:
        raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})")
    if fps <= 0:
        raise ValueError(f"fps ({fps}) must be positive")

    settings = load_settings()
    model = model or settings.cinematography_model

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not configured. Please add it in Settings.")

    # Normalize model name for LiteLLM
    llm_model = model
    if "gemini" in model.lower() and not model.startswith(("gemini/", "vertex_ai/")):
        llm_model = f"gemini/{model}"

    logger.info(f"Analyzing cinematography (video mode) with {model}")

    # Extract clip segment
    temp_video = extract_clip_segment(source_path, start_frame, end_frame, fps)

    try:
        # Check video size before encoding to prevent memory spikes
        file_size_mb = temp_video.stat().st_size / 1024 / 1024
        if file_size_mb > max_video_size_mb:
            raise ValueError(
                f"Video segment too large ({file_size_mb:.1f} MB > {max_video_size_mb} MB); "
                "falling back to frame mode"
            )

        # Encode video
        base64_video = encode_video_base64(temp_video)
        logger.info(f"Encoded video for cinematography: {file_size_mb:.1f} MB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CINEMATOGRAPHY_PROMPT_VIDEO},
                    {
                        "type": "file",
                        "file": {
                            "file_id": f"data:video/mp4;base64,{base64_video}",
                            "format": "video/mp4",
                        },
                    },
                ],
            }
        ]

        response = litellm.completion(
            model=llm_model,
            messages=messages,
            api_key=api_key,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content

        # Parse and validate response
        raw_data = _parse_json_response(response_text)
        normalized = _validate_and_normalize(raw_data)

        # Create CinematographyAnalysis object
        analysis = CinematographyAnalysis(
            shot_size=normalized["shot_size"],
            shot_size_confidence=normalized["shot_size_confidence"],
            camera_angle=normalized["camera_angle"],
            angle_effect=normalized["angle_effect"],
            camera_movement=normalized["camera_movement"],
            movement_direction=normalized["movement_direction"],
            subject_position=normalized["subject_position"],
            headroom=normalized["headroom"],
            lead_room=normalized["lead_room"],
            balance=normalized["balance"],
            subject_count=normalized["subject_count"],
            subject_type=normalized["subject_type"],
            focus_type=normalized["focus_type"],
            background_type=normalized["background_type"],
            lighting_style=normalized["lighting_style"],
            lighting_direction=normalized["lighting_direction"],
            emotional_intensity=normalized["emotional_intensity"],
            suggested_pacing=normalized["suggested_pacing"],
            analysis_model=model,
            analysis_mode="video",
        )

        logger.info(
            f"Cinematography analysis complete: {analysis.shot_size}, "
            f"{analysis.camera_angle}, movement={analysis.camera_movement}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Cinematography video analysis failed: {e}")
        raise RuntimeError(f"Cinematography analysis failed: {e}") from e

    finally:
        # Cleanup temp file
        if temp_video.exists():
            temp_video.unlink()
            logger.debug(f"Cleaned up temp video: {temp_video}")


def analyze_cinematography(
    thumbnail_path: Path,
    source_path: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
) -> CinematographyAnalysis:
    """Analyze cinematography for a clip.

    Uses video mode if source info is provided and mode is "video",
    otherwise uses frame mode with the thumbnail.

    Args:
        thumbnail_path: Path to clip thumbnail (required for frame mode)
        source_path: Path to source video file (optional, for video mode)
        start_frame: Starting frame number (optional, for video mode)
        end_frame: Ending frame number (optional, for video mode)
        fps: Video frame rate (optional, for video mode)
        mode: "frame" or "video" (default: from settings)
        model: VLM model to use (default: from settings)

    Returns:
        CinematographyAnalysis with all fields populated

    Raises:
        ValueError: If API key is not configured
        RuntimeError: If analysis fails
    """
    settings = load_settings()
    mode = mode or settings.cinematography_input_mode
    model = model or settings.cinematography_model

    # Check if video mode is possible
    can_use_video = (
        mode == "video"
        and source_path is not None
        and start_frame is not None
        and end_frame is not None
        and fps is not None
        and source_path.exists()
    )

    if can_use_video:
        try:
            return analyze_cinematography_video(
                source_path=source_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                model=model,
            )
        except Exception as e:
            logger.warning(f"Video mode failed, falling back to frame mode: {e}")
            # Fall through to frame mode

    # Use frame mode
    return analyze_cinematography_frame(thumbnail_path, model=model)
