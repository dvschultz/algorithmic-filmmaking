"""Storyteller: Generate narrative-driven video sequences from clip descriptions.

This module creates narrative sequences by having an LLM analyze clip descriptions,
select clips that fit a coherent story, and arrange them based on the chosen
narrative structure.

Key constraint: clips are selected and ordered based on their descriptions.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NarrativeLine:
    """A single element in the generated narrative.

    Attributes:
        clip_id: ID of the source clip
        description: The clip's description text
        narrative_role: Role in the narrative (e.g., "opening", "rising_action", "climax")
        line_number: Position in narrative (1-indexed)
    """
    clip_id: str
    description: str
    narrative_role: str
    line_number: int


@dataclass
class StorytellerResult:
    """Result of Storyteller narrative generation.

    Attributes:
        narrative_lines: Ordered list of NarrativeLine objects
        theme: The user's theme prompt (if provided)
        structure: The narrative structure used
        target_duration_minutes: Target duration in minutes
        excluded_clip_ids: Clip IDs that were not used in the narrative
    """
    narrative_lines: list[NarrativeLine]
    theme: Optional[str]
    structure: str
    target_duration_minutes: Optional[int]
    excluded_clip_ids: list[str]

    @property
    def clip_order(self) -> list[str]:
        """Return clip IDs in narrative order."""
        return [line.clip_id for line in self.narrative_lines]


# Duration target options with acceptable ranges
DURATION_TARGETS = {
    "10min": {"target": 10, "min": 6, "max": 14},
    "30min": {"target": 30, "min": 20, "max": 40},
    "1hr": {"target": 60, "min": 45, "max": 75},
    "90min": {"target": 90, "min": 70, "max": 110},
    "all": {"target": None, "min": None, "max": None},  # Use all clips
}

# Narrative structure definitions
NARRATIVE_STRUCTURES = {
    "three_act": "Setup (25%) -> Confrontation (50%) -> Resolution (25%)",
    "chronological": "Arrange by time references in descriptions",
    "thematic": "Group similar themes, build through contrast",
    "auto": "Choose the best structure for these clips",
}


def generate_narrative(
    clips_with_descriptions: list[tuple],  # [(Clip, description_text), ...]
    target_duration_minutes: Optional[int],
    narrative_structure: str,  # "three_act", "chronological", "thematic", "auto"
    theme: Optional[str] = None,
    model: Optional[str] = None,
) -> list[NarrativeLine]:
    """Generate a narrative sequence using LLM.

    The LLM selects and orders clips based on their descriptions to create
    a coherent narrative following the specified structure.

    Args:
        clips_with_descriptions: List of (Clip, description) tuples where
            description is the clip's natural language description
        target_duration_minutes: Target duration in minutes (None = use all clips)
        narrative_structure: Structure to follow (three_act, chronological, thematic, auto)
        theme: Optional theme/focus for the narrative
        model: LLM model to use (default: from settings)

    Returns:
        List of NarrativeLine objects in narrative order

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    import litellm
    from core.settings import load_settings, get_llm_api_key

    settings = load_settings()
    model = model or settings.exquisite_corpus_model or "gemini-3-flash-preview"
    temperature = settings.exquisite_corpus_temperature

    logger.info(
        f"Generating narrative: structure='{narrative_structure}', "
        f"target_duration={target_duration_minutes}min, theme='{theme}', model={model}"
    )

    # Build clip inventory with short IDs for LLM communication
    # LLMs tend to truncate long UUIDs, so we use simple indexed IDs
    clip_inventory = {}  # short_id -> {description, duration_seconds}
    short_to_full_id = {}  # short_id -> full clip.id
    full_to_short_id = {}  # full clip.id -> short_id

    for i, (clip, description) in enumerate(clips_with_descriptions, 1):
        short_id = f"c{i}"
        # Get duration in seconds (we need the source fps)
        duration_sec = getattr(clip, '_duration_seconds', None)
        if duration_sec is None:
            # Estimate from frame count assuming 30fps if not available
            duration_sec = clip.duration_frames / 30.0

        clip_inventory[short_id] = {
            "description": description,
            "duration_seconds": round(duration_sec, 1),
        }
        short_to_full_id[short_id] = clip.id
        full_to_short_id[clip.id] = short_id

    logger.debug(f"Clip inventory: {len(clip_inventory)} clips with descriptions")

    # Calculate total available duration
    total_available_seconds = sum(
        clip_inventory[sid]["duration_seconds"] for sid in clip_inventory
    )
    total_available_minutes = total_available_seconds / 60

    # Normalize model name for LiteLLM
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        model = f"gemini/{model}"
    elif "claude" in model.lower() and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        model = f"anthropic/{model}"

    api_key = get_llm_api_key()

    # Build duration guidance
    if target_duration_minutes:
        duration_guidance = f"""
TARGET DURATION: Approximately {target_duration_minutes} minutes.
- Total available clip duration: {total_available_minutes:.1f} minutes
- Select clips to reach approximately {target_duration_minutes} minutes
- It's okay to be within 30% of the target
- If there aren't enough clips, use all available clips"""
    else:
        duration_guidance = """
DURATION: Use all clips that fit the narrative. No duration target."""

    # Build structure guidance
    structure_descriptions = {
        "three_act": """
NARRATIVE STRUCTURE: Three-Act Structure
- Act 1 (Setup, ~25%): Establish setting, characters, situation
- Act 2 (Confrontation, ~50%): Develop conflict, build tension, explore themes
- Act 3 (Resolution, ~25%): Resolve tensions, provide closure or transformation""",
        "chronological": """
NARRATIVE STRUCTURE: Chronological
- Arrange clips by any time references in their descriptions
- Morning -> Day -> Evening -> Night
- Past -> Present -> Future
- Beginning -> Middle -> End of events""",
        "thematic": """
NARRATIVE STRUCTURE: Thematic
- Group clips by similar themes or subjects
- Build through contrast and comparison
- Create visual or conceptual rhymes between sections""",
        "auto": """
NARRATIVE STRUCTURE: Auto (Your Choice)
- Analyze the clips and choose the best structure
- Consider what narrative approach serves these descriptions best
- Report which structure you chose in the output""",
    }

    structure_instruction = structure_descriptions.get(
        narrative_structure, structure_descriptions["auto"]
    )

    # Build theme section
    theme_section = ""
    if theme:
        theme_section = f"""
THEME/FOCUS: {theme}
- Use this theme to guide clip selection and ordering
- Prioritize clips that relate to or evoke this theme
- The narrative should explore or express this theme"""

    # Create the LLM prompt
    system_prompt = f"""You are a film editor creating a narrative sequence from video clips.
Each clip has a description of its visual content.

RULES:
1. SELECT clips that contribute to a coherent narrative
2. You may EXCLUDE clips that don't fit (list them separately)
3. ARRANGE selected clips in narrative order following the structure below
4. Each clip can only be used ONCE
5. Consider pacing - vary intensity and tone
6. The sequence should feel like it tells a story

{structure_instruction}
{duration_guidance}
{theme_section}

OUTPUT FORMAT:
Return a JSON object with:
- "selected": array of clip_ids in narrative order
- "excluded": array of clip_ids not used
- "structure_used": the narrative structure you followed

Example:
{{"selected": ["c1", "c5", "c3"], "excluded": ["c2", "c4"], "structure_used": "three_act"}}

Return ONLY the JSON object, no other text."""

    # Build user prompt with clip inventory
    inventory_text = "\n".join(
        f"- {sid}: \"{info['description']}\" ({info['duration_seconds']}s)"
        for sid, info in clip_inventory.items()
    )

    user_prompt = f"""Create a narrative sequence from these clips:

{inventory_text}

Return the JSON object with selected clips in narrative order."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
    )

    response_text = response.choices[0].message.content.strip()
    logger.debug(f"LLM response: {response_text[:300]}...")

    # Parse response
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Find the JSON object
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx + 1]

        result = json.loads(response_text)

        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")

        selected = result.get("selected", [])
        structure_used = result.get("structure_used", narrative_structure)

        if not isinstance(selected, list):
            raise ValueError("'selected' is not an array")

        # Build narrative lines, mapping short IDs back to full clip IDs
        narrative_lines = []
        for i, short_id in enumerate(selected, 1):
            if short_id in clip_inventory:
                full_clip_id = short_to_full_id[short_id]
                description = clip_inventory[short_id]["description"]

                # Determine narrative role based on position and structure
                role = _determine_narrative_role(i, len(selected), structure_used)

                narrative_lines.append(NarrativeLine(
                    clip_id=full_clip_id,
                    description=description,
                    narrative_role=role,
                    line_number=i,
                ))
            else:
                logger.warning(f"Unknown clip_id in response: {short_id}")

        if not narrative_lines:
            raise ValueError("No valid narrative lines generated")

        logger.info(f"Generated narrative with {len(narrative_lines)} clips using {structure_used}")
        return narrative_lines

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text}")
        raise ValueError("LLM did not return valid JSON. Please try again.")


def _determine_narrative_role(position: int, total: int, structure: str) -> str:
    """Determine the narrative role based on position in sequence.

    Args:
        position: 1-indexed position in the sequence
        total: Total number of clips
        structure: Narrative structure being used

    Returns:
        Narrative role string
    """
    if total == 0:
        return "unknown"

    progress = position / total

    if structure == "three_act":
        if progress <= 0.25:
            return "setup"
        elif progress <= 0.75:
            return "confrontation"
        else:
            return "resolution"
    elif structure == "chronological":
        if progress <= 0.33:
            return "beginning"
        elif progress <= 0.66:
            return "middle"
        else:
            return "end"
    elif structure == "thematic":
        if progress <= 0.5:
            return "theme_a"
        else:
            return "theme_b"
    else:
        # Generic roles
        if position == 1:
            return "opening"
        elif position == total:
            return "closing"
        else:
            return "development"


def sequence_by_narrative(
    narrative_lines: list[NarrativeLine],
    clips_by_id: dict,
    sources_by_id: dict,
) -> list[tuple]:
    """Create a clip sequence matching the narrative order.

    Args:
        narrative_lines: List of NarrativeLine objects in desired order
        clips_by_id: Dict mapping clip_id to Clip objects
        sources_by_id: Dict mapping source_id to Source objects

    Returns:
        List of (Clip, Source) tuples in narrative order
    """
    sequence = []
    for line in narrative_lines:
        if line.clip_id in clips_by_id:
            clip = clips_by_id[line.clip_id]
            source = sources_by_id.get(clip.source_id)
            if source:
                sequence.append((clip, source))
            else:
                logger.warning(f"Source not found for clip {line.clip_id}")
        else:
            logger.warning(f"Clip not found: {line.clip_id}")

    logger.info(f"Created sequence with {len(sequence)} clips")
    return sequence


def calculate_sequence_duration(
    clips: list[tuple],  # [(Clip, Source), ...]
) -> float:
    """Calculate total duration of a clip sequence in seconds.

    Args:
        clips: List of (Clip, Source) tuples

    Returns:
        Total duration in seconds
    """
    total = 0.0
    for clip, source in clips:
        total += clip.duration_seconds(source.fps)
    return total
