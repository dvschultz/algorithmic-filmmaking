"""Exquisite Corpus: Generate poems from extracted video text.

This module creates visual poetry from on-screen text in video clips.
An LLM arranges exact phrases from clips into a poem, then the clips
are sequenced to match the poem's line order.

Key constraint: phrases must be used exactly as extracted, with no modifications.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Poetic form definitions
# Each form has a description for the LLM and optional line count constraints
POETIC_FORMS = {
    "free_verse": {
        "label": "Free Verse",
        "description": "No fixed structure. Arrange phrases for maximum emotional impact.",
        "line_count": None,  # Uses length setting
    },
    "haiku": {
        "label": "Haiku",
        "description": (
            "A 3-line poem. The lines should evoke a single vivid image or moment. "
            "Traditional haiku captures nature or a fleeting sensation. "
            "Since you are using found phrases (not syllable-counted words), "
            "focus on brevity and the juxtaposition of the three chosen phrases."
        ),
        "line_count": 3,
    },
    "limerick": {
        "label": "Limerick",
        "description": (
            "A 5-line poem with a humorous or absurd tone. "
            "Lines 1, 2, and 5 should feel longer or more emphatic. "
            "Lines 3 and 4 should be shorter or punchier. "
            "The overall effect should be playful or comedic."
        ),
        "line_count": 5,
    },
    "couplets": {
        "label": "Couplets",
        "description": (
            "Arrange phrases in pairs. Each pair of consecutive lines should "
            "relate to each other — through contrast, echo, or continuation. "
            "The total number of lines should be even."
        ),
        "line_count": None,
    },
    "triptych": {
        "label": "Triptych",
        "description": (
            "A three-section poem. Divide the phrases into three groups: "
            "an opening section, a contrasting middle, and a resolution. "
            "Leave a visual pause between sections (the sequencer will "
            "handle timing). Each section should have at least 2 phrases."
        ),
        "line_count": None,
    },
    "list_poem": {
        "label": "List Poem",
        "description": (
            "An accumulative list structure. Each line builds on a theme, "
            "creating rhythm through repetition and variation. "
            "The effect should feel like an incantation or inventory."
        ),
        "line_count": None,
    },
    "golden_shovel": {
        "label": "Golden Shovel",
        "description": (
            "Inspired by the Golden Shovel form: choose one phrase as the "
            "'spine' and arrange the other phrases so that reading them "
            "in order builds toward or away from that central phrase. "
            "Place the spine phrase last."
        ),
        "line_count": None,
    },
}


@dataclass
class PoemLine:
    """A single line of the generated poem.

    Attributes:
        text: The exact phrase used (from clip's extracted text)
        clip_id: ID of the source clip
        line_number: Position in poem (1-indexed)
    """
    text: str
    clip_id: str
    line_number: int


@dataclass
class ExquisiteCorpusResult:
    """Result of Exquisite Corpus generation.

    Attributes:
        poem_lines: Ordered list of PoemLine objects
        mood_prompt: The user's mood/vibe prompt
        excluded_clip_ids: Clip IDs that had no text and were excluded
    """
    poem_lines: list[PoemLine]
    mood_prompt: str
    excluded_clip_ids: list[str]

    @property
    def poem_text(self) -> str:
        """Return the poem as formatted text."""
        return "\n".join(line.text for line in self.poem_lines)

    @property
    def clip_order(self) -> list[str]:
        """Return clip IDs in poem order."""
        return [line.clip_id for line in self.poem_lines]


def generate_poem(
    clips_with_text: list[tuple],  # [(clip, extracted_text_string), ...]
    mood_prompt: str,
    model: Optional[str] = None,
    length: str = "medium",
    form: str = "free_verse",
) -> list[PoemLine]:
    """Generate a poem using LLM from extracted clip texts.

    The LLM selects and orders phrases from the provided inventory to create
    a poem that evokes the requested mood. Phrases must be used exactly as
    provided - no modifications allowed.

    Args:
        clips_with_text: List of (Clip, text) tuples where text is the
            extracted on-screen text from the clip
        mood_prompt: User's description of the desired mood/vibe
        model: LLM model to use (default: from settings)
        length: Target poem length - "short" (up to 11 lines), "medium" (12-25 lines),
            or "long" (26+ lines). Default: "medium"
        form: Poetic form key from POETIC_FORMS. Default: "free_verse"

    Returns:
        List of PoemLine objects in poem order

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    from core.llm_client import complete_with_local_fallback
    from core.settings import load_settings

    settings = load_settings()
    model = model or settings.exquisite_corpus_model or f"ollama/{settings.ollama_model}"
    temperature = settings.exquisite_corpus_temperature
    original_model = model

    logger.info(f"Generating poem with mood: '{mood_prompt}', length: '{length}', form: '{form}' using model: {model}, temperature: {temperature}")

    # Build the phrase inventory with short IDs for LLM communication
    # LLMs tend to truncate long UUIDs, so we use simple indexed IDs (c1, c2, ...)
    # and map back to full clip IDs after getting the response
    phrase_inventory = {}  # short_id -> text (for LLM)
    short_to_full_id = {}  # short_id -> full clip.id (for mapping back)
    full_to_short_id = {}  # full clip.id -> short_id (for debugging)

    for i, (clip, text) in enumerate(clips_with_text, 1):
        short_id = f"c{i}"
        phrase_inventory[short_id] = text
        short_to_full_id[short_id] = clip.id
        full_to_short_id[clip.id] = short_id

    logger.debug(f"Phrase inventory: {len(phrase_inventory)} clips with text")

    # Model normalization and API-key resolution are handled by complete_with_local_fallback.

    # Resolve poetic form
    form_def = POETIC_FORMS.get(form, POETIC_FORMS["free_verse"])
    form_instruction = f"POETIC FORM: {form_def['label']}\n{form_def['description']}"

    # Length guidance — fixed-count forms override the length setting
    if form_def.get("line_count"):
        length_instruction = f"Your poem must have exactly {form_def['line_count']} lines."
    else:
        length_guidance = {
            "short": "Create a SHORT poem with up to 11 lines. Be selective and choose impactful phrases.",
            "medium": "Create a MEDIUM poem with 12-25 lines. Balance breadth and focus.",
            "long": "Create a LONG poem with 26 or more lines. Use as many phrases as create a cohesive whole.",
        }
        length_instruction = length_guidance.get(length, length_guidance["medium"])

    # Create the LLM prompt
    system_prompt = f"""You are a poet creating visual poetry from found text.

CRITICAL RULES:
1. You MUST use phrases EXACTLY as provided - no modifications whatsoever
2. Each line of your poem must be one complete phrase from the inventory
3. You cannot split phrases, combine words from different phrases, or change any words
4. You may choose which phrases to use and in what order
5. Create a cohesive poem that evokes the requested mood
6. Consider the visual and sonic qualities of the phrases
7. LENGTH REQUIREMENT: {length_instruction}

{form_instruction}

OUTPUT FORMAT:
Return a JSON array where each element is the clip_id of the phrase to use, in poem order.
Example: ["c1", "c5", "c3"]

Return ONLY the JSON array, no other text."""

    user_prompt = f"""Create a poem with the mood: {mood_prompt}

Available phrases (clip_id: phrase):
{json.dumps(phrase_inventory, indent=2)}

Return ONLY the JSON array of clip_ids in the order they should appear in the poem."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = complete_with_local_fallback(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    response_text = response.choices[0].message.content.strip()
    logger.debug(f"LLM response: {response_text[:200]}...")

    # Parse response
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Handle case where response might have extra text
        # Find the JSON array
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx + 1]

        clip_order = json.loads(response_text)

        if not isinstance(clip_order, list):
            raise ValueError("Response is not a JSON array")

        # Build poem lines, mapping short IDs back to full clip IDs
        poem_lines = []
        for i, short_id in enumerate(clip_order, 1):
            if short_id in phrase_inventory:
                full_clip_id = short_to_full_id[short_id]
                poem_lines.append(PoemLine(
                    text=phrase_inventory[short_id],
                    clip_id=full_clip_id,
                    line_number=i,
                ))
            else:
                logger.warning(f"Unknown clip_id in response: {short_id}")

        if not poem_lines:
            raise ValueError("No valid poem lines generated")

        logger.info(f"Generated poem with {len(poem_lines)} lines")
        return poem_lines

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text}")
        raise ValueError("LLM did not return valid JSON. Please try again.")


def validate_poem_phrases(
    poem_lines: list[PoemLine],
    original_phrases: dict[str, str],  # clip_id -> text
) -> list[tuple[int, str]]:
    """Validate that poem uses exact phrases.

    Checks that each poem line matches the original extracted text exactly.

    Args:
        poem_lines: List of PoemLine objects to validate
        original_phrases: Dict mapping clip_id to original extracted text

    Returns:
        List of (line_number, error_message) tuples for any violations.
        Empty list means all phrases are valid.
    """
    errors = []
    for line in poem_lines:
        if line.clip_id not in original_phrases:
            errors.append((line.line_number, f"Unknown clip: {line.clip_id}"))
        elif line.text != original_phrases[line.clip_id]:
            errors.append((
                line.line_number,
                f"Modified phrase detected. Expected: '{original_phrases[line.clip_id]}'"
            ))
    return errors


def sequence_by_poem(
    poem_lines: list[PoemLine],
    clips_by_id: dict,
    sources_by_id: dict,
) -> list[tuple]:
    """Create a clip sequence matching the poem order.

    Args:
        poem_lines: List of PoemLine objects in desired order
        clips_by_id: Dict mapping clip_id to Clip objects
        sources_by_id: Dict mapping source_id to Source objects

    Returns:
        List of (Clip, Source) tuples in poem order
    """
    sequence = []
    for line in poem_lines:
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
