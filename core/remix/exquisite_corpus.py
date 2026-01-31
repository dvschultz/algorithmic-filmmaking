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

    Returns:
        List of PoemLine objects in poem order

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    import litellm
    from core.settings import load_settings, get_llm_api_key

    settings = load_settings()
    model = model or settings.llm_model or "gpt-4o"
    original_model = model

    logger.info(f"Generating poem with mood: '{mood_prompt}' using model: {model}")

    # Build the phrase inventory
    phrase_inventory = {}
    for clip, text in clips_with_text:
        phrase_inventory[clip.id] = text

    logger.debug(f"Phrase inventory: {len(phrase_inventory)} clips with text")

    # Normalize model name for LiteLLM
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        model = f"gemini/{model}"
    elif "claude" in model.lower() and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        model = f"anthropic/{model}"

    api_key = get_llm_api_key()

    # Create the LLM prompt
    system_prompt = """You are a poet creating visual poetry from found text.

CRITICAL RULES:
1. You MUST use phrases EXACTLY as provided - no modifications whatsoever
2. Each line of your poem must be one complete phrase from the inventory
3. You cannot split phrases, combine words from different phrases, or change any words
4. You may choose which phrases to use and in what order
5. Not all phrases need to be used, but try to use as many as make sense
6. Create a cohesive poem that evokes the requested mood
7. Consider the visual and sonic qualities of the phrases

OUTPUT FORMAT:
Return a JSON array where each element is the clip_id of the phrase to use, in poem order.
Example: ["clip_abc123", "clip_def456", "clip_ghi789"]

Return ONLY the JSON array, no other text."""

    user_prompt = f"""Create a poem with the mood: {mood_prompt}

Available phrases (clip_id: phrase):
{json.dumps(phrase_inventory, indent=2)}

Return ONLY the JSON array of clip_ids in the order they should appear in the poem."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=0.8,  # Some creativity for poetry
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

        # Build poem lines
        poem_lines = []
        for i, clip_id in enumerate(clip_order, 1):
            if clip_id in phrase_inventory:
                poem_lines.append(PoemLine(
                    text=phrase_inventory[clip_id],
                    clip_id=clip_id,
                    line_number=i,
                ))
            else:
                logger.warning(f"Unknown clip_id in response: {clip_id}")

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
