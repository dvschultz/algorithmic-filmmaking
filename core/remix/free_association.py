"""Free Association: LLM-powered iterative sequencer.

The user selects a first clip, then the LLM proposes each next clip based on
clip metadata, providing a rationale for each transition. Accept/reject
interaction with rejection memory per position.

Key design (see docs/plans/2026-04-12-001-feat-free-association-sequencer-plan.md):

Tiered metadata keeps per-step prompt cost bounded (~800 tokens) regardless
of total clip count:
  - Current clip: full metadata block
  - Candidate pool: 12 clips pre-filtered by cosine similarity, each as a
    compact pipe-delimited digest (~20-30 tokens)
  - Sequence history: last 3-5 accepted rationale entries

Local embedding shortlisting absorbs scaling — the LLM only ever sees ~12
candidates regardless of how many clips exist.
"""

import json
import logging
import random
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SHORTLIST_SIZE = 12
DEFAULT_RECENT_RATIONALES = 4  # middle of the 3-5 window


def format_clip_digest(clip: Any) -> str:
    """Format a clip as a compact pipe-delimited digest for LLM prompts.

    Produces a short string (~20-30 tokens) summarizing structured clip fields.
    Missing fields are omitted gracefully — never raises on partial metadata.

    Example output: "CU | warm tones | bright | 2 people | outdoor | dialogue"

    Args:
        clip: A Clip object with optional metadata fields.

    Returns:
        Pipe-delimited string of available structured metadata. Falls back to
        description (truncated) or clip name if no structured fields are set.
    """
    parts: list[str] = []

    if clip.shot_type:
        parts.append(clip.shot_type)

    color_desc = _describe_colors(clip.dominant_colors)
    if color_desc:
        parts.append(color_desc)

    brightness_desc = _describe_brightness(clip.average_brightness)
    if brightness_desc:
        parts.append(brightness_desc)

    if clip.person_count:
        noun = "person" if clip.person_count == 1 else "people"
        parts.append(f"{clip.person_count} {noun}")

    if clip.object_labels:
        top_objects = [label for label in clip.object_labels[:3] if label]
        if top_objects:
            parts.append(", ".join(top_objects))

    transcript_excerpt = _transcript_snippet(clip.transcript)
    if transcript_excerpt:
        parts.append(f'"{transcript_excerpt}"')

    motion_desc = _describe_motion(clip.cinematography)
    if motion_desc:
        parts.append(motion_desc)

    if parts:
        return " | ".join(parts)

    # Fallback: use description or name when structured fields are empty
    if clip.description:
        return clip.description[:80]
    if clip.name:
        return clip.name
    return "no metadata available"


def format_clip_full_metadata(clip: Any) -> str:
    """Format a clip's full metadata as a richer text block for LLM prompts.

    Used for the current clip only — gives the LLM deep context on "where we
    are" in the sequence. Candidate clips use the compact digest instead.

    Args:
        clip: A Clip object.

    Returns:
        Multi-line text block with all available metadata fields.
    """
    lines: list[str] = []

    if clip.description:
        lines.append(f"Description: {clip.description}")
    if clip.shot_type:
        lines.append(f"Shot type: {clip.shot_type}")

    color_desc = _describe_colors(clip.dominant_colors)
    if color_desc:
        lines.append(f"Colors: {color_desc}")

    brightness_desc = _describe_brightness(clip.average_brightness)
    if brightness_desc:
        lines.append(f"Brightness: {brightness_desc}")

    if clip.rms_volume is not None:
        volume_desc = _describe_volume(clip.rms_volume)
        lines.append(f"Audio: {volume_desc}")

    if clip.person_count is not None and clip.person_count > 0:
        noun = "person" if clip.person_count == 1 else "people"
        lines.append(f"People: {clip.person_count} {noun}")

    if clip.object_labels:
        top = ", ".join(clip.object_labels[:5])
        lines.append(f"Objects: {top}")

    if clip.gaze_category:
        lines.append(f"Gaze: {clip.gaze_category}")

    transcript_excerpt = _transcript_snippet(clip.transcript, max_chars=200)
    if transcript_excerpt:
        lines.append(f'Transcript: "{transcript_excerpt}"')

    if clip.extracted_texts:
        texts = [t.text for t in clip.extracted_texts[:3] if getattr(t, "text", "")]
        if texts:
            lines.append(f"On-screen text: {', '.join(texts)}")

    motion_desc = _describe_motion(clip.cinematography)
    if motion_desc:
        lines.append(f"Camera: {motion_desc}")

    if clip.tags:
        lines.append(f"Tags: {', '.join(clip.tags)}")

    return "\n".join(lines) if lines else "(no metadata)"


def shortlist_candidates(
    current_clip: Any,
    pool: list[tuple[Any, Any]],
    k: int = DEFAULT_SHORTLIST_SIZE,
) -> list[tuple[Any, Any]]:
    """Pick the top-k most similar candidates from the pool by cosine similarity.

    Uses existing CLIP/DINOv2 embeddings. Falls back to random sampling when
    the current clip or the pool has no embeddings — the LLM can still make
    useful choices from structured metadata alone.

    Args:
        current_clip: The Clip just accepted (or user-selected first clip).
        pool: Remaining (Clip, Source) tuples available for proposal.
        k: Maximum number of candidates to return.

    Returns:
        Up to k (Clip, Source) tuples ordered by similarity to current_clip
        (most similar first). Returns all of pool if len(pool) <= k.
    """
    if len(pool) <= k:
        return list(pool)

    # Fallback: random sample when embedding comparison isn't possible
    if current_clip.embedding is None:
        logger.info("Current clip has no embedding; shortlisting by random sample")
        return random.sample(pool, k)

    with_emb: list[tuple[int, Any, Any]] = []
    without_emb: list[tuple[Any, Any]] = []
    for i, (clip, source) in enumerate(pool):
        if clip.embedding is not None:
            with_emb.append((i, clip, source))
        else:
            without_emb.append((clip, source))

    if not with_emb:
        logger.info("No pool clips have embeddings; shortlisting by random sample")
        return random.sample(pool, k)

    # Cosine similarity: dot product of L2-normalized vectors
    current_vec = np.array(current_clip.embedding, dtype=np.float32)
    pool_matrix = np.array(
        [clip.embedding for _, clip, _ in with_emb], dtype=np.float32
    )
    similarities = pool_matrix @ current_vec

    # Top-k by similarity (highest first)
    top_indices = np.argsort(-similarities)[:k]
    result = [(with_emb[idx][1], with_emb[idx][2]) for idx in top_indices]

    # If we didn't fill k from embedding-equipped clips, pad with random
    # selections from clips without embeddings
    if len(result) < k and without_emb:
        remaining = k - len(result)
        result.extend(random.sample(without_emb, min(remaining, len(without_emb))))

    return result


def build_id_mapping(
    candidates: list[tuple[Any, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional short-ID <-> UUID mapping for LLM communication.

    LLMs tend to truncate long UUIDs, so we map each candidate to c1, c2, ...
    for the prompt and map back on response parsing.

    Args:
        candidates: List of (Clip, Source) tuples to assign short IDs to.

    Returns:
        (short_to_full, full_to_short) dict pair.
    """
    short_to_full: dict[str, str] = {}
    full_to_short: dict[str, str] = {}
    for i, (clip, _) in enumerate(candidates, start=1):
        short_id = f"c{i}"
        short_to_full[short_id] = clip.id
        full_to_short[clip.id] = short_id
    return short_to_full, full_to_short


def propose_next_clip(
    current_clip_metadata: str,
    candidate_digests: list[tuple[str, str]],
    recent_rationales: list[str],
    rejected_short_ids: list[str],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> tuple[str, str]:
    """Ask the LLM to pick the next clip from the candidate shortlist.

    Args:
        current_clip_metadata: Full metadata text block for the current clip.
        candidate_digests: List of (short_id, digest) tuples for shortlisted
            candidates.
        recent_rationales: Last 3-5 rationale strings from the sequence (empty
            list for the first proposal).
        rejected_short_ids: Short IDs the user rejected for this position.
            The LLM must avoid these.
        model: LLM model name (defaults to settings).
        temperature: Sampling temperature (defaults to settings).

    Returns:
        (clip_short_id, rationale) tuple. The short_id must be validated by
        the caller against the candidate set (handled in the worker).

    Raises:
        ValueError: If the LLM returns no content, malformed JSON, or an ID
            not in the candidate set.
    """
    from core.llm_client import complete_with_local_fallback
    from core.settings import load_settings

    settings = load_settings()
    model = model or settings.exquisite_corpus_model or f"ollama/{settings.ollama_model}"
    if temperature is None:
        temperature = settings.exquisite_corpus_temperature

    # Model normalization and API-key resolution are handled by complete_with_local_fallback.

    candidate_ids = {short_id for short_id, _ in candidate_digests}
    available_ids = sorted(candidate_ids - set(rejected_short_ids))

    system_prompt = """You are a film editor building a sequence one clip at a time through \
free association. You see the current clip's full metadata, compact digests for a shortlist \
of candidate clips, and the rationales for recent transitions.

Pick ONE clip from the candidate shortlist that has the strongest metadata-grounded \
connection to the current clip. Your rationale must reference ACTUAL metadata values \
(shot type, colors, objects, people, motion, etc.) — no abstract or poetic language.

AVOID: candidates already rejected for this position (listed separately).
AVOID: repeating motifs already heavy in recent transitions.

Return ONLY a JSON object with this shape:
{"clip_id": "c3", "rationale": "Both clips share close-up framing with warm tones \
and a single person; the motion rhymes — left-pan to left-pan."}

Keep the rationale to 1-2 sentences. Reference 2-4 concrete metadata dimensions."""

    candidate_block = "\n".join(
        f"  {sid}: {digest}" for sid, digest in candidate_digests
    )

    history_block = ""
    if recent_rationales:
        recent = "\n".join(f"  - {r}" for r in recent_rationales)
        history_block = f"\n\nRECENT TRANSITIONS:\n{recent}"

    rejected_block = ""
    if rejected_short_ids:
        rejected_block = f"\n\nREJECTED AT THIS POSITION (avoid): {', '.join(sorted(rejected_short_ids))}"

    available_block = f"\n\nAVAILABLE TO CHOOSE FROM: {', '.join(available_ids)}"

    user_prompt = f"""CURRENT CLIP:
{current_clip_metadata}

CANDIDATE SHORTLIST:
{candidate_block}{history_block}{rejected_block}{available_block}

Pick one from the available candidates and explain the metadata-grounded connection."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug("Free Association LLM call: model=%s, temp=%s", model, temperature)
    response = complete_with_local_fallback(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # None-safe response extraction (drawing_vlm.py pattern — NOT
    # storyteller.py, which has a latent AttributeError on None content).
    content = response.choices[0].message.content
    if content is None or not str(content).strip():
        raise ValueError("LLM returned no content (possible content filter or rate limit)")
    response_text = str(content).strip()
    logger.debug("LLM response: %s", response_text[:300])

    clip_id, rationale = _parse_proposal_response(response_text)

    if clip_id not in candidate_ids:
        raise ValueError(
            f"LLM returned clip_id '{clip_id}' not in candidate set {sorted(candidate_ids)}"
        )
    if clip_id in rejected_short_ids:
        raise ValueError(
            f"LLM returned clip_id '{clip_id}' which is in the rejected set — retry"
        )

    return clip_id, rationale


def _parse_proposal_response(response_text: str) -> tuple[str, str]:
    """Extract clip_id and rationale from the LLM response JSON.

    Handles markdown code fences and extra surrounding text. Raises ValueError
    on malformed JSON or missing fields.
    """
    text = response_text
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    # Find JSON object boundaries (LLMs sometimes add explanatory prose)
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"No JSON object found in LLM response: {response_text[:200]}")
    text = text[start_idx : end_idx + 1]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned malformed JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError(f"LLM response is not a JSON object: {type(result).__name__}")

    clip_id = result.get("clip_id")
    rationale = result.get("rationale")
    if not clip_id or not isinstance(clip_id, str):
        raise ValueError(f"LLM response missing 'clip_id' field: {result}")
    if not rationale or not isinstance(rationale, str):
        raise ValueError(f"LLM response missing 'rationale' field: {result}")

    return clip_id, rationale


# ---------------------------------------------------------------------------
# Private helpers: metadata descriptors
# ---------------------------------------------------------------------------


def _describe_colors(dominant_colors: Optional[list[tuple[int, int, int]]]) -> str:
    """Map dominant RGB tuples to a short color descriptor."""
    if not dominant_colors:
        return ""
    r, g, b = dominant_colors[0]
    # Warmth from red/blue balance
    if r > b + 30:
        warmth = "warm"
    elif b > r + 30:
        warmth = "cool"
    else:
        warmth = "neutral"
    # Saturation vs grayscale
    max_c, min_c = max(r, g, b), min(r, g, b)
    if max_c - min_c < 25:
        return f"{warmth} grayscale"
    return f"{warmth} tones"


def _describe_brightness(brightness: Optional[float]) -> str:
    if brightness is None:
        return ""
    if brightness < 0.3:
        return "dark"
    if brightness < 0.6:
        return "mid-tone"
    return "bright"


def _describe_volume(rms_volume: float) -> str:
    """Describe RMS volume in dB."""
    if rms_volume < -40:
        return "quiet / near-silent"
    if rms_volume < -20:
        return "moderate"
    return "loud"


def _transcript_snippet(transcript: Any, max_chars: int = 60) -> str:
    """Extract a short excerpt from a clip's transcript, if any."""
    if not transcript:
        return ""
    try:
        text = " ".join(seg.text for seg in transcript if getattr(seg, "text", "")).strip()
    except Exception:
        return ""
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _describe_motion(cinematography: Any) -> str:
    """Describe camera motion from cinematography analysis, if present."""
    if cinematography is None:
        return ""
    # Look for common motion-related attributes without hard-coding the
    # full CinematographyAnalysis schema. Graceful on missing fields.
    for attr in ("camera_motion", "movement", "motion"):
        value = getattr(cinematography, attr, None)
        if value:
            return str(value)
    return ""
