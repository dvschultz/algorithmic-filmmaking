"""Word Sequencer remix wrapper.

Thin shim over ``core/spine/words.py`` that materializes word-instance
sequences into ``SequenceClip`` entries with the documented frame-math
convention.

Frame conversion convention (committed in the plan):

- ``fps = source.fps``; ``ValueError`` if missing (no silent default).
- Handle padding pre-conversion in seconds:
  ``in_seconds  = max(0.0, word.start - handle_frames / fps)``
  ``out_seconds = min(clip_duration_seconds, word.end + handle_frames / fps)``
- Integer-frame snap:
  ``in_point  = floor(in_seconds * fps)``  (keep consonant onset)
  ``out_point = ceil(out_seconds * fps)``  (keep consonant offset)
- All intermediate math uses Python ``float`` after explicit conversion from
  any ``Fraction`` (per the FFmpeg-fractional-duration learning).
- Zero-duration words (``word.start == word.end``) are dropped silently.

The wrapper is a pure function — no Qt, no project mutation. Callers
(``ui/dialogs/word_sequencer_dialog.py``) wrap the returned list in a
``Sequence`` / ``Track`` and assign via ``project.sequence = ...``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from core.spine.words import (
    WordInstance,
    alphabetical,
    build_inventory,
    by_chosen_words,
    by_frequency,
    by_property,
    from_word_list,
)

if TYPE_CHECKING:
    from models.clip import Clip, Source
    from models.sequence import SequenceClip


__all__ = [
    "MissingWordDataError",
    "generate_word_sequence",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingWordDataError(Exception):
    """Raised when any selected clip has segments lacking word-level data.

    ``clip_ids`` lists the offending clip ids. The dialog (U6) catches this
    and triggers alignment.
    """

    def __init__(self, clip_ids: list[str]) -> None:
        self.clip_ids = list(clip_ids)
        super().__init__(
            "The following clip(s) are missing word-level alignment data: "
            + ", ".join(self.clip_ids)
        )


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


_MODES = {
    "alphabetical",
    "by_chosen_words",
    "by_frequency",
    "by_property",
    "from_word_list",
}


def _apply_mode(inv, mode: str, mode_params: dict) -> list[WordInstance]:
    if mode == "alphabetical":
        return alphabetical(inv)
    if mode == "by_chosen_words":
        return by_chosen_words(inv, include=mode_params.get("include", []))
    if mode == "by_frequency":
        return by_frequency(inv, order=mode_params.get("order", "descending"))
    if mode == "by_property":
        return by_property(
            inv,
            key=mode_params.get("key", "length"),
            order=mode_params.get("order", "ascending"),
        )
    if mode == "from_word_list":
        return from_word_list(
            inv,
            sequence=mode_params.get("sequence", []),
            on_missing=mode_params.get("on_missing", "skip"),
        )
    raise ValueError(
        f"Unknown word-sequencer mode {mode!r}; expected one of {sorted(_MODES)}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_word_data(clips: list[tuple[Any, Any]]) -> None:
    """Raise ``MissingWordDataError`` if any selected clip has unaligned segments."""
    missing: list[str] = []
    for clip, _source in clips:
        transcript = getattr(clip, "transcript", None) or []
        if any(getattr(seg, "words", None) is None for seg in transcript):
            missing.append(getattr(clip, "id", ""))
    if missing:
        raise MissingWordDataError(missing)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_word_sequence(
    clips: list[tuple[Any, Any]],
    mode: str,
    mode_params: Optional[dict] = None,
    handle_frames: int = 0,
) -> list["SequenceClip"]:
    """Materialize a word-instance ordering into ``SequenceClip[]``.

    Args:
        clips: List of ``(Clip, Source)`` tuples — the same shape the
            Sequence-tab dialog dispatch passes every other algorithm.
        mode: One of ``alphabetical`` / ``by_chosen_words`` / ``by_frequency``
            / ``by_property`` / ``from_word_list``.
        mode_params: Mode-specific keyword arguments (e.g. ``{"order":
            "descending"}`` for ``by_frequency``).
        handle_frames: Symmetric handle padding in frames (default 0 — raw
            cuts). Clamped to clip bounds.

    Returns:
        A list of ``SequenceClip`` objects with ``in_point`` / ``out_point``
        set to the word-aligned frame boundaries. ``start_frame`` /
        ``track_index`` are left at their defaults — the caller assembles the
        timeline.

    Raises:
        MissingWordDataError: any selected clip has a segment with
            ``words is None``.
        ValueError: ``mode`` is not recognized, or any source has missing fps.
    """
    # Local import keeps the spine boundary check clean and avoids paying
    # the import cost when the module is only being collected by pytest.
    from models.sequence import SequenceClip

    if not clips:
        return []

    mode_params = mode_params or {}

    # Validation must happen BEFORE any frame math — the dialog catches
    # MissingWordDataError and triggers alignment.
    _validate_word_data(clips)

    # Build a quick lookup by clip_id to source/clip pair for the frame math.
    by_clip_id: dict[str, tuple[Any, Any]] = {
        getattr(clip, "id", ""): (clip, source) for clip, source in clips
    }

    inventory = build_inventory(clips)
    instances = _apply_mode(inventory, mode, mode_params)

    result: list[SequenceClip] = []
    for inst in instances:
        # Drop zero-duration words silently (rare ASR artifact).
        if inst.end <= inst.start:
            continue

        clip, source = by_clip_id.get(inst.clip_id, (None, None))
        if clip is None or source is None:
            # Defensive — should never happen since instances come from
            # build_inventory(clips).
            continue

        # fps: explicitly convert Fraction → float per the FFmpeg
        # fractional-duration learning.
        raw_fps = getattr(source, "fps", None)
        if raw_fps is None:
            raise ValueError(
                f"Source {getattr(source, 'id', '?')} has no fps; cannot "
                "convert word boundaries to frames"
            )
        fps = float(raw_fps)
        if fps <= 0:
            raise ValueError(
                f"Source {getattr(source, 'id', '?')} has non-positive fps {fps!r}"
            )

        start_frame = int(getattr(clip, "start_frame", 0))
        end_frame = int(getattr(clip, "end_frame", start_frame))
        clip_duration_seconds = max(0.0, (end_frame - start_frame) / fps)

        handle_seconds = handle_frames / fps
        in_seconds = max(0.0, float(inst.start) - handle_seconds)
        out_seconds = min(clip_duration_seconds, float(inst.end) + handle_seconds)

        in_point = math.floor(in_seconds * fps)
        out_point = math.ceil(out_seconds * fps)

        # Hard-clamp to clip bounds after the floor/ceil snap, in case
        # float rounding pushed past the limits.
        clip_length = end_frame - start_frame
        in_point = max(0, min(in_point, clip_length))
        out_point = max(0, min(out_point, clip_length))

        # Defensive: skip slots that collapsed to zero frames.
        if out_point <= in_point:
            continue

        result.append(
            SequenceClip(
                source_clip_id=getattr(clip, "id", ""),
                source_id=getattr(clip, "source_id", ""),
                in_point=in_point,
                out_point=out_point,
            )
        )

    return result
