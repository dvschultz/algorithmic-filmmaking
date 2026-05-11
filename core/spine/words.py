"""Word inventory + preset ordering modes for the Word Sequencer.

This is the GUI-agnostic shared layer (spine boundary) used by both the
chat/agent tools and ``scene_ripper_mcp`` (future). It holds **no** Qt /
mpv / av / faster_whisper / paddleocr / mlx_vlm imports at module top
level — see ``tests/test_spine_imports.py`` for the enforcement test.

Data model
----------

- ``WordInstance`` — a single occurrence of a word inside a specific clip,
  carrying clip-relative ``start`` / ``end`` (seconds) and back-references to
  the source / clip / transcript indices.

- ``WordInventory`` — frozen container with two pre-built indices: ``by_word``
  (normalized word → instances) and ``by_clip`` (clip id → instances).

Modes
-----

Five preset ordering functions take a ``WordInventory`` and return an
ordered ``list[WordInstance]``:

- ``alphabetical`` — lexical order over ``by_word`` keys.
- ``by_chosen_words(include=[...])`` — subset filter that emits *every*
  instance of each listed word, grouped by include-list order.
- ``by_frequency(order="descending"|"ascending")`` — frequency sort.
- ``by_property(key="length"|"duration"|"log_frequency", order=...)`` —
  property-based sort.
- ``from_word_list(sequence=[...])`` — literal materialization (one slot per
  list entry, repeats allowed).

Word normalization
------------------

Lowercase + strip surrounding ASCII punctuation. Contractions stay whole
(``"don't"`` is one word — faster-whisper's tokenizer already commits to
that). Original (un-normalized) ``text`` is preserved on each
``WordInstance`` so callers can render it verbatim.
"""

from __future__ import annotations

import math
import secrets
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    # Project models — type-only imports keep the spine module loadable
    # without triggering downstream imports we don't need at runtime.
    from core.transcription import TranscriptSegment
    from models.clip import Clip, Source


__all__ = [
    "MissingWordsError",
    "RepeatPolicy",
    "WordInstance",
    "WordInventory",
    "WordMode",
    "alphabetical",
    "build_inventory",
    "by_chosen_words",
    "by_frequency",
    "by_property",
    "compose_with_llm",
    "from_word_list",
    "normalize_word",
]


# Public mode-name Literal — the single source of truth shared by the spine
# functions and the ``core.remix.word_sequencer`` dispatch.
WordMode = Literal[
    "alphabetical",
    "by_chosen_words",
    "by_frequency",
    "by_property",
    "from_word_list",
]


RepeatPolicy = Literal["round-robin", "random", "first", "longest", "shortest"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingWordsError(Exception):
    """Raised by ``from_word_list(on_missing="raise")`` when entries are absent.

    ``missing`` holds the list of words (normalized form) that could not be
    found in the inventory, in input order.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "Word(s) not found in corpus: " + ", ".join(repr(w) for w in self.missing)
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WordInstance:
    """A single occurrence of a word within a specific transcript segment.

    ``start`` and ``end`` are clip-relative seconds — the same frame of
    reference as ``TranscriptSegment.start_time`` set by per-clip
    transcription / alignment.
    """

    source_id: str
    clip_id: str
    segment_index: int  # index into clip.transcript
    word_index: int  # index into clip.transcript[segment_index].words
    start: float  # clip-relative seconds
    end: float  # clip-relative seconds
    text: str  # original (un-normalized) word text


@dataclass(frozen=True)
class WordInventory:
    """Two pre-built indices over the selected clips' words.

    - ``by_word`` is keyed by normalized word string.
    - ``by_clip`` is keyed by ``Clip.id``.

    Both indices share the same underlying ``WordInstance`` objects.
    """

    by_word: dict[str, list[WordInstance]]
    by_clip: dict[str, list[WordInstance]]


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def normalize_word(word: str) -> str:
    """Normalize a single word for indexing.

    Lowercase + strip surrounding ASCII punctuation. Internal punctuation
    (apostrophes inside contractions, hyphens inside hyphenated words) is
    preserved.
    """
    return word.strip().strip(string.punctuation).lower()


# ---------------------------------------------------------------------------
# build_inventory
# ---------------------------------------------------------------------------


def build_inventory(clips: list[tuple["Clip", "Source"]]) -> WordInventory:
    """Build a ``WordInventory`` from a list of ``(Clip, Source)`` tuples.

    Empty / missing transcripts are silently skipped. Words with empty
    normalized text (e.g. a token that was nothing but punctuation) are
    also dropped.
    """
    by_word: dict[str, list[WordInstance]] = {}
    by_clip: dict[str, list[WordInstance]] = {}

    for clip, _source in clips:
        clip_id = getattr(clip, "id", "")
        source_id = getattr(clip, "source_id", "")
        transcript = getattr(clip, "transcript", None)
        if not transcript:
            # No transcript at all → don't materialize a by_clip entry.
            # ``by_clip.get(clip_id, [])`` returns ``[]`` for callers anyway.
            continue

        # Ensure the clip appears in by_clip even when every segment has
        # ``words = []`` (alignment ran but produced no words). Tests rely
        # on the distinction between "clip not visited" and "clip visited
        # but empty".
        clip_bucket = by_clip.setdefault(clip_id, [])

        for seg_idx, segment in enumerate(transcript):
            words = getattr(segment, "words", None) or []
            for word_idx, word in enumerate(words):
                text = getattr(word, "text", "")
                normalized = normalize_word(text)
                if not normalized:
                    continue
                instance = WordInstance(
                    source_id=source_id,
                    clip_id=clip_id,
                    segment_index=seg_idx,
                    word_index=word_idx,
                    start=float(getattr(word, "start", 0.0)),
                    end=float(getattr(word, "end", 0.0)),
                    text=text,
                )
                by_word.setdefault(normalized, []).append(instance)
                clip_bucket.append(instance)

    return WordInventory(by_word=by_word, by_clip=by_clip)


# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------


def alphabetical(inv: WordInventory) -> list[WordInstance]:
    """Lexical order over ``inv.by_word`` keys; encounter order within each word."""
    result: list[WordInstance] = []
    for word_key in sorted(inv.by_word.keys()):
        result.extend(inv.by_word[word_key])
    return result


def by_chosen_words(inv: WordInventory, include: list[str]) -> list[WordInstance]:
    """Emit every instance of each listed word, grouped by include-list order.

    Case-insensitive (comparison on normalized form). Unknown words are
    silently dropped — the dialog can preview which entries matched.
    """
    result: list[WordInstance] = []
    for raw in include:
        key = normalize_word(raw)
        if not key:
            continue
        result.extend(inv.by_word.get(key, []))
    return result


def by_frequency(
    inv: WordInventory,
    order: Literal["descending", "ascending"] = "descending",
) -> list[WordInstance]:
    """Order every instance by the frequency of its parent word.

    Default emits most-frequent first. Within a frequency tier, words sort
    alphabetically to keep the output deterministic.
    """
    if order not in ("descending", "ascending"):
        raise ValueError(f"order must be 'descending' or 'ascending', got {order!r}")

    keys = sorted(
        inv.by_word.keys(),
        key=lambda k: (-len(inv.by_word[k]), k) if order == "descending"
        else (len(inv.by_word[k]), k),
    )
    result: list[WordInstance] = []
    for key in keys:
        result.extend(inv.by_word[key])
    return result


def by_property(
    inv: WordInventory,
    key: Literal["length", "duration", "log_frequency"],
    order: Literal["descending", "ascending"] = "ascending",
) -> list[WordInstance]:
    """Order every instance by a word-derived metric.

    - ``length`` — number of characters in the normalized word.
    - ``duration`` — ``end - start`` of the individual instance.
    - ``log_frequency`` — log of the parent word's corpus frequency.

    Default is ascending (shortest / quickest / rarest first). Ties break
    alphabetically on the normalized form to keep output deterministic.
    """
    if key not in ("length", "duration", "log_frequency"):
        raise ValueError(
            f"key must be 'length', 'duration', or 'log_frequency', got {key!r}"
        )
    if order not in ("descending", "ascending"):
        raise ValueError(f"order must be 'descending' or 'ascending', got {order!r}")

    # Build (instance, normalized_key, metric) tuples and sort.
    rows: list[tuple[WordInstance, str, float]] = []
    for word_key, instances in inv.by_word.items():
        freq = len(instances)
        log_freq = math.log(freq) if freq > 0 else 0.0
        for inst in instances:
            if key == "length":
                metric = float(len(word_key))
            elif key == "duration":
                metric = float(inst.end - inst.start)
            else:  # log_frequency
                metric = log_freq
            rows.append((inst, word_key, metric))

    reverse = (order == "descending")
    rows.sort(key=lambda r: (r[2], r[1]), reverse=reverse)
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Repeat-policy instance picker
# ---------------------------------------------------------------------------


def _pick_instance(
    candidates: list[WordInstance],
    *,
    policy: RepeatPolicy,
    counter: int,
    rng,
) -> WordInstance:
    """Pick a ``WordInstance`` from ``candidates`` under a repeat policy.

    ``counter`` is the per-word emission count (0-based) and is only used
    by ``round-robin``. ``rng`` is a ``random.Random``-shaped object and
    is only used by ``random``.
    """
    if not candidates:
        # Caller should have filtered this out; defensive raise.
        raise ValueError("_pick_instance called with empty candidate list")

    if policy == "round-robin":
        return candidates[counter % len(candidates)]
    if policy == "random":
        return rng.choice(candidates)
    if policy == "first":
        return candidates[0]

    # ``longest`` / ``shortest`` sort the candidates by duration and use
    # ``(clip_id, segment_index, word_index)`` as the tiebreak. Sorting (vs
    # min/max with a complex key) keeps the tiebreak direction explicit:
    # ties always resolve to the earliest (clip_id, segment_index,
    # word_index) regardless of whether we're picking the longest or
    # shortest duration.
    sorted_by_dur = sorted(
        candidates,
        key=lambda inst: (
            inst.end - inst.start,
            inst.clip_id,
            inst.segment_index,
            inst.word_index,
        ),
    )
    if policy == "shortest":
        return sorted_by_dur[0]
    if policy == "longest":
        # Last entry is longest duration; among ties at max duration, the
        # earliest (clip_id, segment_index, word_index) wins because we
        # sort ascending and tie-broken ascending — so we walk back from
        # the end to the first one whose duration matches the max.
        target = sorted_by_dur[-1].end - sorted_by_dur[-1].start
        for inst in sorted_by_dur:
            if (inst.end - inst.start) == target:
                return inst
        return sorted_by_dur[-1]  # pragma: no cover — unreachable
    raise ValueError(
        f"policy must be one of round-robin/random/first/longest/shortest, "
        f"got {policy!r}"
    )


# ---------------------------------------------------------------------------
# LLM-composed mode
# ---------------------------------------------------------------------------


def compose_with_llm(
    inv: WordInventory,
    prompt: str,
    target_length: int,
    repeat_policy: RepeatPolicy = "round-robin",
    seed: Optional[int] = None,
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    system_prompt: Optional[str] = None,
    think: bool = False,
    _completion_fn=None,
) -> list[WordInstance]:
    """Compose an ordered ``list[WordInstance]`` from a user prompt via Ollama.

    The corpus vocabulary (``inv.by_word`` keys) is fed to Ollama as a
    JSON-schema enum so the model can only emit corpus words. The
    resulting words are mapped back to ``WordInstance`` objects under the
    chosen ``repeat_policy``.

    Args:
        inv: ``WordInventory`` built from selected clips.
        prompt: User-supplied generation prompt.
        target_length: Target output length (used in the system prompt).
        repeat_policy: How to choose among multiple corpus instances when
            the LLM emits the same word more than once.

            - ``"round-robin"`` (default): cycle through instances in
              inventory order. State is per ``compose_with_llm`` call.
            - ``"random"``: seeded ``random.choice``. Same ``seed`` →
              identical output.
            - ``"first"``: always the first instance in inventory order.
            - ``"longest"`` / ``"shortest"``: by ``end - start``; ties
              broken deterministically by ``(clip_id, segment_index,
              word_index)``.
        seed: Seed for ``repeat_policy="random"``. ``None`` → cryptographic
            random seed (output non-deterministic; documented).
        model: Ollama model name (defaults to project setting / qwen3:8b).
        api_base: Ollama API base URL.
        temperature: Sampling temperature for the underlying Ollama call.
        timeout: Total HTTP timeout in seconds.
        system_prompt: Override the default frequency-annotated system prompt.
        _completion_fn: Test-only hook. Callable with the same signature as
            ``core.llm_client.complete_with_enum_constraint`` returning a
            list of words. When ``None``, the real client is used.

    Returns:
        Ordered list of ``WordInstance`` objects, one per LLM-emitted word.

    Raises:
        ValueError: empty vocabulary, ``target_length`` < 1, unknown
            ``repeat_policy``.
        core.llm_client.LLMEmptyResponseError: LLM produced no content.
        core.llm_client.OllamaUnreachableError: Ollama not reachable.
        ValueError: LLM somehow emits an OOV word (the ``format`` + enum
            constraint should prevent this; raise loudly so the bug
            surfaces rather than silently dropping the slot).
    """
    import random

    if not inv.by_word:
        raise ValueError("compose_with_llm: inventory is empty (no words to draw from)")
    if target_length < 1:
        raise ValueError(
            f"compose_with_llm: target_length must be >= 1, got {target_length}"
        )
    if repeat_policy not in (
        "round-robin", "random", "first", "longest", "shortest"
    ):
        raise ValueError(
            "repeat_policy must be one of round-robin/random/first/longest/"
            f"shortest, got {repeat_policy!r}"
        )

    vocabulary = sorted(inv.by_word.keys())
    frequencies = {k: len(v) for k, v in inv.by_word.items()}

    # Run the LLM. Lazy import keeps the spine module importable when
    # llm_client's deps aren't available.
    if _completion_fn is None:
        from core.llm_client import complete_with_enum_constraint
        _completion_fn = complete_with_enum_constraint

    words = _completion_fn(
        prompt=prompt,
        vocabulary=vocabulary,
        target_length=target_length,
        system_prompt=system_prompt,
        model=model,
        api_base=api_base,
        temperature=temperature,
        timeout=timeout,
        frequencies=frequencies,
        think=think,
    )

    # Seeded RNG for the random policy.
    if repeat_policy == "random":
        if seed is None:
            # Document: when ``seed is None`` we use a cryptographic seed
            # so runs are not deterministic. Tests can pass an explicit
            # seed for reproducibility.
            seed = int.from_bytes(secrets.token_bytes(8), "big", signed=False)
        rng = random.Random(seed)
    else:
        rng = random.Random(0)  # unused; kept non-None for the picker

    # Round-robin counter is per-word.
    counters: dict[str, int] = {}

    result: list[WordInstance] = []
    for raw in words:
        key = normalize_word(raw)
        candidates = inv.by_word.get(key)
        if not candidates:
            # The enum constraint should prevent OOV emission. If we see
            # one anyway, fail loudly — silently dropping would hide the
            # bug. Covers the AE4 negative case.
            raise ValueError(
                f"LLM emitted out-of-vocabulary word {raw!r} (normalized: "
                f"{key!r}); the format+enum constraint should have prevented "
                "this. Check Ollama / LiteLLM constrained-decoding support."
            )

        counter = counters.get(key, 0)
        instance = _pick_instance(
            candidates,
            policy=repeat_policy,
            counter=counter,
            rng=rng,
        )
        counters[key] = counter + 1
        result.append(instance)

    return result


def from_word_list(
    inv: WordInventory,
    sequence: list[str],
    on_missing: Literal["skip", "raise"] = "skip",
) -> list[WordInstance]:
    """Literal materialization — one slot per ``sequence`` entry, repeats allowed.

    For each slot we return the *first* matching instance in the inventory.
    Higher layers (the LLM-composer's ``repeat_policy``) can rotate among
    multiple instances if they want different picks per slot.

    Unknown words: ``on_missing="skip"`` drops them silently; ``on_missing="raise"``
    raises ``MissingWordsError`` listing every missing entry.
    """
    if on_missing not in ("skip", "raise"):
        raise ValueError(
            f"on_missing must be 'skip' or 'raise', got {on_missing!r}"
        )

    missing: list[str] = []
    result: list[WordInstance] = []
    for raw in sequence:
        key = normalize_word(raw)
        instances = inv.by_word.get(key) if key else None
        if not instances:
            missing.append(key or raw)
            continue
        result.append(instances[0])

    if missing and on_missing == "raise":
        raise MissingWordsError(missing)

    return result
