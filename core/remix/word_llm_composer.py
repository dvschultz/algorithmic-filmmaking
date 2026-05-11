"""LLM Word Composer remix wrapper.

Thin shim over ``core/spine/words.compose_with_llm`` that materializes a
prompt-driven LLM word ordering into ``SequenceClip[]``.

This unit (U5) owns the registration of ``word_llm_composer`` in
``ui/algorithm_config.py`` end-to-end. Frame-math semantics are shared
with the preset-modes path (U4) via
``core/remix/word_sequencer.instances_to_sequence_clips`` — there is
intentionally only one implementation of the floor/ceil/handle/clamp
conversion so the two paths cannot drift.

LLM details (vocabulary constraint, repeat-policy semantics, OOV
handling, system-prompt construction) live in
``core/spine/words.compose_with_llm``; this wrapper only translates the
Sequence-tab's ``[(Clip, Source)]`` input shape into the spine-level
inventory call and then translates the resulting ``WordInstance[]`` into
``SequenceClip[]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from core.remix.word_sequencer import (
    MissingWordDataError,  # re-export for callers / dialogs
    instances_to_sequence_clips,
    validate_word_data,
)
from core.spine.words import (
    RepeatPolicy,
    build_inventory,
    compose_with_llm,
)

if TYPE_CHECKING:
    from models.sequence import SequenceClip


__all__ = [
    "MissingWordDataError",  # re-exported so dialog code can import from one place
    "generate_llm_word_sequence",
]


def generate_llm_word_sequence(
    clips: list[tuple[Any, Any]],
    prompt: str,
    target_length: int,
    repeat_policy: RepeatPolicy = "round-robin",
    seed: Optional[int] = None,
    handle_frames: int = 0,
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    system_prompt: Optional[str] = None,
    think: bool = False,
    _compose_fn=None,
) -> list["SequenceClip"]:
    """Materialize a prompt-driven LLM word ordering into ``SequenceClip[]``.

    Args:
        clips: ``[(Clip, Source), ...]`` — same shape the Sequence-tab
            dialog dispatch hands every other algorithm. The corpus is
            built from these clips' aligned words.
        prompt: User-supplied generation prompt.
        target_length: Target number of words (used in the system prompt).
        repeat_policy: How to choose among multiple corpus instances when
            the LLM emits the same word more than once. See
            ``core.spine.words.compose_with_llm`` for full semantics.
        seed: Seed for ``repeat_policy="random"``.
        handle_frames: Symmetric handle padding (frames). Default 0 — raw
            cuts, per the brainstorm.
        model: Ollama model name (defaults to project setting / qwen3:8b).
        api_base: Ollama API base URL.
        temperature: Sampling temperature for the underlying Ollama call.
        timeout: Total HTTP timeout in seconds.
        system_prompt: Override the default frequency-annotated system prompt.
        _compose_fn: Test-only hook. Callable with the same signature as
            ``core.spine.words.compose_with_llm`` returning ``WordInstance[]``.
            When ``None``, the real spine composer is used.

    Returns:
        ``list[SequenceClip]`` ordered as the LLM emitted words, with
        frame coords from the shared U4/U5 frame-math helper.

    Raises:
        MissingWordDataError: any selected clip has a segment with
            ``words is None``. The dialog catches this and triggers
            alignment.
        ValueError: source has missing or non-positive ``fps``, empty
            corpus, or the LLM emits an OOV word.
        core.llm_client.LLMEmptyResponseError: LLM produced no content.
        core.llm_client.OllamaUnreachableError: Ollama not reachable.
    """
    if not clips:
        return []

    # Validation must happen BEFORE the LLM call — the dialog catches
    # MissingWordDataError and triggers alignment without burning an
    # Ollama call on incomplete data.
    validate_word_data(clips)

    inventory = build_inventory(clips)

    compose = compose_with_llm if _compose_fn is None else _compose_fn
    instances = compose(
        inventory,
        prompt=prompt,
        target_length=target_length,
        repeat_policy=repeat_policy,
        seed=seed,
        model=model,
        api_base=api_base,
        temperature=temperature,
        timeout=timeout,
        system_prompt=system_prompt,
        think=think,
    )

    return instances_to_sequence_clips(instances, clips, handle_frames)
