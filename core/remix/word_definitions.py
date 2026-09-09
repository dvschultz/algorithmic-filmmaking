"""Word Sequencer and LLM Word Composer definitions.

Both materialize word instances through the shared frame-math helper in
``core.remix.word_sequencer`` and emit clip-relative timed entries. The LLM
composer performs its provider call in ``prepare`` and records the composed
word instances as provider output so a recipe replays offline.
"""

from __future__ import annotations

from typing import Any, Sequence

from core.remix.engine import (
    AlgorithmDefinition, ClipInput, ParameterSpec, Prepared, ProposedEntry, SequenceProposal,
)


def _entries_from_sequence_clips(
    sequence_clips, inputs: Sequence[ClipInput], labels: Sequence[Any] | None = None,
) -> tuple[ProposedEntry, ...]:
    """Project materialized word clips onto proposed entries.

    ``labels`` (one per sequence clip) travel with their entry as
    ``provider_output`` so dropped zero-length words never shift the labels.
    """
    by_id = {clip.id: clip for clip, _ in inputs}
    entries = []
    for index, entry in enumerate(sequence_clips):
        clip = by_id.get(entry.source_clip_id)
        if clip is None:
            raise ValueError(f"Word sequence referenced unknown clip {entry.source_clip_id!r}")
        in_offset = entry.in_point - clip.start_frame
        out_offset = entry.out_point - clip.start_frame
        if out_offset <= in_offset:
            continue
        entries.append(ProposedEntry(
            clip.id, entry.source_id, max(0, in_offset), min(out_offset, clip.duration_frames),
            provider_output=labels[index] if labels is not None else None,
        ))
    return tuple(entries)


class WordSequencerDefinition(AlgorithmDefinition):
    key = "word_sequencer"
    version = 1
    kind = "timed"
    allow_duplicates = True
    prerequisites = ("transcribe",)
    parameters = (
        ParameterSpec(
            "mode", "string", "alphabetical", "Word ordering mode",
            choices=("alphabetical", "by_chosen_words", "by_frequency", "by_property", "from_word_list"),
        ),
        ParameterSpec("include", "array", [], "by_chosen_words: words to include, in order"),
        ParameterSpec("order", "string", "descending", "by_frequency/by_property: ascending or descending", choices=("ascending", "descending")),
        ParameterSpec("key", "string", "length", "by_property: length, duration, or log_frequency", choices=("length", "duration", "log_frequency")),
        ParameterSpec("sequence", "array", [], "from_word_list: words in the order to cut"),
        ParameterSpec("on_missing", "string", "skip", "from_word_list: skip or raise on missing words", choices=("skip", "raise")),
        ParameterSpec("handle_frames", "integer", 0, "Symmetric handle padding in frames", minimum=0, maximum=600),
    )

    def generate(self, inputs, parameters, rng, context=None):
        from core.remix.word_sequencer import generate_word_sequence

        mode = parameters["mode"]
        mode_params: dict[str, Any] = {}
        if mode == "by_chosen_words":
            mode_params["include"] = [str(w) for w in parameters["include"]]
        elif mode == "by_frequency":
            mode_params["order"] = parameters["order"]
        elif mode == "by_property":
            mode_params = {"key": parameters["key"], "order": parameters["order"]}
        elif mode == "from_word_list":
            mode_params = {"sequence": [str(w) for w in parameters["sequence"]], "on_missing": parameters["on_missing"]}
        sequence_clips = generate_word_sequence(
            list(inputs), mode=mode, mode_params=mode_params, handle_frames=parameters["handle_frames"],
        )
        entries = _entries_from_sequence_clips(sequence_clips, inputs)
        return SequenceProposal("timed", entries, notes=(f"{len(entries)} words placed",))


class WordLLMComposerDefinition(AlgorithmDefinition):
    key = "word_llm_composer"
    version = 1
    kind = "provider"
    seeded = True
    provider = True
    long_running = True
    allow_duplicates = True
    prerequisites = ("transcribe",)
    parameters = (
        ParameterSpec("prompt", "string", "", "What the local LLM should compose from the corpus"),
        ParameterSpec("target_length", "integer", 12, "Target word count", minimum=1, maximum=500),
        ParameterSpec(
            "repeat_policy", "string", "round-robin", "How repeated words pick an instance",
            choices=("round-robin", "random", "first", "longest", "shortest"),
        ),
        ParameterSpec("handle_frames", "integer", 0, "Symmetric handle padding in frames", minimum=0, maximum=600),
        ParameterSpec("model", "string", "", "Ollama model name; empty uses the project setting"),
        ParameterSpec("temperature", "number", 0.7, "Sampling temperature", minimum=0.0, maximum=2.0),
        ParameterSpec("think", "boolean", False, "Enable the model's thinking mode"),
    )

    def prepare(self, inputs, parameters, *, cancel_event=None, progress=None, resources=None):
        from core.remix.word_sequencer import validate_word_data
        from core.spine.words import build_inventory, compose_with_llm

        if not parameters["prompt"].strip():
            raise ValueError("prompt is required")
        clips = list(inputs)
        validate_word_data(clips)
        inventory = build_inventory(clips)
        resources = resources or {}
        compose = resources.get("compose_fn") or compose_with_llm
        if progress:
            progress(f"Calling local LLM to generate {parameters['target_length']} words...")
        instances = compose(
            inventory,
            prompt=parameters["prompt"],
            target_length=parameters["target_length"],
            repeat_policy=parameters["repeat_policy"],
            seed=resources.get("seed"),
            model=parameters["model"] or None,
            api_base=resources.get("api_base"),
            temperature=parameters["temperature"],
            timeout=float(resources.get("timeout", 120.0)),
            system_prompt=resources.get("system_prompt"),
            think=parameters["think"],
        )
        words = [
            {"text": w.text, "clip_id": w.clip_id, "source_id": w.source_id,
             "segment_index": w.segment_index, "word_index": w.word_index,
             "start": float(w.start), "end": float(w.end)}
            for w in instances
        ]
        return Prepared(clips, {"words": words})

    def generate(self, inputs, parameters, rng, context=None):
        from core.remix.word_sequencer import instances_to_sequence_clips
        from core.spine.words import WordInstance

        words = list((context or {}).get("words") or [])
        by_id = {clip.id for clip, _ in inputs}
        # instances_to_sequence_clips drops zero-duration words and words whose
        # clip is absent; keep the same words here so labels stay aligned.
        kept = [w for w in words if w["end"] > w["start"] and w["clip_id"] in by_id]
        instances = [
            WordInstance(
                source_id=w["source_id"], clip_id=w["clip_id"], segment_index=w["segment_index"],
                word_index=w["word_index"], start=w["start"], end=w["end"], text=w["text"],
            )
            for w in kept
        ]
        sequence_clips = instances_to_sequence_clips(instances, list(inputs), parameters["handle_frames"])
        if len(sequence_clips) != len(kept):
            raise ValueError("Composed words and materialized clips diverged")
        entries = _entries_from_sequence_clips(
            sequence_clips, inputs, labels=[{"word": w["text"]} for w in kept],
        )
        return SequenceProposal(
            "provider", entries,
            provider_outputs={"words": words, "composed_text": " ".join(w["text"] for w in words)},
            notes=(f"{len(entries)} words composed",),
        )

