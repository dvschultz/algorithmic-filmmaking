"""Exquisite Corpus and Storyteller: language-model-composed orderings.

Both call their provider in ``prepare`` and record the composed lines as
provider output, so recipes replay offline. The dialogs let users reorder the
composed lines before committing; that reorder is captured as the
``order_override`` parameter, and the dialog hands the already composed lines
back through ``resources`` so committing never repeats the provider call.
"""

from __future__ import annotations

from typing import Sequence

from core.remix.engine import (
    AlgorithmDefinition, ClipInput, ParameterSpec, Prepared, ProposedEntry, SequenceProposal,
)


def _ordered_pairs(
    inputs: Sequence[ClipInput], provider_order: list[str], override: list, *, allow_repeats: bool,
) -> list[ClipInput]:
    by_id = {clip.id: (clip, source) for clip, source in inputs}
    order = [str(c) for c in override] if override else list(provider_order)
    unknown = [c for c in order if c not in by_id]
    if unknown:
        raise ValueError(f"Ordered clips are not among the inputs: {', '.join(unknown[:5])}")
    if not allow_repeats and len(set(order)) != len(order):
        raise ValueError("Ordered clips must not repeat")
    return [by_id[c] for c in order]


class ExquisiteCorpusDefinition(AlgorithmDefinition):
    key = "exquisite_corpus"
    version = 1
    kind = "provider"
    provider = True
    allow_duplicates = True
    prerequisites = ("extract_text",)
    parameters = (
        ParameterSpec("mood", "string", "", "Mood or vibe the poem should evoke"),
        ParameterSpec("length", "string", "medium", "Poem length", choices=("short", "medium", "long")),
        ParameterSpec("form", "string", "free_verse", "Poetic form key from POETIC_FORMS"),
        ParameterSpec("model", "string", "", "LLM model override; empty uses settings"),
        ParameterSpec("order_override", "array", [], "Clip ids in final order when the composed order was edited"),
    )

    def prepare(self, inputs, parameters, *, cancel_event=None, progress=None, resources=None):
        from core.remix.exquisite_corpus import POETIC_FORMS, generate_poem

        if parameters["form"] not in POETIC_FORMS:
            raise ValueError(f"Unknown poetic form {parameters['form']!r}")
        resources = resources or {}
        texts: dict[str, str] = dict(resources.get("clip_texts") or {})
        clips_with_text = []
        for clip, _ in inputs:
            text = texts.get(clip.id) if clip.id in texts else getattr(clip, "combined_text", None)
            if text:
                clips_with_text.append((clip, text))
        if not clips_with_text:
            raise ValueError("No clips have extracted text. Run text extraction first.")
        lines = resources.get("poem_lines")
        if lines is None:
            if progress:
                progress("Generating poem...")
            poem = generate_poem(
                clips_with_text, parameters["mood"], model=parameters["model"] or None,
                length=parameters["length"], form=parameters["form"],
            )
            lines = [{"text": line.text, "clip_id": line.clip_id, "line_number": line.line_number} for line in poem]
        excluded = [clip.id for clip, _ in inputs if clip.id not in {c.id for c, _ in clips_with_text}]
        return Prepared(list(inputs), {"poem": list(lines), "excluded_clip_ids": excluded})

    def generate(self, inputs, parameters, rng, context=None):
        context = context or {}
        poem = list(context.get("poem") or [])
        ordered = _ordered_pairs(inputs, [line["clip_id"] for line in poem], parameters["order_override"], allow_repeats=True)
        text_by_clip: dict[str, list[str]] = {}
        for line in poem:
            text_by_clip.setdefault(line["clip_id"], []).append(line["text"])
        entries = []
        for clip, source in ordered:
            texts = text_by_clip.get(clip.id) or []
            entries.append(ProposedEntry(clip.id, source.id, provider_output={"line": texts.pop(0) if texts else None}))
        notes = [f"{len(poem)} poem lines"]
        if context.get("excluded_clip_ids"):
            notes.append(f"{len(context['excluded_clip_ids'])} clips had no text")
        return SequenceProposal(
            "provider", tuple(entries), notes=tuple(notes),
            provider_outputs={"poem": poem, "poem_text": "\n".join(line["text"] for line in poem)},
        )


class StorytellerDefinition(AlgorithmDefinition):
    key = "storyteller"
    version = 1
    kind = "provider"
    provider = True
    prerequisites = ("describe",)
    parameters = (
        ParameterSpec("theme", "string", "", "Optional theme or focus for the narrative"),
        ParameterSpec(
            "structure", "string", "auto", "Narrative structure",
            choices=("three_act", "chronological", "thematic", "auto"),
        ),
        ParameterSpec("target_duration_minutes", "integer", 0, "Target duration in minutes; 0 uses all clips", minimum=0, maximum=600),
        ParameterSpec("model", "string", "", "LLM model override; empty uses settings"),
        ParameterSpec("order_override", "array", [], "Clip ids in final order when the composed order was edited"),
    )

    def prepare(self, inputs, parameters, *, cancel_event=None, progress=None, resources=None):
        from core.remix.storyteller import generate_narrative

        resources = resources or {}
        clips_with_desc = []
        for clip, source in inputs:
            if clip.description:
                clip._duration_seconds = clip.duration_seconds(source.fps)
                clips_with_desc.append((clip, clip.description))
        if not clips_with_desc:
            raise ValueError("No clips have descriptions. Run Describe analysis first.")
        lines = resources.get("narrative_lines")
        if lines is None:
            if progress:
                progress("Analyzing clip descriptions...")
            narrative = generate_narrative(
                clips_with_desc,
                target_duration_minutes=parameters["target_duration_minutes"] or None,
                narrative_structure=parameters["structure"],
                theme=parameters["theme"] or None,
                model=parameters["model"] or None,
            )
            lines = [
                {"clip_id": line.clip_id, "description": line.description,
                 "narrative_role": line.narrative_role, "line_number": line.line_number}
                for line in narrative
            ]
        excluded = [clip.id for clip, _ in inputs if clip.id not in {line["clip_id"] for line in lines}]
        return Prepared(list(inputs), {"narrative": list(lines), "excluded_clip_ids": excluded})

    def generate(self, inputs, parameters, rng, context=None):
        context = context or {}
        narrative = list(context.get("narrative") or [])
        ordered = _ordered_pairs(inputs, [line["clip_id"] for line in narrative], parameters["order_override"], allow_repeats=False)
        roles = {line["clip_id"]: line["narrative_role"] for line in narrative}
        entries = tuple(
            ProposedEntry(clip.id, source.id, provider_output={"narrative_role": roles.get(clip.id)})
            for clip, source in ordered
        )
        notes = [f"{len(narrative)} narrative beats ({parameters['structure']})"]
        if context.get("excluded_clip_ids"):
            notes.append(f"{len(context['excluded_clip_ids'])} clips left out")
        return SequenceProposal(
            "provider", entries, notes=tuple(notes),
            provider_outputs={"narrative": narrative},
        )

