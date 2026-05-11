---
date: 2026-05-11
topic: word-sequencer
---

# Word Sequencer

## Summary

A new sequencer that cuts user-selected source video(s) into word-level fragments using forced-alignment-augmented transcription and resequences those fragments under five ordering rules — alphabetical (Lenka Clayton-style), chosen-words supercut, statistical (frequency / length / other), user-curated word lists, and LLM-composed sentences with strict vocabulary constraint. Joins are raw hard cuts.

---

## Problem Frame

Scene Ripper's sequencers today operate at the *clip* level — they reorder scene-detected clips. There is a class of editorial work, exemplified by Lenka Clayton's *Qaeda Quality Question Quickly Quickly Quiet* (presidential speeches rearranged alphabetically by word), that operates at the *word* level: cuts happen mid-sentence, sometimes mid-breath, and the unit of meaning becomes the spoken word rather than the shot. There is no current path through the app to produce work of this kind without manual word-by-word cutting in an external NLE.

The pain is twofold. First, the unit of cut is wrong — extracting a single word from a clip is a per-cut manual operation in tools like Premiere or DaVinci, and a 5-minute speech yields hundreds of cuts. Second, even with timestamps from existing transcription, cross-attention-derived word timestamps (which is what `faster-whisper` and `lightning-whisper-mlx` produce by default) drift by 50–150ms and routinely slice consonants and clip plosives, making spliced output unintelligible or aesthetically broken for any composition more deliberate than abstract collage.

The work this brainstorm scopes is the smallest path to making word-as-cut-unit a first-class capability in Scene Ripper, accurate enough that LLM-composed sentence-collage output (the most ambitious of the proposed modes) produces recognizable speech rather than noise.

---

## Pipeline

```
Source video
    │
    ▼
Transcribe (faster-whisper / lightning-whisper-mlx)  ← existing
    │
    ▼
Forced alignment post-step (wav2vec2-style)          ← NEW, opt-in
    │
    ▼
Structured per-word timestamps on the transcript     ← schema upgrade
    │
    ├──► Word Sequencer (preset modes) ───────────────┐
    │      alphabetical / chosen-words / frequency /  │
    │      by-property / user-curated list            │
    │                                                 ├──► SequenceClip[]
    └──► LLM-Composed Word Sequencer                  │      with tight in/out
            LLM (strict-decode) → word list ──────────┘      at word boundaries
                                                              │
                                                              ▼
                                                        Existing render
                                                        (no changes)
```

---

## Actors

- A1. **Filmmaker / artist (primary user)**: selects source(s), runs transcription with alignment, runs the word sequencer in one of the supported modes, reviews the resulting sequence in the timeline, renders.
- A2. **LLM (strict-decode capable)**: in LLM-composed mode, receives a prompt and the corpus vocabulary, emits text constrained to that vocabulary.

---

## Key Flows

- F1. **Run an alphabetical word film**
  - **Trigger:** User picks one or more sources and chooses "Word Sequencer → Alphabetical".
  - **Actors:** A1
  - **Steps:** (1) User selects sources; (2) if any selected source lacks word-aligned transcription, app offers to run alignment; (3) sequencer builds word inventory, sorts alphabetically, materializes `SequenceClip[]`; (4) sequence appears in Sequence tab.
  - **Outcome:** A timeline whose clips are individual spoken words in alphabetical order from the selected corpus.
  - **Covered by:** R6, R7, R8, R9

- F2. **Compose a sentence-collage with an LLM**
  - **Trigger:** User picks one or more sources and chooses "Word Sequencer → LLM-Composed", provides a topic/seed prompt.
  - **Actors:** A1, A2
  - **Steps:** (1) User provides prompt; (2) algorithm builds vocabulary from corpus; (3) LLM generates constrained to that vocabulary; (4) each generated word is mapped to a corpus instance via the configured selection policy; (5) `SequenceClip[]` is materialized.
  - **Outcome:** A timeline whose clips, played in order, produce the LLM-composed sentence in the speaker's actual voice using only words the speaker actually said.
  - **Covered by:** R10, R11, R12, R13

---

## Requirements

**Word-timestamping foundation**
- R1. The transcription pipeline gains an optional forced-alignment post-step that produces word-level start/end timestamps with substantially better accuracy than cross-attention-derived timestamps (target: ~20–30ms typical error).
- R2. Forced alignment is opt-in per transcription run; existing transcripts and existing features that consume them remain unaffected when alignment is not run.
- R3. Existing transcripts can be retro-aligned without re-running the transcription engine itself.
- R4. Per-word timestamp data is stored as structured data on the existing transcript field of the source or clip; no new top-level entity is introduced.
- R5. Forced alignment is language-aware. If the source's language is not supported by the available alignment model, the feature surfaces a clear message and the source is unavailable to the Word Sequencer (existing transcription continues to work).

**Sequencer — preset ordering modes**
- R6. A new sequencer ("Word Sequencer") operates on one or more user-selected sources and produces a `SequenceClip[]` whose entries reference word-level slices of those sources.
- R7. The preset-ordering algorithm supports five modes selectable from its config dialog: alphabetical, chosen-words (user supplies a list of words to include and the order they should play), by-frequency (most-frequent → least-frequent or reverse), by-property (word length, word duration, or other word-derived metric), and user-curated ordered list (user types or pastes an exact word sequence to materialize).
- R8. The sequencer outputs `SequenceClip` entries with in/out points set to the aligned word boundaries; joins are raw hard cuts with no audio crossfade, video smoothing, or held frames.
- R9. If a selected source has no aligned word data, the sequencer surfaces the missing alignment and offers to run alignment before producing the sequence.

**Sequencer — LLM-composed mode**
- R10. A separate algorithm ("LLM Word Composer" or similar) accepts a user-provided prompt or topic seed and produces a sentence/paragraph whose every word is drawn from the corpus's word inventory.
- R11. The vocabulary constraint is enforced at decode time via grammar-constrained decoding (or equivalent strict-decoding mechanism), not as a post-generation filter or substitution pass.
- R12. The generated word sequence is materialized into a `SequenceClip[]` by mapping each word to an instance in the corpus inventory.
- R13. When a generated word has multiple instances in the corpus, the algorithm uses a configurable selection policy (default: round-robin so repeated words use different physical instances; alternatives: random with seed, fixed-first, longest/shortest duration).

**Integration**
- R14. The sequencer integrates with the existing Sequence tab and follows the existing algorithm config-dialog pattern in `ui/dialogs/`.
- R15. The output is consumable by the existing render pipeline without changes to that pipeline.
- R16. New heavy ML dependencies (the alignment model, optionally a local LLM runtime) are installed on demand through the existing `core/feature_registry.py` pattern, not bundled with the base install.

---

## Acceptance Examples

- AE1. **Covers R5.** Given a French-language source and an alignment model that supports only English, when the user runs the Word Sequencer on it, the app surfaces "alignment model does not support French; source unavailable for word-level sequencing", and existing transcription of that source continues to work for other features.
- AE2. **Covers R9.** Given a selected source that has a transcript but no word-level alignment, when the user runs the Word Sequencer on it, the app offers "this source has no word-level alignment; run alignment now?" rather than failing or producing inaccurate output.
- AE3. **Covers R13.** Given a corpus containing 12 instances of the word "the" and an LLM-composed sentence that uses "the" four times, when the sequence is materialized with the default round-robin policy, the four "the" `SequenceClip` entries reference four different source instances (in inventory order), not the same instance four times.
- AE4. **Covers R11.** Given an LLM-composed mode run where the corpus does not contain the word "however", when the LLM generation completes, no `SequenceClip` references "however" — and the LLM never emitted "however" in the first place (i.e., constraint was enforced at decode, not patched after).

---

## Success Criteria

- A user can take a single 3–5 minute speech source, run transcription + alignment + the Word Sequencer in alphabetical mode, and produce a renderable timeline of every spoken word in alphabetical order on the first attempt, with most word boundaries intelligibly clean (no audible plosive-clipping or consonant-loss at the splices).
- A user can run the LLM-composed mode against the same corpus with a one-line prompt (e.g., "compose a sentence about silence") and produce a ~10–30-word sentence-collage timeline that is recognizably intelligible speech in the source speaker's voice, using only words the speaker actually said.
- A downstream planning agent reading this doc can identify which existing modules are touched (transcription pipeline modules, `core/remix/` for sequencers, `ui/dialogs/` for config UI, `core/feature_registry.py` for dependency gating) without needing to invent product behavior or scope.

---

## Scope Boundaries

**Rejected for this product** (not planned):
- Sub-word / phoneme-level cuts.
- Audio crossfades, freeze-frames, or any join smoothing at word boundaries.
- Project-wide or arbitrary-clip corpus selection (corpus is always one or more chosen sources).
- Replacing the existing transcription stack with a single WhisperX-style canonical pipeline (Approach C rejected — surgical addition of alignment is preferred to engine replacement).
- "Say it like this" / prosody- or emotion-aware variant picking (e.g., "use only angry-sounding instances of this word").
- Cloud-LLM-first strict-decode mode (cloud APIs that don't expose grammar-constrained decoding are not part of the primary path; see Key Decisions).

**Deferred future enhancements** (documented; not built in v1):
- **Speaker diarization** — would enable single-speaker corpora and per-speaker word grouping (e.g., "alphabetical of only the interviewer's words"). Could be added later as an optional analysis pass via `pyannote.audio` or by extracting WhisperX's diarization component. Not required for v1; flagged here so the v1 design does not accidentally preclude it.

---

## Key Decisions

- **Foundation = forced alignment as a post-step (Approach B), not WhisperX replacement (Approach C) and not "ship on cross-attention" (Approach A).** Rationale: Approach B surgically fixes accuracy without disturbing the working `faster-whisper` / `lightning-whisper-mlx` transcription path (preserving the Apple Silicon fast path). Approach A's ~50–150ms drift fails the LLM-composed mode's intelligibility bar; rebuilding on Approach B later would require re-doing the splicing-assumptions work.
- **LLM mode uses strict-at-decode-time constraint, not filter-time substitution.** Rationale: guarantees every emitted word is splicable from the corpus, aligns with the artistic premise (the speaker only ever says words they said), and avoids the substitution mode's drift away from corpus discipline.
- **Word-timestamp data lives on the existing transcript field; no new entity.** Rationale: minimal schema disruption; existing serialization/migration paths absorb the change naturally; word data is intrinsically transcript-shaped.
- **Preset-ordering modes and LLM-composed mode ship as two separate algorithms in `core/remix/`, not one mode-flagged algorithm.** Rationale: the input shapes are fundamentally different (presets read inventory; LLM consumes a topic prompt and requires an LLM runtime), and config dialogs would be confusingly overloaded if combined.
- **Local LLMs (Ollama / llama.cpp) are the primary path for LLM-composed mode.** Rationale: grammar-constrained decoding is not exposed by the major cloud LLM APIs (OpenAI / Anthropic / Gemini); cloud-hosted-inference providers like Together AI and Fireworks AI do support it via API and remain a viable future fallback, but local is the cleanest primary path (offline-capable, no rate limits, no per-query cost during aesthetic iteration).

---

## Dependencies / Assumptions

- A wav2vec2-style alignment model (~360MB–1GB) must be downloadable at first-run and is language-specific; multi-language workflows may require multiple model downloads or per-source language gating.
- A local LLM runtime with grammar-constrained-decoding support (e.g., `llama.cpp` directly or via a wrapper like Ollama) is assumed available for the LLM-composed mode; UI affordance for that mode is gated on detection of a compatible local LLM through `core/feature_registry.py`.
- Existing transcription engines produce per-segment or per-word coarse timestamps that the forced-alignment step can use as initial anchors, so alignment doesn't have to align from scratch.
- The existing `SequenceClip` model fields (`source_id`, `in_point`, `out_point`) are sufficient to express word-level slices; no schema change to that model is anticipated.
- The render pipeline already correctly handles very short `SequenceClip` durations (sub-second), which it does today for frame-level extraction; word-level slices fall in the same range.

---

## Outstanding Questions

### Deferred to Planning

- [Affects R1, R3][Needs research] Which forced-alignment library / model gives the best cross-platform consistency (MPS on Apple Silicon, CUDA on Linux/Windows, CPU fallback) and integrates most cleanly with the existing transcription pipeline? Candidates include `huggingface transformers` wav2vec2 variants, extracting the alignment subset from WhisperX, `ctc-segmentation`, and `aeneas`. Trade-offs span accuracy, install footprint, language coverage, and runtime.
- [Affects R10–R13][Technical] Which local LLM stack is the right primary target given the existing LiteLLM abstraction? Ollama exposes JSON-schema constraint but limited GBNF; `llama.cpp` directly supports full GBNF; `Outlines` provides grammar-based decoding on top of `transformers`. Integration shape (subprocess, HTTP server, in-process) is a planning decision.
- [Affects R8][Needs research] How many frames of "handle" should be included before/after each aligned word boundary to avoid clipping plosives even when alignment is accurate to ~20–30ms? A small empirical study (1, 2, 3 frames) on representative speech sources would resolve this; exposing it as a per-run parameter is a reasonable fallback.
- [Affects R12][Technical] In LLM-composed mode, how is the "vocabulary" presented to the model — as a flat list, as a frequency-weighted list, with part-of-speech tags, with sample sentences? Affects output fluency; pure planning question.
- [Affects R7][User decision, deferred to planning] For "by-property" mode, what is the default property and what is the user-selectable set (word length / word duration / log frequency / something else)? Defer to planning, ship with a sensible default.
