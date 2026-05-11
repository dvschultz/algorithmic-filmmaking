---
title: "feat: Word Sequencer with forced-alignment-augmented transcription"
type: feat
status: completed
date: 2026-05-11
origin: docs/brainstorms/word-sequencer-requirements.md
---

# feat: Word Sequencer with forced-alignment-augmented transcription

## Summary

This plan delivers the Word Sequencer feature by surfacing `faster-whisper`'s already-computed word timestamps (currently discarded in `core/transcription.py`), adding `ctc-forced-aligner` as an opt-in accuracy upgrade and as the required path for MLX-transcribed sources, building a word-inventory and two new sequencer algorithms in `core/spine/` + `core/remix/`, and wiring the LLM-composed mode through Ollama's existing JSON-schema-enum constraint via `core/llm_client.py` — keeping new dependencies minimal and the LLM stack inside the existing client abstraction.

---

## Problem Frame

See origin: `docs/brainstorms/word-sequencer-requirements.md`. Plan-specific framing: the foundation lift is smaller than the brainstorm assumed because `faster-whisper` already computes word timestamps that the current pipeline throws away (`core/transcription.py:446`, `core/transcription.py:599`). Reclaiming that data covers most non-MLX sources at minimal cost; forced alignment becomes a targeted upgrade rather than the only path.

---

## Requirements

**Word-timestamping foundation**
- R1. Transcription pipeline gains an optional forced-alignment post-step producing word-level start/end timestamps at ~20–30ms typical error.
- R2. Forced alignment is opt-in per transcription run; existing transcripts and downstream features remain unaffected when alignment is not run.
- R3. Existing transcripts can be retro-aligned without re-running the transcription engine.
- R4. Per-word timestamp data is stored as structured data on the existing transcript field; no new top-level entity is introduced.
- R5. Forced alignment is language-aware; if the source's language is not supported, the feature surfaces a clear message and the source is unavailable to the Word Sequencer.

**Sequencer — preset ordering modes**
- R6. A new sequencer operates on user-selected source(s) and produces `SequenceClip[]` whose entries reference word-level slices.
- R7. Five preset modes selectable from the config dialog: **alphabetical** (every word in the corpus in lexical order); **chosen-words** (subset filter — keep only instances of words on the user-supplied include list; output is grouped by include-list order, all instances of word A then all instances of word B, etc., with no repeats from the user list); **by-frequency** (every word ordered most-frequent → least-frequent, or reverse); **by-property** (word length, word duration, or other word-derived metric); **user-curated ordered list** (user supplies an exact word sequence — including repeats — that the algorithm materializes literally, e.g., `["the", "the", "the", "sky"]`). chosen-words and user-curated differ in two ways: (1) chosen-words plays *every* instance of the listed words, user-curated plays *one slot per list entry*; (2) chosen-words is grouped by include-list word, user-curated is verbatim.
- R8. Output `SequenceClip` entries have in/out points set to aligned word boundaries; joins are raw hard cuts (no crossfade, no smoothing, no held frames).
- R9. If a selected source has no aligned word data, the sequencer surfaces the missing alignment and offers to run alignment before producing the sequence.

**Sequencer — LLM-composed mode**
- R10. A separate algorithm accepts a user-provided prompt and produces a sentence/paragraph whose every word is drawn from the corpus inventory.
- R11. Vocabulary constraint is enforced at decode time via Ollama's JSON-schema-enum mechanism (Tier 1); GBNF (Tier 2) is deferred.
- R12. Generated word sequence is materialized into `SequenceClip[]` by mapping each word to a corpus instance.
- R13. When a generated word has multiple corpus instances, the algorithm uses a configurable selection policy (default: round-robin).

**Integration**
- R14. The sequencer integrates with the existing Sequence tab and follows the existing algorithm config-dialog pattern.
- R15. Output is consumable by the existing render pipeline without changes.
- R16. New heavy ML deps install on demand through `core/feature_registry.py`.

**Origin actors:** A1 (filmmaker/artist), A2 (LLM)
**Origin flows:** F1 (alphabetical word film), F2 (LLM-composed sentence-collage)
**Origin acceptance examples:** AE1 (covers R5, unsupported language), AE2 (covers R9, missing alignment), AE3 (covers R13, round-robin instance selection), AE4 (covers R11, no OOV emission)

---

## Scope Boundaries

- Sub-word / phoneme-level cuts.
- Audio crossfades, freeze-frames, or join smoothing.
- Project-wide / arbitrary-clip corpus selection (corpus is always one or more chosen sources).
- Replacing the existing transcription stack with WhisperX or any single canonical pipeline.
- "Say it like this" / prosody-aware variant picking.
- Cloud-LLM strict-decode path (major cloud APIs don't expose decode-time grammar).
- Render pipeline changes (per R15).

### Deferred to Follow-Up Work

- **Tier 2 grammar-constrained decoding via `llama-cpp-python` GBNF.** Deferred to a follow-up plan, gated on validation that Tier 1 (Ollama JSON-schema enum) is too slow at production corpus scales (see Open Questions). New `feature_registry` entry name reserved (`constrained_decoding_grammar`) but not implemented.
- **Speaker diarization.** Deferred at brainstorm tier; would enable single-speaker corpora and per-speaker word grouping. Plan-time decision: do not preclude it — `Source` and `Clip` already have IDs that future diarization can attach per-segment.

---

## Context & Research

### Relevant Code and Patterns

- `core/transcription.py:446, 599` — `faster-whisper` is already called with `word_timestamps=True`; the per-word `.words` list is currently unused.
- `core/transcription.py:257` — `TranscriptSegment` dataclass lives here (NOT in `models/clip.py`, which only TYPE_CHECKING-imports it). Current fields: `start_time / end_time / text / confidence`. Round-trip via `to_dict` (line 268) / `from_dict` (line 275). `models/clip.py:261` references it via forward declaration on `Clip.transcript: Optional[list["TranscriptSegment"]]`.
- `core/remix/staccato.py` — canonical sequencer-algorithm pattern: pure function, no Qt imports, input is `list[(Clip, Source)]`, output is algorithm-specific data that the dialog converts to `SequenceClip[]`.
- `ui/dialogs/staccato_dialog.py` — canonical algorithm-dialog pattern: modal `QDialog`, owns its `CancellableWorker`, calls `check_feature_ready` + `install_for_feature` inside the worker, builds final `Sequence` and assigns via `project.sequence = ...`.
- `ui/algorithm_config.py` — algorithm registration with keys `label`, `description`, `required_analysis`, `is_dialog`, `categories`, `allow_duplicates`.
- `ui/workers/transcription_worker.py` — analysis-worker reference: `ThreadPoolExecutor` but forces serial when backend is `mlx-whisper` (not thread-safe). Pre-loads model with progress reporting. Same constraints apply to forced alignment.
- `core/feature_registry.py:172` (`FEATURE_DEPS`), `:203` (`transcribe_mlx` reference entry), `:72-123` (`_validate_feature_runtime`) — pattern for new heavy-dep features. Must mirror in `core/dependency_manager.py`.
- `core/llm_client.py` — `ProviderType.LOCAL` exists, default model `qwen3:8b`, `check_ollama_health()` at line 201, `complete_with_local_fallback()` at line 495. JSON-mode supported; no grammar/logit-bias passthrough today.
- `core/spine/` + `tests/test_spine_imports.py` — boundary test forbids top-level imports of PySide6, mpv, av, faster_whisper, paddleocr, mlx_vlm in spine modules. Heavy imports must be inside function bodies.
- `core/project.py:620+` — `project.sequences`, `project.sequence` getter/setter; sequence-builders assign via `project.sequence = new_seq`, observers fire automatically.
- `.claude/rules/sequencer-algorithms.md` — sequencer-algorithm registry rules.
- `.claude/rules/ui-consistency.md` — UI sizing (32px min heights, 140px label width, wrap forms in `QScrollArea`).

### Institutional Learnings

- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` — new workers need the guard-flag-plus-`Qt.UniqueConnection` pattern; reset the guard at click-start so re-runs work.
- `docs/solutions/ui-bugs/cut-tab-clips-hidden-after-thumbnail-failure.md` — sub-second FFmpeg cuts fail with fractional/rational timestamps. Normalize all word boundaries to finite decimal seconds before constructing FFmpeg args.
- `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md` — route everything through the single canonical `Sequence`; no parallel widget-local copies.
- `docs/solutions/logic-errors/circular-import-config-consolidation.md` — registering the new algorithm in both `ui/algorithm_config.py` and `ui/widgets/sorting_card_grid.py` (if separate `ALGORITHMS` registry exists) requires extraction to a neutral module first.
- `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md` — LLM-composed sentence stems could produce shell metacharacters; use the existing FFmpeg escape helper.
- `docs/solutions/runtime-errors/macos-libmpv-pyav-ffmpeg-dylib-collision-20260504.md` — keep `torch`/`transformers` lazy-imported inside function bodies in spine modules; top-level heavy imports break the spine boundary test and risk macOS dyld collisions.
- `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md` — Ollama / FFmpeg subprocesses must wrap iteration in `try/finally` with explicit kill + wait; sequence cancellation with hundreds of words otherwise orphans processes.

### External References

- `ctc-forced-aligner` (MahmoudAshraf97) — chosen forced-alignment library; cleaner standalone API than WhisperX, multilingual via Meta MMS, CPU/CUDA support. CPU-only on Apple Silicon (MPS wav2vec2 is unreliable in 2026; force fp32).
- Ollama `format` parameter — JSON-schema-enum mode enforces token-level constraints; Ollama has explicitly closed all arbitrary-GBNF PRs (issue #6237, PR #1606), so `format`+enum is the only constrained-decoding API through Ollama. Non-parallelized masking → validate latency empirically at corpus scale.
- LiteLLM has no `grammar` passthrough parameter but does forward `response_format`. May need to bypass LiteLLM and call Ollama's `/api/generate` directly for the `format` parameter — verify during U5.

---

## Key Technical Decisions

- **Surface faster-whisper's discarded word data as the cheap default; layer `ctc-forced-aligner` as opt-in upgrade and as required path for MLX-transcribed sources.** Rationale: `core/transcription.py` already computes the data; reclaiming it covers non-MLX sources at zero compute cost. Forced alignment becomes a quality knob, not a hard prerequisite.
- **`ctc-forced-aligner` (over WhisperX, torchaudio, Outlines, aeneas).** Rationale: purpose-built for the "I have a transcript; align it" path, healthier maintenance than torchaudio's deprecating forced-alignment APIs, cleaner Python integration than WhisperX (no Whisper-engine coupling).
- **Strict-decode via Ollama `format`+JSON-schema-enum for v1; `llama-cpp-python` GBNF deferred.** Rationale: zero new dependencies, works through the existing `core/llm_client.py` Ollama provider. Tier 2 is the documented escape hatch if Tier 1 perf is unacceptable at corpus scales.
- **CPU-only alignment on Apple Silicon; force fp32.** Rationale: PyTorch MPS support for wav2vec2 forced-alignment is flaky in 2026. Promising MPS would mislead users; CPU on M-series is acceptable for typical short-corpus runs.
- **Spine layout: alignment runtime in `core/analysis/alignment.py`; inventory + ordering + composer plumbing in `core/spine/words.py`; thin algorithm wrappers in `core/remix/`.** Rationale: keeps `core/spine/words.py` reusable from the MCP server (and future agent tools) without GUI or heavy-ML coupling; `core/analysis/alignment.py` can do top-level heavy imports because it's not spine-boundary-restricted; `core/remix/` wrappers match `staccato.py` shape so the algorithm registry treats this like any other sequencer.
- **Schema upgrade: extend `TranscriptSegment` with optional `words: list[WordTimestamp]`; no new top-level entity (per R4).** Rationale: backward-compatible serialization; word data is intrinsically transcript-shaped.
- **Register both algorithms with `is_dialog=True`.** Rationale: a known footgun (CLAUDE.md "sequence overwrite by generic handlers") affects dialog sequencers when the generic non-dialog handler also runs.
- **Default handle-frame count = 0** (truly raw cuts, per brainstorm), exposed as a dialog parameter for user experimentation.

---

## Open Questions

### Resolved During Planning

- **Which forced-alignment library?** → `ctc-forced-aligner`.
- **Which local LLM stack for strict-decode?** → Ollama with `format`+JSON-schema-enum (Tier 1); `llama-cpp-python` GBNF deferred.
- **Default by-property mode?** → Word length, ascending. User-selectable: length / duration / log-frequency, with ascending/descending companion select.
- **Vocabulary presentation to LLM?** → Flat list with frequency annotations in the system prompt.
- **Default handle-frame count?** → 0 (raw); exposed as dialog spinner.
- **Where does the alignment runtime live?** → `core/analysis/alignment.py` (not spine — can import heavy deps at module level).
- **Where does `WordTimestamp` live?** → Colocated with `TranscriptSegment` in `core/transcription.py:257`. `models/clip.py` only TYPE_CHECKING-imports it, so a new dataclass added there would create a circular import. (Closes a feasibility finding.)
- **How is the source-language signal captured for MLX clips?** → `_parse_mlx_result` already reads `result["language"]` and discards it; U1 captures it into `TranscriptSegment.language`. Same field receives the faster-whisper detected language from the previously-discarded `info` object. (Closes a feasibility finding.)
- **Alignment audio strategy: per-clip extracted WAV vs full-source audio?** → Per-clip, matching `transcribe_clip`'s existing pattern at `core/transcription.py:560–578`. `WordTimestamp.start/end` are clip-relative seconds — same frame of reference as `TranscriptSegment.start_time`. (Closes a feasibility finding.)
- **Word-boundary-to-frame conversion convention?** → Floor on `in_point`, ceil on `out_point` (keeps the consonant onset and offset audible); all intermediate math in `float` after explicit `Fraction` conversion; handle-frame padding clamped to clip bounds. See U4 Approach. (Closes a feasibility finding.)
- **Clip-boundary attribution for words straddling a Clip?** → Not reachable: alignment runs per-clip, so every word is intrinsically attributed to its parent clip. (Closes a feasibility finding.)
- **How are repeated mode entries handled (`by_frequency(reverse=False)` ambiguity)?** → Renamed to `order: Literal["descending", "ascending"]` to eliminate boolean ambiguity. (Closes a coherence finding.)
- **Algorithm registration ownership across U4/U5/U6?** → U4 owns `word_sequencer` registration; U5 owns `word_llm_composer` registration; U6 does not modify `ui/algorithm_config.py` at all. (Closes a coherence finding.)
- **`METADATA_CHECKS` updates?** → U4 adds `transcription_with_words` to `core/cost_estimates.py`'s METADATA_CHECKS, OPERATION_LABELS, TIME_PER_CLIP, COST_PER_CLIP so the cost-confirmation gate surfaces the new dependency. `local_llm` is *not* in `required_analysis` — Ollama health is a runtime check, not per-clip metadata. (Closes a feasibility finding.)
- **Dialog UX conventions (source picker shape, conditional field visibility, error display, missing-alignment behavior, empty-corpus state, Ollama-absent state)?** → All specified in U6 "Shared dialog conventions". (Closes the design-lens findings.)
- **`chosen-words` vs `user-curated ordered list` semantic distinction?** → chosen-words is a subset filter that emits every instance of each listed word grouped by include-list order; user-curated is a verbatim materialization of an exact word sequence with one slot per entry, repeats allowed. R7 carries the full definition; U4 implements as two distinct functions (`by_chosen_words` vs `from_word_list`). (Closes a coherence finding.)
- **U4's dependency on U3?** → U4 has a data-shape dependency on U1 (uses the new `WordTimestamp` and `language` fields) but only a runtime-data dependency on U3 (the worker that populates words). `core/spine/words.py` does not import the U3 worker, and U4 can be implemented + tested against mocked `Clip.transcript[*].words`. (Closes a coherence finding.)
- **Retroactive alignment skip granularity?** → Per-clip (re-align if any `TranscriptSegment.words is None`). See U3 Approach. (Closes a feasibility finding.)
- **`ui/algorithm_config.py` neutral module?** → Verified during planning to already exist. Both `ui/tabs/sequence_tab.py` and `ui/widgets/sorting_card_grid.py` consume from it. No pre-extraction work needed.

### Deferred to Implementation

- **[Affects U5][Needs validation] Ollama enum-constraint latency at corpus scale.** With ~5000-item enums, masking is non-parallelized in current Ollama builds; if generation latency is unacceptable for typical artistic corpora (single 5-minute speech → ~500 unique words; multiple sources → 2000–5000 unique words), Tier 2 (`llama-cpp-python` GBNF) becomes a follow-up plan. Measure during U5 implementation; document findings. **Go/no-go threshold for v1 ship:** for a 1000-word corpus and a 20-word target, full-sentence generation should complete under ~30s on `qwen3:8b` on representative hardware. Slower than that → flag for Tier 2.
- **[Affects U5][Technical] Does LiteLLM strip Ollama's `format` parameter?** If LiteLLM doesn't forward `format` to Ollama, call `/api/generate` directly (still through `core/llm_client.py` as an Ollama-specific helper). Verify by inspecting LiteLLM's Ollama provider during U5; fall back to direct HTTP if needed.
- **[Affects U4][Needs research-at-impl-time] Word normalization** (case, punctuation, contractions). Start with lowercase + strip surrounding ASCII punctuation, contractions intact (`"don't"` is one word); revisit if grouping behavior is surprising.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
Source video
    │
    ▼
Transcribe (existing: faster-whisper / lightning-whisper-mlx)
    │
    ├─ faster-whisper path ─► capture .words AND info.language    ◄── U1
    └─ mlx-whisper path ────► capture result["language"]; words=None
    │
    ▼
TranscriptSegment.words (Optional)
TranscriptSegment.language (Optional)                             ◄── U1 schema
    │
    ▼
(optional for faster-whisper / required for MLX)
Forced alignment post-step                                        ◄── U2, U3
   ctc-forced-aligner: per-clip extracted WAV
   → list[WordTimestamp], clip-relative seconds
   language read from segments[0].language
    │
    ▼
Word inventory build                                              ◄── U4
   build_inventory(clips: list[(Clip, Source)]) -> WordInventory
   { by_word: {normalized: [WordInstance]},
     by_clip: {clip_id: [WordInstance]} }
    │
    ├──► Preset modes                                             ◄── U4
    │       alphabetical / by_chosen_words / by_frequency /
    │       by_property / from_word_list
    │       │
    │       ▼
    │     list[WordInstance]  ───►  SequenceClip[]
    │                                 (floor(start*fps)/ceil(end*fps) frames,
    │                                  clip-relative, handle padding clamped)
    │
    └──► LLM-composed mode                                        ◄── U5
            Ollama format=JSON-schema-enum(corpus)
            → {"words": [...]}
            │
            ▼
          map each emitted word → WordInstance via repeat-policy
            │
            ▼
          SequenceClip[]
    │
    ▼
project.sequence = Sequence(tracks=[Track(clips=SequenceClip[])])
    │
    ▼
existing render pipeline (unchanged — R15)
```

---

## Implementation Units

### U1. TranscriptSegment word-data schema, language capture, and surface faster-whisper's existing word timestamps

**Goal:** Extend `TranscriptSegment` (which lives in `core/transcription.py:257`) with optional `words` and `language` fields, round-trip them through serialization, capture the per-word data already returned by `faster-whisper`, and capture the detected source language from both backends so forced alignment in U2 has a reliable signal.

**Requirements:** R1, R2, R3, R4, R5

**Dependencies:** None

**Files:**
- Modify: `core/transcription.py` (add `WordTimestamp` dataclass colocated with `TranscriptSegment` at line 257; add `words: Optional[list[WordTimestamp]]` and `language: Optional[str]` fields to `TranscriptSegment`; update `to_dict` line 268 and `from_dict` line 275; at line 445 and 599 capture `.words` from `faster-whisper` segments into `WordTimestamp` objects; capture `info.language` from `faster-whisper`'s `info` object — currently discarded — and `result["language"]` from `_parse_mlx_result` at line 626)
- Modify: `models/clip.py` (no field changes; existing TYPE_CHECKING import at line 15 picks up the new `TranscriptSegment` shape automatically; verify the `from_dict` path at line 493–508 forwards the new fields through `TranscriptSegment.from_dict` unchanged)
- Test: `tests/test_transcription.py` (TranscriptSegment serialization round-trip with words + language; faster-whisper word_timestamps population; MLX language capture)

**Approach:**
- New `@dataclass WordTimestamp(start: float, end: float, text: str, probability: Optional[float] = None)` colocated with `TranscriptSegment` in `core/transcription.py`. Keep `probability` optional because forced alignment surfaces a different confidence metric than `faster-whisper`.
- `TranscriptSegment.words: Optional[list[WordTimestamp]]` — `None` means "no word data surfaced yet for this segment" (MLX case before alignment, or pre-feature project files). Empty list `[]` means "we tried alignment and produced no words" (rare — silence/instrumental). The two are deliberately not conflated.
- `TranscriptSegment.language: Optional[str]` — ISO 639-1 code (e.g., `"en"`, `"fr"`). `None` only when neither backend produced one (older project files). faster-whisper's `info.language` and MLX's `result["language"]` populate it. Single value per segment because both backends produce one language per call; per-segment storage means future per-segment diarization can override without schema change.
- `from_dict` accepts the absence of either field (back-compat with old project files).
- In `core/transcription.py:445` (faster-whisper `transcribe_video`) and `:599` (faster-whisper `transcribe_clip`), the loop building `TranscriptSegment` objects also constructs a `WordTimestamp` per word from segment `.words` and stores `info.language` (returned by faster-whisper's `transcribe()` alongside segments) into every segment from that call.
- In `_parse_mlx_result` at `:626`, read `result["language"]` (currently unused) and write it onto every emitted `TranscriptSegment`. The MLX backend still leaves `words = None` — alignment fills that in U2.

**Execution note:** Test the serialization round-trip first — old-format files must still load.

**Patterns to follow:**
- Existing `TranscriptSegment` shape and `to_dict` / `from_dict` style.

**Test scenarios:**
- Happy path: A `TranscriptSegment` constructed with a list of `WordTimestamp` and `language="en"` round-trips through `to_dict` → `from_dict` identically.
- Happy path: An old-format dict (no `words` and no `language` keys) deserializes to a `TranscriptSegment` with `words = None` and `language = None`.
- Happy path: After `transcribe_video()` with the faster-whisper backend on an English source, every `TranscriptSegment` has a non-empty `words` list AND `language = "en"`.
- Happy path: After `transcribe_video()` with the MLX backend on a French source, every `TranscriptSegment` has `words = None` AND `language = "fr"` (language was previously discarded; this is the fix).
- Edge case: A `Clip` with `transcript = None` continues to serialize/deserialize correctly (no regression on clips without transcripts).
- Edge case: `words = []` (empty list — alignment ran but produced no words) is preserved across round-trip and distinguished from `words = None`.

**Verification:**
- Loading an existing project from disk works unchanged.
- `tests/test_spine_imports.py` continues to pass (schema change adds dataclass fields only; no new heavy imports).
- After `transcribe_video()` runs on a fresh MLX-backend source, that source is now ready for U2 alignment without the user supplying a language argument (the segment carries it).

---

### U2. Forced-alignment runtime module

**Goal:** Implement `align_words(audio_path, transcript_segments, language) -> list[WordTimestamp]` using `ctc-forced-aligner`. Register the heavy dep through `core/feature_registry.py`.

**Requirements:** R1, R3, R5

**Dependencies:** U1

**Files:**
- Create: `core/analysis/alignment.py`
- Modify: `core/feature_registry.py` (add `word_alignment` entry to `FEATURE_DEPS`; add `ensure_word_alignment_runtime_available()` branch in `_validate_feature_runtime`)
- Modify: `core/dependency_manager.py` (mirror the new feature entry)
- Test: `tests/test_alignment.py`

**Approach:**
- Pure module — no Qt, no project imports. Inputs are paths and pure-data segments; output is pure-data words.
- Public API: `align_words(audio_path: str, transcript_segments: list[TranscriptSegment]) -> list[WordTimestamp]`. **Language is read off `transcript_segments[0].language`** (which U1 populates from both backends). When `language is None` (only true for legacy project files predating U1), raise `LanguageUnknownError` with the hint to re-transcribe; do not silently default to English. On a language the alignment model doesn't support, raise `UnsupportedLanguageError(language=...)`.
- **Audio strategy: per-clip extracted WAV, matching the existing `transcribe_clip` pattern at `core/transcription.py:560–578`.** Per-clip extraction means `WordTimestamp.start` / `WordTimestamp.end` are **clip-relative seconds** (the same frame of reference as `TranscriptSegment.start_time`), which preserves the existing transcript-time contract and avoids a source-vs-clip offset translation later. Trade-off: one FFmpeg pass per clip vs one per source. Accepted because (a) it mirrors existing transcription behavior, (b) per-clip cancellation is naturally granular, and (c) clip-boundary attribution becomes trivial (every word belongs to the clip whose audio produced it).
- `ctc-forced-aligner` and `torch` imports go inside the function body (avoids paying their startup cost when this module is loaded but not used; also keeps the module importable without the heavy deps for unit tests that mock the engine).
- Force fp32 on Apple Silicon; do not attempt MPS — MPS wav2vec2 is unreliable in 2026.
- Model download via `feature_registry.install_for_feature("word_alignment", progress_cb)` on first use.

**Patterns to follow:**
- `core/analysis/audio.py` — analysis module with lazy heavy-dep imports, clean public API.
- `core/feature_registry.py:203` (`transcribe_mlx`) — native-install gating pattern.

**Test scenarios:**
- Happy path: align a known short clip-audio + transcript with the model mocked → returns a list of `WordTimestamp` whose count matches the transcript word count. Timestamps are clip-relative seconds (start of clip = 0.0).
- Happy path: word boundaries returned are within `[0, clip_duration]` (no times beyond clip audio length).
- Happy path: language is read from `segments[0].language` (`"en"`) and passed to the alignment model.
- Edge case: empty transcript → returns empty list (no model load attempted, no FFmpeg extract).
- Edge case: very short clip audio (< 1 s) does not crash.
- Error path: `segments[0].language is None` raises `LanguageUnknownError` (legacy transcript pre-U1).
- Error path: unsupported language (e.g., a language outside the MMS multilingual coverage) raises `UnsupportedLanguageError` with the language name. *Covers AE1.*
- Error path: missing model triggers `install_for_feature` (mock the install path).
- Integration: per-clip audio is extracted by FFmpeg, alignment runs, FFmpeg temp WAV is cleaned up afterward (no leftover temp files).

**Verification:**
- Module imports successfully on a clean Python env without `ctc-forced-aligner` installed (lazy-import contract holds).
- `core/feature_registry.py` correctly detects the `word_alignment` feature as missing/installed.

---

### U3. ForcedAlignmentWorker and project-level wiring

**Goal:** A new `CancellableWorker` subclass that runs alignment over a list of sources, surfaces progress, populates `TranscriptSegment.words` through the project, and is safe to cancel mid-run.

**Requirements:** R1, R2, R3, R5, R9

**Dependencies:** U1, U2

**Files:**
- Create: `ui/workers/forced_alignment_worker.py`
- Modify: `ui/tabs/analyze_tab.py` (add "Run word-level alignment" affordance — or wherever transcription is run today; locate the existing transcription trigger and add the alignment action adjacent to it)
- Test: `tests/test_forced_alignment_worker.py`

**Approach:**
- Inherits `CancellableWorker` (`ui/workers/base.py`).
- Signals: `progress(int, int)`, `clip_aligned(str, list)` (`clip_id`, `WordTimestamp` list as data), `alignment_completed()`, inherited `error(str)`.
- Serial per-clip (alignment is CPU-bound on Mac; not thread-safe per `core/analysis/alignment.py`).
- Pre-load the alignment model with progress reporting before the work loop (pattern from `ui/workers/transcription_worker.py:162-174`).
- Calls `core.feature_registry.check_feature_ready("word_alignment")` and `install_for_feature("word_alignment", ...)` inside the worker before starting.
- **Skip predicate (per-clip granularity):** for each clip, run alignment when *any* `TranscriptSegment.words is None`. When all segments already have `words` (even empty lists), skip the clip entirely — its alignment is fresh. This is per-clip not per-segment because alignment-per-clip is the FFmpeg-extract granularity from U2; reusing a model-loaded process to re-align only some segments of the same clip is more complex than it's worth. Edge case: if a prior cancellation interrupted *between segments of the same clip*, the clip has a mix of populated and `None` segments — current rule re-aligns the whole clip, which is correct (idempotent re-alignment produces the same word boundaries).
- After each clip alignment, the worker emits `clip_aligned(clip_id, words)` and the main thread writes the words back through the project (use the existing pattern for project mutation — never mutate models directly from the worker).
- Uses `Qt.UniqueConnection` on the `finished` signal handler and a guard flag pattern to prevent duplicate signal delivery on re-runs (per `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`).
- On a worker rerun, reset the guard flag at click-start so re-runs work.

**Execution note:** Write a failing test for the cancel-mid-run signal contract first.

**Patterns to follow:**
- `ui/workers/transcription_worker.py` (model preload + per-item processing + serial-execution constraint).
- Guard-flag-plus-`Qt.UniqueConnection` from the QThread-destroyed-duplicate-signal-delivery learning.

**Test scenarios:**
- Happy path: Worker processes 3 clips → emits `clip_aligned` three times, then `alignment_completed` once.
- Happy path: `progress(i, total)` emits incrementally across the run.
- Edge case: Empty clip list → emits `alignment_completed` immediately without any `clip_aligned`.
- Error path: `worker.cancel()` called mid-run → no further `clip_aligned` signals; `finished` still fires exactly once; cleanup runs.
- Error path: alignment raises inside the worker → `error(message)` emits; `alignment_completed` does not fire; the worker is clean for reuse on next run.
- Integration: After a successful run, the affected `TranscriptSegment.words` is populated on the project (project observer fires).
- Integration: Re-running on already-aligned sources is a no-op (no model re-load if the data is fresh).

**Verification:**
- Cancelling a 100-clip alignment partway through leaves the partial results persisted (no all-or-nothing semantics — partial progress is real progress).
- `Qt.UniqueConnection` is in place on the finished handler; double-clicking "Run alignment" doesn't double-fire downstream effects.

---

### U4. Word inventory + preset-modes algorithm + algorithm registration

**Goal:** Build `core/spine/words.py` (inventory + ordering modes), the thin remix wrapper `core/remix/word_sequencer.py` that materializes mode output into `SequenceClip[]`, and register the algorithm canonically (this unit *owns* the `word_sequencer` registration end-to-end — U6 does not re-touch it).

**Requirements:** R6, R7, R8, R9, R14

**Dependencies:** U1 (data-shape dependency). U3 is a runtime-data prerequisite (clips must be aligned before the algorithm produces useful output) but **not** a code dependency — `core/spine/words.py` does not import the U3 worker, and U4 can be implemented and tested before U3 lands by mocking `Clip.transcript[*].words`.

**Files:**
- Create: `core/spine/words.py`
- Create: `core/remix/word_sequencer.py`
- Modify: `ui/algorithm_config.py` — register `word_sequencer` with `label="Word Sequencer"`, `is_dialog=True`, `required_analysis=["transcription_with_words"]`, `categories=["word", "experimental"]`, `allow_duplicates=False`. **This is the authoritative registration; no later unit modifies this entry.**
- Modify: `core/cost_estimates.py` — add the `transcription_with_words` entries: a `METADATA_CHECKS["transcription_with_words"]` lambda (`lambda clip: bool(clip.transcript and any(seg.words for seg in clip.transcript))`), an `OPERATION_LABELS["transcription_with_words"]` label (e.g., `"Word-level alignment"`), and conservative `TIME_PER_CLIP` / `COST_PER_CLIP` rows so the cost-confirmation gate surfaces this dependency. Without these entries the gate silently skips the new algorithm.
- Test: `tests/test_spine_words.py`
- Test: `tests/test_word_sequencer.py`
- Test: `tests/test_cost_estimates.py` (or wherever cost-estimate tests live — add a case proving the new key is recognized)

**Approach:**

*Input contract — clips, not sources.* The Sequence tab's dispatch (`ui/tabs/sequence_tab.py:1426` for the staccato pattern) hands the dialog `list[(Clip, Source)]`. The word sequencer accepts the same shape; **the corpus is the *selected clips' words*, not "every word in the source"**. This is the simpler product semantics: "make a word film from these clips I selected" rather than "make a word film from every source that happens to have at least one clip selected". The dialog deduplicates `Source` instances internally for the alignment-gating step.

*`core/spine/words.py`:*
  - `@dataclass WordInstance(source_id: str, clip_id: str, segment_index: int, word_index: int, start: float, end: float, text: str)`. `start` / `end` are **clip-relative seconds** (the same frame of reference as `TranscriptSegment.start_time` set by U2's per-clip alignment).
  - `@dataclass(frozen=True) WordInventory(by_word: dict[str, list[WordInstance]], by_clip: dict[str, list[WordInstance]])` — both indices are populated at build time so callers can iterate either way without re-walking the data.
  - `build_inventory(clips: list[tuple[Clip, Source]]) -> WordInventory`. Word normalization: lowercase + strip surrounding ASCII punctuation; preserve original `text` on the instance. Contractions stay as-is (`"don't"` is one word, not `"do"` + `"n't"`) — `faster-whisper`'s tokenizer already commits to that.
  - Mode functions:
    - `alphabetical(inv) -> list[WordInstance]` — lexical order over `inv.by_word.keys()`, then encounter order within each word.
    - `by_chosen_words(inv, include: list[str]) -> list[WordInstance]` — emits every instance of each listed word, **grouped by include-list order**. Unknown words in `include` are silently dropped (no error — the dialog can preview which were dropped). This is "play every 'never', then every 'always', then every 'silence'."
    - `by_frequency(inv, order: Literal["descending", "ascending"] = "descending") -> list[WordInstance]` — named keyword (not `reverse=` bool) to remove ambiguity about what `False` means. Default emits most-frequent first.
    - `by_property(inv, key: Literal["length", "duration", "log_frequency"], order: Literal["descending", "ascending"] = "ascending") -> list[WordInstance]` — default emits shortest first when `key="length"`.
    - `from_word_list(inv, sequence: list[str], on_missing: Literal["skip", "raise"] = "skip") -> list[WordInstance]` — literal materialization of `sequence`, one slot per entry, repeats allowed (e.g., `["the","the","the","sky"]` produces 4 slots). Instance picked per the higher-layer repeat policy (round-robin / random / etc.).
  - No top-level imports of PySide6, mpv, av, faster_whisper, paddleocr, mlx_vlm — verify against `tests/test_spine_imports.py`.

*`core/remix/word_sequencer.py`:*
  - `generate_word_sequence(clips, mode, mode_params, handle_frames=0) -> list[SequenceClip]`.
  - **Frame conversion convention (committed):**
    - Source fps is the source of truth. Read from `Source.fps`; if missing, raise rather than guess.
    - `WordTimestamp.start_clip_relative_seconds` → `clip_relative_in_seconds = max(0.0, start - (handle_frames / fps))`.
    - `WordTimestamp.end_clip_relative_seconds` → `clip_relative_out_seconds = min(clip_duration_seconds, end + (handle_frames / fps))`.
    - Convert to integer frames: `in_point = floor(clip_relative_in_seconds * fps)` (keep the consonant onset audible).
    - `out_point = ceil(clip_relative_out_seconds * fps)` (keep the consonant offset audible).
    - All intermediate math uses Python `float` after explicit conversion from any `Fraction` (per the FFmpeg-fractional-duration learning).
    - `SequenceClip.source_clip_id` is the parent `Clip.id`; `SequenceClip.source_id` is `Clip.source_id`.
    - Clip-boundary attribution is implicit because alignment runs per-clip in U2: every `WordInstance` already belongs to exactly one clip. No straddling case.
  - Validates all selected clips have aligned word data (`all(seg.words is not None for seg in clip.transcript or [])`); raises `MissingWordDataError(clip_ids=[...])` if any are missing. The dialog catches this and triggers alignment.

**Patterns to follow:**
- `core/remix/staccato.py` (pure function, no Qt, returns algorithm-shaped data).
- Spine module boundary discipline.

**Test scenarios:**
- Happy path: Inventory built from 2 clips spanning 1 source with known segment+word data shows correct grouping by normalized word and correct `by_clip` index.
- Happy path: `alphabetical` returns word instances in lexical order over `by_word` keys.
- Happy path: `by_frequency(order="descending")` returns most-frequent word first; `order="ascending"` reverses it.
- Happy path: `by_property(key="length")` defaults to ascending (shortest first); explicit `order="descending"` reverses.
- Happy path: `by_chosen_words(["never", "always"])` returns every instance of "never" (grouped together), then every instance of "always" — preserving include-list order.
- Happy path: `from_word_list(["the", "the", "the", "sky"])` returns exactly 4 instance slots in that order.
- Edge case: Empty `clips` list → empty inventory → empty result list; no crash.
- Edge case: `by_chosen_words(["xyz"])` where "xyz" is not in corpus → empty result (silent drop is the default; no exception).
- Edge case: `from_word_list(..., on_missing="raise")` with missing words raises `MissingWordsError` listing the absentees.
- Edge case: Word boundary at clip start (`start = 0.0`) with `handle_frames=2` — `in_point` clamps to 0 (not negative).
- Edge case: Word boundary at clip end with `handle_frames=2` — `out_point` clamps to `clip_duration * fps` (not past the end).
- Edge case: Source with `fps` missing raises with a clear message (no silent default).
- Edge case: Word with `start == end` (zero-duration word — rare ASR artifact) is dropped, not materialized as a 0-frame `SequenceClip`.
- Error path: `generate_word_sequence` called on a `Clip` with any `TranscriptSegment.words = None` raises `MissingWordDataError(clip_ids=[...])`. *Covers AE2 (the dialog catches this and triggers alignment).*
- Integration: `tests/test_spine_imports.py` passes (no spine boundary violation).
- Integration: `core/cost_estimates.py` recognizes the `transcription_with_words` key (test in `tests/test_cost_estimates.py` confirms the cost gate surfaces it).

**Verification:**
- Running the full pipeline on a 30-second test source produces a valid `SequenceClip[]` whose timing maps to expected frame boundaries.
- Spine boundary test continues to pass.

---

### U5. LLM-composed mode algorithm + algorithm registration

**Goal:** Add the LLM word composer to `core/spine/words.py` and the remix wrapper `core/remix/word_llm_composer.py`. Use Ollama's `format`+JSON-schema-enum for strict vocabulary constraint. Register the algorithm canonically (this unit *owns* the `word_llm_composer` registration — U6 does not re-touch it).

**Requirements:** R10, R11, R12, R13, R14

**Dependencies:** U4

**Files:**
- Modify: `core/spine/words.py` (add `compose_with_llm(inventory, prompt, target_length, repeat_policy, seed=None) -> list[WordInstance]`)
- Modify: `core/llm_client.py` (add `complete_with_enum_constraint(prompt, vocabulary, ...)` helper; may need to bypass LiteLLM and call Ollama's `/api/generate` directly with `format` payload if LiteLLM strips the parameter)
- Create: `core/remix/word_llm_composer.py`
- Modify: `ui/algorithm_config.py` — register `word_llm_composer` with `label="LLM Word Composer"`, `is_dialog=True`, `required_analysis=["transcription_with_words"]`, `categories=["word", "experimental", "llm"]`, `allow_duplicates=False`. **This is the authoritative registration; no later unit modifies this entry.** Note: `local_llm` does NOT belong in `required_analysis` — that key drives `METADATA_CHECKS` (per-clip data dependencies), not runtime-service availability. Ollama availability is checked at dialog open and inside the worker via `check_ollama_health()` (see U6 Dialog UX Conventions for the disabled-algorithm-entry UX when Ollama is absent).
- Test: `tests/test_word_llm_composer.py`
- Test: `tests/test_llm_client_constrained.py`

**Approach:**
- Schema for the constrained call: `{"type": "object", "properties": {"words": {"type": "array", "items": {"type": "string", "enum": [...vocabulary]}}}, "required": ["words"]}`.
- Validate response: non-`None`, non-empty `words` list. Per a recurring LLM gotcha, never trust the LLM not to return `None` content.
- Repeat-policy implementation: `"round-robin"` (default — cycle through instances per word), `"random"` (seeded for determinism), `"first"` (always use the first instance), `"longest"` / `"shortest"` (by `WordInstance.end - start`).
- System prompt includes frequency annotations of the vocabulary so the model has usage hints (e.g., `"the (42), a (38), and (28), ..."`).
- On Ollama unavailable: surface a clear error with a "check Ollama health" hint (reuse `check_ollama_health()` from `core/llm_client.py:201`).
- Subprocess discipline: any subprocess spawned to talk to Ollama or to wrap a fallback path must use `try/finally` with explicit kill+wait (per the subprocess-cleanup-on-exception learning).
- **Implementation-time measurement:** record decode latency for a ~5000-item enum on `qwen3:8b`. If unacceptable (e.g., >30s for a 20-word sentence), flag for Tier 2 follow-up and document findings inline.

**Execution note:** Stub LLM responses in unit tests; an integration test against a real local Ollama runs only when `SCENE_RIPPER_OLLAMA_INTEGRATION=1` is set in the env.

**Patterns to follow:**
- `core/llm_client.py` — existing Ollama provider patterns, `check_ollama_health()`, `complete_with_local_fallback()`.
- JSON-mode usage in `core/analysis/cinematography.py:565` and `:698`.

**Test scenarios:**
- Happy path: Stubbed LLM returns `{"words": ["i", "have", "always", "loved", "silence"]}` → 5 `SequenceClip` entries, each from a corpus instance.
- Happy path: With 12 instances of "the" in the corpus and an LLM output using "the" 4 times under `round-robin`, the 4 emitted `SequenceClip`s reference 4 different source instances (in inventory order). *Covers AE3.*
- Happy path: With `repeat_policy="random"` and a fixed seed, two runs produce identical SequenceClip sequences.
- Edge case: LLM emits a single-word response → 1 SequenceClip.
- Edge case: Corpus has only one instance of a word that the LLM emits 3 times → round-robin reuses the single instance three times; behavior is documented and tested.
- Error path: LLM returns `None` content → raises `LLMEmptyResponseError` with a clear message (do not silently produce an empty sequence).
- Error path: Ollama unreachable → raises with the Ollama health-check hint.
- Error path: LLM somehow emits an OOV word (should not happen if `format`+enum is enforced) → fail loudly so the bug surfaces, not silently drop. *Covers AE4 (when the constraint works correctly, no OOV emission occurs).*
- Integration: Real Ollama integration test (gated by env var) confirms `format` parameter survives the transport and the enum is enforced.

**Verification:**
- A test sentence runs end-to-end against a stubbed LLM and produces a renderable `SequenceClip[]`.
- Real-Ollama integration test confirms decode latency is recorded; the test asserts a soft-upper-bound (warning, not failure) so future regressions are visible.

---

### U6. Dialogs and end-to-end smoke

**Goal:** Two QDialog subclasses (preset modes + LLM composer); wire them into `ui/tabs/sequence_tab.py`'s algorithm-dispatch ladder; end-to-end smoke test from Sequence tab through render. Algorithm registration in `ui/algorithm_config.py` is *not* in this unit — U4 and U5 each register their own entry authoritatively.

**Requirements:** R6, R7, R8, R9, R12, R14, R15, R16

**Dependencies:** U4, U5

**Files:**
- Create: `ui/dialogs/word_sequencer_dialog.py`
- Create: `ui/dialogs/word_llm_composer_dialog.py`
- Modify: `ui/tabs/sequence_tab.py` — add two branches to the hand-maintained algorithm-dispatch if-ladder around `sequence_tab.py:778-807` (the `_show_<algorithm>_dialog` pattern), instantiating each new dialog with `(clips=clips, project=self.project, parent=self)` per the staccato pattern at `sequence_tab.py:1426`.
- Modify (only if needed): `ui/widgets/sorting_card_grid.py` — should pick up the new algorithms automatically if it reads from `ui/algorithm_config.py`. Verify during U6.
- Test: `tests/test_word_sequencer_dialog.py`
- Test: `tests/test_word_llm_composer_dialog.py`

**Approach:**

*Shared dialog conventions* (apply to both Word Sequencer and LLM Composer):

- **Source picker.** `QListWidget` with checkboxes; one row per `Source` represented in the user's selected clips (deduplicate by `Source.id`). Each row shows source filename, total duration, and an alignment-status badge: `✓ aligned`, `… needs alignment`, or `⚠ unsupported language`. Default state: every row checked. The user can uncheck rows to exclude sources without leaving the dialog. Pre-selection mirrors the Sequence tab's current selection.
- **Disabled rows** (unsupported-language sources). The row is checkable=False with greyed-out text and a tooltip carrying the AE1 message ("alignment model does not support \<language\>; source unavailable for word-level sequencing"). The user can still proceed with the remaining sources; the Accept button stays enabled as long as at least one checked row is available.
- **Conditional field visibility.** Mode-specific controls are wrapped in `QWidget` containers that toggle `.setVisible()` on mode change. When hidden, **user-entered values are preserved** (the widget keeps its state in memory). The dialog's `QScrollArea` recomputes layout on visibility change so the dialog height stays sane.
- **Missing-alignment behavior.** When the user accepts and any selected source lacks word data, the dialog **auto-runs alignment** (no extra confirmation modal) and shows an inline progress indicator (a `QProgressBar` below the form, replacing the action buttons until completion). The user can cancel via a button that swaps in during alignment; cancellation aborts the worker per U3's contract and the dialog returns to its pre-Accept state. The "always prompt vs auto-run" question is **resolved: auto-run**, with the cancel button as the escape valve.
- **LLM Composer generation state.** When the user accepts the LLM Composer, the dialog enters a generating state with the same in-dialog `QProgressBar` pattern. The prompt input becomes read-only during generation; the Accept button is replaced by a Cancel button that interrupts the in-flight LLM call (subprocess kill per the cleanup-on-exception learning). Empty / `None` responses surface as an inline error label without dismissing the dialog.
- **Error display.** A single `QLabel` with the project's error style (red text, leading icon) sits below the action button row. It's hidden when empty and `.setText()` populates it for any user-surfaced error: `MissingWordDataError` ("source X has no word data — alignment ran but produced no segments — try re-transcribing"), Ollama-unreachable, `LLMEmptyResponseError`, install-feature failures. Inline label, not `QMessageBox` popup — the user keeps the dialog state visible.
- **Empty corpus.** When every checked source's clips combined produce zero words (silence, instrumental-only, all words filtered out by chosen-words), the Accept button is disabled and the inline error label reads "no words in selected sources — adjust your selection or include list".
- **Ollama-absent state (LLM Composer only).** Detected at dialog construction via `check_ollama_health()`. When Ollama is unreachable, the dialog opens with a single inline message ("Local LLM not detected — install or start Ollama, then re-open this dialog") and a "Open feature install" button that invokes `install_for_feature("ollama_runtime", ...)`. The Accept button is disabled until Ollama is healthy. The algorithm-picker entry in the Sequence tab is *not* greyed out in v1 — the Ollama check happens at dialog open, not at picker render, because we'd otherwise need to poll Ollama health on every Sequence tab refresh.

*Per-dialog parameters:*

- **Word Sequencer dialog parameters:**
  - Source picker (shared convention above).
  - Mode select — `QComboBox` with 5 entries: Alphabetical, Chosen Words, By Frequency, By Property, User-Curated Ordered List.
  - **Chosen-words input** (visible when mode = Chosen Words): multi-line `QPlainTextEdit`, one word per line (also accept comma- or whitespace-separated; normalize on parse). Below it: a read-only label showing "in corpus: N / M" as the user types, so they see which entries match. Case-insensitive matching.
  - **Property select** (visible when mode = By Property): `QComboBox` with `length` / `duration` / `log_frequency`; companion `QComboBox` for `ascending` / `descending` (default ascending).
  - **User-curated word list** (visible when mode = User-Curated Ordered List): multi-line `QPlainTextEdit`, same parser as chosen-words but order- and repetition-preserving. Below it: "N slots; M unrecognized" preview.
  - Handle-frame count: `QSpinBox`, default 0, range 0–10.
- **LLM Composer dialog parameters:**
  - Source picker (shared convention above).
  - Prompt input: multi-line `QPlainTextEdit`; placeholder "e.g., 'compose a sentence about silence'".
  - Target length: `QSpinBox`, default 20 (words), range 1–200.
  - Repeat policy select: `QComboBox` with round-robin / random with seed / first / longest / shortest. Default round-robin.
  - Seed: `QSpinBox`, visible only when repeat policy = random; default 0; range 0–2^31.

*Dialog mechanics* (both):
  - Modal `QDialog` subclass following `ui/dialogs/staccato_dialog.py`.
  - Owns workers internally (the `ForcedAlignmentWorker` from U3 when alignment is needed; a lightweight `LLMComposerWorker` for the LLM generation step that calls into spine for the actual sequencing).
  - Calls `check_feature_ready("word_alignment")` / `install_for_feature` inside the worker, not in dialog code (per the staccato pattern).
  - On accept: builds a complete `Sequence` and assigns via `project.sequence = ...` (no parallel widget-local copies; per `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`).
  - UI sizing follows `.claude/rules/ui-consistency.md` (`UISizes`, 32px min heights, 140px label width). Forms wrapped in `QScrollArea`.

*Analyze tab affordance* (for U3 follow-through visible in U6 wiring):
  - Add a `QPushButton "Run Word-Level Alignment"` to `ui/tabs/analyze_tab.py`, placed in the existing transcription button row, immediately after the "Transcribe" button.
  - Disabled when no clips have a transcript (`Clip.transcript is None` for all selected). Enabled otherwise.
  - On click: dispatches the `ForcedAlignmentWorker` over selected clips and shows progress in the tab's existing analysis-progress widget (same widget the transcription button uses).

**Execution note:** End-to-end integration test the dialog → worker → spine → SequenceClip → render path with a tiny seeded source.

**Patterns to follow:**
- `ui/dialogs/staccato_dialog.py` (best reference — has a worker, feature-registry preflight, post-accept Sequence assignment).
- `.claude/rules/ui-consistency.md`.

**Test scenarios:**
- Happy path: Opening the Word Sequencer dialog with aligned clips selected shows the source picker with all rows checked and `✓ aligned` badges; selecting "alphabetical" and accepting produces a timeline. *Covers F1.*
- Happy path: Opening the LLM Composer with aligned clips, entering a prompt, and accepting produces a timeline. *Covers F2.*
- Happy path: Mixed faster-whisper (word data present) + MLX (word data absent) source picker shows `✓ aligned` and `… needs alignment` badges respectively; accepting auto-runs alignment on the MLX sources only (inline progress, no extra modal).
- Happy path: Switching between modes preserves the user-entered chosen-words text and the user-curated text independently (each retains its value when hidden).
- Edge case: Unsupported-language source appears in the picker with row disabled, `⚠ unsupported language` badge, tooltip carrying the AE1 message; Accept stays enabled because other sources are checked and valid. *Covers AE1.*
- Edge case: All checked sources are unsupported-language → Accept disabled, inline error label explains why.
- Edge case: Word Sequencer with chosen-words mode and an include list whose entries are all absent from the corpus → Accept disabled, error label reads "0 of N include-list words found in corpus".
- Edge case: Empty corpus (zero words in selected clips) → Accept disabled, error label "no words in selected sources".
- Edge case: Cancelling alignment mid-run via the in-dialog cancel button → worker.cancel() fires; dialog returns to pre-Accept state with partial alignment results persisted in the project; no orphan worker.
- Edge case: Cancelling LLM generation mid-run → in-flight Ollama call is interrupted (subprocess kill per cleanup learning); dialog returns to pre-Accept state.
- Edge case: LLM Composer opened with Ollama down → dialog opens with installer-prompt and disabled Accept; clicking "Open feature install" invokes the feature_registry flow.
- Edge case: LLM returns `None` content → inline error label shows the empty-response message; dialog stays open so the user can retry or adjust the prompt.
- Edge case: A source with `fps = None` → dialog shows the source as disabled with tooltip "source missing fps metadata"; surfaces at picker render so the user sees it before Accept.
- Integration: Full path clips → dialog → alignment → sequencer → `SequenceClip[]` → render produces a playable mp4 from a 30-second seeded source.
- Integration: Existing algorithm dialogs (Staccato, etc.) continue to work unchanged — regression check.
- Integration: Sub-second `SequenceClip` durations are FFmpeg-safe (timestamps normalized to finite decimal seconds per the fractional-duration learning); spot-check that the rendered output contains all expected words.
- Integration: Analyze tab's "Run Word-Level Alignment" button is disabled when selected clips have no transcripts, enabled otherwise; clicking it dispatches the same `ForcedAlignmentWorker` and shows progress in the tab's existing analysis-progress widget.

**Verification:**
- End-to-end smoke from a 30-second source with 50 words produces a 50-cut alphabetical timeline that renders to a playable mp4.
- The two new algorithms appear in the Sequence tab algorithm picker.
- Existing algorithms still work (no regression).

---

## System-Wide Impact

- **Interaction graph:** The new `ForcedAlignmentWorker` integrates with project observers; the new `complete_with_enum_constraint` helper on `LLMClient` is a new public method that chat tools / agent flows should *not* hit accidentally (the word composer is a pure algorithm, not a chat tool). The new dialogs add two entries to `ui/algorithm_config.py` and corresponding wiring in `ui/tabs/sequence_tab.py`.
- **Error propagation:** Alignment errors propagate via `worker.error(str)`. LLM errors propagate via algorithm-level exceptions caught in the dialog and surfaced as user messages. `MissingWordDataError` and `UnsupportedLanguageError` are caught by the dialogs and routed to user-visible prompts (offer to align, disable the source).
- **State lifecycle risks:** Alignment is per-clip and partial results persist. Mid-run cancellation leaves the already-aligned clips with `words` populated and the rest at `None`. Re-running picks up from where it left off (no extra work for already-aligned clips).
- **API surface parity:** `core/spine/words.py` is shaped so `scene_ripper_mcp/tools/` could add `align_words` and `word_sequence` MCP tools in a follow-up without changes to the spine code. Not implemented in this plan.
- **Integration coverage:** Spine boundary test (`tests/test_spine_imports.py`) must continue to pass — `core/spine/words.py` lazy-imports any heavy ML dependencies.
- **Unchanged invariants:**
  - Render pipeline is not touched (per R15).
  - `SequenceClip` model is not changed. (R4 governs *word-timestamp storage location* — "no new top-level entity"; the fact that `SequenceClip` is unchanged is a separate plan-level observation reinforcing the brainstorm's "no render pipeline changes" guarantee, not a direct R4 consequence.)
  - `Source` and `Clip` schemas are unchanged. (The schema change is on `TranscriptSegment`, which is referenced from `Clip.transcript` but not part of the `Clip` dataclass itself.)
  - Existing transcripts without word data continue to work for current features (per R2).

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Ollama `format`+enum decode latency is unacceptable at corpus scale (~5000+ enum items, non-parallel masking) | Measure in U5; if unacceptable, Tier 2 (`llama-cpp-python` GBNF) is a documented follow-up plan. Tier 1 unblocks shipping for typical short-corpus runs. |
| LiteLLM strips Ollama's `format` parameter when forwarding | Verify in U5; if stripped, bypass LiteLLM for the constrained-decode call and hit Ollama's `/api/generate` directly through `core/llm_client.py`. |
| Forced alignment is CPU-only on Apple Silicon and slow for long sources | Worker reports per-clip progress; cancellation is safe; partial results persist. UI sets expectations clearly. |
| Word-boundary splices still clip plosives at ~20–30ms accuracy | Handle-frame parameter is the user-facing escape valve (default 0 per brainstorm; spinner exposed). |
| Sub-second `SequenceClip` durations break FFmpeg with fractional-second args | Normalize all timestamps to finite decimal seconds before constructing FFmpeg args (per existing learning). Covered by U4 approach and U6 integration test. |
| Worker replaced before previous run finished → crash | Guard-flag + `Qt.UniqueConnection` pattern in U3 (per existing learning). |
| Dialog "sequence overwrite by generic handlers" footgun | Register both algorithms with `is_dialog=True` and audit `ui/tabs/sequence_tab.py` for any non-dialog fallback path that might clobber the sequence post-accept. |
| LLM returns `None` content → silent empty sequence | Explicit validation in U5; raise `LLMEmptyResponseError`. |
| Subprocess orphan on cancellation (Ollama HTTP, FFmpeg extracts) | `try/finally` with explicit kill+wait (per existing learning); apply in U5 and any U6 render paths that spawn subprocesses. |
| Algorithm registry circular import (historical) | Verified mitigated — `ui/algorithm_config.py` is already the neutral module; no extraction needed. |
| MLX-transcribed corpora silently mis-align as English when source is non-English | U1 captures `result["language"]` from MLX (currently discarded) into `TranscriptSegment.language`; U2 reads `segments[0].language` and raises `LanguageUnknownError` rather than silently defaulting to "en". |
| `core/cost_estimates.py` silently skips the new algorithm because `transcription_with_words` isn't in `METADATA_CHECKS` | U4 adds the entry (lambda + label + time + cost rows); test in `tests/test_cost_estimates.py` asserts the key is recognized. |
| `WordTimestamp` accidentally added to `models/clip.py` causes circular import | U1 colocates `WordTimestamp` with `TranscriptSegment` in `core/transcription.py`; `models/clip.py` continues to TYPE_CHECKING-import only. |
| Algorithm registration written from multiple units causes drift | U4 and U5 each own one entry end-to-end; U6 does not modify `ui/algorithm_config.py`. |

---

## Documentation / Operational Notes

- Update `docs/user-guide/` with a new page describing the Word Sequencer (alignment opt-in, the five preset modes, the LLM-composed mode). Use the existing `/sync-feature-docs` or `/sync-sequencer-docs` skill to keep `docs/user-guide/sequencers.md` in step.
- Update `.claude/rules/sequencer-algorithms.md` to include the two new algorithms.
- The new feature-registry entries (`word_alignment`; reserved `constrained_decoding_grammar`) need to be documented in any developer-facing notes about adding ML features.

---

## Sources & References

- **Origin document:** [docs/brainstorms/word-sequencer-requirements.md](docs/brainstorms/word-sequencer-requirements.md)
- Related code: `core/transcription.py:257` (`TranscriptSegment`), `core/transcription.py:445`/`:599` (word_timestamps call sites), `core/transcription.py:626` (`_parse_mlx_result`), `core/remix/staccato.py`, `ui/dialogs/staccato_dialog.py`, `core/llm_client.py:201` (`check_ollama_health`), `core/feature_registry.py:172`/`:203`, `core/cost_estimates.py:77` (`METADATA_CHECKS`), `ui/workers/transcription_worker.py`, `ui/tabs/sequence_tab.py:1426` (dialog dispatch pattern)
- Related rules: `.claude/rules/sequencer-algorithms.md`, `.claude/rules/ui-consistency.md`
- Related learnings: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`, `docs/solutions/ui-bugs/cut-tab-clips-hidden-after-thumbnail-failure.md`, `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`, `docs/solutions/logic-errors/circular-import-config-consolidation.md`, `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`, `docs/solutions/runtime-errors/macos-libmpv-pyav-ffmpeg-dylib-collision-20260504.md`, `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`
- External: `ctc-forced-aligner` (MahmoudAshraf97/ctc-forced-aligner on GitHub); Ollama `format` documentation; LiteLLM Input Params docs
