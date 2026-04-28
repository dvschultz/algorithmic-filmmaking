---
title: "feat: Cassette Tape sequencer (phrase-driven transcript matcher)"
type: feat
status: draft
date: 2026-04-27
---

# Cassette Tape sequencer

## Problem

Scene Ripper has 16 sequencer algorithms but no way to assemble a
sequence around **what people are actually saying**. Users with
transcribed footage who want to find clips matching a specific list of
phrases (a quote, a lyric, a list of recurring lines) have to filter
manually one phrase at a time and stitch results into a sequence by
hand. There's no tool that takes a list of target phrases and surfaces
the clips that say each one (or come close).

Cassette Tape fills that gap. It's a quote-finder / clip-browser
dressed as a sequencer: user supplies N phrases, picks how many matches
to pull per phrase, reviews match confidence, and gets a sequence built
from the matched transcript segments.

## Users

- Editors working with transcribed interview, documentary, or
  found-footage material who want to cite specific lines.
- Researchers building reference reels around recurring phrases or
  motifs in a corpus.
- Creative remixers building dialogue-driven supercuts where the
  source of each cut is "a clip that says X".

## Use case framing

This is **not** narrative composition (Storyteller already handles
that). The mental model is **search results playlist**: user enters a
list of queries, gets back a sequence of clips that match. Scoring
controls quality; phrase grouping preserves the user's intent. Confidence
is shown so users can prune obviously-bad matches before committing.

## Requirements

### R1. Phrase entry

- R1.1. The setup screen presents a list of **phrase rows**. Each row
  has a text input for the phrase and a **count slider (1–5)** that
  controls how many matches that phrase contributes to the final
  sequence.
- R1.2. Users can **add and remove rows**. Empty rows are ignored at
  generation time. Default starting state: 3 empty rows.
- R1.3. The slider's tick marks are integer values 1, 2, 3, 4, 5. Default
  midpoint: 3.
- R1.4. Every phrase row's slider is independent — different phrases can
  pull different counts in the same run.

### R2. Source clips

- R2.1. Cassette Tape only considers clips with a non-empty
  `transcript`. Clips without transcription are silently excluded from
  the candidate pool. (Algorithm reference table calls out
  `transcribe` as required analysis — same convention as other
  transcript-aware features.)
- R2.2. Disabled clips are excluded from matching (consistent with
  every other sequencer).
- R2.3. The candidate pool is the project's full set of enabled,
  transcribed clips — not a tab-local selection. (Aligns with how
  existing sequencers operate on the project pool.)

### R3. Matching

- R3.1. The matching unit is the **TranscriptSegment**. Each (clip,
  segment) pair is scored against each phrase independently. A clip
  with multiple segments contributes multiple match candidates.
- R3.2. Similarity scoring uses `rapidfuzz.fuzz.partial_ratio`
  (case-insensitive) by default. Rationale: transcript segments are
  often longer than the user's phrase, and `partial_ratio` finds the
  best-matching substring within the segment — which mirrors what
  "this clip says my phrase somewhere" feels like to a user.
  Alternative algorithms (`token_sort_ratio`, embedding-based) are
  worth evaluating during planning if `partial_ratio` produces
  unexpected results, but `partial_ratio` is the v1 default.
- R3.3. For each phrase, the top **N best-scoring (clip, segment)
  pairs** are retained, where N is that phrase's slider value. There
  is no minimum-score threshold — the user controls quality on the
  review screen by toggling off bad matches (see R4).
- R3.4. Tie-breaking when scores are equal: prefer earlier
  `segment.start` within the source clip, then earlier source clip
  creation order. (Deterministic and avoids randomness across runs.)
- R3.5. The same (clip, segment) can match multiple phrases. The user
  may want this (different phrases happen to match the same line) or
  not (duplicate playback). v1 behavior: **allow duplicates** across
  phrase groups. The review screen makes them visible so the user can
  toggle one off if undesired.

### R4. Review / confidence screen

- R4.1. After matching runs, the user sees a **review screen** showing
  every match the algorithm produced, grouped by phrase, in score order
  (best first within each phrase).
- R4.2. Each match displays:
  - the phrase it matched;
  - the **source clip** thumbnail and name;
  - the **matched transcript segment text** (the actual words the
    clip says);
  - the **confidence score** (0–100, rendered as a numeric badge plus
    a colored bar — green ≥ 80, yellow 50–79, red < 50);
  - a **toggle (checkbox)** that includes/excludes the match from the
    final sequence. Defaults to enabled.
- R4.3. Users can scroll through all matches, untoggle low-confidence
  picks, then click **Generate sequence**. Only enabled matches end up
  in the sequence.
- R4.4. The review screen has **Cancel** (return to setup) and **Back**
  (re-enter phrase setup with prior values preserved) buttons. Users
  who want different counts re-tune sliders on the setup screen, not
  the review screen.

### R5. Sequence output

- R5.1. The final sequence contains one `SequenceClip` per
  enabled-after-review match.
- R5.2. Each `SequenceClip` is a **sub-clip** of the source: its
  `in_point` and `out_point` correspond to the matched
  `TranscriptSegment`'s `start`/`end` (converted from seconds to
  source frames using the source's fps).
- R5.3. Sequence order: **phrase order, grouped**. Phrase 1's enabled
  matches first (best to worst within the phrase), then phrase 2's
  matches, etc. Empty phrase groups (all matches disabled) are
  skipped.
- R5.4. The sequence replaces the project's current sequence
  (consistent with how other dialog-based sequencers behave —
  Storyteller, Exquisite Corpus, Free Association).

### R6. Edge cases

- R6.1. **No transcribed clips in project**: setup screen shows an
  inline message ("This project has no transcribed clips. Run Analyze
  → Transcribe first.") and the Generate button is disabled.
- R6.2. **No phrases entered**: Generate button is disabled until at
  least one non-empty phrase row exists.
- R6.3. **Phrase has no acceptable matches at all**: the review screen
  still shows the top N picks, however bad. The user can toggle them
  all off to skip that phrase.
- R6.4. **Phrase count exceeds available segments**: if N=5 but only 3
  segments exist in the corpus, return all 3. Don't pad.
- R6.5. **Cancellation mid-match**: the worker should be a
  `CancellableWorker` so users can abort matching on large corpora.

## UX flow

1. **Sequence tab → algorithm dropdown → "Cassette Tape"**: opens the
   setup dialog (modal `QDialog`, like other dialog-based
   sequencers).
2. **Setup screen**: user enters phrases, tunes count sliders, clicks
   **Find matches**.
3. **Progress**: a brief progress indicator while matching runs (fast
   for typical projects but worth surfacing, especially on cancel).
4. **Review screen**: matches grouped by phrase, with toggle and
   confidence display. User prunes, clicks **Generate sequence**.
5. The dialog closes; the project sequence is replaced; the timeline
   shows the new sub-clip sequence.

## Out of scope (v1)

- **Saving / loading phrase lists** as presets. Useful future feature
  but not required for v1.
- **Word-level timing** for tighter sub-clip boundaries (extracting
  only the exact words rather than the whole segment). v1 uses
  segment boundaries as recorded by faster-whisper / lightning-whisper.
  Word-level slicing is a v2 polish.
- **Embedding-based semantic matching** (sentence-transformers).
  Defer until users hit the limits of `partial_ratio`. The
  string-distance approach is faster, deterministic, and has no extra
  model dependency.
- **Per-phrase minimum-score threshold** (we discussed and explicitly
  rejected this — count semantics + manual review screen pruning is
  the agreed approach).
- **Auto-trim of leading/trailing silence** within the matched
  segment.
- **Matching across phrase boundaries** (e.g., phrase that spans two
  consecutive segments). v1 scores per segment; v2 could explore
  segment concatenation if needed.

## Dependencies / assumptions

- `rapidfuzz` is already a project dependency (used in
  `core/scene_detect.py:232`).
- `Clip.transcript` is `list[TranscriptSegment]` with each segment
  carrying `start`, `end`, and `text` fields. Confirmed in
  `models/clip.py:227` and `core/transcription.py`.
- `TranscriptSegment.start` / `end` are seconds, not frames —
  conversion to frames will use the source clip's fps.
- The `Clip` data model already has `disabled`, so source filtering
  in R2.2 is straightforward.
- The Sequence tab's algorithm dropdown and dialog routing
  infrastructure already exists (Storyteller, Exquisite Corpus, etc.
  follow this pattern).

## Success criteria

A user with a transcribed project can:

1. Open Cassette Tape from the Sequence tab.
2. Enter 3–5 phrases and pick a count for each.
3. See a review screen where every match is annotated with the
   transcript snippet and a 0–100 confidence score.
4. Untoggle 1–2 obviously-bad matches.
5. Click Generate and immediately see a sequence on the timeline that
   plays back: phrase 1's clips in score order, then phrase 2's, etc.,
   each clip trimmed to just the line that matched.

The whole flow should take under 60 seconds for a 50-clip project
with 5 phrases. Matching itself should be sub-second per phrase on
typical project sizes — the slow path (if any) is faster-whisper at
analysis time, not matching here.

## Open questions

- **OQ1.** Should confidence scores below ~30 be auto-untoggled on the
  review screen? Reduces "click 5 boxes" tedium when the slider is
  set high but the corpus is small. Defer; v1 ships with all matches
  toggled on.
- **OQ2.** How should the review screen handle very long transcript
  segments (e.g., a 30-word segment matched on 2 words)? Truncate
  display with an expand button? Highlight the matched substring in
  the rendered text? v1 default: render full segment, highlight
  matched substring (`rapidfuzz` exposes the matching position).
- **OQ3.** Algorithm key in `ui/algorithm_config.py`: propose
  `"cassette_tape"`. Required analysis: `["transcribe"]`. Dialog:
  `True`. Confirm during planning.

## Next step

Hand off to `/ce-plan` to design the implementation: dialog UI, worker
threading, `core/remix/cassette_tape.py` module, and registration in
`ui/algorithm_config.py` + `ui/main_window.py` dialog routing.
