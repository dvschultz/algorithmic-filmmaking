---
title: "feat: Cassette Tape sequencer"
type: feat
status: active
date: 2026-04-27
origin: docs/brainstorms/2026-04-27-cassette-tape-sequencer-requirements.md
---

# Cassette Tape sequencer

## Overview

Add a new dialog-based sequencer that lets the user enter a list of phrases
and pulls the closest matches from transcribed clips, returning a sequence of
sub-clips trimmed to the matched transcript segments. Two-step UX:
phrase entry + per-phrase count slider, then a review screen showing each
match with confidence so the user can prune before generating.

---

## Problem Frame

Scene Ripper has 16 sequencers but none operate on **what's being said**.
Users with transcribed footage who want to assemble a sequence around
specific quotes have to filter manually one phrase at a time. Cassette Tape
fills that gap as a quote-finder / clip-browser dressed as a sequencer
(see origin: `docs/brainstorms/2026-04-27-cassette-tape-sequencer-requirements.md`).

---

## Requirements Trace

- R1. Phrase entry rows with per-phrase count slider (1–5), add/remove rows,
  default 3 empty rows.
- R2. Source pool = enabled clips with non-empty `transcript`; project-wide.
- R3. Match unit = `TranscriptSegment`; score with
  `rapidfuzz.fuzz.partial_ratio` (case-insensitive). Top-N best per phrase,
  no threshold.
- R4. Review screen: every match grouped by phrase, with phrase, source clip,
  transcript snippet, 0–100 confidence (numeric badge + colored bar:
  green ≥ 80 / yellow 50–79 / red < 50), and an include/exclude toggle.
- R5. Output sequence = enabled-after-review matches as `SequenceClip`
  sub-clips with `in_point`/`out_point` derived from segment start/end
  (seconds → frames via source fps); ordered phrase-then-score; replaces
  current sequence.
- R6. Edge cases: no transcribed clips → setup-screen message; no phrases →
  Generate disabled; cancellable worker.

---

## Scope Boundaries

- **Non-goals (v1):** preset save/load of phrase lists; word-level sub-clip
  boundaries; embedding-based semantic matching; per-phrase score threshold;
  auto-trim of leading/trailing silence; matching across segment boundaries.
- **Auto-untoggle below confidence ~30:** deferred (origin OQ1) — v1 leaves
  every match enabled by default; user prunes manually.

### Deferred to Follow-Up Work

- None. The feature ships as a single PR.

---

## Context & Research

### Relevant Code and Patterns

- **Sequencer module pattern:** `core/remix/storyteller.py`,
  `core/remix/exquisite_corpus.py` — module exposes a generation function
  plus a `sequence_by_*` helper that returns clip tuples.
- **Sub-clip in/out emission pattern:** `core/remix/signature_style.py` +
  `ui/tabs/sequence_tab.py:1301-1313` — the dialog emits 4-tuples
  `(clip, source, in_point, out_point)` and the apply method calls
  `timeline.scene.add_clip_to_track(...)` with
  `in_point=clip.start_frame + in_point`. Cassette Tape will follow this
  pattern exactly.
- **Multi-step dialog pattern:** `ui/dialogs/storyteller_dialog.py` —
  `QDialog` + `QStackedWidget`, page constants (`PAGE_CONFIG`,
  `PAGE_PROGRESS`, `PAGE_PREVIEW`), `QThread` worker with
  `progress / finished_* / error` signals, dialog emits a `sequence_ready`
  signal that the sequence tab connects to.
- **Algorithm registration:** `ui/algorithm_config.py` — add a
  `cassette_tape` entry with `is_dialog: True`,
  `required_analysis: ["transcribe"]`, `categories: ["arrange", "audio"]`.
- **Dialog dispatch:** `ui/tabs/sequence_tab.py:776-786` — `_on_card_clicked`
  routes dialog algorithms by name; add a `cassette_tape` branch calling a
  new `_show_cassette_tape_dialog(clips)` helper. Mirror the `apply_*`
  receiver pattern at `:1290-1325`.
- **Intention-workflow re-entry:** `ui/main_window.py:8166-8186` already
  handles `is_dialog` routing back through `sequence_tab._on_card_clicked`
  — no changes needed there for Cassette Tape (transcribe is a known
  analysis op).
- **Transcript model:** `core/transcription.py:239` —
  `TranscriptSegment(start_time, end_time, text, confidence)`, all in
  seconds. Helper `Clip.get_transcript_text()` at `models/clip.py:295`.
- **Fuzzy scoring:** `core/scene_detect.py:232` already uses
  `rapidfuzz.fuzz.ratio`. Cassette Tape uses `partial_ratio` for
  substring-aware matching. `rapidfuzz` exposes the matched substring
  position via `fuzz.partial_ratio_alignment` for highlight rendering.

### Institutional Learnings

- **`docs/solutions/` skills relevant here:**
  - `wire-sequencer-algorithm` — registration dance for new sequencers
    (algorithm_config + sequence_tab dispatch + apply method).
  - `pyside6-qthread-finished-signal-shadowing` — don't name a signal
    `finished` (clashes with `QThread.finished`); use `finished_matches`
    or similar.
  - `litellm-empty-response-validation` — N/A here (no LLM).
  - `pyside6-duplicate-signal-guard` — relevant if the dialog is
    re-opened; ensure worker signals connect once per worker instance.
- **Signal naming:** Use `matches_ready = Signal(list)` and
  `sequence_ready = Signal(list)` to avoid `QThread.finished` shadowing.

### External References

- `rapidfuzz` docs: `fuzz.partial_ratio` and `fuzz.partial_ratio_alignment`
  are stable, already-installed APIs. No external research needed —
  pattern is local and well-understood.

---

## Key Technical Decisions

- **Match unit = `TranscriptSegment` (one row per segment), not whole-clip
  transcript.** Rationale: gives the user tight quote boundaries for free
  (R3, R5) without needing word-level timing. Whole-clip matching would
  blur which segment matched, making the review screen misleading.
- **`partial_ratio` over `ratio` or `token_sort_ratio`.** Rationale: the
  user types short phrases ("I love you") to find them inside longer
  segments ("well I love you very much"). `partial_ratio` finds the best
  substring; `ratio` would penalize length differences; `token_sort_ratio`
  is too forgiving on word order, which inverts user intent.
- **No threshold; user prunes manually on review screen.** Carried from
  origin (R3, R4). Removes the "0 matches returned" UX pain.
- **Dialog emits 4-tuples, not `SequenceClip` objects.** Match the
  `signature_style` precedent so the sequence tab's
  `add_clip_to_track` integration works without a new code path.
- **Deterministic tie-breaking:** equal scores break by earlier
  `segment.start_time` then earlier source-clip creation order. Avoids
  test flakiness and surprising result reordering between runs.
- **Dialog dimensions and page constants** mirror `StorytellerDialog`
  (3 pages: setup / progress / review) for consistency.
- **Algorithm key = `cassette_tape`.** Required analysis =
  `["transcribe"]`. Categories = `["arrange", "audio"]`.
- **Highlight matched substring on review screen** (origin OQ2). Use
  `rapidfuzz.fuzz.partial_ratio_alignment` to get the matched range and
  render with bold/colored span in the segment text.

---

## Open Questions

### Resolved During Planning

- **OQ3 — algorithm key + required analysis:** `cassette_tape`,
  `required_analysis: ["transcribe"]`, `is_dialog: True`,
  `categories: ["arrange", "audio"]`.
- **OQ2 — long-segment display:** render full segment text with the
  matched substring highlighted (bold + theme accent color). Use
  `partial_ratio_alignment.dest_start/dest_end` for the highlight range.
  No truncation in v1.
- **Worker signal naming:** `matches_ready = Signal(list)` (not
  `finished`) to avoid `QThread.finished` shadowing.

### Deferred to Implementation

- **Exact widget choice for the per-row controls:** likely
  `QLineEdit` + `QSlider(Horizontal)` + `QLabel` for current count + an
  `×` remove button per row. Final layout decided when wiring.
- **Review screen widget:** likely a scrolling `QVBoxLayout` of phrase
  group headers + per-match rows with embedded `QCheckBox`. Could
  alternatively use `QTableView`; defer until first pass shows the rough
  density.
- **Threading model for matching:** plan calls for a `CancellableWorker`
  (R6.5). Since `partial_ratio` over a few hundred segments is
  millisecond-fast, the worker may not be strictly necessary for
  performance — but it future-proofs for large corpora and matches the
  pattern other dialog sequencers use. Implement as a worker.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
USER FLOW
─────────
Sequence tab
   └─ click "Cassette Tape" card
       └─ sequence_tab._on_card_clicked("cassette_tape")
           └─ _show_cassette_tape_dialog(clips)
               └─ CassetteTapeDialog
                   ├─ PAGE_SETUP  →  user enters phrases + counts, clicks Find Matches
                   ├─ PAGE_PROGRESS  →  CassetteTapeWorker runs match_phrases()
                   └─ PAGE_REVIEW  →  user toggles, clicks Generate
                       └─ emit sequence_ready([(clip, source, in_frame, out_frame), ...])
                           └─ sequence_tab._apply_cassette_tape_sequence()
                               └─ timeline.scene.add_clip_to_track(...) per match


DATA FLOW
─────────
phrases = ["I love you", "thank you", ...]      # from setup page
counts  = [3, 5, ...]                           # per-phrase slider values
clips_with_transcripts = [clip for clip in available_clips
                          if clip.transcript and not clip.disabled]

# Build (clip, segment) candidate set once
candidates = [(clip, seg) for clip in clips_with_transcripts
                          for seg in clip.transcript]

# Score per phrase, keep top-N
matches_per_phrase = {}
for phrase, n in zip(phrases, counts):
    scored = [
        (score(phrase, seg.text), clip, seg)
        for clip, seg in candidates
    ]
    scored.sort(key=lambda r: (-r[0], seg.start_time, clip_index(clip)))
    matches_per_phrase[phrase] = scored[:n]

# Review screen renders matches_per_phrase with toggles
# On Generate: build SequenceClip tuples for enabled matches in phrase order
```

---

## Implementation Units

- U1. **Core matching module: `core/remix/cassette_tape.py`**

**Goal:** Pure-logic module that scores phrases against
`(clip, TranscriptSegment)` pairs and returns top-N matches per phrase.
No Qt, no threading.

**Requirements:** R2, R3, R5

**Dependencies:** none

**Files:**
- Create: `core/remix/cassette_tape.py`
- Test: `tests/test_cassette_tape.py`

**Approach:**
- Define a `MatchResult` dataclass: `phrase`, `clip_id`, `segment_index`,
  `segment` (TranscriptSegment), `score` (0–100 int), `match_start`,
  `match_end` (substring positions for highlight).
- `match_phrases(phrases: list[tuple[str, int]], clips: list[Clip])
  -> dict[str, list[MatchResult]]` — input is `(phrase, count)` pairs;
  output preserves phrase order in the dict (Python 3.7+).
- Internal: build candidate `(clip, segment)` list from clips with
  non-empty `transcript`; for each phrase score every candidate via
  `rapidfuzz.fuzz.partial_ratio` (lowercased on both sides); sort with
  the documented tie-break; slice to N.
- Use `rapidfuzz.fuzz.partial_ratio_alignment` to get
  `match_start/match_end` on the segment text for highlight rendering.
- `build_sequence_data(matches_in_order: list[MatchResult],
  clips_by_id: dict, sources_by_id: dict, ...) ->
  list[tuple[Clip, Source, int, int]]` — converts segment seconds to
  frames using `source.fps`; in_frame = `round(seg.start_time * fps)`,
  out_frame = `round(seg.end_time * fps)` (relative to clip start, to
  match the `signature_style` convention where the sequence tab adds
  `clip.start_frame`).

**Patterns to follow:**
- `core/remix/signature_style.py` for the build-sequence-data shape.
- `core/scene_detect.py:232` for `rapidfuzz` usage.

**Test scenarios:**
- *Happy path:* phrase "thank you", segment "well thank you for coming",
  expect score ≥ 90 and `match_start`/`match_end` cover "thank you".
- *Happy path:* phrase "I love you" with 3 matches available, count=2,
  returns the 2 highest-scoring `(clip, segment)` pairs.
- *Edge case:* clip with `transcript=None` is silently excluded.
- *Edge case:* clip with `disabled=True` is silently excluded.
- *Edge case:* phrase matches no segment well (top score < 30); top-N
  is still returned (no threshold).
- *Edge case:* count > available segments → returns all available
  without padding.
- *Edge case:* same phrase appears in two clips with identical scores —
  tie-broken by earlier `segment.start_time` then earlier clip index;
  deterministic across runs.
- *Happy path:* `build_sequence_data` converts seconds to frames using
  source fps; for fps=24 and seg.start_time=2.5, in_frame=60.
- *Edge case:* segment with start_time=0.0 yields in_frame=0.

**Verification:**
- `pytest tests/test_cassette_tape.py` passes.
- `match_phrases` returns a dict keyed by phrase order, each value a
  list of `MatchResult` with length ≤ requested count.
- No PySide6, Qt, or threading imports in the module.

---

- U2. **Algorithm registration: `ui/algorithm_config.py`**

**Goal:** Register `cassette_tape` so it appears in the Sequence tab card
grid with the correct required-analysis gating.

**Requirements:** R1, R2

**Dependencies:** U1 (logic exists, even if dialog isn't wired yet —
keeps the registration valid).

**Files:**
- Modify: `ui/algorithm_config.py`

**Approach:**
- Add a `cassette_tape` entry to `ALGORITHM_CONFIG`:
  - `label`: "Cassette Tape"
  - `description`: e.g., "Find clips that say specific phrases — the
    transcript-driven mixtape"
  - `required_analysis`: `["transcribe"]`
  - `is_dialog`: `True`
  - `allow_duplicates`: `False`
  - `categories`: `["arrange", "audio"]`
- No icon required (existing entries leave the field as empty string).

**Patterns to follow:**
- Other dialog entries: `exquisite_corpus`, `storyteller`,
  `signature_style`.

**Test scenarios:**
- Test expectation: none — pure config addition. Covered indirectly by
  U3's dialog dispatch tests, which look up the algorithm by key.

**Verification:**
- `cassette_tape` appears in the Sequence tab algorithm card grid when
  the project has at least one clip with a transcript.
- Card is gated correctly when no clips have transcripts (existing
  `required_analysis` machinery handles this).

---

- U3. **Dialog: `ui/dialogs/cassette_tape_dialog.py`**

**Goal:** Three-page modal dialog (setup / progress / review) that
collects phrases and counts, runs matching in a worker, lets the user
toggle individual matches, and emits `sequence_ready` with the chosen
sub-clip tuples.

**Requirements:** R1, R4, R5, R6

**Dependencies:** U1

**Files:**
- Create: `ui/dialogs/cassette_tape_dialog.py`
- Test: `tests/test_cassette_tape_dialog.py`

**Approach:**
- Mirror `StorytellerDialog` structure: `QDialog` + `QStackedWidget`,
  three page constants (`PAGE_SETUP=0`, `PAGE_PROGRESS=1`,
  `PAGE_REVIEW=2`).
- **Setup page:** scrollable list of phrase rows; `+` adds a row, `×`
  removes a row, default 3 empty rows; each row has a `QLineEdit` and a
  `QSlider(1,5)` plus a count label; "Find Matches" button at the
  bottom — disabled until at least one phrase has non-empty text.
  Header above the list shows a one-line message if no transcribed
  clips exist (R6.1) and disables Find Matches.
- **Progress page:** simple status label + a `QProgressBar` (or
  indeterminate spinner) and a Cancel button.
- **Review page:** scrolling layout grouped by phrase. Each group shows
  the phrase header (and "no acceptable matches" if all matches scored
  near zero — visual hint, not a hard filter); per-match rows show
  thumbnail, clip name, segment snippet (with the matched substring
  highlighted), confidence badge + colored bar, and an enabled-by-default
  `QCheckBox`. "Generate Sequence" button (disabled if zero rows
  enabled), "Back" returns to setup with prior values preserved,
  "Cancel" closes the dialog.
- **Worker:** `CassetteTapeWorker(QThread, CancellableWorker mixin)` with
  signals `progress = Signal(int, int)` (n, total),
  `matches_ready = Signal(dict)` (phrase → [MatchResult]),
  `error = Signal(str)`. Worker calls `match_phrases` from U1.
  Avoid the `finished` name to prevent QThread.finished shadowing.
- **Public signal:** `sequence_ready = Signal(list)` emitting
  `list[tuple[Clip, Source, int, int]]` once the user clicks Generate.
- Confidence color thresholds: green ≥ 80, yellow 50–79, red < 50;
  read from `ui/theme.py` if existing accent colors fit, otherwise
  introduce three constants in the dialog module.
- Worker connection happens at instantiation; if dialog is reopened,
  create a new worker each run (avoids the duplicate-signal-guard
  pattern).

**Patterns to follow:**
- `ui/dialogs/storyteller_dialog.py` for page management and worker
  signal layout.
- `ui/workers/base.py:CancellableWorker` for cancellation.
- `ui/theme.py:UISizes` for slider/button heights and label widths
  (per `.claude/rules/ui-consistency.md`).
- For review-screen styling, mirror the per-match cards used in
  `ui/dialogs/free_association_dialog.py` if a similar list-of-cards
  surface exists; otherwise build a minimal layout.

**Test scenarios:**
- *Happy path:* dialog initializes with 3 empty rows; Find Matches is
  disabled until a phrase is entered.
- *Happy path:* entering 2 phrases and clicking Find Matches advances
  to the progress page.
- *Happy path:* worker emits `matches_ready`; dialog advances to review
  with all match toggles checked by default.
- *Happy path:* user unchecks one match; clicking Generate emits
  `sequence_ready` with the remaining matches.
- *Edge case:* no transcribed clips supplied → setup page shows the
  no-transcripts message and Find Matches is disabled.
- *Edge case:* user clicks Cancel during progress → worker is
  requested-cancel, dialog closes without emitting `sequence_ready`.
- *Edge case:* worker emits `error("...")` → dialog returns to setup
  with the error displayed in a banner.
- *Edge case:* user toggles every match off → Generate is disabled.
- *Edge case:* dialog closed and reopened in same session → new worker
  is created; signals do not double-fire.
- *Integration:* `partial_ratio_alignment` highlight range is non-empty
  on a real `(phrase, segment)` pair (verifies U1↔U3 contract).

**Verification:**
- `pytest tests/test_cassette_tape_dialog.py` passes.
- Manually opening the dialog from the Sequence tab card produces a
  usable flow end-to-end on a small project with transcribed clips.

---

- U4. **Dialog dispatch + sequence application: `ui/tabs/sequence_tab.py`**

**Goal:** Wire the algorithm card click to open the dialog, and add the
`_apply_cassette_tape_sequence` handler that turns the dialog's
4-tuples into timeline clips.

**Requirements:** R5

**Dependencies:** U2, U3

**Files:**
- Modify: `ui/tabs/sequence_tab.py`
- Test: covered by U3's dialog tests (which assert `sequence_ready`
  shape) plus an existing-style smoke test in
  `tests/test_sequence_tab_dialog_routing.py` if that file already
  exists; otherwise add a minimal targeted test in
  `tests/test_cassette_tape_dialog.py` covering the wiring contract.

**Approach:**
- Add `from ui.dialogs import CassetteTapeDialog` (and update
  `ui/dialogs/__init__.py` to export it).
- In `_on_card_clicked`, after the existing `is_dialog` checks
  (storyteller, exquisite_corpus, etc.), add an `elif algorithm ==
  "cassette_tape": self._show_cassette_tape_dialog(clips); return`.
- New helper `_show_cassette_tape_dialog(clips)` — mirror
  `_show_storyteller_dialog` shape: instantiate dialog with
  `clips_with_sources`, connect `sequence_ready` to
  `self._apply_cassette_tape_sequence`, call `dialog.exec()`.
- New handler `_apply_cassette_tape_sequence(sequence_data)` — mirror
  `_apply_signature_style_sequence` at `:1290-1325`:
  - clear timeline, set fps from first clip's source
  - for each `(clip, source, in_point, out_point)` tuple, call
    `timeline.scene.add_clip_to_track(track_index=0,
    source_clip_id=clip.id, source_id=source.id,
    start_frame=current_frame, in_point=clip.start_frame + in_point,
    out_point=clip.start_frame + out_point, thumbnail_path=...)`
  - advance `current_frame` by `out_point - in_point`
  - emit `clip_added` per tuple
  - update timeline preview, algorithm dropdown label ("Cassette Tape"),
    fit zoom
- Update `ui/dialogs/__init__.py` to export `CassetteTapeDialog`.

**Patterns to follow:**
- `_show_storyteller_dialog` and `_apply_signature_style_sequence` in
  the same file.
- Existing `from ui.dialogs import …` block at line 26 of
  `ui/tabs/sequence_tab.py`.

**Test scenarios:**
- *Happy path:* feeding a 3-tuple sequence into
  `_apply_cassette_tape_sequence` puts 3 clips on the timeline at
  correct `start_frame` offsets and matching `in_point`/`out_point`.
- *Edge case:* empty `sequence_data` does not crash and shows a
  user-visible "no clips" status (mirror existing apply methods).
- *Integration:* end-to-end through dialog → apply: covered by U3's
  manual verification step plus dialog test asserting the emitted
  shape.

**Verification:**
- `pytest tests/` is green for new and existing tests.
- Manually clicking the Cassette Tape card on a transcribed project
  routes to the dialog and a generated sequence appears on the
  timeline.

---

- U5. **User-facing docs: `docs/user-guide/sequencers.md`**

**Goal:** Document the new sequencer in the user guide so it's
discoverable.

**Requirements:** documentation hygiene per `CLAUDE.md`'s
`sync-sequencer-docs` skill convention.

**Dependencies:** U2 (registration must be in place so the doc
reference is correct).

**Files:**
- Modify: `docs/user-guide/sequencers.md`

**Approach:**
- Add a "Cassette Tape" section following the existing per-sequencer
  format (label, when to use, required analysis, dialog flow,
  examples).
- Include the use case framing from the brainstorm: quote-finder, not
  narrative composition.

**Patterns to follow:**
- Existing per-sequencer sections in `docs/user-guide/sequencers.md`.
- Optionally invoke the `sync-sequencer-docs` skill to ensure
  alignment with `ui/algorithm_config.py`.

**Test scenarios:**
- Test expectation: none — documentation only.

**Verification:**
- The new section renders correctly in markdown preview.
- The label, required analysis, and behavior description match U2's
  registration entry verbatim.

---

## System-Wide Impact

- **Interaction graph:** new dialog plugs into the existing Sequence
  tab card-click → dialog → sequence_ready chain. No new entry points.
- **Error propagation:** worker errors surface on the dialog's error
  banner; no upstream handler changes.
- **State lifecycle risks:** dialog closing mid-match must cancel the
  worker (R6.5). Reopening the dialog must not reuse a stale worker —
  instantiate fresh per run.
- **API surface parity:** the existing intention-workflow re-entry at
  `ui/main_window.py:8166-8186` already routes any `is_dialog`
  algorithm via `sequence_tab._on_card_clicked`, so the new dialog is
  reachable from both the Sequence-tab card and the
  intention-workflow path without extra plumbing.
- **Integration coverage:** U3's `partial_ratio_alignment` integration
  test guards the U1↔U3 contract; U4's apply method test guards
  U3↔U4 contract.
- **Unchanged invariants:** existing sequencers, `SequenceClip` schema,
  timeline `add_clip_to_track` contract, transcribe analysis pipeline,
  and `ALGORITHM_CONFIG`'s shape are all preserved.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `partial_ratio` produces unintuitive scores on very short phrases (e.g., "hi" matches almost everything). | Document in the user guide that 2–3-word phrases work best; consider an opt-in min-phrase-length warning in v2. |
| Long transcript segments inflate score variance and make highlights span most of the visible text. | Highlight-only rendering (no truncation) + per-match toggle let users prune; v2 can revisit truncation-with-expand. |
| `QThread` worker reuse triggers duplicate signal connections (see `pyside6-duplicate-signal-guard` learning). | Instantiate a fresh worker per Find-Matches click; do not store the worker on the dialog beyond one run. |
| Signal name `finished` would shadow `QThread.finished`. | Use `matches_ready` for the worker, `sequence_ready` for the dialog (see `pyside6-qthread-finished-signal-shadowing` learning). |
| Float→frame conversion can produce a 1-frame gap or overlap if `start_time`/`end_time` aren't aligned with frame boundaries. | Use `round()` consistently for both bounds; tests cover the 0.0 and mid-frame edge cases. |
| User pastes hundreds of phrases. | The matching loop is O(phrases × candidates × short-string-cost) — fast, but the review screen could become unwieldy. v1 documents the practical limit (~10 phrases) in the user guide; no hard cap. |

---

## Documentation / Operational Notes

- Update `docs/user-guide/sequencers.md` (U5).
- Optional: add a `docs/solutions/` learning if the
  `partial_ratio_alignment` highlight integration uncovers anything
  surprising during U3 implementation.
- No migration, monitoring, or rollout concerns — feature is purely
  additive and project-local.

---

## Sources & References

- **Origin document:** `docs/brainstorms/2026-04-27-cassette-tape-sequencer-requirements.md`
- Related code:
  - `core/remix/signature_style.py` (sub-clip in/out emission)
  - `ui/dialogs/storyteller_dialog.py` (multi-step dialog pattern)
  - `core/transcription.py:239` (TranscriptSegment shape)
  - `models/clip.py:227` (Clip.transcript)
  - `ui/tabs/sequence_tab.py:1290-1325` (apply pattern)
  - `ui/algorithm_config.py` (registration shape)
- Related skills:
  - `wire-sequencer-algorithm`
  - `pyside6-qthread-finished-signal-shadowing`
  - `pyside6-duplicate-signal-guard`
  - `sync-sequencer-docs`
