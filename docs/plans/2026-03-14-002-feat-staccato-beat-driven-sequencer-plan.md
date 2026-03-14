---
title: "feat: Add Staccato beat-driven sequencer"
type: feat
status: completed
date: 2026-03-14
origin: docs/brainstorms/2026-03-14-staccato-beat-driven-sequencer-brainstorm.md
---

# feat: Add Staccato Beat-Driven Sequencer

## Overview

A new sequencer algorithm that cuts video clips to the rhythm of a user-provided music track. Onset strength at each beat drives visual contrast between consecutive clips — stronger transients trigger bigger visual jumps, measured by DINOv2 embedding distance.

## Problem Statement / Motivation

Scene Ripper has `core/remix/audio_sync.py` with beat/onset alignment utilities and `core/analysis/audio.py` with librosa-based beat detection, but no sequencer uses them. Users have no way to create rhythm-driven edits. Staccato turns unused audio infrastructure into a creative tool with a unique onset-strength-to-visual-contrast mapping.

(see brainstorm: docs/brainstorms/2026-03-14-staccato-beat-driven-sequencer-brainstorm.md)

## Proposed Solution

Add `staccato` to `ALGORITHM_CONFIG` as a dialog-based algorithm. The dialog accepts a music file, analyzes beats/onsets with librosa, displays a waveform with beat markers, then generates a sequence where each clip fills one beat interval. Clip selection is driven by onset strength: stronger onsets → greater DINOv2 embedding distance from the previous clip.

## Technical Considerations

### onset_strengths — New Field on AudioAnalysis

`core/analysis/audio.py`'s `AudioAnalysis` dataclass currently stores `onset_times` but not onset strengths. Add:

```python
onset_strengths: list[float] = field(default_factory=list)  # normalized [0,1] per onset
```

In `analyze_audio()`, after `librosa.onset.onset_detect()`:

```python
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_strength_values = onset_env[onset_frames].tolist()
if onset_strength_values:
    max_str = max(onset_strength_values)
    if max_str > 0:
        onset_strength_values = [s / max_str for s in onset_strength_values]
```

Update `to_dict()` and `from_dict()` accordingly.

### Clip Selection Algorithm

For each beat interval, given the onset strength `s` at that cut point:

1. Compute `target_distance = s * max_embedding_distance` (linear mapping)
2. From the clip pool, find the clip whose cosine distance from the previous clip is closest to `target_distance`
3. Prefer clips longer than the beat interval (to avoid looping)
4. If a clip is shorter than the interval, it loops
5. Clips can repeat (necessary when beats > clips)

Cosine distance between two DINOv2 embeddings: `1 - cosine_similarity(a, b)`. Since embeddings are already L2-normalized (`core/analysis/embeddings.py`), this simplifies to `1 - dot(a, b)`.

### Waveform Visualization

Build a custom `QWidget` with `paintEvent` that:
- Loads audio samples via librosa (downsampled to widget pixel width)
- Draws waveform as a filled polygon or vertical bars
- Overlays beat/onset markers as vertical lines
- Uses theme colors from `ui/theme.py`

Reference patterns: `ui/widgets/range_slider.py:173` (custom QPainter with anti-aliasing), `ui/widgets/drawing_canvas.py` (interactive QPainter canvas).

### QThread Worker Safety

Per documented learning (`docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`):
- Add guard flag (`self._handler_executed`) at the top of every `finished` slot
- Use `Qt.UniqueConnection` when wiring workers inside the dialog
- Reset guard when user kicks off a new operation

### Dialog Architecture

Follow the Dice Roll pattern (`ui/dialogs/dice_roll_dialog.py`):
- Worker class inheriting `CancellableWorker`
- Dialog class inheriting `QDialog` with `QStackedWidget` (config page → progress page)
- `sequence_ready = Signal(list)` emitting `list[(Clip, Source)]`

**Dialog config page controls:**
1. Music file picker button + label showing selected filename
2. Waveform widget (populated after file is loaded and analyzed)
3. Onset sensitivity slider (maps to librosa onset detection `delta` parameter)
4. Beat strategy dropdown: Beats, Downbeats, Onsets
5. Info labels: detected beat count, estimated clip count, music duration
6. Generate button

**Dialog flow:**
1. User selects music file → background worker analyzes audio → waveform + markers displayed
2. User adjusts sensitivity/strategy → re-analyze → waveform updates
3. User clicks Generate → worker auto-computes missing embeddings → runs clip matching → emits sequence

### Sequence Tab Integration

Add to `ui/tabs/sequence_tab.py`:
- Import `StaccatoDialog` from `ui/dialogs`
- Add `if algorithm == "staccato":` branch in the `_apply_algorithm` if/elif chain (~line 515)
- Add `_show_staccato_dialog(clips)` method
- Add `_apply_staccato_sequence(sequence_clips)` slot delegating to `_apply_dialog_sequence()`

## Acceptance Criteria

- [x] `staccato` entry added to `ALGORITHM_CONFIG` with `is_dialog: True`, `required_analysis: ["embeddings"]`
- [x] `AudioAnalysis` gains `onset_strengths` field, computed and serialized
- [x] `StaccatoDialog` opens from Sequence tab when Staccato is selected
- [x] Dialog accepts MP3/WAV/FLAC/M4A/AAC/OGG files via file picker
- [x] Waveform visualization displays audio with beat/onset markers overlaid
- [x] Onset sensitivity slider controls number of detected cut points
- [x] Beat strategy dropdown switches between beats, downbeats, onsets
- [x] Clips are assigned to beat intervals using onset-strength-to-embedding-distance mapping
- [x] Stronger onsets produce larger visual jumps between consecutive clips
- [x] Clips are trimmed to fit beat intervals; shorter clips loop
- [x] Clips can repeat when beat count exceeds clip count
- [x] DINOv2 embeddings auto-computed if missing (pattern from `_auto_compute_embeddings`)
- [x] Worker uses `CancellableWorker` base with guard flag pattern
- [x] Sequence emitted as `list[(Clip, Source)]` via `sequence_ready` signal
- [x] Help menu updated if sequencer doc skill triggers (via `/sync-sequencer-docs`)

## Implementation Phases

### Phase 1: AudioAnalysis Enhancement

**Files:**
- `core/analysis/audio.py` — Add `onset_strengths` field and computation

**Tasks:**
- [x] Add `onset_strengths: list[float]` to `AudioAnalysis` dataclass
- [x] Compute onset strength envelope in `analyze_audio()` using `librosa.onset.onset_strength()`
- [x] Sample envelope at detected onset frames, normalize to [0, 1]
- [x] Update `to_dict()` and `from_dict()` for serialization
- [x] Add `onset_strength_at(time)` method using nearest-neighbor lookup
- [x] Write tests for onset strength extraction

### Phase 2: Core Algorithm

**Files:**
- `core/remix/staccato.py` — New file with the beat-matching algorithm

**Tasks:**
- [x] Implement `generate_staccato_sequence(clips, audio_analysis, strategy)` function
- [x] Implement onset-strength-to-target-distance mapping (linear)
- [x] Implement clip selector: given previous clip embedding, target distance, and clip pool, find best match
- [x] Handle clip reuse (allow repeats)
- [x] Handle clip trimming (compute in/out points for each beat interval)
- [x] Handle short clip looping (mark clips that need to loop)
- [x] Write unit tests for the algorithm with mocked embeddings and audio analysis

### Phase 3: Waveform Widget

**Files:**
- `ui/widgets/waveform_widget.py` — New custom QWidget

**Tasks:**
- [x] Create `WaveformWidget(QWidget)` with custom `paintEvent`
- [x] Accept audio samples (numpy array) and beat/onset markers
- [x] Draw waveform as filled polygon using QPainter
- [x] Draw beat markers as vertical lines (theme accent color)
- [x] Draw onset markers as vertical lines (different color/style)
- [x] Support `set_audio_data()` to update from worker results
- [x] Use theme colors from `ui/theme.py`

### Phase 4: Dialog

**Files:**
- `ui/dialogs/staccato_dialog.py` — New dialog + worker
- `ui/dialogs/__init__.py` — Export

**Tasks:**
- [x] Create `StaccatoWorker(CancellableWorker)` with two phases:
  - Phase 1: Analyze music file (librosa) — emit audio data for waveform
  - Phase 2: Auto-compute embeddings + run clip matching — emit sequence
- [x] Create `StaccatoDialog(QDialog)` with `QStackedWidget`:
  - Page 0: Config (file picker, waveform, sensitivity slider, strategy dropdown, info labels, generate button)
  - Page 1: Progress (progress bar, cancel button)
- [x] Implement file picker with filter for audio formats
- [x] Wire sensitivity slider and strategy dropdown to re-analyze on change
- [x] Apply guard flag pattern for worker `finished` slots
- [x] Emit `sequence_ready = Signal(list)` with `list[(Clip, Source)]`
- [x] Export from `ui/dialogs/__init__.py`

### Phase 5: Integration

**Files:**
- `ui/algorithm_config.py` — Add config entry
- `ui/tabs/sequence_tab.py` — Add routing and slots
- `core/remix/__init__.py` — Export if needed

**Tasks:**
- [x] Add `"staccato"` to `ALGORITHM_CONFIG`:
  ```python
  "staccato": {
      "icon": "\U0001f3b5",  # musical note
      "label": "Staccato",
      "description": "Cut clips to the rhythm of a music track",
      "allow_duplicates": True,
      "required_analysis": ["embeddings"],
      "is_dialog": True,
  }
  ```
- [x] Add routing in `sequence_tab.py` `_apply_algorithm` if/elif chain
- [x] Add `_show_staccato_dialog()` and `_apply_staccato_sequence()` methods
- [x] Import `StaccatoDialog` in `sequence_tab.py`
- [x] Run `/sync-sequencer-docs` to update `docs/user-guide/sequencers.md`

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| librosa is already a dependency (`requirements.txt`) | No new deps needed. Verify lazy import pattern is followed. |
| DINOv2 embedding computation can be slow for many clips | Show progress in dialog. Same pattern as Human Centipede — users expect this. |
| Onset detection sensitivity tuning may be confusing | Default to sensible value. Label the slider with "Fewer Cuts" ↔ "More Cuts". |
| Waveform widget is net-new UI code | Keep it simple — static display, no interaction. QPainter patterns exist in the codebase. |
| QThread worker can deliver duplicate signals | Apply documented guard flag + UniqueConnection pattern. |
| Short clips looping may look bad | Prefer longer clips first. Looping is the fallback, not the default behavior. |

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-14-staccato-beat-driven-sequencer-brainstorm.md](docs/brainstorms/2026-03-14-staccato-beat-driven-sequencer-brainstorm.md) — Key decisions: onset strength → visual contrast mapping, DINOv2 embeddings, waveform visualization, one clip per beat interval with looping.

### Internal References

- Dialog sequencer pattern: `ui/dialogs/dice_roll_dialog.py` (simplest example)
- Dialog routing: `ui/tabs/sequence_tab.py:511-532` (if/elif chain)
- DINOv2 embeddings: `core/analysis/embeddings.py:93` (`extract_clip_embeddings_batch`)
- Auto-compute pattern: `core/remix/__init__.py:397` (`_auto_compute_embeddings`)
- Audio analysis: `core/analysis/audio.py:250` (`analyze_audio`)
- Music file analysis: `core/analysis/audio.py:418` (`analyze_music_file`)
- Audio sync utilities: `core/remix/audio_sync.py` (alignment functions)
- Algorithm config: `ui/algorithm_config.py`
- CancellableWorker base: `ui/workers/base.py`
- QPainter reference: `ui/widgets/range_slider.py:173`
- Worker safety learning: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- Cache invalidation learning: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
