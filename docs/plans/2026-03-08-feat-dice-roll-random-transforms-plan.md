---
title: "feat: Add Random Transform Options to Dice Roll Sequencer"
type: feat
status: completed
date: 2026-03-08
---

# feat: Add Random Transform Options to Dice Roll Sequencer

## Overview

Add three checkbox options to the Dice Roll (shuffle) algorithm: **Random Horizontal Flip**, **Random Vertical Flip**, and **Random Reverse Playback**. When enabled, each clip in the shuffled sequence has a 50% chance of receiving the transform. Transforms are stored on `SequenceClip`, applied by FFmpeg at export time, and accessible via the agent API.

## Problem Statement / Motivation

The Dice Roll algorithm currently only randomizes clip **order**. Experimental filmmakers want more chaos — randomly mirroring and reversing clips creates disorienting, collage-like sequences that break visual continuity. This is a natural extension of the shuffle concept: randomize not just order but orientation and temporal direction.

## Proposed Solution

Add per-clip transform flags (`hflip`, `vflip`, `reverse`) to the `SequenceClip` model. When the user clicks the Dice Roll card, show checkboxes for each transform option in the timeline header (alongside direction dropdowns). On generation/regeneration, randomly assign transforms to clips based on checked options. At export, FFmpeg applies the corresponding filters per clip segment.

### UI Flow

1. User clicks "Dice Roll" card → sequence generates normally (shuffle order)
2. Timeline header shows three checkboxes: "Random H-Flip", "Random V-Flip", "Random Reverse"
3. User checks desired options → sequence **regenerates** with transforms randomly assigned
4. Checkboxes persist while Dice Roll is the active algorithm; hidden for other algorithms
5. Re-clicking Dice Roll or changing seed re-randomizes both order and transforms

This follows the existing **direction dropdown pattern** — options appear in the timeline header after generation and trigger regeneration on change.

## Technical Approach

### 1. Model: Add Transform Fields to SequenceClip

**File:** `models/sequence.py`

```python
@dataclass
class SequenceClip:
    # ... existing fields ...
    hflip: bool = False
    vflip: bool = False
    reverse: bool = False
```

- [x] Add `hflip: bool = False`, `vflip: bool = False`, `reverse: bool = False` to `SequenceClip`
- [x] Update `to_dict()` — conditionally serialize (only when `True`, like `hold_frames`)
- [x] Update `from_dict()` — default to `False` for backward compatibility
- [x] Update `Sequence.to_dict()` / `from_dict()` if needed (likely no changes — clips handle themselves)

### 2. Algorithm Config: Declare Transform Options

**File:** `ui/algorithm_config.py`

- [x] Add `"transform_options"` key to the `"shuffle"` entry in `ALGORITHM_CONFIG`:

```python
"shuffle": {
    "icon": "🎲",
    "label": "Dice Roll",
    "description": "Randomly shuffle clips into a new order",
    "allow_duplicates": False,
    "required_analysis": [],
    "transform_options": ["hflip", "vflip", "reverse"],
},
```

This declares which algorithms support which transforms. Only shuffle has it for now, but the pattern is generic.

### 3. UI: Transform Checkboxes in Timeline Header

**File:** `ui/tabs/sequence_tab.py`

Follow the `_DIRECTION_OPTIONS` / `_update_direction_dropdown` pattern:

- [x] Add `_TRANSFORM_OPTIONS` dict mapping algorithm keys to available transforms:

```python
_TRANSFORM_OPTIONS = {
    "shuffle": [
        ("Random H-Flip", "hflip"),
        ("Random V-Flip", "vflip"),
        ("Random Reverse", "reverse"),
    ],
}
```

- [x] Create three `QCheckBox` widgets in the timeline header area
- [x] Add `_update_transform_checkboxes(algorithm)` method — shows/hides checkboxes based on whether the algorithm has entries in `_TRANSFORM_OPTIONS`
- [x] Connect checkbox `stateChanged` signals to `_on_transform_option_changed` which calls `_regenerate_sequence`
- [x] Pass active transform options through to the sequence generation pipeline

### 4. Transform Assignment in Sequence Generation

**File:** `core/remix/__init__.py`

- [x] Add `transform_options: dict[str, bool] | None = None` parameter to `generate_sequence()`
- [x] After shuffle ordering, if `transform_options` is provided, iterate clips and assign transforms:

```python
def _assign_random_transforms(
    clips: list[tuple],
    transform_options: dict[str, bool],
    rng: random.Random,
) -> dict[str, dict[str, bool]]:
    """Assign random transforms to clips. Returns {clip_id: {hflip, vflip, reverse}}."""
    transforms = {}
    for clip, source in clips:
        t = {}
        if transform_options.get("hflip"):
            t["hflip"] = rng.random() < 0.5
        if transform_options.get("vflip"):
            t["vflip"] = rng.random() < 0.5
        if transform_options.get("reverse"):
            t["reverse"] = rng.random() < 0.5
        if any(t.values()):
            transforms[clip.id] = t
    return transforms
```

- [x] Return transforms alongside the clip list (add to return value or as a second return)
- [x] Use the same `random.Random` instance seeded by the sequence seed so transforms are deterministic per seed

### 5. Thread Transforms Through Workers

**File:** `ui/workers/sequence_worker.py`

- [x] Accept `transform_options` in `SequenceWorker.__init__`
- [x] Pass to `generate_sequence()`
- [x] Include transform map in the `finished` signal result

**File:** `ui/tabs/sequence_tab.py`

- [x] `_apply_algorithm` passes current checkbox state to `SequenceWorker`
- [x] `_on_sequence_ready` reads transform map and sets flags on `SequenceClip` objects when populating the timeline

### 6. FFmpeg Export: Apply Transform Filters

**File:** `core/sequence_export.py`

- [x] Update `_build_video_filter()` to accept a `SequenceClip` (or its transform flags) and append filters:

```python
# Filter order: scale → hflip → vflip → reverse → chromatic bar
if seq_clip.hflip:
    parts.append("hflip")
if seq_clip.vflip:
    parts.append("vflip")
if seq_clip.reverse:
    parts.append("reverse")
```

- [x] For `reverse`: also apply `areverse` audio filter so audio matches video direction
- [x] **Safety limit**: Skip `reverse` for clips longer than 15 seconds (log a warning). The `reverse` filter buffers entire clip in memory (~900MB for 5s of 1080p30).
- [x] Update `_export_clip_segment` to pass `SequenceClip` to the filter builder (currently only passes coordinates)

### 7. Agent Tool Parity

**File:** `core/chat_tools.py`

- [x] Add `random_hflip: bool = False`, `random_vflip: bool = False`, `random_reverse: bool = False` parameters to `generate_remix` tool
- [x] Pass these as `transform_options` dict to `generate_sequence()`
- [x] Update `list_sorting_algorithms` to include transform options for shuffle
- [x] Update system prompt description if needed

### 8. Tests

- [x] **Model round-trip**: `SequenceClip` with transforms serializes/deserializes correctly; old project files without transforms load with `False` defaults
- [x] **Transform assignment**: Given a seed, verify transforms are deterministic; verify ~50% distribution over many clips
- [x] **FFmpeg filter chain**: `_build_video_filter` produces correct filter strings for all transform combinations (8 combos: none, h, v, r, hv, hr, vr, hvr)
- [x] **Reverse safety limit**: Clips over 15 seconds skip `reverse`; short clips get it
- [x] **Agent tool**: `generate_remix` with transform flags produces clips with transforms set

## Acceptance Criteria

- [x] Clicking Dice Roll card generates a shuffled sequence (existing behavior preserved)
- [x] Three checkboxes appear in the timeline header when Dice Roll is active
- [x] Checking a box regenerates the sequence with ~50% of clips receiving that transform
- [x] Checkboxes are hidden when switching to a different algorithm
- [x] Exported video applies hflip/vflip/reverse filters correctly per clip
- [x] Reversed clips have both video and audio reversed
- [x] Clips longer than 15 seconds skip the reverse transform
- [x] Agent `generate_remix` tool supports `random_hflip`, `random_vflip`, `random_reverse` parameters
- [x] Project save/load preserves transform flags on SequenceClip
- [x] Old project files without transform fields load without errors

## Technical Considerations

- **FFmpeg `reverse` memory**: Buffers entire clip in RAM. 15-second safety limit prevents OOM. Log a warning when skipping.
- **Filter ordering**: `scale → hflip → vflip → reverse → chromatic_bar`. Reverse must come before chromatic bar overlay (otherwise the bar gets reversed). hflip/vflip before reverse so the flip is visible in both forward and reversed playback.
- **Audio reverse**: Use `areverse` filter in the audio filter chain alongside `reverse` in the video chain. Requires split audio/video filtering.
- **Deterministic transforms**: Use the same `random.Random(seed)` for both shuffle ordering and transform assignment, so a given seed always produces the same result.
- **Backward compatibility**: New `SequenceClip` fields default to `False`. `from_dict()` uses `.get("hflip", False)` pattern.
- **Preview limitation**: Video player preview will NOT show transforms — they only apply at FFmpeg export time. This matches how chromatic bar overlay works (also export-only). No changes needed to the video player.

## Dependencies & Risks

- **No new dependencies** — uses existing FFmpeg filters and Python stdlib `random`
- **Risk: Long clip reverse OOM** — mitigated by 15-second safety limit
- **Risk: Audio/video sync after reverse** — mitigated by applying `areverse` alongside `reverse`
- **Scope containment**: `sort_sequence` agent tool and MCP `shuffle_sequence` do NOT get transform support (they reorder existing clips in-place; transforms are a generation-time concept)

## Files to Modify

| File | Change |
|------|--------|
| `models/sequence.py` | Add `hflip`, `vflip`, `reverse` fields to `SequenceClip` |
| `ui/algorithm_config.py` | Add `transform_options` to shuffle config |
| `ui/tabs/sequence_tab.py` | Add checkboxes, `_TRANSFORM_OPTIONS`, regeneration logic |
| `core/remix/__init__.py` | Add `transform_options` param, `_assign_random_transforms()` |
| `ui/workers/sequence_worker.py` | Thread `transform_options` through worker |
| `core/sequence_export.py` | Add hflip/vflip/reverse to `_build_video_filter()`, reverse safety limit |
| `core/chat_tools.py` | Add transform params to `generate_remix`, update `list_sorting_algorithms` |
| `tests/` | New test files for model, transforms, filter chain, agent tool |

## Sources & References

- Similar pattern: `_DIRECTION_OPTIONS` in `ui/tabs/sequence_tab.py` — direction dropdown shown/hidden per algorithm
- Similar pattern: `show_chromatic_color_bar` on `Sequence` model — export-only visual effect
- FFmpeg filter docs: `hflip`, `vflip`, `reverse`, `areverse` are standard filters
- Existing filter chain: `_build_video_filter()` in `core/sequence_export.py`
