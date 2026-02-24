---
title: "feat: Reference-Guided Remixing"
type: feat
status: completed
date: 2026-02-23
origin: docs/brainstorms/2026-02-23-buzzy-analysis-and-artistic-alternative.md
---

# Reference-Guided Remixing

## Overview

Add a reference-guided matching algorithm to the Sequence tab. A reference video is decomposed into its structural DNA across artist-selected dimensions (color, shot scale, audio energy, rhythm, embeddings, cinematography). The same analysis runs on the artist's clip library. The tool matches clips to reference moments via weighted multi-dimensional distance, generating a rough sequence the artist refines.

This is **not** a viral content cloner. The artist's taste is the curation — the tool extends their visual sensibility by letting them remix their own footage through the structure of work that interests them. (See brainstorm: `docs/brainstorms/2026-02-23-buzzy-analysis-and-artistic-alternative.md`)

## Problem Statement / Motivation

Scene Ripper currently has 12 sequencer algorithms, all of which sort/chain clips along a **single dimension** (color, brightness, similarity, etc.). There is no way to say "arrange my clips to follow the structure of this other video" — matching across multiple weighted dimensions simultaneously.

Reference-guided remixing is the most-requested capability for experimental filmmakers who study existing work and want to respond to it with their own footage. The brainstorm identifies this as the core differentiator from tools like Buzzy that clone viral templates.

## Proposed Solution

### Core Loop

```
Reference Video ──→ Scene Detect ──→ Analyze (selected dims) ──→ Feature Vectors
                                                                        │
User Footage ────→ Scene Detect ──→ Analyze (same dims) ──→ Feature Vectors
                                                                        │
                                              Weighted Distance Matching ←┘
                                                        │
                                              Rough Sequence ──→ Artist Refinement
```

### Where It Lives

**The Sequence tab.** A new algorithm card ("Reference Guide") opens a configuration dialog where the artist selects the reference source and adjusts dimension weights. Output lands on the timeline like any other algorithm. No new tabs needed. (See brainstorm: "Where This Lives in Scene Ripper")

### Existing Infrastructure Reused

| Dimension | Existing Field on Clip | Module | Reuse Level |
|-----------|----------------------|--------|-------------|
| Color (hue) | `dominant_colors` | `core/analysis/color.py:200` | Direct |
| Brightness | `average_brightness` | `core/analysis/color.py:353` | Direct |
| Shot scale | `cinematography.shot_size` (10-class) or `shot_type` (5-class) | `core/analysis/cinematography.py`, `core/analysis/shots.py` | Direct |
| Audio energy | `rms_volume` | `core/analysis/audio.py:345` | Direct |
| Visual embedding | `embedding` (768-dim DINOv2) | `core/analysis/embeddings.py:91` | Direct |
| Camera movement | `cinematography.camera_movement` (8 categories) | `core/analysis/cinematography.py` | Direct |
| Duration/rhythm | `duration_frames` / `duration_seconds()` | `models/clip.py:225` | Direct |
| Description | `description` | VLM analysis | Deferred (needs sentence embeddings) |

All 7 MVP dimensions already have extraction functions and clip storage fields. **No new analysis modules needed for Phase 1.**

## Technical Approach

### The Matching Algorithm

Core function in `core/remix/reference_match.py`:

```python
def reference_guided_match(
    reference_clips: list[Clip],
    user_clips: list[Clip],
    weights: dict[str, float],       # dimension -> 0.0-1.0
    sources_by_id: dict[str, Source], # for FPS lookup
    allow_repeats: bool = False,
    match_reference_timing: bool = False,
) -> list[tuple[Clip, Clip, float]]:
    """
    For each reference clip, find the best-matching user clip.
    Returns list of (reference_clip, matched_user_clip, distance).
    """
```

**Key design decisions** (from brainstorm + SpecFlow analysis):

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Assignment strategy | **Greedy, sequential** (process reference clips in order) | Matches brainstorm pseudocode. First positions get best matches, which matters artistically for opening shots. |
| Allow Repeats off | 1:1 assignment — each user clip used at most once | Standard behavior. If ref has more clips than user, later positions get no match (gaps). |
| Allow Repeats on | Per-position argmin — same clip can appear multiple times | Simple, no limit on repeats in V1. |
| Normalization | **Min-max per dimension** across union of ref + user clips | Ensures all dimensions contribute equally at weight=1.0. |
| Duration comparison | **Always in seconds** (float), never frames | FPS varies per source (`models/clip.py:235` requires source FPS). |
| Categorical dimensions | Exact match: distance=0 if same, distance=1 if different | Camera movement has no inherent ordering. |
| Ordinal dimensions | Linear distance on numeric scale | Shot scale uses existing `_SHOT_SIZE_PROXIMITY` mapping (1-10). |
| Embedding dimensions | Cosine distance (0 to 2 range, normalized to 0-1) | Consistent with `similarity_chain.py:16`. |
| Insufficient user clips | Leave gaps in sequence, show warning | Artist fills manually or enables repeats. |
| Short user clips | Use clip's natural duration (don't extend) | No freeze-frame or looping in V1. |
| Text description matching | **Deferred to Phase 2** | Needs sentence embedding model; MVP uses visual embeddings. |

### Feature Vector Extraction

```python
def extract_feature_vector(
    clip: Clip,
    source: Source,
    active_dimensions: list[str],
) -> dict[str, Any]:
    """Extract normalized values for each active dimension."""
    vector = {}
    if "color" in active_dimensions and clip.dominant_colors:
        vector["color_hue"] = get_primary_hue(clip.dominant_colors) / 360.0
    if "brightness" in active_dimensions and clip.average_brightness is not None:
        vector["brightness"] = clip.average_brightness  # already 0-1
    if "shot_scale" in active_dimensions:
        vector["shot_scale"] = get_proximity_score(clip) / 10.0  # normalized 0-1
    if "audio" in active_dimensions and clip.rms_volume is not None:
        vector["audio"] = clip.rms_volume  # will be min-max normalized
    if "embedding" in active_dimensions and clip.embedding:
        vector["embedding"] = clip.embedding  # cosine distance computed separately
    if "movement" in active_dimensions and clip.cinematography:
        vector["movement"] = clip.cinematography.camera_movement  # categorical
    if "duration" in active_dimensions:
        vector["duration"] = clip.duration_seconds(source.fps)  # min-max normalized
    return vector
```

### Distance Function

```python
def weighted_distance(
    ref_vector: dict, user_vector: dict, weights: dict[str, float],
    normalizers: dict[str, tuple[float, float]],  # min, max per dim
) -> float:
    total = 0.0
    total_weight = 0.0
    for dim, weight in weights.items():
        if weight == 0 or dim not in ref_vector or dim not in user_vector:
            continue
        if dim == "embedding":
            dist = cosine_distance(ref_vector[dim], user_vector[dim])
        elif dim == "movement":
            dist = 0.0 if ref_vector[dim] == user_vector[dim] else 1.0
        else:
            # Scalar: min-max normalize then absolute difference
            lo, hi = normalizers[dim]
            r = (ref_vector[dim] - lo) / (hi - lo) if hi > lo else 0.5
            u = (user_vector[dim] - lo) / (hi - lo) if hi > lo else 0.5
            dist = abs(r - u)
        total += weight * dist
        total_weight += weight
    return total / total_weight if total_weight > 0 else float('inf')
```

### Data Model Extensions

**`models/sequence.py` — Sequence class:**
```python
# New fields on Sequence
reference_source_id: Optional[str] = None    # which source was the guide
dimension_weights: Optional[dict[str, float]] = None  # slider values
allow_repeats: bool = False
match_reference_timing: bool = False
```

**`ui/algorithm_config.py` — new card:**
```python
"reference_guided": {
    "icon": "🎯",
    "label": "Reference Guide",
    "description": "Match your clips to a reference video's structure",
    "allow_duplicates": True,  # controlled by allow_repeats toggle
    "required_analysis": [],   # dynamic — depends on selected dimensions
    "is_dialog": True,         # opens config dialog, like Storyteller
}
```

### UI: Reference Configuration Dialog

Follow the dialog pattern from Exquisite Corpus / Storyteller (not a new inline state). The dialog opens when the card is clicked:

```
┌─ Reference Guide Configuration ─────────────────┐
│                                                   │
│  Reference Source: [▼ Source dropdown         ]    │
│                                                   │
│  ─── Dimension Weights ──────────────────────     │
│                                                   │
│  ☑ Color        ████████░░  80%                   │
│  ☑ Brightness   ████░░░░░░  40%                   │
│  ☑ Shot Scale   ██████░░░░  60%                   │
│  ☑ Audio Energy ██░░░░░░░░  20%                   │
│  ☑ Visual Match ██████████  100%                  │
│  ☑ Movement     ████░░░░░░  40%                   │
│  ☑ Duration     ██████░░░░  60%                   │
│                                                   │
│  ─── Options ────────────────────────────────     │
│                                                   │
│  ☐ Allow Repeats                                  │
│  ☐ Match Reference Timing                         │
│                                                   │
│  Reference: 47 clips  │  Your footage: 312 clips  │
│                                                   │
│           [ Cancel ]  [ Generate Sequence ]        │
└───────────────────────────────────────────────────┘
```

**Source dropdown**: Lists all sources in the project. Selecting one designates it as the reference — its clips become the template, all other clips become the matching pool.

**Dimension checkboxes + sliders**: Each dimension has a checkbox (on/off) and a slider (0-100%). Unchecked dimensions are excluded from matching entirely (weight=0). Disabled dimensions show a tooltip explaining what analysis is needed.

**Clip counts**: Show reference clip count and user pool clip count so the artist understands the ratio.

**Generate button**: Triggers cost estimation (are both pools analyzed for active dimensions?), then runs the matching algorithm in `SequenceWorker`.

### Cost Estimation Extension

Extend `estimate_sequence_cost()` in `core/cost_estimates.py` to accept:
- `active_dimensions: list[str]` — only estimate costs for these dimensions
- Support two clip pools (reference clips + user clips) in a single estimate

The dialog shows cost before running, consistent with the existing `STATE_CONFIRM` pattern.

### Integration with generate_sequence()

Add to the dispatcher in `core/remix/__init__.py`:

```python
elif algorithm == "reference_guided":
    from core.remix.reference_match import reference_guided_match
    # reference_clips and user_clips separated by reference_source_id
    # weights, allow_repeats, match_reference_timing passed through
    matched = reference_guided_match(
        reference_clips, user_clips, weights, sources_by_id,
        allow_repeats, match_reference_timing,
    )
    return [(clip, source) for _, clip, _ in matched if clip]
```

### Agent API Extension

Add to `SequenceTab` agent methods:

```python
def generate_reference_guided(
    self,
    reference_source_id: str,
    weights: dict[str, float],
    allow_repeats: bool = False,
    match_reference_timing: bool = False,
) -> dict:
    """Agent-callable reference-guided sequence generation."""
```

Add corresponding tool definition in `core/chat_tools.py`.

## System-Wide Impact

- **Sequence model**: New optional fields (`reference_source_id`, `dimension_weights`, `allow_repeats`, `match_reference_timing`). Backward-compatible — all default to None/False.
- **Cost estimation**: Extended API with `active_dimensions` parameter. Existing callers unaffected (parameter is optional).
- **Algorithm dispatcher**: One new `elif` branch. No changes to existing algorithms.
- **Sequence tab**: New dialog class. Existing state machine unchanged (dialog pattern, not new state).
- **Algorithm config**: One new entry. Grid auto-layouts.
- **Project save/load**: New Sequence fields serialize/deserialize via existing dataclass-based JSON.

## Acceptance Criteria

### Phase 1: Core Algorithm + UI (MVP)

- [x] New file `core/remix/reference_match.py` with `reference_guided_match()` function
- [x] Feature vector extraction handles all 7 dimensions (color, brightness, shot_scale, audio, embedding, movement, duration)
- [x] Min-max normalization produces equal influence at equal weights
- [x] Greedy matching processes reference clips in order
- [x] "Allow Repeats" toggle switches between 1:1 assignment and per-position matching
- [x] Reference configuration dialog with source dropdown, dimension sliders, options
- [x] Disabled dimensions show tooltip about required analysis
- [x] Dialog shows clip counts for reference and user pools
- [x] Cost estimation runs for active dimensions on both clip pools
- [x] Matching runs in `SequenceWorker` background thread with progress
- [x] Output sequence appears on timeline
- [x] Sequence model persists `reference_source_id` and `dimension_weights`
- [x] Algorithm card "Reference Guide" appears in card grid
- [x] Agent tool `generate_reference_guided()` works
- [x] Handles edge cases: insufficient user clips (gaps + warning), all sliders at zero (error), single-clip pools

### Phase 2: Visualization — Analysis Strips

- [ ] Color trajectory strip: gradient of dominant colors per clip, rendered below timeline
- [ ] Rhythm strip: cut timing visualized as beat pattern alongside duration bars
- [ ] Strips visible for any sequence, not just reference-guided
- [ ] Strips toggle on/off from timeline view controls

### Phase 3: Comparison View

- [ ] Side-by-side analysis overlay: import two sources, view their color + rhythm strips together
- [ ] Dual-track preview: reference sequence above, matched sequence below
- [ ] Visual alignment markers showing which reference clip maps to which user clip

### Phase 4: Movement Analysis + Text Matching

- [ ] Optical flow classification as new analysis dimension (or extend cinematography VLM)
- [ ] Sentence embedding model for description matching
- [ ] Both integrated as additional matching dimensions

## Implementation Phases

### Phase 1: Core Algorithm + Minimal UI — MVP

**New files:**
- `core/remix/reference_match.py` — matching algorithm, feature extraction, distance function
- `ui/dialogs/reference_guide_dialog.py` — configuration dialog

**Modified files:**
- `models/sequence.py` — add `reference_source_id`, `dimension_weights`, `allow_repeats`, `match_reference_timing` fields
- `core/remix/__init__.py` — add `reference_guided` case to `generate_sequence()`
- `ui/algorithm_config.py` — add `"reference_guided"` entry
- `ui/widgets/sorting_card_grid.py` — card auto-added via config
- `ui/tabs/sequence_tab.py` — wire dialog to card click, handle dialog result
- `core/cost_estimates.py` — extend `estimate_sequence_cost()` with `active_dimensions` param
- `core/chat_tools.py` — add agent tool definition
- `core/tool_executor.py` — wire agent tool

**Tests:**
- `tests/test_reference_match.py` — unit tests for matching algorithm, normalization, edge cases
- `tests/test_sequence_model.py` — serialization of new fields

### Phase 2: Visualization — Analysis Strips

**New files:**
- `ui/timeline/analysis_strip.py` — base class for analysis visualization strips
- `ui/timeline/color_strip.py` — color trajectory gradient strip
- `ui/timeline/rhythm_strip.py` — cut timing beat pattern strip

**Modified files:**
- `ui/timeline/timeline_widget.py` — add strip rendering below tracks
- `ui/timeline/timeline_scene.py` — layout strips in scene

### Phase 3: Comparison View

**New files:**
- `ui/widgets/comparison_view.py` — side-by-side analysis overlay

**Modified files:**
- `ui/tabs/sequence_tab.py` — add comparison mode toggle
- `ui/timeline/timeline_widget.py` — support dual-track display

### Phase 4: Movement Analysis + Text Matching

**New files:**
- `core/analysis/text_embeddings.py` — sentence embedding extraction (new dependency: `sentence-transformers`)

**Modified files:**
- `core/remix/reference_match.py` — add text embedding distance
- `models/clip.py` — add `description_embedding` field
- `requirements.txt` — add `sentence-transformers` (optional dependency)

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| Normalization producing unintuitive weight behavior | Unit test with synthetic data: equal weights should produce equal influence across dimensions |
| Large clip libraries (500+) making matching slow | Algorithm is O(R * U) per dimension; profile and consider numpy vectorization if >2s |
| Cinematography analysis required for movement/shot_scale but expensive (VLM) | Dialog disables those sliders if clips lack analysis; shows cost estimate |
| Text matching requires new dependency | Deferred to Phase 4; MVP works with 6 non-text dimensions |
| Sequence overwrite by generic handlers | Follow dialog-based pattern from Storyteller; set `sequence.algorithm = "reference_guided"` to prevent fallback overwrite |

## Success Metrics

- Artist can generate a reference-guided sequence from existing analyzed clips in <30 seconds
- Adjusting weight sliders produces noticeably different sequences from the same inputs
- Algorithm correctly handles the ratio mismatch (more ref clips than user clips, and vice versa)
- Round-trip save/load preserves reference source and weight configuration

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-02-23-buzzy-analysis-and-artistic-alternative.md](docs/brainstorms/2026-02-23-buzzy-analysis-and-artistic-alternative.md) — Key decisions carried forward: reference-guided matching as core differentiator, lives in Sequence tab (no new tab), artist controls dimension weights via sliders, MVP builds on existing analysis infrastructure.

### Internal References

- Matching algorithm precedent: `core/remix/similarity_chain.py` (greedy nearest-neighbor), `core/remix/match_cut.py` (boundary matching + 2-opt)
- Auto-compute pattern: `core/remix/__init__.py:315-403` (`_auto_compute_brightness`, `_auto_compute_volume`, etc.)
- Dialog algorithm pattern: `ui/tabs/sequence_tab.py` (Storyteller/Exquisite Corpus flow)
- Cost estimation: `core/cost_estimates.py:159` (`estimate_sequence_cost()`)
- Proximity score mapping: `core/remix/__init__.py:38-49` (`_SHOT_SIZE_PROXIMITY`)
- Feature vector dimensions: `models/clip.py:189-222`, `models/cinematography.py:182-487`
- Embedding distance: `core/remix/similarity_chain.py:16` (`_cosine_distance_matrix()`)
- Sequence model: `models/sequence.py:132-212`
- Algorithm config: `ui/algorithm_config.py`

### Related Plans

- [Scream Scenes Sequencer Algorithms](docs/plans/2026-02-07-feat-scream-scenes-sequencer-algorithms-plan.md) — established the card-based algorithm pattern, auto-compute, and cost estimation flow that this feature extends
