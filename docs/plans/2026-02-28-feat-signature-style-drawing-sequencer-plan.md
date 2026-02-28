---
title: "feat: Add Signature Style drawing-based sequencer"
type: feat
status: completed
date: 2026-02-28
origin: docs/brainstorms/2026-02-28-signature-style-brainstorm.md
---

# feat: Add Signature Style Drawing-Based Sequencer

## Overview

"Signature Style" is a new sequencing algorithm that interprets a visual drawing as an editing guide. Users draw (or import an image) on a canvas, and the system reads the drawing left-to-right to determine clip selection, pacing, and arrangement. Two interpretation modes share a single canvas: **Parametric** (pixel-level reading of position and color) and **VLM** (vision-language model interpretation of visual meaning).

This lives as a new algorithm card in the existing Sequence tab, opening a large modal dialog when selected. (see brainstorm: `docs/brainstorms/2026-02-28-signature-style-brainstorm.md`)

## Problem Statement / Motivation

Current sequencing algorithms are either fully automatic (shuffle, color sort, brightness) or text-driven (storyteller). There is no way for users to express an editing vision through a visual/gestural medium. Filmmakers and artists think visually — a drawing of an emotional arc, a color progression, or an abstract composition is a more natural way to communicate editing intent than dropdown menus or text prompts.

## Proposed Solution

### Core Architecture

```
Drawing (canvas/image)
    │
    ├── Parametric Mode ──→ Local pixel sampling ──→ DrawingSegment[]
    │                                                      │
    └── VLM Mode ──→ Local slicing + VLM interpretation ──→ DrawingSegment[]
                                                           │
                                                    Clip Matcher
                                                    (weighted multi-criteria)
                                                           │
                                                    (Clip, Source)[]
                                                           │
                                                    Timeline / Sequence
```

Both modes produce a shared intermediate data structure (`DrawingSegment`), which is then matched against the clip pool by a single shared matching algorithm.

### DrawingSegment — The Core Abstraction

```python
@dataclass
class DrawingSegment:
    x_start: int                          # Pixel position on canvas
    x_end: int                            # Pixel position on canvas
    target_duration_seconds: float        # Derived from output duration + segment proportion
    target_pacing: float                  # 0.0 (slow/long holds) to 1.0 (fast/short cuts)
    target_color: Optional[tuple[int,int,int]]  # RGB, None if B&W/no color
    is_bw: bool                           # True if this region is B&W
    # VLM-only fields (None in parametric mode):
    shot_type: Optional[str]
    energy: Optional[float]               # 0.0-1.0
    brightness: Optional[float]           # 0.0-1.0
    color_mood: Optional[str]             # "warm", "cool", "neutral", etc.
```

This decouples drawing interpretation from clip selection. Both modes produce `DrawingSegment[]`, and the matching algorithm consumes them uniformly.

### Mode Details

#### Parametric Mode

1. **Canvas sampling**: Sample the drawing at a user-configurable granularity (a "granularity" slider in the dialog controls how many X-positions are sampled).
2. **Y-axis reading**: At each sample point, read the highest non-background pixel's Y position. Map to pacing: `pacing = 1.0 - (y / canvas_height)` (top = fast, bottom = slow).
3. **Color reading**: Sample the average color in a small region around the sample point. If saturation < 15.0 in HSV (using existing `_SATURATION_THRESHOLD`), mark as B&W.
4. **Segment merging**: Adjacent samples with similar pacing (within threshold) and similar color (within color distance threshold) are merged into a single segment.
5. **B&W handling**: B&W segments prefer clips with `source.color_profile in ("grayscale", "sepia")`. If no B&W clips exist, fall back to pacing-only matching (see brainstorm: Resolved Questions).

#### VLM Mode

1. **Local slicing**: Color histogram difference along X-axis detects visual change boundaries. Produces initial slice boundaries.
2. **Whole-image VLM call**: Full drawing sent to VLM for overall mood/theme interpretation. Establishes context.
3. **Per-slice VLM calls**: Each slice sent with whole-image context. VLM can suggest merging adjacent slices or splitting complex ones. VLM returns chain-of-thought description followed by structured JSON.
4. **JSON parsing**: Use JSON mode where available (OpenAI, Anthropic), fall back to regex extraction. Unknown `shot_type` values mapped to nearest known value from `SHOT_TYPES`.
5. **Segment construction**: VLM JSON fields mapped to `DrawingSegment` fields.

### Clip Matching Algorithm

A weighted multi-criteria matching system:

1. **Criteria dimensions**: duration fit, color distance, shot type match, brightness match, energy match (VLM only).
2. **Duration is a weighted factor, not a hard constraint**: Duration closeness competes with analysis matching. If an 8-second clip is the best match on all other criteria for a 2-second target, it wins but gets trimmed via `SequenceClip.in_point/out_point` (see brainstorm: resolved during planning).
3. **Greedy assignment with reuse**: For each segment (left to right), find the best-matching clip from the pool. Clips can be reused (see brainstorm: Key Decisions).
4. **Color distance metric**: HSV hue distance for chromatic colors. Below the saturation threshold, clips are treated as "neutral" and excluded from color matching.
5. **Multi-color clips**: Match against the most dominant color (first in `dominant_colors` list).

### UI Architecture

**Large modal QDialog** (80% of screen size) with this layout:

```
┌──────────────────────────────────────────────────────────┐
│  Duration: [__2:30__]  FPS: [30▼]  Mode: [Parametric|VLM]│
│  Granularity: [────●──────] (parametric only)            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    Drawing Canvas                         │
│              (QWidget with QPainter)                      │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  [🖊 Pen] [⬜ Eraser] [🎨 Color] [↩ Undo] [↪ Redo]      │
│  [📂 Import Image]                                       │
├──────────────────────────────────────────────────────────┤
│  [Cancel]                              [Generate Sequence]│
└──────────────────────────────────────────────────────────┘
```

- **Canvas**: `QWidget` with custom `paintEvent()` / `mouseMoveEvent()` for drawing. Simpler than `QGraphicsView` for stroke capture.
- **Tools**: Pen, eraser, color picker (standard `QColorDialog`), undo/redo stack.
- **Import**: File picker for PNG/JPG/BMP/TIFF. Image loaded onto canvas, strokes drawn on top.
- **Mode toggle**: Switches between Parametric and VLM. Drawing persists across mode switches.
- **Granularity slider**: Only visible in parametric mode. Controls sampling density.

### Auto-Analysis on Demand

When the user clicks "Generate":

1. Check which metadata is needed (parametric: `dominant_colors`; VLM: `dominant_colors` + `shot_type` + `description`).
2. If clips are missing required metadata, show a confirmation: "X clips need color analysis. Analyze now?"
3. Run analysis as a sub-step of the generation worker, with progress feedback.
4. Proceed to sequence generation after analysis completes.

The `required_analysis` field in `ALGORITHM_CONFIG` is set to `["colors"]` (the minimum). VLM mode's additional requirements are handled dynamically within the dialog.

### Source Material

Supports both clips (from scene detection) and individual frames (see brainstorm: Key Decisions).

- **Clips**: Use native FPS. Duration setting determines how many clips and their pacing.
- **Frames**: FPS setting determines hold duration (`hold_frames = target_duration * fps`). Creates frame-based `SequenceClip` entries via `frame_id`.

## Technical Considerations

### Architecture Impacts

- New algorithm entry in `ALGORITHM_CONFIG` with `is_dialog: True`
- New dialog class routes through existing `_on_card_clicked()` handler in sequence tab
- Drawing interpretation and clip matching are pure functions in `core/remix/signature_style.py` — no GUI dependencies
- Dialog produces `(Clip, Source)[]` list via existing `sequence_ready` pattern

### Performance Implications

- **Parametric mode**: Pure local computation, negligible time (<1s for any reasonable clip pool)
- **VLM mode**: N+1 API calls where N = number of slices. Each call ~2-5 seconds. For a drawing with 15 slices, expect 30-75 seconds. Progress bar with per-slice status is essential.
- **Auto-analysis**: Color analysis is fast (~0.1s/clip). VLM-based analysis (describe, shots) is slow (~3-5s/clip). Cost estimation dialog gates expensive operations.

### Security Considerations

- Image import: Validate file format via QImage (rejects non-image files). Limit dimensions to prevent memory exhaustion (cap at 4096x4096).
- VLM prompts: User drawings sent to external VLM API. Content filtering may reject some drawings — handle gracefully with error message.
- No file path injection risk — canvas data is rendered to in-memory image, not file paths.

## System-Wide Impact

### Interaction Graph

```
User clicks "Signature Style" card
  → _on_card_clicked("signature_style") in sequence_tab.py
  → SignatureStyleDialog.exec()
  → User draws + clicks Generate
  → SignatureStyleWorker started (QThread)
    → [Optional] Auto-analysis sub-workers for missing metadata
    → Drawing interpretation (parametric or VLM)
    → Clip matching
  → Worker emits sequence_ready(sorted_clips)
  → sequence_tab._on_sequence_ready() populates timeline
```

### Error Propagation

- VLM API errors → caught in worker → emitted as `error` signal → dialog shows error message with retry option
- Malformed VLM JSON → fallback to regex parsing → if still fails, skip that slice and log warning
- Content filtering → VLM returns None content → error message: "Drawing could not be interpreted. Try a different drawing or switch to Parametric mode."
- Canvas import errors → QImage load failure → show "Could not load image" message

### State Lifecycle Risks

- Drawing is NOT saved with the project in v1. The `Sequence.algorithm` field records `"signature_style"` but the drawing itself is not persisted. Re-generating requires re-drawing or re-importing. This is an acceptable v1 limitation.
- No orphaned state risk — the dialog is modal, and generation either succeeds (producing a sequence) or fails (user stays in dialog).

### API Surface Parity

- Agent tools: `set_sorting_algorithm("signature_style")` should open the dialog. The agent cannot draw, but could trigger generation with an imported image path. This is a v2 enhancement.
- CLI: No CLI support needed for v1. This is inherently a visual/interactive feature.

### Integration Test Scenarios

1. **Parametric with full color metadata**: Draw a red→blue gradient line. Verify clips are arranged with red-dominant clips first, blue-dominant clips last.
2. **Parametric with B&W drawing on color clips**: Draw a B&W line. Verify system falls back to pacing-only (no color matching).
3. **VLM with simple drawing**: Draw three distinct colored blocks. Verify VLM produces three slices with corresponding color/mood interpretations.
4. **Clip reuse**: Set granularity high with a small clip pool. Verify clips are reused to fill all segments.
5. **Duration trimming**: Target 2s segments with 8s clips. Verify SequenceClip in/out points trim correctly.

## Acceptance Criteria

### Functional Requirements

- [x] "Signature Style" card appears in Sequence tab algorithm grid
- [x] Clicking card opens a large modal dialog with drawing canvas
- [x] Canvas supports pen, eraser, color picker, undo/redo
- [x] Users can import PNG/JPG images onto the canvas
- [x] Duration and FPS inputs above the canvas control output parameters
- [x] Parametric mode: drawing is sampled left-to-right, Y=pacing, color=color match
- [x] VLM mode: drawing is sliced locally, slices sent to VLM, JSON responses parsed
- [x] Mode toggle switches between parametric and VLM without losing drawing
- [x] Granularity slider controls segment count in parametric mode
- [x] Generated sequence appears in the timeline like any other algorithm
- [x] Clips can be reused when segments exceed clip pool size
- [x] Clips are trimmed via in/out points when target duration differs from clip duration
- [x] B&W drawing regions prefer B&W footage, fall back to pacing-only
- [x] Auto-analysis triggers when clips lack required metadata
- [ ] Both clips and individual frames are supported as source material

### Non-Functional Requirements

- [x] Parametric generation completes in <2 seconds for pools up to 500 clips
- [x] VLM mode shows per-slice progress ("Interpreting slice 3 of 12...")
- [x] Canvas drawing feels responsive (no lag on mouse move)
- [x] Image import rejects non-image files and caps dimensions at 4096x4096
- [x] VLM JSON parsing handles malformed responses gracefully

### Quality Gates

- [x] Unit tests for `DrawingSegment` construction from parametric sampling
- [x] Unit tests for clip matching algorithm with synthetic segments and clips
- [x] Unit tests for VLM JSON parsing (valid, partial, malformed)
- [x] Integration test for end-to-end parametric mode with real clip metadata
- [ ] Manual QA of canvas drawing responsiveness and tool behavior

## Dependencies & Prerequisites

- **Existing infrastructure**: All dependencies are already in `requirements.txt` (PySide6, Pillow, numpy, LiteLLM)
- **No new dependencies needed**
- **Requires**: Clips with `dominant_colors` metadata for color matching (auto-triggered if missing)
- **VLM mode requires**: A vision-capable model configured in settings (OpenAI GPT-4V, Anthropic Claude 3, Gemini Pro Vision, or local Ollama with vision model)

## Implementation Phases

### Phase 1: Core Algorithm + Parametric Mode

**Files to create:**
- `core/remix/signature_style.py` — Drawing interpretation + clip matching
- `core/remix/drawing_segment.py` — `DrawingSegment` dataclass

**Files to modify:**
- `core/remix/__init__.py` — Add `"signature_style"` dispatch in `generate_sequence()`
- `ui/algorithm_config.py` — Register algorithm card

**Deliverables:**
- `DrawingSegment` dataclass
- Parametric sampling function: canvas image → `DrawingSegment[]`
- Segment merging function
- Weighted clip matcher: `DrawingSegment[]` + clip pool → `(Clip, Source)[]`
- Color distance utility (HSV-based)
- Unit tests for all pure functions

**Success criteria:** Given a test image and a clip pool with known metadata, the algorithm produces a correctly ordered sequence matching expected pacing and color patterns.

### Phase 2: Dialog UI + Canvas

**Files to create:**
- `ui/dialogs/signature_style_dialog.py` — Main dialog
- `ui/widgets/drawing_canvas.py` — Canvas widget with pen/eraser/color/undo

**Files to modify:**
- `ui/tabs/sequence_tab.py` — Route `"signature_style"` card click to dialog

**Deliverables:**
- Drawing canvas widget with mouse/stylus input
- Tool bar (pen, eraser, color picker, undo/redo)
- Image import (file picker → load onto canvas)
- Duration/FPS inputs
- Mode toggle (parametric/VLM)
- Granularity slider (parametric only)
- "Generate" button wired to Phase 1 algorithm
- Progress feedback during generation

**Success criteria:** User can open dialog, draw on canvas, click Generate, and see a sequence appear in the timeline.

### Phase 3: VLM Mode

**Files to create:**
- `core/remix/drawing_vlm.py` — VLM interpretation logic (slicing, prompting, JSON parsing)

**Files to modify:**
- `core/remix/signature_style.py` — VLM mode branch in main entry point
- `ui/dialogs/signature_style_dialog.py` — VLM-specific progress feedback

**Deliverables:**
- Local adaptive slicing (color histogram diff along X-axis)
- Whole-image VLM prompt + parsing
- Per-slice VLM prompt (with merge/split suggestions) + JSON parsing
- JSON validation with fallback for malformed responses
- `DrawingSegment[]` construction from VLM output
- VLM cost estimation in confirm dialog
- Progress: "Interpreting slice 3 of 12..."
- Unit tests for JSON parsing, slice boundary detection

**Success criteria:** User can toggle to VLM mode, draw an abstract composition, and get a sequence where clip selection reflects the VLM's interpretation of the drawing's visual qualities.

### Phase 4: Polish + Frame Support

**Files to modify:**
- `ui/dialogs/signature_style_dialog.py` — Frame-specific FPS handling
- `core/remix/signature_style.py` — Frame-based `SequenceClip` creation

**Deliverables:**
- Frame-based source material support (`frame_id` + `hold_frames`)
- Auto-analysis trigger when metadata is missing
- Error handling for all edge cases (blank canvas, VLM failures, no matching clips)
- Canvas dimension limits for imported images

**Success criteria:** Full feature working end-to-end with both clips and frames, with graceful handling of all error paths.

## Alternative Approaches Considered

1. **Two separate algorithm cards** — Rejected: clutters grid, harder to discover both modes (see brainstorm: Why This Approach)
2. **Parametric-only, VLM as enhancement** — Rejected: limits creative expression for abstract drawings (see brainstorm)
3. **Inline state in Sequence tab** — Rejected in favor of modal dialog for consistency with Storyteller/Reference Guide pattern and simpler state management
4. **VLM-based adaptive slicing** — Rejected as primary method: unreliable for pixel coordinates. Hybrid approach chosen: local slicing with VLM merge/split suggestions

## Risk Analysis & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| VLM returns inconsistent JSON | Medium | High | JSON schema validation, regex fallback, default values for missing fields |
| Drawing canvas feels laggy | High | Low | QPainter is fast for stroke rendering; avoid per-pixel operations during drawing |
| Color matching produces poor perceptual results | Medium | Medium | Use HSV hue distance, not RGB euclidean; tune thresholds with real footage |
| Users don't understand Y-axis = pacing | Medium | Medium | Add Y-axis label ("Fast cuts ↑ / Slow holds ↓") and tooltip on canvas |
| VLM content filtering rejects drawings | Low | Low | Catch None responses, show friendly error, suggest parametric mode |
| Clip pool too small for drawing complexity | Low | Medium | Clip reuse enabled; show warning if pool < 5 clips |

## Future Considerations

- **Drawing persistence**: Save drawing image with project for re-generation (v2)
- **Agent integration**: Allow chat agent to trigger Signature Style with an image path (v2)
- **Pressure sensitivity**: Tablet pressure → line thickness → additional matching dimension (v2)
- **Eyedropper tool**: Sample colors from clip thumbnails for precise color matching (v2)
- **Preview before commit**: Show proposed sequence in dialog before populating timeline (v2)
- **Algorithm dropdown redo**: Allow re-triggering Signature Style from timeline header (v2)

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-02-28-signature-style-brainstorm.md](docs/brainstorms/2026-02-28-signature-style-brainstorm.md) — Key decisions carried forward: unified canvas with mode toggle, parametric mapping (X=time, Y=pacing, color=dominant match), adaptive VLM slicing, auto-analyze on demand, allow clip reuse

### Internal References

- Algorithm registration pattern: `ui/algorithm_config.py`
- Dialog algorithm pattern: `ui/tabs/sequence_tab.py` (`_on_card_clicked()` for `is_dialog` algorithms)
- Storyteller dialog (reference): `ui/dialogs/storyteller_dialog.py`
- Color analysis utilities: `core/analysis/color.py` (`extract_dominant_colors`, `_SATURATION_THRESHOLD`)
- VLM integration: `core/llm_client.py` (multi-provider, base64 image support)
- Worker pattern: `ui/workers/base.py` (`CancellableWorker`)
- Sequence model: `models/sequence.py` (`SequenceClip`, `frame_id`, `in_point`/`out_point`)
- Remix dispatch: `core/remix/__init__.py` (`generate_sequence()`)
- Reference-guided matching: `core/remix/reference_match.py` (weighted distance pattern)
- QGraphicsScene gotcha: `docs/solutions/runtime-errors/qgraphicsscene-missing-items-20260124.md` (always call `rebuild()` during init)
- Duplicate signal guard: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` (guard flags for workers)
