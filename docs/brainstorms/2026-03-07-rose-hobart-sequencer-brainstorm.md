# Rose Hobart Sequencer

**Date:** 2026-03-07
**Status:** Brainstorm

## What We're Building

A new sequencer algorithm called "Rose Hobart" — named after Joseph Cornell's 1936 film where he re-edited *East of Borneo* to isolate shots of actress Rose Hobart. This sequencer takes a reference image of a person and filters the clip pool to keep only clips where that person appears, producing a focused supercut of a single individual across all source footage.

### Core Behavior

- **Strict filter**: Binary keep/discard. A clip either contains the person or it doesn't.
- **One person per run**: User provides a single reference image. Run multiple times for different people.
- **User-selected ordering**: After filtering, the user chooses how to order the kept clips — original order, by duration, by color, by brightness, by match confidence, or random.

## Why This Approach

### Face Embeddings (InsightFace/ArcFace)

Chosen over VLM-based matching and CLIP/SigLIP similarity for the best balance of speed, cost, and accuracy:

| Approach | Price | Speed | Quality |
|---|---|---|---|
| **Face embeddings** | Free, local | ~20-50ms/frame | Excellent for visible faces |
| VLM matching | $6-18/run (cloud) | 1-10s/frame | Most flexible but slow, non-deterministic |
| CLIP/SigLIP crops | Free, local | ~50-100ms/crop | High false positive rate |

**Accepted limitation**: The person's face must be at least partially visible. This aligns with the Rose Hobart concept — Cornell was isolating shots of an actress's face.

## Key Decisions

### UX Pattern: Dialog-based + Standalone Analysis

- **Primary path**: Dialog-based algorithm (like Storyteller/Signature Style). User clicks Rose Hobart card, dialog opens, user picks a reference image via file picker, sets options, and generates.
- **Secondary path**: Face embedding extraction is also available as a standalone analysis operation in the Analyze tab. This lets users pre-compute face data and reuse it across multiple runs.

### Frame Sampling

- Sample a frame **every 1 second** of clip duration (default, user can override in dialog).
- ~30 samples for a 30-second clip. Good balance of recall and speed.

### Matching Sensitivity

- **Three named presets** instead of a raw threshold slider:
  - **Strict** — Fewer matches, very high confidence only
  - **Balanced** — Default, good accuracy
  - **Loose** — More matches, may include borderline cases

### Data Persistence

- Face embeddings are **persisted on each Clip object** and saved to the project file.
- First run extracts embeddings (slow); subsequent runs filter instantly.
- Required for the Analyze tab integration anyway.

### Reference Image Input

- **File picker** (standard file dialog). User browses for an image on disk.
- Support **up to 3 reference images** of the same person from different angles. InsightFace averages multiple embeddings for more robust identity matching.
- No frame-from-video-player selection (can add later if needed).

### Post-Filter Ordering Options

In the dialog, user picks from:
- **Original order** — Chronological/source order (default)
- **By duration** — Shortest or longest first
- **By color** — Rainbow, warm-to-cool, etc.
- **By brightness** — Light to dark or dark to light
- **By confidence** — Highest face match confidence first
- **Random** — Shuffled

### Model Dependency

- **InsightFace** (~250MB model) auto-downloads on first use, following the existing pattern used for YOLO/DINOv2.
- Show a progress indicator during download.
- `insightface` added to requirements.txt; model weights download lazily.

## Technical Integration Points

### New Analysis Operation

Add `face_embeddings` to `ANALYSIS_OPERATIONS` in `core/analysis_operations.py`:
- Module: `core/analysis/faces.py` (new)
- Extracts face bounding boxes + 512-dim ArcFace embeddings per detected face
- Stores on `Clip` as a list of face embedding vectors
- Available in Analyze tab as "Detect Faces"

### New Algorithm Entry

Add `rose_hobart` to `ALGORITHM_CONFIG` in `ui/algorithm_config.py`:
- `is_dialog: True`
- `required_analysis: []` (dialog handles its own analysis)
- Icon suggestion: a film frame or portrait silhouette

### New Dialog

`ui/dialogs/rose_hobart_dialog.py`:
- File picker for reference images (1-3 images via multi-file picker or add button)
- Preview of selected reference image(s)
- Sensitivity preset dropdown (Strict / Balanced / Loose)
- Ordering dropdown (Original, Duration, Color, Brightness, Confidence, Random)
- Sampling interval override (optional)
- Progress bar during face matching
- **Match results summary**: Always show count of matched clips. If 0 matches, let user adjust sensitivity or cancel. No separate preview grid — generate directly after matching.

### Clip Model Changes

Add to `models/clip.py`:
- `face_embeddings: Optional[list]` — List of face embedding dicts, each with `bbox`, `embedding` (512-dim vector), `confidence`

## Resolved Questions

1. **Match preview before generating?** No — generate directly. Faster workflow; user can undo/re-run.
2. **Multiple reference images?** Yes — up to 3 images of the same person. Average embeddings for more robust matching.
3. **No clips match?** Show match count and let user adjust sensitivity or cancel. No hard error.
