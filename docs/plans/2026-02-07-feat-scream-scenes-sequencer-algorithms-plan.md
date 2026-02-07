---
title: "feat: Scream Scenes-Inspired Sequencer Algorithms"
type: feat
date: 2026-02-07
source: "XX Scream Scenes Director's Commentary transcript"
---

# Scream Scenes-Inspired Sequencer Algorithms

## Overview

Based on the director's commentary for the Scream Scenes project (31 days of experimental videos from ~3600 horror film clips), this plan identifies 6 new sequencer algorithms that map to real editing techniques used in the project. Scoped to algorithms only (clip ordering) — filter workflows and clip effects are out of scope.

## Design Decisions

These decisions were made during planning and should be followed during implementation:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Brightness sampling | Sample 3-5 frames per clip | More accurate than single thumbnail — clips can change brightness over duration |
| Similarity chain source constraint | Free chain (no source constraint) | Pure visual similarity drives ordering — same-source adjacency is fine if clips look alike |
| Silent clips in volume sort | Exclude from sequence entirely | Volume gradient is meaningless for clips without audio |
| Match cut approach | Static frame similarity only (v1) | Motion-aware matching is a separate system to build later |
| Chain starting clip | User's current selection, or first clip | If user has a clip selected in UI, use it as chain start |
| Color cycle filtered clips | Dropped entirely | Sequence only contains clips with strong color identity |
| Missing analysis data | Auto-analyze on algorithm selection | When user picks an algorithm that needs data, run analysis automatically before sorting |
| Card grid layout | Flat grid, all 13 cards visible | No tabs or grouping — keep it simple |

## Full Technique Inventory

All 31 days mapped to technique categories:

### Already Supported

| Day | Title | Technique | Existing Algorithm |
|-----|-------|-----------|-------------------|
| 1 | Stochastic Terror | Random subset with fixed seed | `shuffle` with seed |
| 3 | My Favorite Murder Rock | All clips from one film, in order | `sequential` (filter to source) |
| 7 | Scream Scenes | All clips from one franchise, in order | `sequential` (filter to source) |
| 8 | Suspense | Sort clips by duration, long→short | `duration` |
| 23 | A Cycle of Violence | Sort by dominant color | `color` (partially — see #6 below) |
| 31 | Body Parts | Every clip, randomly sequenced | `shuffle` |

### New Algorithms (This Plan)

| Day | Title | Technique | Proposed Algorithm |
|-----|-------|-----------|-------------------|
| 25 | Into the Dark | Sort by brightness, bright→dark | `brightness` |
| 21 | Crescendo | Sort by audio volume, quiet→loud | `volume` |
| 14 | Up Close and Personal | Sort by camera-to-subject distance | `proximity` |
| 29 | Human Centipede | Each next clip is most visually similar to current | `similarity_chain` |
| 20, 26 | Twister / The Brood | Match first/last frame similarity at cut points | `match_cut` |
| 23 | A Cycle of Violence | Filter to monotone clips, cycle through hue wheel | `color_cycle` |

### Out of Scope (documented for future reference)

**Filter workflows** (Days 4, 5, 10-12, 18): Thematic search, demographic tagging, motion type filtering — these are filter-then-sort workflows, not new sort algorithms.

**Clip effects** (Days 2, 6, 13, 15, 16, 17, 19, 22, 28): Frame interpolation, audio isolation, glitch, style transfer, audio-reactive effects — per-clip transforms, not ordering.

**Physical/hybrid** (Days 9, 24, 27, 30): Paper printing, film boiling, hand-traced animation, StyleGAN training — not automatable as sequencer algorithms.

**Motion-aware match cuts** (Day 20 motion direction aspect): Optical flow-based motion matching is a separate, more complex system that should be built as its own feature after static match cuts prove useful.

---

## Algorithm Specifications

### 1. Brightness Gradient

**Inspired by:** Day 25, "Into the Dark"

**Concept:** Sort clips from bright to dark (or vice versa), creating a visual arc of descending into darkness or emerging into light.

**Algorithm key:** `brightness`
**UI card name:** "Into the Dark"
**UI card description:** "Arrange clips from light to shadow, or shadow to light"

**Direction options:**
- `"bright_to_dark"` (default) — starts luminous, ends in darkness
- `"dark_to_bright"` — emergence from darkness

**Sort value:** Average luminance across 3-5 sampled frames from the clip.

**Computation:**
```
1. For each clip, sample 3-5 evenly-spaced frames from source video
   - Frame indices: start + i * (duration / (num_samples + 1)) for i in 1..num_samples
2. Convert each frame to grayscale (single channel)
3. Compute mean pixel value per frame (0-255 scale)
4. Average across all sampled frames
5. Normalize to 0.0-1.0
6. Cache on clip model for repeated sorts
```

**Auto-analysis behavior:** When user selects the Brightness algorithm and clips don't have `average_brightness` cached, compute it automatically before sorting. This is fast (a few seconds for hundreds of clips) since it only reads frames and computes means.

**New clip field:** `average_brightness: Optional[float]` — cached luminance value (0.0-1.0)

**Files to modify:**
- `core/remix/__init__.py` — add `brightness` case to `generate_sequence()`
- `core/analysis/color.py` — add `get_average_brightness(source_path, start_frame, end_frame, fps, num_samples=5) -> float`
- `models/clip.py` — add `average_brightness` field
- `ui/tabs/sequence_tab.py` — add card + direction dropdown entries
- `ui/widgets/sorting_card_grid.py` — add card definition

**Acceptance criteria:**
- [x] Samples 3-5 frames per clip, not just the thumbnail
- [x] Clips without source video available sort to end (brightness=0.5 default)
- [x] Direction dropdown shows "Bright to Dark" / "Dark to Bright"
- [x] Card always available (brightness auto-computed on demand)
- [x] Auto-analysis runs with progress indicator when data is missing
- [x] Agent tool `generate_remix(algorithm="brightness", direction="bright_to_dark")` works
- [x] Brightness values are cached on the clip and persisted in project save

---

### 2. Volume Gradient

**Inspired by:** Day 21, "Crescendo"

**Concept:** Sort clips by their audio loudness, building from whispers to screams (or the reverse). Clips without audio are excluded entirely.

**Algorithm key:** `volume`
**UI card name:** "Crescendo"
**UI card description:** "Build from silence to thunder, or thunder to silence"

**Direction options:**
- `"quiet_to_loud"` (default) — crescendo
- `"loud_to_quiet"` — decrescendo

**Sort value:** Mean RMS audio level of the clip's audio track in dB.

**Computation:**
```
1. Extract audio segment for clip:
   ffmpeg -ss {start_time} -t {duration} -i {source_path} -vn -f wav pipe:1
2. Compute RMS level:
   ffmpeg -i pipe:0 -af volumedetect -f null -
3. Parse "mean_volume: -XX.X dB" from stderr output
4. Store as float (higher = louder, range typically -60 to 0)
5. If no audio stream exists in source → clip is excluded from sequence
```

**Silent clip handling:** Clips from sources with no audio track are **excluded** from the output sequence entirely. The UI should show a notice: "X clips excluded (no audio track)".

**Auto-analysis behavior:** When user selects the Crescendo algorithm, auto-extract volume for all clips that don't have it cached. Show progress bar during extraction.

**New clip field:** `rms_volume: Optional[float]` — mean volume in dB (typically -60 to 0), `None` means not yet analyzed

**Files to modify:**
- `core/remix/__init__.py` — add `volume` case, filter out clips where `rms_volume is None` after analysis
- `core/analysis/audio.py` — add `extract_clip_volume(source_path, start_seconds, duration_seconds) -> Optional[float]` (returns None if no audio)
- `models/clip.py` — add `rms_volume` field
- `ui/tabs/sequence_tab.py` — add card + direction dropdown, show exclusion notice
- `ui/widgets/sorting_card_grid.py` — add card definition

**Acceptance criteria:**
- [x] Volume extraction uses FFmpeg `volumedetect` (no new dependencies)
- [x] Clips without audio track are excluded from sequence (not sorted to end)
- [x] UI shows "N clips excluded (no audio)" when clips are dropped
- [x] Card always available (volume auto-analyzed on demand)
- [x] Auto-analysis runs with progress indicator
- [x] Agent tool `generate_remix(algorithm="volume", direction="quiet_to_loud")` works
- [x] Volume values cached on clip and persisted in project save

---

### 3. Proximity Sorting

**Inspired by:** Day 14, "Up Close and Personal"

**Concept:** Sort clips by how close the camera is to the subject — from wide establishing shots to extreme close-ups (or the reverse). A continuous gradient rather than the 5 discrete buckets of the existing Focal Ladder (shot_type) algorithm.

**Algorithm key:** `proximity`
**UI card name:** "Up Close and Personal"
**UI card description:** "Glide from distant vistas to intimate close-ups"

**Direction options:**
- `"far_to_close"` (default) — wide shots first, close-ups last
- `"close_to_far"` — close-ups first, pulling back

**Sort value:** Numeric proximity score derived from existing analysis data.

**Score mapping (10-class cinematography, preferred):**
```
shot_size → numeric score:
  "extreme long shot"  → 1.0
  "very long shot"     → 2.0
  "long shot"          → 3.0
  "medium long shot"   → 4.0
  "medium shot"        → 5.0
  "medium close-up"    → 6.0
  "close-up"           → 7.0
  "big close-up"       → 8.0
  "extreme close-up"   → 9.0
  "insert"             → 10.0
```

**Fallback mapping (5-class shot_type):**
```
  "wide shot"          → 2.0
  "full shot"          → 4.0
  "medium shot"        → 5.0
  "close-up"           → 7.0
  "extreme close-up"   → 9.0
```

**Resolution order:** Check `clip.cinematography.shot_size` first (10-class, finer). If not available, use `clip.shot_type` (5-class). If neither exists, use default score 5.0 (middle).

**No auto-analysis for this one** — proximity depends on shot_type or cinematography analysis which are heavier operations (VLM calls). Card is disabled if no clips have the required data, prompting user to run analysis first.

**Files to modify:**
- `core/remix/__init__.py` — add `proximity` case with dual mapping
- `ui/tabs/sequence_tab.py` — add card + direction dropdown
- `ui/widgets/sorting_card_grid.py` — add card (disabled if no clips have shot_type or cinematography)

**Acceptance criteria:**
- [x] Uses 10-class cinematography.shot_size when available
- [x] Falls back to 5-class shot_type gracefully
- [x] Clips without any shot classification sort to middle (score=5.0)
- [x] Card disabled if zero clips have shot_type or cinematography analysis
- [x] Produces a noticeably smoother gradient than Focal Ladder
- [x] Agent tool `generate_remix(algorithm="proximity", direction="far_to_close")` works

---

### 4. Visual Similarity Chain

**Inspired by:** Day 29, "Human Centipede"

**Concept:** Starting from one clip, find the most visually similar clip in the dataset, then find the most similar to *that* one, and so on — creating a chain where every cut feels intentional because adjacent clips share visual qualities.

**Algorithm key:** `similarity_chain`
**UI card name:** "Human Centipede"
**UI card description:** "Chain clips together by visual similarity — each cut flows into the next"

**No direction options.** No source constraint — the chain jumps freely between films based purely on visual similarity.

**Starting clip:** If the user has a clip selected in the UI, that clip starts the chain. Otherwise, the first clip in the input list is used.

**Algorithm:**
```
1. Ensure all clips have CLIP embeddings (auto-analyze if missing)
2. Determine start clip:
   - If user has a clip selected → use that
   - Else → use first clip in input list
3. visited = {start}
4. sequence = [start]
5. While unvisited clips remain:
     current = sequence[-1]
     next = argmin(cosine_distance(current.embedding, c.embedding))
            for c in clips if c not in visited
     sequence.append(next)
     visited.add(next)
6. Return sequence
```

**CLIP model:** OpenAI CLIP ViT-B/32 — already loaded and used for shot classification in `core/analysis/shots.py`. Reuse the same model loading infrastructure; do not load a second instance.

**Embedding storage:** Store per-clip as `list[float]` (512 values). For 3600 clips ≈ 7MB — trivial for project files.

**Similarity computation:** Brute-force cosine similarity matrix. For N clips:
- N=100: instant
- N=1000: <0.5 seconds
- N=3600: ~2-3 seconds
- N>5000: consider FAISS index (not needed for v1)

**Auto-analysis behavior:** When user selects this algorithm, auto-compute CLIP embeddings for all clips that don't have them. This is the heaviest auto-analysis (~0.5-1 second per clip on CPU). Show progress bar with ETA.

**New clip field:** `embedding: Optional[list[float]]` — CLIP ViT-B/32 embedding vector (512 dimensions)

**New files:**
- `core/analysis/embeddings.py`:
  - `extract_clip_embedding(thumbnail_path: Path) -> list[float]` — single clip
  - `extract_clip_embeddings_batch(thumbnail_paths: list[Path]) -> list[list[float]]` — batch for efficiency
  - Uses same CLIP model as `core/analysis/shots.py`
- `core/remix/similarity_chain.py`:
  - `similarity_chain(clips_with_embeddings, start_clip_id=None) -> list[Clip]`
  - Greedy nearest-neighbor traversal
  - Returns reordered clip list

**Files to modify:**
- `core/remix/__init__.py` — add `similarity_chain` case
- `models/clip.py` — add `embedding` field (excluded from lightweight serialization by default, loaded on demand)
- `ui/tabs/sequence_tab.py` — add card, pass selected clip as start
- `ui/widgets/sorting_card_grid.py` — add card definition

**Acceptance criteria:**
- [x] Reuses CLIP model from `core/analysis/shots.py` (single model instance)
- [x] Embedding extraction supports batch mode for efficiency
- [x] Greedy chain completes in <5 seconds for 3600 clips
- [x] If user has a clip selected, it's used as chain starting point
- [x] Card available (auto-analyzes on selection)
- [x] Auto-analysis shows progress bar with ETA
- [x] Adjacent clips in output are visually similar (subjective but verifiable)
- [x] Embeddings cached on clip and persisted in project save
- [x] Agent tool `generate_remix(algorithm="similarity_chain")` works

---

### 5. Match Cut Sequencing

**Inspired by:** Days 20 and 26, "Twister" and "The Brood / Better Call Saul"

**Concept:** Find clips whose ending frames match the starting frames of other clips — composition, pose, framing — creating the illusion of continuous movement across cuts from different films. This is static frame similarity only; motion-aware matching is a future separate system.

**Algorithm key:** `match_cut`
**UI card name:** "Match Cut"
**UI card description:** "Find hidden connections between clips — where one ending meets another's beginning"

**How it differs from Similarity Chain:** The similarity chain compares overall visual appearance of clips (thumbnail/keyframe). Match Cut specifically compares the *last frame* of clip A to the *first frame* of clip B, optimizing for smooth transitions at cut points.

**Starting clip:** Same as similarity chain — user's selected clip if any, otherwise first clip.

**Algorithm:**
```
1. Ensure all clips have first_frame_embedding and last_frame_embedding
   (auto-analyze if missing — extracts actual frames from source video)
2. Determine start clip (same logic as similarity_chain)
3. Greedy chain using boundary frame similarity:
   visited = {start}
   sequence = [start]
   While unvisited clips remain:
     current = sequence[-1]
     next = argmin(cosine_distance(
              current.last_frame_embedding,
              c.first_frame_embedding
            )) for c not in visited
     sequence.append(next)
     visited.add(next)
4. Optional 2-opt refinement:
   - Swap pairs of clips and check if total chain cost decreases
   - Run for up to 1000 iterations or until no improvement
5. Return sequence
```

**Frame extraction:** Unlike similarity_chain (which embeds thumbnails), match_cut must extract actual first and last frames from the source video:
```
First frame: ffmpeg -ss {start_time} -i {source} -frames:v 1 -f image2pipe pipe:1
Last frame:  ffmpeg -ss {end_time - 1/fps} -i {source} -frames:v 1 -f image2pipe pipe:1
```

**New clip fields:**
- `first_frame_embedding: Optional[list[float]]` — CLIP embedding of clip's first frame
- `last_frame_embedding: Optional[list[float]]` — CLIP embedding of clip's last frame

**Builds on #4:** Shares the CLIP model and embedding extraction infrastructure from `core/analysis/embeddings.py`.

**New files:**
- `core/remix/match_cut.py`:
  - `match_cut_chain(clips, start_clip_id=None) -> list[Clip]`
  - Greedy nearest-neighbor using boundary embeddings
  - Optional 2-opt refinement pass

**Files to modify:**
- `core/analysis/embeddings.py` — add `extract_boundary_embeddings(source_path, start_frame, end_frame, fps) -> tuple[list[float], list[float]]`
- `core/remix/__init__.py` — add `match_cut` case
- `models/clip.py` — add `first_frame_embedding`, `last_frame_embedding` fields
- `ui/tabs/sequence_tab.py` — add card, pass selected clip as start
- `ui/widgets/sorting_card_grid.py` — add card definition

**Acceptance criteria:**
- [x] Extracts actual first and last frames from source video (not thumbnail)
- [x] Reuses CLIP model and infrastructure from `core/analysis/embeddings.py`
- [x] 2-opt refinement improves chain quality vs. pure greedy
- [x] Card available (auto-analyzes on selection)
- [x] Auto-analysis shows progress bar (slower than similarity_chain since it reads video frames)
- [x] If user has a clip selected, it starts the chain
- [x] Boundary embeddings cached on clip and persisted in project save
- [x] Agent tool `generate_remix(algorithm="match_cut")` works

---

### 6. Color Cycle

**Inspired by:** Day 23, "A Cycle of Violence"

**Concept:** A curated variant of color sorting — first filter to clips that are predominantly one color (high saturation, low variance), then cycle the survivors through the hue wheel. Clips without strong color identity are excluded entirely.

**Algorithm key:** `color_cycle`
**UI card name:** "Color Cycle"
**UI card description:** "Curate clips with strong color identity and cycle through the spectrum"

**How it differs from existing `color`:** The existing `color` algorithm sorts *all* clips by hue, including clips with muddy/mixed/desaturated colors. Color Cycle first filters to clips where one color dominates, producing a shorter but visually cohesive sequence of pure color transitions.

**Direction options:**
- `"spectrum"` (default) — linear hue progression (red→orange→yellow→...→violet)
- `"complementary"` — alternate between complementary colors for maximum contrast

**Algorithm:**
```
1. For each clip with dominant_colors:
   a. Convert dominant_colors (RGB) to HSV
   b. Compute color purity score:
      - mean_saturation = average S value across dominant colors
      - color_variance = standard deviation of H values across dominant colors
      - purity = mean_saturation * (1.0 - min(color_variance / 60.0, 1.0))
      - Range: 0.0 (muddy/mixed) to 1.0 (single strong color)
   c. If purity < threshold (default 0.4): exclude clip from sequence

2. For "spectrum" direction:
   - Sort remaining clips by primary hue (0-360°)

3. For "complementary" direction:
   - Sort by hue
   - Interleave: pick clip from bottom, then top, alternating
   - e.g., [red, violet, orange, blue, yellow, cyan, ...]
```

**No new analysis needed** — uses existing `dominant_colors` field. Color purity computed on the fly.

**Excluded clips handling:** The UI should show "N clips included (M excluded — low color purity)" so the user understands why the sequence is shorter than the input.

**Files to modify:**
- `core/remix/__init__.py` — add `color_cycle` case
- `core/analysis/color.py` — add `compute_color_purity(dominant_colors: list[tuple[int,int,int]]) -> float`
- `ui/tabs/sequence_tab.py` — add card + direction dropdown, show inclusion/exclusion notice
- `ui/widgets/sorting_card_grid.py` — add card (disabled if no clips have dominant_colors)

**Acceptance criteria:**
- [x] Only includes clips with purity score above threshold
- [x] Excluded clips are NOT in the output sequence
- [x] UI shows "N included, M excluded" notice
- [x] "Complementary" direction alternates between hue opposites
- [x] Card disabled if no clips have `dominant_colors`
- [x] Agent tool `generate_remix(algorithm="color_cycle", direction="spectrum")` works

---

## Auto-Analysis Behavior

When user selects an algorithm that requires data the clips don't yet have, the system should auto-analyze before sorting.

| Algorithm | Required Data | Auto-Analysis Action | Speed |
|-----------|--------------|---------------------|-------|
| `brightness` | `average_brightness` | Sample 3-5 frames, compute mean luminance | Fast (~0.1s/clip) |
| `volume` | `rms_volume` | FFmpeg volumedetect per clip | Fast (~0.2s/clip) |
| `proximity` | `shot_type` or `cinematography` | **No auto-analysis** — these require VLM calls. Card disabled instead. | N/A |
| `similarity_chain` | `embedding` | CLIP embedding per thumbnail | Medium (~0.5-1s/clip CPU) |
| `match_cut` | `first/last_frame_embedding` | Extract frames + CLIP embedding | Slow (~1-2s/clip, disk I/O) |
| `color_cycle` | `dominant_colors` | **No auto-analysis** — card disabled if no color data. | N/A |

**Auto-analysis UX flow:**
```
User clicks algorithm card
  → Check if required data exists on clips
  → If yes: run algorithm immediately
  → If partial: show dialog "X of Y clips need analysis. Run now?"
     → User confirms → run with progress bar → then sort
     → User cancels → return to card grid
  → If none: show dialog "This algorithm requires [data]. Analyze all clips now?"
     → Same flow as partial
```

**Progress display:**
- Show progress bar in the sequence tab header area
- Format: "Analyzing brightness... 42/100 clips"
- Allow cancellation (partial results are still usable)

---

## Implementation Order

```
Phase 1 — Quick wins (no new analysis infrastructure)
├── 1. Brightness Gradient     ~3-4 hours   (frame sampling + luminance + auto-analysis)
├── 3. Proximity Sorting       ~2-3 hours   (dual mapping of existing data)
└── 6. Color Cycle             ~3-4 hours   (purity filter + sort + exclusion UI)

Phase 2 — Audio analysis
└── 2. Volume Gradient         ~4-5 hours   (FFmpeg volumedetect + exclusion logic + auto-analysis)

Phase 3 — Embedding infrastructure
└── 4. Similarity Chain        ~10-12 hours (CLIP embeddings module + greedy chain + auto-analysis + batch mode)

Phase 4 — Builds on Phase 3
└── 5. Match Cut               ~8-10 hours  (boundary frame extraction + chain + 2-opt + auto-analysis)
```

**Phase 1** ships 3 new algorithms in ~1-2 days.
**Phase 2** adds audio-based sorting with ~1 day of work.
**Phase 3** is the cornerstone — CLIP embeddings unlock the most powerful algorithms. ~2 days.
**Phase 4** extends embeddings to cut-point matching. ~1-2 days.

**Total estimated effort:** ~30-38 hours across all 4 phases.

## UI Card Grid (After Implementation)

The Sequence tab card grid would show 13 algorithm cards in a flat grid:

| Card | Algorithm | Requires Analysis | Direction Options | Auto-Analyze |
|------|-----------|-------------------|-------------------|--------------|
| Dice Roll | `shuffle` | None | — | — |
| Time Capsule | `sequential` | None | — | — |
| Chromatic Flow | `color` | `dominant_colors` | Rainbow / Warm→Cool / Cool→Warm | No |
| **Color Cycle** | `color_cycle` | `dominant_colors` | Spectrum / Complementary | No |
| Tempo Shift | `duration` | None | Short First / Long First | — |
| **Into the Dark** | `brightness` | `average_brightness` | Bright→Dark / Dark→Bright | Yes |
| **Crescendo** | `volume` | `rms_volume` | Quiet→Loud / Loud→Quiet | Yes |
| Focal Ladder | `shot_type` | `shot_type` | — | No |
| **Up Close and Personal** | `proximity` | `shot_type` | Far→Close / Close→Far | No |
| **Human Centipede** | `similarity_chain` | `embedding` | — | Yes |
| **Match Cut** | `match_cut` | boundary embeddings | — | Yes |
| Exquisite Corpus | `exquisite_corpus` | `extracted_texts` | (dialog) | — |
| Storyteller | `storyteller` | `description` | (dialog) | — |

## References

- **Transcript:** `XX scream_scenes_(director's_commentary)_transcript.txt`
- **Existing algorithms:** `core/remix/__init__.py`
- **CLIP model (reuse):** `core/analysis/shots.py` (already loads CLIP ViT-B/32)
- **Audio infrastructure:** `core/remix/audio_sync.py`, `core/analysis/audio.py`
- **Color analysis:** `core/analysis/color.py`
- **Shot analysis:** `core/analysis/shots.py`
- **Cinematography VLM:** `core/analysis/cinematography.py`
- **Clip model:** `models/clip.py`
- **Sequence model:** `models/sequence.py`
