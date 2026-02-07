---
title: "feat: Auto-detect B&W footage for optimized scene detection"
type: feat
date: 2026-02-05
---

# Auto-Detect B&W Footage for Optimized Scene Detection

## Overview

Scene detection currently uses fixed parameters that treat all video content identically. On black-and-white footage, the default `AdaptiveDetector` and `ContentDetector` compute a weighted sum of `delta_hue`, `delta_sat`, and `delta_lum` — but for grayscale content, the hue and saturation channels are pure noise from compression artifacts. This noise dilutes the real luma signal and causes the detector to miss genuine cuts.

The fix: automatically detect grayscale/B&W video before running scene detection, then pass `luma_only=True` to the PySceneDetect detector. This eliminates two noise channels and lets the luma signal drive detection cleanly.

## Problem Statement

A user running the shuffle intention flow on a black-and-white video experienced a large number of missed cuts. The root cause is that PySceneDetect's default `ContentDetector`/`AdaptiveDetector` weighting gives equal weight to `delta_hue`, `delta_sat`, and `delta_lum`. On B&W footage:

- `delta_hue` is **noise** — no meaningful hue exists, but codec artifacts create random fluctuations
- `delta_sat` is **noise** — saturation is near-zero but fluctuates randomly
- `delta_lum` is the **only real signal** — brightness changes between cuts

With default weights (1.0 each for hue/sat/lum), two-thirds of the `content_val` score is noise. This both dilutes real cuts (missed detections) and can create spurious triggers (false positives).

PySceneDetect natively supports `luma_only=True` on both detectors, which zeroes out hue and saturation channels. This is the intended solution for grayscale content.

## Proposed Solution

Add a grayscale pre-check that runs inside `SceneDetector` before detector construction:

1. **Sample 10 evenly-spaced frames** from the video using OpenCV
2. **Downsample each to 160px wide** for speed
3. **Measure mean HSV saturation** per frame
4. **Classify** the video: `grayscale`, `sepia`, `mixed`, or `color`
5. **If grayscale or sepia**: pass `luma_only=True` to the detector
6. **Store classification** on the `Source` model as `color_profile`

**Key design decisions:**
- **`luma_only` only** — the auto-detection sets `luma_only=True` but never overrides the user's threshold/sensitivity. Users control how sensitive detection is; the system only ensures the right channels are used.
- **Inside `SceneDetector`** — all 7+ entry points (GUI, CLI, agent tools, MCP) get the behavior automatically.
- **Persisted on `Source`** — the classification is stored so it can be queried later by the agent, displayed in the UI, and survives project save/load.

## Technical Approach

### Architecture

The pre-check runs inside `SceneDetector.detect_scenes()` and `detect_scenes_with_progress()`, after the video is opened but before the PySceneDetect detector is constructed. This means every code path that runs scene detection gets automatic B&W optimization.

```
SceneDetector.detect_scenes[_with_progress]()
    ├── Open video (existing)
    ├── NEW: Run grayscale pre-check (if config.luma_only is None)
    │     ├── Sample 10 frames via cv2.VideoCapture
    │     ├── Downsample to 160px, convert to HSV
    │     ├── Measure mean saturation per frame
    │     └── Classify: grayscale / sepia / mixed / color
    ├── NEW: Set luma_only=True if grayscale or sepia
    ├── Construct AdaptiveDetector / ContentDetector (add luma_only param)
    └── Run detection (existing)
```

### Data Flow

```
DetectionConfig
    luma_only: Optional[bool] = None  # None=auto, True/False=explicit

SceneDetector.__init__(config)
    self.config = config

SceneDetector.detect_scenes(video_path)
    video = VideoStreamCv2(video_path)
    source = Source(...)

    # NEW: Auto-detect grayscale
    if self.config.luma_only is None:
        classification = detect_video_color_profile(video_path)
        source.color_profile = classification.classification
        use_luma_only = classification.is_grayscale
    else:
        use_luma_only = self.config.luma_only

    # Construct detector with luma_only
    detector = AdaptiveDetector(
        ...,
        luma_only=use_luma_only,
    )
```

### Implementation Phases

#### Phase 1: Grayscale Detection Function

Add `detect_video_color_profile()` to `core/analysis/color.py`.

**New function in `core/analysis/color.py`:**

```python
@dataclass
class ColorProfileResult:
    """Result of video color profile detection."""
    is_grayscale: bool
    classification: str  # "grayscale", "sepia", "mixed", "color"
    mean_saturation: float
    frame_saturations: list[float]

def detect_video_color_profile(
    video_path: Path,
    num_samples: int = 10,
    saturation_threshold: float = 12.0,
    grayscale_ratio_threshold: float = 0.95,
    downsample_width: int = 160,
) -> ColorProfileResult:
    """Detect whether a video is grayscale, sepia, mixed, or color.

    Samples N evenly-spaced frames, downsamples for speed,
    and checks HSV saturation to classify video content.
    """
    ...
```

**Classification logic:**

| Condition | Classification | `is_grayscale` |
|-----------|---------------|-----------------|
| ≥95% of frames have mean saturation < 12.0 | `grayscale` | `True` |
| ≥95% grayscale + mean sat 3-40 + hue std < 20° + mean hue 15-45° | `sepia` | `True` |
| 30-95% of frames are grayscale | `mixed` | `False` |
| <30% of frames are grayscale | `color` | `False` |

**Performance:** ~50-150ms for 10 frames at 160px wide. Negligible vs. detection time.

#### Phase 2: DetectionConfig + SceneDetector Integration

**Modify `core/scene_detect.py`:**

1. Add `luma_only: Optional[bool] = None` field to `DetectionConfig`
2. In `detect_scenes()` and `detect_scenes_with_progress()`:
   - After opening the video, if `self.config.luma_only is None`, run `detect_video_color_profile()`
   - Set the resolved `luma_only` value
   - Store `color_profile` on the returned `Source` object
3. Pass `luma_only` to both `AdaptiveDetector(luma_only=...)` and `ContentDetector(luma_only=...)`
4. Add `DetectionConfig.grayscale()` class method preset
5. Skip pre-check for karaoke mode (KaraokeTextDetector doesn't use PySceneDetect detectors)

**Add to `DetectionConfig`:**

```python
@dataclass
class DetectionConfig:
    threshold: float = 3.0
    min_scene_length: int = 15
    use_adaptive: bool = True
    min_content_val: float = 15.0
    window_width: int = 2
    luma_only: Optional[bool] = None  # None=auto-detect, True/False=explicit
```

#### Phase 3: Source Model Persistence

**Modify `models/clip.py`:**

1. Add `color_profile: Optional[str] = None` field to `Source`
2. Update `Source.to_dict()` to include `color_profile`
3. Update `Source.from_dict()` to read `color_profile`

Valid values: `"grayscale"`, `"sepia"`, `"mixed"`, `"color"`, `None` (not yet checked)

#### Phase 4: CLI + Agent Parameter Passthrough

**Modify `cli/commands/detect.py`:**

- Add `--luma-only` / `--no-luma-only` flag (default: `None` for auto-detect)
- Pass to `DetectionConfig(luma_only=...)`

**Modify `core/chat_tools.py`:**

- Add optional `luma_only: Optional[bool] = None` parameter to `detect_scenes`, `detect_scenes_live`, `detect_all_unanalyzed` tools
- Pass through to `DetectionConfig`

#### Phase 5: Logging + Progress

- Log classification at `logger.info` level: `"Video color profile: grayscale (mean saturation: 2.3)"`
- In `detect_scenes_with_progress()`, emit a status message: `progress_callback(0.05, "Checking video color profile...")`
- If grayscale/sepia detected, log: `"Using luma-only detection for grayscale content"`

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| **Video < 10 frames** | Sample all available frames; if < 3 frames readable, skip pre-check and use standard params |
| **Frame read failures** | Skip unreadable frames; classify based on successfully-read frames; if <50% readable, skip pre-check |
| **Sepia-toned footage** | Classified as `sepia`, treated same as `grayscale` (`luma_only=True`) since hue/sat channels carry no useful scene-change info |
| **Desaturated color** (e.g., heavy color grading) | Mean saturation typically 15-30, above threshold of 12 → classified as `color`, standard params used |
| **Mixed B&W + color** (e.g., flashbacks) | Classified as `mixed` → standard params used (safe default since color sections need hue/sat) |
| **Colored overlays on B&W** (logos, timecodes) | Small colored regions barely affect mean saturation → still classified as `grayscale` |
| **Digitized film with scanner color cast** | Slight warm/cool cast keeps mean sat 5-15; most below threshold → classified as `grayscale` |
| **Karaoke mode** | Pre-check skipped entirely (KaraokeTextDetector doesn't use PySceneDetect) |
| **User explicit `luma_only=True/False`** | Skips auto-detection, uses explicit value |
| **Interlaced archival footage** | 160px downsampling averages out interlacing artifacts; no special handling needed |
| **Night footage (very dark, low saturation)** | Dark color footage often has saturation 20-40 → classified as `color`; truly monochromatic dark footage → classified as `grayscale` correctly |
| **Batch detection (detect_all_unanalyzed)** | Pre-check runs per-source (each video classified independently) |
| **Re-detection on same source** | Pre-check runs fresh each time; `color_profile` on Source is updated |

## Acceptance Criteria

### Functional

- [x] B&W video scene detection uses `luma_only=True` automatically without user intervention
- [x] Sepia-toned video is detected and treated the same as grayscale
- [x] Color video detection behavior is unchanged (no regression)
- [x] Mixed B&W/color video uses standard parameters (safe default)
- [x] `color_profile` field on `Source` is populated after detection and persists in project save/load
- [x] All entry points benefit: GUI, CLI, agent tools (detect_scenes, detect_scenes_live, detect_all_unanalyzed)
- [x] User can explicitly override with `luma_only=True` or `luma_only=False` via CLI flag and agent tool parameter
- [x] Pre-check is skipped for karaoke detection mode
- [x] Auto-detection never modifies the user's threshold/sensitivity setting

### Non-Functional

- [x] Pre-check completes in < 500ms for any video
- [x] No new external dependencies (uses existing cv2, numpy)
- [x] Classification is logged at `info` level

## Files to Modify

| File | Changes |
|------|---------|
| `core/analysis/color.py` | Add `ColorProfileResult` dataclass and `detect_video_color_profile()` function |
| `core/scene_detect.py` | Add `luma_only` field to `DetectionConfig`; call pre-check in both `detect_scenes()` and `detect_scenes_with_progress()`; pass `luma_only` to detectors; add `DetectionConfig.grayscale()` preset |
| `models/clip.py` | Add `color_profile: Optional[str] = None` to `Source`; update `to_dict()` and `from_dict()` |
| `cli/commands/detect.py` | Add `--luma-only` / `--no-luma-only` CLI flag |
| `core/chat_tools.py` | Add `luma_only` parameter to `detect_scenes`, `detect_scenes_live`, `detect_all_unanalyzed` tools |
| `tests/test_color_profile.py` | **New file** — unit tests for `detect_video_color_profile()` |

## Testing Strategy

Per the "Prove It Pattern" in CLAUDE.md:

1. **Unit tests for `detect_video_color_profile()`:**
   - Create synthetic test frames: pure gray, sepia-toned, full color, mixed
   - Write to temporary video files with OpenCV
   - Assert classification matches expected result
   - Test edge cases: very short video (< 10 frames), unreadable frames

2. **Integration test for `SceneDetector` with B&W video:**
   - Create a short synthetic B&W video with known scene cuts (e.g., alternating bright/dark frames)
   - Run detection with default config (`luma_only=None`)
   - Assert `luma_only` was auto-set to `True`
   - Assert `source.color_profile == "grayscale"`

3. **Regression test for color video:**
   - Create a short synthetic color video
   - Run detection with default config
   - Assert behavior unchanged from current defaults

## References

### Internal

- Scene detection engine: `core/scene_detect.py:20-71` (DetectionConfig), `core/scene_detect.py:340-351` (detector construction)
- Color analysis: `core/analysis/color.py` (existing HSV utilities)
- Source model: `models/clip.py:56-155`
- Detection worker: `ui/main_window.py:110` (DetectionWorker)
- Agent tools: `core/chat_tools.py:2673` (detect_scenes), `core/chat_tools.py:3096` (detect_scenes_live)
- CLI detect: `cli/commands/detect.py:48`

### External

- [PySceneDetect `luma_only` parameter](https://www.scenedetect.com/docs/latest/api/detectors.html) — native grayscale support on both AdaptiveDetector and ContentDetector
- [PySceneDetect `Components` weights](https://github.com/Breakthrough/PySceneDetect) — `delta_hue`, `delta_sat`, `delta_lum`, `delta_edges` weighting
- [FFmpeg `signalstats` filter](https://ffmpeg.org/ffmpeg-filters.html#signalstats) — alternative approach using SATAVG metric (not used; OpenCV is faster for this use case)

### Institutional Learnings

- **Source ID mismatch** (`docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`): Scene detection workers create new Source objects with auto-generated UUIDs. Sync source_id on clips before storing.
- **Subprocess cleanup** (`docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`): Wrap cv2.VideoCapture in try/finally to release resources.
- **QThread duplicate signals** (`docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`): Guard flags on signal handlers to prevent duplicate worker creation.
