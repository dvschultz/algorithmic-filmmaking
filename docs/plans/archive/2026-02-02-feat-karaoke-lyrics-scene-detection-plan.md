---
title: "feat: Karaoke Lyrics Scene Detection"
type: feat
date: 2026-02-02
---

# feat: Karaoke Lyrics Scene Detection

## Overview

Add a new scene detection mode that detects cuts based on when on-screen text changes, rather than visual scene changes. This is designed for karaoke-style videos where lyrics overlay changes but the background remains consistent.

## Problem Statement / Motivation

The current scene detection (via PySceneDetect's AdaptiveDetector) detects visual scene changes. For videos like karaoke performances, music videos with lyrics, or educational content with subtitles, the visual scene may remain constant while the meaningful content (text) changes.

Users working with such content need to:
- Split videos at text change boundaries, not visual boundaries
- Isolate individual lyrics lines or subtitle segments
- Create clips based on textual content rather than visual transitions

The existing Exquisite Corpus integration doesn't handle this use case well because it relies on standard scene detection.

## Proposed Solution

Add a new `KaraokeTextDetector` that:
1. Monitors a configurable region of interest (ROI) for text - typically the bottom 25% for karaoke/subtitles
2. Uses fast pixel-difference detection to identify when the ROI changes
3. Runs OCR only when pixels change (performance optimization)
4. Compares extracted text using fuzzy matching to confirm actual text change
5. Registers scene cuts when text similarity drops below threshold

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| OCR Library | PaddleOCR 3.x | Fastest with GPU, best accuracy, handles multiple languages |
| Text Comparison | RapidFuzz | Fast Levenshtein distance, handles partial matches |
| Pre-filter | Pixel difference | Skip redundant OCR calls, ~10x faster |
| ROI Default | Full frame | Text position varies by video; user can restrict via config |
| Integration | New detector mode | Keep separate from existing AdaptiveDetector |
| Confirmation | 3 frames | Reduces false positives from OCR jitter during fade-in/out |
| Cut offset | 5 frames | Compensates for OCR detecting text mid-fade |

## Technical Approach

### Architecture Overview

```
Frame → ROI Extraction → Pixel Change Check → OCR (if changed) → Text Comparison → Scene Cut
                              ↓ (no change)
                           Skip frame
```

### Files to Create/Modify

| File | Changes |
|------|---------|
| `core/scene_detect.py` | Add `KaraokeTextDetector` class, update `DetectionConfig` |
| `core/analysis/text_detection.py` | Add ROI extraction helper |
| `ui/tabs/cut_tab.py` | Add detector mode selection UI |
| `core/chat_tools.py` | Update `detect_scenes` tool for new mode |
| `requirements.txt` | Add `paddleocr`, `rapidfuzz` |

### New Detector Implementation

```python
# core/scene_detect.py

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class KaraokeDetectionConfig:
    """Configuration for karaoke/text-based scene detection."""
    roi_top_percent: float = 0.0  # Start of ROI (0.0 = full frame, 0.75 = bottom 25%)
    roi_bottom_percent: float = 1.0  # End of ROI
    roi_left_percent: float = 0.0  # Full width by default
    roi_right_percent: float = 1.0
    text_similarity_threshold: float = 60.0  # Below this = new scene (0-100)
    pixel_change_threshold: float = 0.02  # 2% pixel difference triggers OCR
    min_scene_frames: int = 15  # Minimum frames between cuts (~0.5 sec at 30fps)
    confirm_frames: int = 3  # Require N consecutive frames to confirm text change
    cut_offset: int = 5  # Shift cuts backward to catch fade-in starts
    device: str = "gpu:0"  # PaddleOCR 3.x device: "gpu:0", "gpu:1", or "cpu"
    language: str = "en"  # OCR language


class KaraokeTextDetector:
    """Detect scene changes based on on-screen text changes.

    Optimized for karaoke-style videos where text overlays change
    but the visual background remains consistent.
    """

    def __init__(self, config: KaraokeDetectionConfig = None):
        self.config = config or KaraokeDetectionConfig()
        self._ocr = None  # Lazy init
        self._last_roi = None
        self._last_text = ""
        self._frames_since_cut = 0
        self._scene_cuts = []
        # Confirmation frame tracking
        self._pending_text = ""
        self._pending_count = 0

    def _get_ocr(self):
        """Lazy-initialize PaddleOCR 3.x."""
        if self._ocr is None:
            import os
            import logging
            os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "WARNING")
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            logging.getLogger("paddle").setLevel(logging.WARNING)

            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(device=self.config.device)
        return self._ocr

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region of interest from frame."""
        h, w = frame.shape[:2]
        top = int(h * self.config.roi_top_percent)
        bottom = int(h * self.config.roi_bottom_percent)
        left = int(w * self.config.roi_left_percent)
        right = int(w * self.config.roi_right_percent)
        return frame[top:bottom, left:right]

    def _pixels_changed(self, roi: np.ndarray) -> bool:
        """Check if ROI pixels changed significantly from last frame."""
        if self._last_roi is None:
            self._last_roi = roi.copy()
            return True

        # Calculate normalized pixel difference
        diff = np.abs(roi.astype(float) - self._last_roi.astype(float))
        change_ratio = np.mean(diff) / 255.0

        self._last_roi = roi.copy()
        return change_ratio > self.config.pixel_change_threshold

    def _extract_text(self, roi: np.ndarray) -> str:
        """Extract text from ROI using PaddleOCR 3.x."""
        ocr = self._get_ocr()
        result = ocr.predict(roi)  # PaddleOCR 3.x API

        if not result:
            return ""

        # PaddleOCR 3.x returns list of dicts with 'rec_texts' key
        texts = []
        for item in result:
            if isinstance(item, dict) and "rec_texts" in item:
                texts.extend(item["rec_texts"])
            elif isinstance(item, list):
                # Fallback for older format
                for line in item:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        texts.append(text)

        return " ".join(texts)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-100)."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0

        from rapidfuzz import fuzz
        return fuzz.ratio(text1.lower(), text2.lower())

    def process_frame(self, frame_num: int, frame: np.ndarray) -> bool:
        """Process a frame and return True if scene cut detected.

        Args:
            frame_num: Current frame number
            frame: BGR image array

        Returns:
            True if this frame starts a new scene
        """
        self._frames_since_cut += 1

        # Enforce minimum scene length
        if self._frames_since_cut < self.config.min_scene_frames:
            return False

        # Extract ROI
        roi = self._extract_roi(frame)

        # Fast path: skip if pixels haven't changed
        if not self._pixels_changed(roi):
            return False

        # Run OCR
        current_text = self._extract_text(roi)

        # Compare with previous text
        similarity = self._text_similarity(current_text, self._last_text)

        # Detect scene cut if text changed significantly
        if similarity < self.config.text_similarity_threshold:
            # Confirmation frames: require N consecutive frames with same new text
            if current_text == self._pending_text:
                self._pending_count += 1
            else:
                self._pending_text = current_text
                self._pending_count = 1

            # Only register cut after confirmation threshold reached
            if self._pending_count >= self.config.confirm_frames:
                self._last_text = current_text
                self._frames_since_cut = 0
                # Apply cut offset to catch fade-in starts
                actual_cut_frame = max(0, frame_num - self.config.cut_offset)
                self._scene_cuts.append(actual_cut_frame)
                # Reset confirmation state
                self._pending_text = ""
                self._pending_count = 0
                return True
        else:
            # Text is similar to last confirmed text, reset pending state
            self._pending_text = ""
            self._pending_count = 0
            self._last_text = current_text

        return False

    def get_scene_list(self) -> list[int]:
        """Return list of frame numbers where scenes start."""
        return self._scene_cuts.copy()

    def reset(self):
        """Reset detector state for new video."""
        self._last_roi = None
        self._last_text = ""
        self._frames_since_cut = 0
        self._scene_cuts = []
        self._pending_text = ""
        self._pending_count = 0
```

### Integration with Existing SceneDetector

```python
# core/scene_detect.py - Update existing class

class SceneDetector:
    """Scene detection wrapper with multiple detector modes."""

    def detect_scenes(
        self,
        video_path: Path,
        config: DetectionConfig = None,
        karaoke_config: KaraokeDetectionConfig = None,
        mode: str = "adaptive",  # "adaptive", "content", "karaoke"
        progress_callback: Callable = None,
    ) -> list[tuple[int, int]]:
        """Detect scenes in video.

        Args:
            video_path: Path to video file
            config: Standard detection config (for adaptive/content modes)
            karaoke_config: Karaoke detection config (for karaoke mode)
            mode: Detection mode - "adaptive", "content", or "karaoke"
            progress_callback: Optional callback(current, total)

        Returns:
            List of (start_frame, end_frame) tuples
        """
        if mode == "karaoke":
            return self._detect_karaoke_scenes(
                video_path, karaoke_config, progress_callback
            )
        else:
            return self._detect_visual_scenes(
                video_path, config, mode, progress_callback
            )

    def _detect_karaoke_scenes(
        self,
        video_path: Path,
        config: KaraokeDetectionConfig,
        progress_callback: Callable,
    ) -> list[tuple[int, int]]:
        """Detect scenes based on text changes."""
        import cv2

        config = config or KaraokeDetectionConfig()
        detector = KaraokeTextDetector(config)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detector.process_frame(frame_num, frame)

            if progress_callback and frame_num % 30 == 0:
                progress_callback(frame_num, total_frames)

            frame_num += 1

        cap.release()

        # Convert cut points to scene ranges
        cuts = [0] + detector.get_scene_list() + [total_frames]
        scenes = []
        for i in range(len(cuts) - 1):
            scenes.append((cuts[i], cuts[i + 1] - 1))

        return scenes
```

### UI Integration

Add detector mode selection to Cut tab:

```python
# ui/tabs/cut_tab.py

class CutTab(BaseTab):
    def _setup_detection_controls(self):
        # Existing controls...

        # Add detector mode dropdown
        mode_label = QLabel("Detection Mode:")
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItem("Visual (Adaptive)", "adaptive")
        self.mode_dropdown.addItem("Visual (Content)", "content")
        self.mode_dropdown.addItem("Text/Karaoke", "karaoke")
        self.mode_dropdown.currentIndexChanged.connect(self._on_mode_changed)

        # Karaoke-specific options (hidden by default)
        self.karaoke_options = QWidget()
        karaoke_layout = QFormLayout(self.karaoke_options)

        self.roi_top_spin = QDoubleSpinBox()
        self.roi_top_spin.setRange(0.0, 1.0)
        self.roi_top_spin.setValue(0.0)  # Full frame by default
        self.roi_top_spin.setSingleStep(0.05)
        karaoke_layout.addRow("ROI Top:", self.roi_top_spin)

        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setRange(0.0, 100.0)
        self.text_threshold_spin.setValue(60.0)  # Tuned default
        karaoke_layout.addRow("Text Similarity:", self.text_threshold_spin)

        self.confirm_frames_spin = QSpinBox()
        self.confirm_frames_spin.setRange(1, 10)
        self.confirm_frames_spin.setValue(3)  # Reduces OCR jitter false positives
        karaoke_layout.addRow("Confirm Frames:", self.confirm_frames_spin)

        self.cut_offset_spin = QSpinBox()
        self.cut_offset_spin.setRange(0, 30)
        self.cut_offset_spin.setValue(5)  # Compensates for fade-in detection delay
        karaoke_layout.addRow("Cut Offset:", self.cut_offset_spin)

        self.karaoke_options.setVisible(False)

    def _on_mode_changed(self, index):
        mode = self.mode_dropdown.currentData()
        self.karaoke_options.setVisible(mode == "karaoke")
        # Show/hide standard sensitivity slider
        self.sensitivity_widget.setVisible(mode != "karaoke")
```

### Agent Tool Update

```python
# core/chat_tools.py

@tools.register(
    description="Detect scenes in a video. Supports visual detection (adaptive/content) "
                "and text-based detection (karaoke) for videos with changing text overlays. "
                "For karaoke mode, specify roi_top (0.0-1.0) for text region.",
    requires_project=True,
)
def detect_scenes(
    project,
    main_window,
    source_id: str,
    mode: str = "adaptive",  # "adaptive", "content", "karaoke"
    sensitivity: float = 3.0,  # For visual modes
    roi_top: float = 0.0,  # For karaoke mode - where text region starts (0.0 = full frame)
    text_threshold: float = 60.0,  # For karaoke mode - similarity threshold
    confirm_frames: int = 3,  # For karaoke mode - frames to confirm text change
    cut_offset: int = 5,  # For karaoke mode - shift cuts backward for fade-in
) -> dict:
    """Detect scenes in a source video."""
    from core.scene_detect import (
        SceneDetector, DetectionConfig, KaraokeDetectionConfig
    )

    source = project.get_source(source_id)
    if not source:
        return {"success": False, "error": f"Source not found: {source_id}"}

    detector = SceneDetector()

    if mode == "karaoke":
        config = KaraokeDetectionConfig(
            roi_top_percent=roi_top,
            text_similarity_threshold=text_threshold,
            confirm_frames=confirm_frames,
            cut_offset=cut_offset,
        )
        scenes = detector.detect_scenes(
            source.file_path,
            karaoke_config=config,
            mode="karaoke",
        )
    else:
        config = DetectionConfig(threshold=sensitivity)
        scenes = detector.detect_scenes(
            source.file_path,
            config=config,
            mode=mode,
        )

    # Create clips from detected scenes
    clips = create_clips_from_scenes(project, source, scenes)

    return {
        "success": True,
        "result": {
            "mode": mode,
            "scene_count": len(scenes),
            "clip_ids": [c.id for c in clips],
        }
    }
```

## Acceptance Criteria

### Functional Requirements

- [ ] New `KaraokeTextDetector` class detects scenes based on text changes
- [ ] ROI is configurable (default: bottom 25% of frame)
- [ ] Text similarity threshold is configurable (default: 75%)
- [ ] Pixel-change pre-filter skips frames without visual change in ROI
- [ ] Minimum scene length prevents excessive cuts
- [ ] Cut tab UI allows selecting detection mode
- [ ] Karaoke mode shows relevant options, hides sensitivity slider
- [ ] Agent tool supports `mode="karaoke"` parameter
- [ ] Works with multiple languages via PaddleOCR

### Edge Cases

- [ ] No text in ROI: no scene cuts detected
- [ ] Same text repeated: no false cuts
- [ ] Partial text visibility: handled via fuzzy matching
- [ ] GPU unavailable: falls back to CPU
- [ ] Empty video: returns single scene

### Performance

- [ ] Pixel pre-filter reduces OCR calls by ~80%
- [ ] Processing speed: target 5-10 fps (acceptable for batch processing)
- [ ] Memory usage: single frame + ROI in memory at a time

## Prototype Learnings

The standalone prototype (`prototypes/karaoke_text_detector.py`) validated the approach and revealed important tunings.

### PaddleOCR 3.x API Changes

**CRITICAL:** PaddleOCR 3.x has breaking API changes from 2.x:

| Parameter | PaddleOCR 2.x | PaddleOCR 3.x |
|-----------|---------------|---------------|
| GPU selection | `use_gpu=True` | `device="gpu:0"` or `device="cpu"` |
| Angle classification | `use_angle_cls=True` | Removed (always enabled) |
| Logging | `show_log=False` | Use `logging` module instead |
| OCR method | `ocr(image, cls=True)` | `predict(image)` |

**Correct initialization:**
```python
import os
import logging
os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "WARNING")
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)

from paddleocr import PaddleOCR
ocr = PaddleOCR(device="gpu:0")  # or device="cpu"
```

### Tuned Parameters

Testing revealed these optimal defaults for karaoke/lyrics content:

| Parameter | Original | Tuned | Rationale |
|-----------|----------|-------|-----------|
| `text_similarity_threshold` | 75.0 | **60.0** | More sensitive catches text during fade-in |
| `min_scene_frames` | 30 | **15** | Allows faster text changes (~0.5s at 30fps) |
| `pixel_change_threshold` | 0.03 | **0.02** | Catches subtle fade-in changes |
| `confirm_frames` | N/A | **3** | NEW: Reduces OCR jitter false positives |
| `cut_offset` | N/A | **5** | NEW: Compensates for fade-in detection delay |

### Confirmation Frames Pattern

**Problem:** OCR jitter causes false positives. When text is fading in/out, OCR may briefly detect partial or corrupted text, triggering spurious cuts.

**Solution:** Require N consecutive frames showing the same new text before registering a cut.

```python
# Track potential new text
if similarity < threshold:
    if current_text == self._pending_text:
        self._pending_count += 1
    else:
        self._pending_text = current_text
        self._pending_count = 1

    # Only register cut after N confirmations
    if self._pending_count >= self.config.confirm_frames:
        self._scene_cuts.append(frame_num)
        self._pending_text = ""
        self._pending_count = 0
```

### Cut Offset Pattern

**Problem:** OCR detects text mid-fade-in, so detected cut points are several frames after the visual transition begins.

**Solution:** Shift cut points backward by a configurable offset.

```python
# When registering a cut, apply offset
actual_cut_frame = max(0, frame_num - self.config.cut_offset)
self._scene_cuts.append(actual_cut_frame)
```

### Frame-Accurate Export

**CRITICAL:** When splitting videos at detected cut points, use re-encoding for frame-accurate cuts:

```python
# WRONG - stream copy cuts on keyframes only (±2-10 seconds off)
cmd = ["ffmpeg", "-y", "-ss", start, "-i", video, "-t", duration, "-c", "copy", output]

# CORRECT - re-encode for frame-accurate cuts
cmd = [
    "ffmpeg", "-y",
    "-ss", f"{start_time:.3f}",
    "-i", str(video_path),
    "-t", f"{duration:.3f}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-c:a", "aac", "-b:a", "192k",
    "-avoid_negative_ts", "make_zero",
    str(output_file),
]
```

See skill `ffmpeg-frame-accurate-cutting` for full documentation.

## Dependencies & Risks

**New Dependencies:**
- `paddleocr>=3.0.0` - OCR engine (PaddleOCR 3.x required)
- `paddlepaddle` - PaddlePaddle backend (install separately: `pip install paddlepaddle`)
- `rapidfuzz>=3.0.0` - Fast fuzzy string matching

**Risks:**
| Risk | Mitigation |
|------|------------|
| PaddleOCR large download (~1GB) | Lazy-load, cache model |
| GPU memory issues | Fallback to CPU, configurable |
| OCR accuracy varies | Fuzzy matching handles minor errors |
| Non-Latin scripts | PaddleOCR supports 80+ languages |

## Success Metrics

- Users can detect scene cuts at text change boundaries
- False positive rate < 10% for clean karaoke videos
- Processing time < 3x real-time on CPU, < 1x on GPU
- Agent can trigger karaoke detection via chat

## Implementation Notes

### Performance Optimization Strategy

1. **Frame skipping**: Process every Nth frame for faster initial pass
2. **Pixel pre-filter**: Only run OCR when ROI pixels change >3%
3. **Batch OCR**: Queue multiple ROIs for batch inference (future)
4. **Resolution scaling**: Downscale ROI before OCR (maintain readability)

### Existing Code Reuse

- `core/analysis/text_detection.py` - EAST-based text region detection for validation
- `core/analysis/ocr.py` - Fallback Tesseract OCR if PaddleOCR unavailable
- `core/scene_detect.py` - Base infrastructure, video reading patterns

### Alternative Approaches Considered

1. **EAST + Template Matching**: Fast but requires known fonts
2. **Deep learning text spotting**: Most accurate but slowest
3. **Contour-based detection**: Fails on complex backgrounds
4. **VLM-based detection**: Too slow for frame-by-frame analysis

**Chosen approach** (PaddleOCR + pixel pre-filter) balances accuracy and speed.

## References & Research

### Internal References

- **Prototype script**: `prototypes/karaoke_text_detector.py` - Validated approach and tuned parameters
- **Prototype plan**: `docs/plans/2026-02-02-proto-karaoke-text-detector.md`
- FFmpeg frame-accurate cutting skill: `~/.claude/skills/ffmpeg-frame-accurate-cutting/SKILL.md`
- Existing OCR: `core/analysis/ocr.py:52-155`
- Text detection: `core/analysis/text_detection.py:132-212`
- Scene detection: `core/scene_detect.py:72-236`
- Exquisite Corpus: `core/remix/exquisite_corpus.py`

### External References

- PySceneDetect custom detectors: https://www.scenedetect.com/docs/latest/api/detectors.html
- PaddleOCR documentation: https://paddlepaddle.github.io/PaddleOCR/
- RapidFuzz documentation: https://rapidfuzz.github.io/RapidFuzz/

### Research Findings Summary

From best practices research and prototype validation:
- ROI extraction + pixel change detection + OCR is the recommended pattern
- PaddleOCR 3.x provides best speed/accuracy tradeoff with GPU (note API changes from 2.x)
- Levenshtein distance (via RapidFuzz) handles OCR errors gracefully
- **60% similarity threshold** works better than 75% for catching fade-in text changes
- **Confirmation frames (3)** are essential to reduce OCR jitter false positives
- **Cut offset (5 frames)** compensates for OCR detecting text mid-fade
- **Min scene frames (15)** allows faster lyric changes while preventing noise
- Full-frame ROI (roi_top=0.0) is more flexible than fixed bottom-25%; text position varies by video
