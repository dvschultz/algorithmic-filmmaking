---
title: "proto: Karaoke Text Detector Prototype"
type: proto
date: 2026-02-02
---

# proto: Karaoke Text Detector Prototype

## Overview

Build a standalone prototype script to validate the karaoke/text-based scene detection approach before integrating into the main application.

## Goal

Prove the concept works with a minimal, self-contained script that:
1. Takes a video file as input
2. Detects scene cuts based on text changes
3. Outputs cut timestamps and extracted text
4. Provides visual feedback (ROI preview)

## Deliverable

**Script**: `prototypes/karaoke_text_detector.py`

## Dependencies

```bash
pip install paddleocr rapidfuzz opencv-python numpy
```

Note: First run will download PaddleOCR models (~1GB).

## Usage

```bash
# Basic usage - scans full frame for text anywhere
python prototypes/karaoke_text_detector.py video.mp4

# Show preview before processing
python prototypes/karaoke_text_detector.py video.mp4 --preview

# Output to JSON
python prototypes/karaoke_text_detector.py video.mp4 --output cuts.json

# Verbose mode (show detected text)
python prototypes/karaoke_text_detector.py video.mp4 -v

# Restrict to bottom 25% (faster, for standard subtitles)
python prototypes/karaoke_text_detector.py video.mp4 --roi-top 0.75

# Speed up with frame skipping
python prototypes/karaoke_text_detector.py video.mp4 --frame-skip 3

# Force CPU mode
python prototypes/karaoke_text_detector.py video.mp4 --cpu
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--roi-top` | 0.0 | Top of text region (0.0=full frame, 0.75=bottom 25%) |
| `--roi-bottom` | 1.0 | Bottom of text region |
| `--threshold` | 75.0 | Text similarity threshold (0-100) |
| `--min-scene` | 30 | Minimum frames between cuts |
| `--frame-skip` | 1 | Process every Nth frame |
| `--language` | en | OCR language code |
| `--cpu` | false | Disable GPU acceleration |
| `--preview` | false | Show ROI preview window |
| `--output` | - | JSON output file path |
| `--verbose` | false | Print detected text |

## Expected Output

```
Processing: karaoke_video.mp4
ROI: top=0.75, bottom=1.0
Threshold: 75.0%
Min scene: 30 frames

Progress: 100.0% (3600/3600 frames)

Processing stats:
  Frames processed: 3600
  OCR calls: 287
  OCR skip ratio: 92.0%

Detected 12 scene cuts:

Cut 1: Frame 150 (00:00:05:00)
  Similarity: 23.5%

Cut 2: Frame 320 (00:00:10:20)
  Similarity: 45.2%
...
```

## Acceptance Criteria

- [x] Script runs standalone without main app
- [x] Detects cuts when text changes in ROI
- [x] Pixel pre-filter reduces OCR calls (target: 80%+ skip ratio)
- [x] Preview mode shows ROI overlay
- [x] JSON output includes all cut data
- [x] Verbose mode shows before/after text
- [x] Stats show OCR efficiency

## Testing

Test with the example karaoke frames provided by user:
1. Frame with "I SHOULDN'T HAVE LET YOU GO"
2. Frame with "THAT SOMETHING WASN'T RIGHT HERE"

Should detect a cut between these two states.

## Success Criteria

- OCR skip ratio > 80% (pixel pre-filter working)
- Correctly identifies text change boundaries
- Processing speed acceptable for iteration (~5-10 fps)
- False positive rate low on clean karaoke content

## Next Steps

After prototype validation:
1. Tune thresholds based on real-world testing
2. Integrate into main app per `2026-02-02-feat-karaoke-lyrics-scene-detection-plan.md`
3. Add UI controls to Cut tab
4. Add agent tool support
