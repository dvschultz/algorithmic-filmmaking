---
title: "feat: Whisper Transcription with faster-whisper"
type: feat
date: 2026-01-24
priority: P1
---

# feat: Whisper Transcription with faster-whisper

## Overview

Add speech-to-text transcription for video clips using `faster-whisper`, enabling text search across clips, dialogue-based sequencing, and transcript export. This is a P1 priority feature from the course mapping.

## Problem Statement

Users working with dialogue-heavy footage have no way to:
1. Search clips by spoken words
2. Find clips containing specific phrases
3. Build sequences based on dialogue content
4. Export transcripts for subtitling or documentation

## Proposed Solution

Integrate `faster-whisper` (CTranslate2-optimized Whisper) with:
- User-selectable model sizes (tiny → large)
- Background transcription worker following existing patterns
- Transcript storage in clip metadata
- Text search in Analyze tab
- Optional transcript display overlay

## Technical Approach

### Why faster-whisper

| Feature | faster-whisper | Original Whisper | whisper.cpp |
|---------|---------------|------------------|-------------|
| Speed | 4x faster | Baseline | 2x faster |
| CPU Support | ✅ Excellent | ✅ Works | ✅ Excellent |
| Python API | ✅ Native | ✅ Native | ❌ Bindings |
| Quantization | ✅ int8 | ❌ No | ✅ Yes |
| Memory | Lower | Higher | Lower |
| Active Dev | ✅ Yes | ✅ Yes | ✅ Yes |

### Model Options

```python
WHISPER_MODELS = {
    "tiny.en": {"size": "39MB", "speed": "~32x", "accuracy": "Basic", "vram": "<1GB"},
    "small.en": {"size": "244MB", "speed": "~15x", "accuracy": "Good", "vram": "~1GB"},
    "medium.en": {"size": "769MB", "speed": "~5x", "accuracy": "Better", "vram": "~2GB"},
    "large-v3": {"size": "1.5GB", "speed": "~2x", "accuracy": "Best", "vram": "~4GB"},
}
```

**Default:** `small.en` (good balance of speed/accuracy for most footage)

### Data Model Changes

```python
# models/clip.py - Add transcript field
@dataclass
class Clip:
    # ... existing fields ...
    transcript: Optional[list[TranscriptSegment]] = None

@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start_time: float  # seconds from clip start
    end_time: float
    text: str
    confidence: float = 0.0
    words: Optional[list[dict]] = None  # word-level timestamps
```

### Files to Create/Modify

| File | Change |
|------|--------|
| `core/transcription.py` | **New** - faster-whisper wrapper |
| `models/clip.py` | Add `transcript` field and `TranscriptSegment` |
| `ui/main_window.py` | Add `TranscriptionWorker` |
| `core/settings.py` | Add transcription model settings |
| `ui/widgets/settings_dialog.py` | Add model selection UI |

### Core Module: `core/transcription.py`

```python
"""Speech transcription using faster-whisper."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lazy load to avoid startup delay
_model = None
_model_name = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0


def get_model(model_name: str = "small.en"):
    """Get or load the Whisper model (lazy loading)."""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {model_name}")
        # Use int8 quantization for CPU, float16 for GPU
        _model = WhisperModel(
            model_name,
            device="auto",  # auto-detect CPU/GPU
            compute_type="int8",  # Use int8 for CPU efficiency
        )
        _model_name = model_name
        logger.info(f"Whisper model loaded: {model_name}")

    return _model


def transcribe_video(
    video_path: Path,
    model_name: str = "small.en",
    language: str = "en",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to video file
        model_name: Whisper model to use
        language: Language code (e.g., "en", "es", "auto")
        progress_callback: Optional callback(progress, message)

    Returns:
        List of TranscriptSegment objects
    """
    model = get_model(model_name)

    if progress_callback:
        progress_callback(0.1, "Extracting audio...")

    # faster-whisper can process video files directly
    segments, info = model.transcribe(
        str(video_path),
        language=language if language != "auto" else None,
        word_timestamps=True,
        vad_filter=True,  # Filter out non-speech
    )

    if progress_callback:
        progress_callback(0.5, "Processing segments...")

    results = []
    for segment in segments:
        results.append(TranscriptSegment(
            start_time=segment.start,
            end_time=segment.end,
            text=segment.text.strip(),
            confidence=segment.avg_logprob,
        ))

    if progress_callback:
        progress_callback(1.0, f"Transcribed {len(results)} segments")

    return results


def transcribe_clip(
    source_path: Path,
    start_time: float,
    end_time: float,
    model_name: str = "small.en",
) -> list[TranscriptSegment]:
    """Transcribe a specific clip range from a video.

    For efficiency, extracts the audio segment first.
    """
    import subprocess
    import tempfile

    # Extract audio segment to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Extract audio segment with FFmpeg
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", str(source_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # Whisper expects 16kHz
            "-ac", "1",  # Mono
            str(tmp_path),
        ], capture_output=True, check=True)

        # Transcribe the segment
        segments = transcribe_video(tmp_path, model_name)

        # Adjust timestamps relative to clip start
        # (already relative since we extracted the segment)
        return segments

    finally:
        tmp_path.unlink(missing_ok=True)
```

### Worker: `TranscriptionWorker`

```python
class TranscriptionWorker(QThread):
    """Background worker for transcribing clips."""

    progress = Signal(int, int)  # current, total
    transcript_ready = Signal(str, list)  # clip_id, segments
    finished = Signal()
    error = Signal(str)  # error message

    def __init__(self, clips: list[Clip], source: Source, model_name: str = "small.en"):
        super().__init__()
        self.clips = clips
        self.source = source
        self.model_name = model_name
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.transcription import transcribe_clip

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                break

            try:
                segments = transcribe_clip(
                    self.source.file_path,
                    clip.start_time(self.source.fps),
                    clip.end_time(self.source.fps),
                    self.model_name,
                )
                self.transcript_ready.emit(clip.id, segments)
            except Exception as e:
                logger.warning(f"Transcription failed for {clip.id}: {e}")

            self.progress.emit(i + 1, total)

        self.finished.emit()
```

### Settings Addition

```python
# core/settings.py - Add to Settings dataclass

# Transcription settings
transcription_model: str = "small.en"  # tiny.en, small.en, medium.en, large-v3
transcription_language: str = "en"  # en, auto, or specific language code
auto_transcribe: bool = False  # Auto-transcribe on detection
```

### UI Integration

#### Analyze Tab - Transcribe Button

Add "Transcribe All" button next to existing analysis buttons:

```python
# In Analyze tab toolbar
self.transcribe_btn = QPushButton("Transcribe")
self.transcribe_btn.setToolTip("Transcribe speech in all clips")
self.transcribe_btn.clicked.connect(self._on_transcribe_click)
```

#### Clip Browser - Transcript Display

Show transcript snippet on clip hover or in detail panel:

```python
# In clip detail/hover tooltip
if clip.transcript:
    text = " ".join(seg.text for seg in clip.transcript[:3])
    if len(clip.transcript) > 3:
        text += "..."
    transcript_label.setText(f'"{text}"')
```

#### Search Integration

Add text search to filter clips by transcript content:

```python
def filter_by_transcript(self, query: str) -> list[Clip]:
    """Filter clips containing the search query in transcript."""
    query_lower = query.lower()
    return [
        clip for clip in self.clips
        if clip.transcript and any(
            query_lower in seg.text.lower()
            for seg in clip.transcript
        )
    ]
```

## Implementation Steps

### Step 1: Add Dependency

```bash
pip install faster-whisper
```

Add to `requirements.txt`:
```
faster-whisper>=1.0.0
```

### Step 2: Create Core Module

Create `core/transcription.py` with:
- [x] `TranscriptSegment` dataclass
- [x] `get_model()` lazy loader
- [x] `transcribe_video()` function
- [x] `transcribe_clip()` function for clip ranges

### Step 3: Update Data Model

Modify `models/clip.py`:
- [x] Add `transcript: Optional[list[TranscriptSegment]]` field
- [x] Import `TranscriptSegment` from core.transcription

### Step 4: Add TranscriptionWorker

Add to `ui/main_window.py`:
- [x] `TranscriptionWorker` class (follows ColorAnalysisWorker pattern)
- [x] `self.transcription_worker` instance variable
- [x] `_on_transcribe_click()` handler
- [x] `_on_transcript_ready()` slot
- [x] Transcribe button in Analyze tab toolbar

### Step 5: Add Settings

Modify `core/settings.py`:
- [x] Add `transcription_model` setting
- [x] Add `transcription_language` setting
- [x] Add `auto_transcribe` setting

### Step 6: Settings UI

Modify settings dialog:
- [x] Model selection dropdown
- [x] Language selection (en, auto, or list)
- [x] Auto-transcribe checkbox

### Step 7: Search Integration

Add to Analyze tab:
- [x] Search text input
- [x] Filter by transcript content
- [x] Transcript overlay on hover

## Acceptance Criteria

- [x] Can transcribe all clips with one button click
- [x] Model selection in settings (tiny → large)
- [x] Progress indicator during transcription
- [x] Transcript text visible on clip hover/detail
- [x] Can search/filter clips by transcript content
- [x] Cancellation works correctly
- [x] Works on CPU without GPU
- [x] Handles clips with no speech gracefully

## Future Enhancements (Out of Scope)

- Subtitle/SRT export
- Word-level timestamp display
- Transcript editing
- Speaker diarization
- Sentence similarity search (Phase 5)

## Dependencies

```
faster-whisper>=1.0.0
```

Note: faster-whisper will pull in:
- ctranslate2
- tokenizers
- huggingface-hub (for model downloads)

First model download will be ~244MB for small.en.

## References

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Model Card](https://huggingface.co/openai/whisper-small.en)
- [Course: Dialogue Extraction notebook](docs/research/ITP%20Algorithmic%20Filmmaking.md)
- [Existing pattern: ColorAnalysisWorker](ui/main_window.py:176)
- [Existing pattern: ShotTypeWorker](ui/main_window.py:213)

---

*Generated: 2026-01-24*
