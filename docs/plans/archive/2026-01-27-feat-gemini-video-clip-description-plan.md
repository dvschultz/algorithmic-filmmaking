---
title: "feat: Send video clips to Gemini for description"
type: feat
date: 2026-01-27
---

# Send Video Clips to Gemini for Description

## Overview

Enhance the Describe feature to send actual video clips (not just single frames) to Gemini models for richer temporal understanding. When Gemini is the configured cloud model, extract and upload the clip segment; fall back to current frame-based behavior for other models.

## Problem Statement / Motivation

Currently, the Describe feature extracts a single frame at 1/3 into each clip and sends it to VLMs. This loses temporal information:
- Motion and action context
- Audio cues (dialogue, music, sound effects)
- Scene transitions within a clip
- Pacing and rhythm

Gemini 2.5 models support video input natively with strong temporal understanding. By sending the actual clip, we get significantly richer descriptions that capture what's *happening*, not just what a single frame *looks like*.

## Proposed Solution

### High-Level Approach

1. **Detect Gemini model**: Check if configured cloud model is a Gemini variant
2. **Extract clip segment**: Use FFmpeg to extract the clip's frame range to a temp video file
3. **Upload and analyze**: Send video to Gemini via appropriate method (inline or Files API)
4. **Fall back gracefully**: Use frame-based for non-Gemini models or on failure

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Gemini detection | Check if model name contains "gemini" (case-insensitive) | Simple, covers all variants |
| Video extraction | FFmpeg re-encode to MP4 | Frame-accurate boundaries, universal format |
| Upload method | Inline base64 for <20MB, Files API for larger | Balance simplicity and capability |
| Tool API design | Automatic detection, no new parameters | Agent doesn't need to know implementation details |
| Failure handling | Fall back to frame-based, log warning | Resilience over consistency |
| User override | Add setting "Use video for Gemini descriptions" (default: true) | Cost control for users |

## Technical Approach

### Architecture

```
User clicks "Describe" (Gemini configured)
    │
    ▼
DescriptionWorker starts
    │
    ├──► is_video_capable_model(model) ──► False ──► Current frame-based path
    │
    ▼ True
    │
extract_clip_segment(source_path, start_frame, end_frame, fps)
    │
    ▼
temp_video_path (e.g., /tmp/clip_abc123.mp4)
    │
    ▼
describe_video_gemini(video_path, prompt, model)
    │
    ├──► file_size < 20MB ──► Inline base64 upload
    │
    ├──► file_size >= 20MB ──► Gemini Files API upload
    │
    ▼
LLM response with temporal description
    │
    ▼
Cleanup temp file
```

### File Changes

| File | Changes |
|------|---------|
| `core/analysis/description.py` | Add `is_video_capable_model()`, `describe_video_gemini()`, `extract_clip_segment()` |
| `core/settings.py` | Add `use_video_for_gemini: bool = True` setting |
| `ui/settings_dialog.py` | Add checkbox for video description setting |
| `ui/main_window.py` | Update DescriptionWorker to pass source reference |
| `models/clip.py` | Update `description_frames` field semantics |

### Implementation Details

#### 1. Gemini Detection Function

```python
# core/analysis/description.py

def is_video_capable_model(model: str) -> bool:
    """Check if model supports video input."""
    model_lower = model.lower()
    # Gemini models support video natively
    if "gemini" in model_lower:
        return True
    # Future: Add other video-capable models here
    return False
```

#### 2. Clip Extraction Function

```python
# core/analysis/description.py

def extract_clip_segment(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_dir: Optional[Path] = None,
) -> Path:
    """Extract clip segment from source video using FFmpeg.

    Args:
        source_path: Path to source video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Video frame rate
        output_dir: Directory for temp file (default: system temp)

    Returns:
        Path to extracted video segment (MP4 format)
    """
    import tempfile
    import subprocess

    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())

    # Generate unique filename
    output_path = output_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4"

    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-ss", str(start_time),
        "-i", str(source_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extraction failed: {result.stderr}")

    return output_path
```

#### 3. Video Description Function

```python
# core/analysis/description.py

def describe_video_gemini(
    video_path: Path,
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> tuple[str, str]:
    """Send video to Gemini for description.

    Args:
        video_path: Path to video file
        prompt: Description prompt
        model: Gemini model name

    Returns:
        Tuple of (description, model_name)
    """
    import google.generativeai as genai

    # Get API key
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not configured")

    genai.configure(api_key=api_key)

    file_size = video_path.stat().st_size

    if file_size < 20 * 1024 * 1024:  # < 20MB
        # Inline upload
        video_file = genai.upload_file(video_path)
    else:
        # Files API for larger files
        video_file = genai.upload_file(video_path)
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")

    # Normalize model name (remove gemini/ prefix if present for SDK)
    model_name = model.replace("gemini/", "")

    gen_model = genai.GenerativeModel(model_name)
    response = gen_model.generate_content([video_file, prompt])

    # Cleanup uploaded file
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass  # Best effort cleanup

    return response.text, f"{model} (video)"
```

#### 4. Updated describe_frame Function

```python
# core/analysis/description.py

def describe_frame(
    image_path: Path,
    tier: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    # New parameters for video support
    source_path: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> tuple[str, str]:
    """Describe a video frame or clip.

    If source video info is provided and model supports video,
    extracts and sends the clip. Otherwise uses single frame.
    """
    settings = load_settings()
    tier = tier or settings.description_model_tier

    if tier == "cloud":
        model = settings.description_model_cloud

        # Check for video-capable model with video parameters
        if (
            is_video_capable_model(model)
            and settings.use_video_for_gemini
            and source_path is not None
            and start_frame is not None
            and end_frame is not None
            and fps is not None
        ):
            try:
                # Extract and describe video
                temp_video = extract_clip_segment(
                    source_path, start_frame, end_frame, fps
                )
                try:
                    return describe_video_gemini(temp_video, prompt, model)
                finally:
                    # Cleanup temp file
                    if temp_video.exists():
                        temp_video.unlink()
            except Exception as e:
                logger.warning(f"Video description failed, falling back to frame: {e}")
                # Fall through to frame-based

        return describe_frame_cloud(image_path, prompt)

    # ... existing cpu/gpu handling
```

#### 5. DescriptionWorker Updates

```python
# ui/main_window.py - DescriptionWorker

class DescriptionWorker(QThread):
    def __init__(
        self,
        clips: list[Clip],
        sources: dict[str, Source],  # NEW: source lookup
        tier: str,
        prompt: Optional[str] = None,
    ):
        super().__init__()
        self.clips = clips
        self.sources = sources  # clip.source_id -> Source
        self.tier = tier
        self.prompt = prompt or DEFAULT_PROMPT

    def run(self):
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                break

            self.progress.emit(i + 1, len(self.clips))

            # Get source for video extraction
            source = self.sources.get(clip.source_id)

            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    description, model = describe_frame(
                        clip.thumbnail_path,
                        tier=self.tier,
                        prompt=self.prompt,
                        # Pass video info for potential video extraction
                        source_path=source.file_path if source else None,
                        start_frame=clip.start_frame,
                        end_frame=clip.end_frame,
                        fps=source.fps if source else None,
                    )

                    if description and not description.startswith("Error"):
                        self.description_ready.emit(clip.id, description, model)
                        self.success_count += 1
                    else:
                        self.error.emit(clip.id, description)
                        self.error_count += 1
            except Exception as e:
                self.error.emit(clip.id, str(e))
                self.error_count += 1
```

#### 6. Settings Addition

```python
# core/settings.py

@dataclass
class Settings:
    # ... existing fields ...

    # Video description settings
    use_video_for_gemini: bool = True  # Send video clips to Gemini instead of frames
```

## Acceptance Criteria

### Functional Requirements

- [x] When Gemini is configured as cloud model and `use_video_for_gemini` is true, clips are sent as video
- [x] When non-Gemini model is configured, single frame behavior is preserved
- [x] When video extraction/upload fails, gracefully falls back to frame-based
- [x] Settings dialog has checkbox to toggle video description behavior
- [x] `description_model` field indicates "(video)" suffix when video was used
- [x] Agent tool `describe_content_live` automatically uses video when appropriate (uses updated DescriptionWorker)

### Non-Functional Requirements

- [x] Video extraction completes within 30 seconds for clips under 2 minutes
- [x] Memory usage stays reasonable (no full video in memory)
- [x] Temp files are always cleaned up, even on errors
- [ ] Progress feedback during extraction and upload phases

### Quality Gates

- [x] Unit tests for `is_video_capable_model()` with various model name formats
- [ ] Integration test for extraction + upload flow (mocked Gemini API)
- [ ] Manual test with real Gemini API key and sample clips

## Dependencies & Risks

### Dependencies

| Dependency | Required For | Status |
|------------|--------------|--------|
| `litellm` | Gemini video upload (via file type) | Already installed |
| FFmpeg | Clip extraction | Already required |
| Gemini API key | Video upload | User-configured |

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gemini API changes | Low | Medium | Pin SDK version, monitor changelogs |
| High costs surprise users | Medium | Medium | Add warning for long clips, document costs |
| Large files fail upload | Low | Low | Fallback to frame-based |
| Temp disk space issues | Low | Low | Use system temp, cleanup immediately |

## Future Considerations

1. **Duration-based switching**: Auto-fall back to multi-frame sampling for very long clips (>5 min)
2. **Cost estimation**: Show estimated cost before batch operations
3. **YouTube URL support**: For clips from YouTube sources, use URL+timestamp instead of re-upload
4. **Caching**: Cache extracted clips for re-description with different prompts
5. **Audio-only mode**: Option to include audio context without video for cheaper analysis

## Implementation MVP

### describe_frame signature change

```python
# core/analysis/description.py

def describe_frame(
    image_path: Path,
    tier: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    source_path: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> tuple[str, str]:
    ...
```

### is_video_capable_model helper

```python
# core/analysis/description.py

def is_video_capable_model(model: str) -> bool:
    """Check if model supports video input."""
    return "gemini" in model.lower()
```

### extract_clip_segment function

```python
# core/analysis/description.py

def extract_clip_segment(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> Path:
    """Extract clip segment to temp MP4 file."""
    ...
```

### describe_video_gemini function

```python
# core/analysis/description.py

def describe_video_gemini(
    video_path: Path,
    prompt: str,
    model: str,
) -> tuple[str, str]:
    """Send video to Gemini for description using native SDK."""
    ...
```

### Settings field

```python
# core/settings.py

use_video_for_gemini: bool = True
```

## References

### Internal References

- Frame extraction: `core/thumbnail.py:generate_clip_thumbnail()`
- Description flow: `core/analysis/description.py:describe_frame()`
- Worker pattern: `ui/main_window.py:DescriptionWorker`
- Gemini routing: `core/analysis/description.py:describe_frame_cloud()` (line ~150)
- Settings: `core/settings.py:Settings`

### External References

- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/vision?lang=python#video)
- [Gemini Files API](https://ai.google.dev/gemini-api/docs/files)
- [google-generativeai SDK](https://github.com/google-gemini/generative-ai-python)

### Institutional Learnings Applied

- Gemini API routing: Always use `gemini/` prefix for LiteLLM, but strip for native SDK
- Worker patterns: Use guard flags, cleanup in finally blocks
- Agent tools: Return parameters, let signal handler execute
