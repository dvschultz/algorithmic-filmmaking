---
title: Film Language Integration Documentation
type: feat
date: 2026-02-01
status: draft
tags: [documentation, film-language, analysis, cinematography, education]
deepened: 2026-02-01
---

# Film Language Integration Documentation

## Enhancement Summary

**Deepened on:** 2026-02-01
**Sections enhanced:** 10
**Research agents used:** 13 (kieran-python-reviewer, architecture-strategist, performance-oracle, data-integrity-guardian, agent-native-reviewer, code-simplicity-reviewer, best-practices-researcher ×2, general-purpose ×3, learnings-researcher, framework-docs-researcher)

### Key Improvements

1. **Phased Approach Confirmed:** Start with Phase 1 (tooltips/glossary) and Phase 2 (expanded cinematography), then Phase 3-5
2. **Agent-Native is Critical:** Every feature MUST have a corresponding agent tool—filmmakers and agents should have identical capabilities
3. **Audio Reframed:** Phase 3 shifts from "audio classification" to "audio-guided sequencing" (beat detection, rhythm-based editing)
4. **Sequence Analysis Confirmed:** Phase 4 is valuable for continuity checking and pacing analysis
5. **Scale: 1000s of Clips:** Performance architecture must handle very large projects (1000+ clips)—use embeddings, pagination, background processing

### User Requirements Clarified

| Aspect | Decision |
|--------|----------|
| Glossary detail | Brief (1-2 sentences) |
| Glossary search | Yes, filterable by category |
| Audio sources | Music files (MP3, WAV) imported separately |
| Beat alignment | Both suggest and auto-arrange, user choice |
| Multiple audio tracks | Not for now |
| Continuity warnings | Advisory (soft warnings) |
| Pacing feedback | All: genre comparison, visualization, suggestions |
| Project scale | 1000s of clips |
| Primary use case | All equally (music videos, documentary, narrative) |

### New Considerations Discovered

- **VLM Prompt Engineering:** Add "unknown" as valid enum value for all fields; remove confidence scores (unreliable); use provider-specific prompts for Gemini vs OpenAI
- **Thread Safety:** Use frozen dataclasses for ThreadPoolExecutor tasks; apply CancellableFFmpegProcess wrapper pattern
- **Backward Compatibility:** Existing `from_dict()` pattern with `.get()` is correct; maintain for all new fields
- **Relevant Learnings Applied:** QThread signal guard flags, subprocess cleanup on exception, FFmpeg path escaping, source ID mismatch fixes

---

## Overview

This plan documents how Scene Ripper can integrate film language concepts from traditional cinematography education into its automated video analysis pipeline. The goal is to create a comprehensive reference that maps film theory to practical implementation, enabling users to analyze and understand footage using the vocabulary and frameworks of professional filmmakers.

**Source Material:** [LearnAboutFilm.com Film Language Guide](https://learnaboutfilm.com/film-language/)

## Problem Statement / Motivation

Scene Ripper provides powerful automated video analysis, but users may not understand how the detected metadata relates to professional film language. By documenting the mapping between film theory and our analysis capabilities, we:

1. **Educate users** on cinematography concepts while they work
2. **Validate our taxonomy** against established film education standards
3. **Identify gaps** where additional analysis could add value
4. **Enable better filtering/sorting** based on film language concepts
5. **Support creative workflows** by surfacing the "why" behind shot choices

## Current State Analysis

### Already Implemented in Scene Ripper

| Film Concept | Implementation | Location |
|--------------|----------------|----------|
| Shot Size (10 types) | `CinematographyAnalysis.shot_size` | `models/cinematography.py:147` |
| Camera Angle (6 types) | `CinematographyAnalysis.camera_angle` | `models/cinematography.py:148` |
| Camera Movement (8 types) | `CinematographyAnalysis.camera_movement` | `models/cinematography.py:150` |
| Subject Position (rule of thirds) | `CinematographyAnalysis.subject_position` | `models/cinematography.py:154` |
| Headroom | `CinematographyAnalysis.headroom` | `models/cinematography.py:155` |
| Lead/Looking Room | `CinematographyAnalysis.lead_room` | `models/cinematography.py:156` |
| Composition Balance | `CinematographyAnalysis.balance` | `models/cinematography.py:157` |
| Focus Type | `CinematographyAnalysis.focus_type` | `models/cinematography.py:160` |
| Lighting Style | `CinematographyAnalysis.lighting_style` | `models/cinematography.py:162` |
| Lighting Direction | `CinematographyAnalysis.lighting_direction` | `models/cinematography.py:163` |
| Color Palette | `Clip.dominant_colors` | `models/clip.py:168` |
| Speech Content | `Clip.transcript` | `models/clip.py:170` |
| Object Detection | `Clip.detected_objects` | `models/clip.py:175` |
| Scene Description | `Clip.description` | `models/clip.py:178` |

### Not Yet Implemented

| Film Concept | Complexity | Notes |
|--------------|------------|-------|
| Sound Design Analysis | High | Music mood, diegetic classification |
| Editing Rhythm/Pacing | Medium | Sequence-level analysis |
| Continuity Validation | High | Multi-clip relationship analysis |
| Lens Characteristics | Medium | Wide-angle distortion, depth estimation |
| Temporal Analysis | High | Cross-cutting, parallel editing detection |

---

## Film Language Taxonomy

### 1. Shot Sizes

**Film Education Standard** (from LearnAboutFilm):

| Shot Type | Description | Emotional Effect |
|-----------|-------------|------------------|
| Extreme Long Shot (ELS) | Setting dominates, people tiny | Isolation, insignificance, scope |
| Long Shot (LS) | Head to toe | Context, action, group dynamics |
| Medium Long Shot (MLS) | Three-quarter body | Closer engagement, some context |
| Medium Shot (MS) | Hips to head | Conversation, casual observation |
| Medium Close-Up (MCU) | Head and shoulders | Presentation, direct address |
| Close-Up (CU) | Head and bit of shoulders | Emotion, reaction, intimacy |
| Big Close-Up (BCU) | Face fills frame | Intense emotion, threat |
| Extreme Close-Up (ECU) | Part of face (eyes, mouth) | Maximum intensity |
| Insert | Important detail | Clarification, emphasis |

**Scene Ripper Implementation:**

```python
# models/cinematography.py - SHOT_SIZE_CHOICES
SHOT_SIZE_CHOICES = [
    "ELS",    # Extreme Long Shot
    "VLS",    # Very Long Shot (variant)
    "LS",     # Long Shot
    "MLS",    # Medium Long Shot
    "MS",     # Medium Shot
    "MCU",    # Medium Close-Up
    "CU",     # Close-Up
    "BCU",    # Big Close-Up
    "ECU",    # Extreme Close-Up
    "Insert"  # Insert shot
]
```

**Mapping:** Direct 1:1 mapping. Scene Ripper's taxonomy matches film education standards exactly.

**Usage Principle:** Progressive tightening intensifies emotional scenes. VLM analysis detects shot size from visual composition.

#### Research Insights

**Best Practices:**
- Use `Literal` type hints for enum-like string fields for IDE autocomplete and type checking
- Consider adding `"unknown"` as valid value for VLM classification failures (prevents hallucination)
- Store display names separately from enum values for localization support

**VLM Prompt Optimization:**
```python
# Provider-specific prompts improve accuracy
SHOT_SIZE_PROMPT_GEMINI = """..."""  # Gemini handles detailed JSON schemas well
SHOT_SIZE_PROMPT_OPENAI = """..."""  # GPT-4o prefers simpler enum lists
```

**References:**
- [LearnAboutFilm Shot Sizes](https://learnaboutfilm.com/film-language/picture/shotsize/)

---

### 2. Composition & Framing

**Film Education Standard:**

| Technique | Description | Effect |
|-----------|-------------|--------|
| Rule of Thirds | Subject 1/3 from edge | Natural, pleasing |
| Centered | Subject dead center | Formal, confrontational, unusual |
| Headroom | Space above head | Balance; too much/little feels wrong |
| Looking Space (Nose Room) | Space in gaze direction | Prevents "boxed in" feeling |
| Dutch Angle | Tilted horizon | Unease, disorientation |
| Unbalanced | Deliberate asymmetry | Tension, discomfort |

**Scene Ripper Implementation:**

```python
# models/cinematography.py - Composition fields
subject_position: str   # "left_third", "center", "right_third", "distributed"
headroom: str           # "tight", "normal", "excessive", "n/a"
lead_room: str          # "tight", "normal", "excessive", "n/a"
balance: str            # "balanced", "left_heavy", "right_heavy", "symmetrical"
```

**Gap Identified:** Dutch angle not explicitly detected. The `camera_angle` field has `dutch_angle` as an option, but this is angle relative to subject, not horizon tilt.

**Recommendation:**
```python
# Proposed addition to CinematographyAnalysis
dutch_tilt: str  # "none", "slight", "moderate", "extreme"
```

#### Research Insights

**Implementation Pattern:**
```python
from typing import Literal

DutchTiltType = Literal["none", "slight", "moderate", "extreme", "unknown"]

@dataclass(frozen=True)
class CompositionAnalysis:
    subject_position: str
    headroom: str
    lead_room: str
    balance: str
    dutch_tilt: DutchTiltType = "unknown"
```

**Edge Cases:**
- Handheld footage may have incidental tilt that isn't artistic choice—flag as `"unknown"` when confidence is low
- Rolling shutter artifacts can look like dutch tilt—VLM should be trained to distinguish

**References:**
- [LearnAboutFilm Composition](https://learnaboutfilm.com/film-language/picture/frame-it-right/)

---

### 3. Camera Position & Angle

**Film Education Standard:**

| Angle | Description | Psychological Effect |
|-------|-------------|---------------------|
| Low Angle | Camera below, pointing up | Power, heroism, threat |
| Eye Level | Camera at subject height | Neutral, documentary |
| High Angle | Camera above, pointing down | Weakness, vulnerability |
| Bird's Eye | Directly overhead | Omniscience, spatial overview |
| Worm's Eye | Ground level looking up | Extreme power, awe |

**Camera Position:**

| Position | Description | Effect |
|----------|-------------|--------|
| Frontal | Head-on | Engagement, confrontation |
| Three-Quarter | 45° angle | Standard coverage, balanced |
| Side (Profile) | 90° | Observer, detachment |
| Back | Behind subject | Vulnerability, hidden emotion |

**Scene Ripper Implementation:**

```python
# models/cinematography.py - CAMERA_ANGLE_CHOICES
CAMERA_ANGLE_CHOICES = [
    "low_angle",   # Power, heroism
    "eye_level",   # Neutral
    "high_angle",  # Vulnerability
    "dutch_angle", # Disorientation
    "birds_eye",   # Overhead
    "worms_eye"    # Extreme low
]

# Derived emotional effect
ANGLE_EFFECT_CHOICES = [
    "power",           # Low angle effect
    "neutral",         # Eye level
    "vulnerability",   # High angle
    "disorientation",  # Dutch angle
    "omniscience",     # Bird's eye
    "extreme_power"    # Worm's eye
]
```

**Gap Identified:** Camera position relative to subject (frontal/three-quarter/side/back) not currently detected.

**Recommendation:**
```python
# Proposed addition to CinematographyAnalysis
camera_position: str  # "frontal", "three_quarter", "profile", "back"
```

#### Research Insights

**VLM Prompt Strategy:**
- Camera position detection is less reliable than angle detection
- Include visual cues in prompt: "Look for subject's face visibility, ear visibility, nose direction"
- Add `"unknown"` option for shots without clear subject orientation

**Architecture Note:**
Separating `camera_angle` (vertical relationship) from `camera_position` (horizontal relationship) aligns with professional cinematography terminology and avoids conflating distinct concepts.

---

### 4. Camera Movement

**Film Education Standard:**

| Movement | Description | Effect |
|----------|-------------|--------|
| Static | No movement | Stability, observation |
| Pan | Camera turns left/right | Scan, follow, reveal |
| Tilt | Camera moves up/down | Reveal height, follow vertical action |
| Track/Dolly | Camera moves through space | Intensity, involvement, follow |
| Crane | Vertical movement through space | Grandeur, reveal, establish |
| Arc | Circle around subject | 360° view, dramatic emphasis |
| Handheld | Unstable, human feel | Urgency, documentary, chaos |
| Steadicam | Smooth handheld | Follow action smoothly |

**Movement Direction Semantics:**

| Direction | Meaning |
|-----------|---------|
| Track In | Build intensity, approach threat |
| Track Out | Reveal context, retreat |
| X-axis (left/right) | Natural journey movement |
| Y-axis (up/down) | Struggle, aspiration |
| Z-axis (toward/away) | Involvement, threat |

**Scene Ripper Implementation:**

```python
# models/cinematography.py - Movement fields
camera_movement: str    # "static", "pan", "tilt", "track", "handheld", "crane", "arc", "n/a"
movement_direction: str # "left", "right", "up", "down", "forward", "backward", "clockwise", "counterclockwise"
```

**Mapping:** Good coverage. Steadicam grouped with handheld; could be separated.

**Note:** Movement detection requires video mode analysis (not frame mode). Currently only Gemini supports video input.

#### Research Insights

**Video Analysis Performance:**
- Gemini video analysis is expensive—consider sampling frames for movement detection rather than full video upload
- Movement direction can often be inferred from 2-3 frame comparison (optical flow)

**Implementation Pattern (from existing worker):**
```python
# From ui/workers/cinematography_worker.py - use frozen dataclass for thread safety
@dataclass(frozen=True)
class ClipAnalysisTask:
    clip_id: str
    video_path: Path
    analysis_type: str
```

**References:**
- [LearnAboutFilm Camera Movement](https://learnaboutfilm.com/film-language/picture/movement/)

---

### 5. Lens Usage

**Film Education Standard:**

| Lens Type | Characteristics | Use Case |
|-----------|-----------------|----------|
| Wide-Angle | Broad view, exaggerated perspective | Interiors, dramatic close-ups |
| Telephoto | Compressed perspective, shallow DoF | Flattering portraits, distance shots |
| Normal | Natural perspective | General use |

**Focus Techniques:**

| Technique | Description | Effect |
|-----------|-------------|--------|
| Deep Focus | Everything sharp | Equal attention across scene |
| Shallow Focus | Subject sharp, background blur | Isolation, emphasis |
| Rack/Pull Focus | Focus shifts mid-shot | Redirect attention |

**Scene Ripper Implementation:**

```python
# models/cinematography.py - Focus fields
focus_type: str      # "deep", "shallow", "rack_focus"
background_type: str # "blurred", "sharp", "cluttered", "plain"
```

**Gap Identified:** Lens type (wide/normal/telephoto) not detected. Could be estimated from:
- Perspective distortion analysis
- Depth of field characteristics
- Field of view estimation

**Recommendation:**
```python
# Proposed addition to CinematographyAnalysis
estimated_lens_type: str  # "wide", "normal", "telephoto", "unknown"
perspective_distortion: str  # "strong", "moderate", "none"
```

#### Research Insights

**Detection Approach:**
- VLMs can reliably detect perspective distortion (wide-angle barrel distortion, telephoto compression)
- Depth of field is more reliable indicator than perspective for lens type
- Consider combining multiple signals for higher confidence

**Edge Cases:**
- Anamorphic lenses have unique distortion—may need separate classification
- Vintage lenses with character may not fit standard categories

---

### 6. Light and Color

**Film Education Standard:**

**Light Size:**

| Type | Description | Effect |
|------|-------------|--------|
| Hard Light | Direct source, sharp shadows | Drama, harsh reality |
| Soft Light | Diffused, even illumination | Flattering, neutral |
| Medium | Window, lantern | Balance of softness and shape |

**Light Position:**

| Position | Effect |
|----------|--------|
| Front | Flat, characterless |
| Three-Quarter | Depth, modeling |
| Side | Atmospheric, dramatic |
| Below | Unsettling, horror |
| Back/Rim | Separation, silhouette |

**Three-Point Lighting:**
1. **Key Light**: Main source, provides shape
2. **Fill Light**: Reduces shadows (typically half key brightness)
3. **Rim/Back Light**: Edge separation

**High-Key vs Low-Key:**
- High-Key: Bright, minimal shadows (comedy, happiness)
- Low-Key: Dramatic shadows (thriller, mystery)

**Scene Ripper Implementation:**

```python
# models/cinematography.py - Lighting fields
lighting_style: str     # "high_key", "low_key", "natural", "dramatic"
lighting_direction: str # "front", "three_quarter", "side", "back", "below"
```

**Color Analysis:**

```python
# core/analysis/color.py
def extract_dominant_colors(image_path) -> list[tuple[int, int, int]]
def classify_color_palette(colors) -> str  # "warm", "cool", "neutral", "vibrant"
```

**Gap Identified:**
- Light hardness (hard/soft) not detected
- Color temperature (warm/cool K value) not detected
- Three-point lighting identification not attempted

**Recommendation:**
```python
# Proposed additions to CinematographyAnalysis
light_quality: str       # "hard", "soft", "mixed"
color_temperature: str   # "warm", "neutral", "cool"
contrast_ratio: str      # "high", "medium", "low"
```

#### Research Insights

**Color Temperature Detection:**
- Can be computed algorithmically from pixel values (correlated color temperature)
- VLM detection more reliable for artistic intent (warm vs cool "feeling")
- Consider storing both: `color_temp_kelvin: Optional[int]` and `color_mood: str`

**Light Quality Indicators:**
- Shadow edge sharpness is primary indicator (sharp = hard, gradual = soft)
- Highlight falloff patterns also indicate light quality
- Multiple light sources complicate classification—use `"mixed"`

**References:**
- [LearnAboutFilm Lighting](https://learnaboutfilm.com/film-language/picture/light-and-colour/)

---

### 7. Sound

**Film Education Standard:**

**Sound Categories:**

| Type | Description | Example |
|------|-------------|---------|
| Diegetic | Exists in scene world | Dialogue, footsteps, wind |
| Non-Diegetic | Added externally | Score, narrator voiceover |
| Synchronous | Matches visible action | Lips moving with dialogue |
| Asynchronous | Off-screen source | Doorbell before door shown |

**Music Functions:**

| Function | Description |
|----------|-------------|
| Setting | Establish time/place (baroque = 18th century) |
| Emotion | Control mood (tension, joy, sadness) |
| Leitmotif | Character themes |
| Hits | Music synchronized to action |
| Contrapuntal | Music contradicts visuals (ironic) |
| Parallel | Music reinforces visuals |

**Sound Techniques:**

| Technique | Description |
|-----------|-------------|
| Split Edit | Sound and picture change at different times |
| Sound Bridge | Sound continues across scene transition |
| Silence | Powerful pause; absence creates impact |

**Scene Ripper Current State:**

- **Transcription**: Whisper-based speech-to-text (`Clip.transcript`)
- **No music analysis**: Cannot detect genre, mood, or presence
- **No sound design analysis**: Cannot classify diegetic vs non-diegetic
- **No audio level analysis**: Cannot detect silence, volume changes

**Gap Identified:** Sound analysis is the largest gap in film language coverage.

**Recommendation - Audio-Guided Sequencing (Practical Filmmaker Focus):**

Rather than classifying audio properties, focus on using audio to drive editing decisions:

```python
# Proposed AudioSyncData for sequencing
@dataclass
class AudioSyncData:
    """Audio analysis for sequencing, not classification."""
    tempo_bpm: Optional[float]           # Detected tempo
    beat_times: list[float]              # Timestamps of beats
    onset_times: list[float]             # Timestamps of transients/hits
    downbeat_times: list[float]          # Strong beats (measure starts)

    def nearest_beat(self, time: float) -> float:
        """Find nearest beat to given timestamp."""
        if not self.beat_times:
            return time
        return min(self.beat_times, key=lambda b: abs(b - time))
```

**Use Cases:**
- **Music video editing**: Align cuts to beats
- **Montage creation**: Rhythmic pacing from music track
- **Trailer editing**: Hit cuts on musical accents
- **Documentary**: Match pacing to score

#### Research Insights

**Recommended Audio Stack for Beat Detection:**

| Library | Purpose | Why |
|---------|---------|-----|
| `librosa` | Beat tracking, onset detection, tempo | Industry standard, well-documented |
| `madmom` | Advanced beat detection | More accurate for complex music, optional |

**Beat Detection Pattern:**
```python
import librosa

def detect_beats(audio_path: Path) -> dict:
    """Extract beat timestamps and tempo from audio."""
    y, sr = librosa.load(str(audio_path), sr=22050)

    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Get onset (transient) times for cut points
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return {
        "tempo_bpm": float(tempo),
        "beat_times": beat_times.tolist(),
        "onset_times": onset_times.tolist(),
    }
```

**Audio-Guided Sequencing Algorithm:**
```python
def align_clips_to_beats(
    clips: list[Clip],
    beat_times: list[float],
    strategy: str = "nearest"
) -> list[SequenceClip]:
    """
    Align clip cuts to beat timestamps.

    Strategies:
    - "nearest": Cut at nearest beat to natural clip boundary
    - "on_beat": Force cuts exactly on beats (may trim clips)
    - "downbeat": Prefer strong beats (every 4th beat in 4/4)
    """
```

**Agent Tools Required:**
```python
@tool
def detect_audio_beats(audio_path: str) -> dict:
    """
    Analyze music file for beats and tempo.

    Args:
        audio_path: Path to music file (MP3, WAV)

    Returns:
        {"success": True, "tempo_bpm": float, "beat_count": int, "duration_seconds": float}
    """

@tool
def suggest_beat_aligned_cuts(
    sequence_id: str,
    audio_path: str
) -> dict:
    """
    Suggest cut points aligned to beats (human reviews before applying).

    Returns:
        {"success": True, "suggestions": [{"clip_id": str, "current_cut": float, "suggested_cut": float}]}
    """

@tool
def auto_align_sequence_to_beats(
    sequence_id: str,
    audio_path: str,
    strategy: str = "nearest"  # "nearest", "on_beat", "downbeat"
) -> dict:
    """
    Automatically align sequence cuts to beats (fully automated).

    Returns:
        {"success": True, "adjusted_clips": int, "tempo_bpm": float}
    """
```

**User Choice: Suggest vs Auto-Align**
- `suggest_beat_aligned_cuts`: Shows recommendations, human decides
- `auto_align_sequence_to_beats`: Fully automated adjustment

**Audio Source: Music Files Only (for now)**
- Supports MP3, WAV, FLAC, AAC
- User imports music file separately from video sources
- Single audio track per sequence (no layering)

**Edge Cases:**
- Music with irregular tempo: Use onset detection instead of beat tracking
- No clear beats: Fall back to onset/transient detection
- Very long music files: Process in chunks, cache beat data

**Performance:**
- Beat detection is CPU-bound—run in background worker
- Cache beat analysis per audio file (beats don't change)
- For long audio (>10 min), process in chunks

**References:**
- [librosa beat tracking](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
- [madmom beat detection](https://madmom.readthedocs.io/en/latest/)

---

### 8. Continuity

**Film Education Standard:**

| Rule | Description | Breaking Effect |
|------|-------------|-----------------|
| 180° Rule | Camera stays on one side of action axis | Disorienting if crossed |
| 30° Rule | Camera moves 30°+ between shots | Avoids jump cuts |
| Match on Action | Cut during movement | Hides edit |
| Eyeline Match | Gaze direction consistent | Maintains spatial logic |
| Shot-Reverse Shot | Alternating angles in dialogue | Shows both participants |

**Scene Ripper Current State:**

Continuity analysis requires **sequence-level analysis** - examining relationships between clips, not individual clips.

**Gap Identified:** Scene Ripper analyzes clips individually; no cross-clip relationship analysis exists.

**Recommendation - Sequence Analysis Feature:**
```python
# Proposed SequenceAnalysis dataclass
@dataclass
class SequenceAnalysis:
    """Analysis of clip relationships within a sequence."""

    # Continuity checks
    axis_violations: list[tuple[str, str]]  # Pairs of clips that violate 180° rule
    jump_cut_risks: list[tuple[str, str]]   # Similar shots that may jump cut

    # Pacing analysis
    average_shot_duration: float
    shot_duration_variance: float
    pacing_classification: str  # "fast", "medium", "slow", "varied"

    # Visual consistency
    color_consistency: float    # 0-1 how similar color palettes are
    lighting_consistency: float # 0-1 how similar lighting is
```

#### Research Insights

**Architecture Decision: Lazy Computation**

SequenceAnalysis should **NOT** be persisted to the project file. Instead:
- Compute on-demand when user views Sequence tab or requests report
- Cache in memory with invalidation when sequence changes
- Reason: Sequence changes frequently; persisting creates sync issues

```python
class Sequence:
    _analysis_cache: Optional[SequenceAnalysis] = None

    @property
    def analysis(self) -> SequenceAnalysis:
        if self._analysis_cache is None:
            self._analysis_cache = self._compute_analysis()
        return self._analysis_cache

    def invalidate_analysis(self):
        """Call when clips are added/removed/reordered."""
        self._analysis_cache = None
```

**Performance Guard (O(N²) Warning):**
```python
def analyze_sequence(clips: list[Clip]) -> SequenceAnalysis:
    if len(clips) > 100:
        # Use embedding similarity + ANN instead of pairwise comparison
        return _analyze_large_sequence_with_embeddings(clips)
    return _analyze_sequence_pairwise(clips)
```

**Cross-Source Handling:**
- Normalize visual features per-source before comparison
- Flag cross-source transitions as intentional (user placed them)
- Compute within-source consistency separately

**YAGNI Consideration:**
> Sequence analysis is complex and may not be needed. Start with basic pacing stats (average duration, variance) before implementing continuity detection.

---

### 9. Editing

**Film Education Standard:**

**Cut Types:**

| Type | Description | Use |
|------|-------------|-----|
| Cut | Instant transition | Continuous action |
| Jump Cut | Similar shots cut together | Stylistic disorientation |
| Cross Dissolve | Images blend | Time passage, journey |
| Fade Out/In | To/from black | Major time passage |
| Wipe | One image pushes another | Rarely used (dated) |

**Advanced Techniques:**

| Technique | Description |
|-----------|-------------|
| Edit on Action | Cut during movement to hide edit |
| J-Cut | Audio precedes video |
| L-Cut | Video precedes audio |
| Cutaway | Insert shot to hide edit |
| Cross-Cutting | Alternate between locations |
| Montage | Rapid sequence showing time/change |

**Shot Duration Principle:**
> "A shot should stay on screen long enough for people to understand it, but not so long that they get bored."

- Close-ups: Can be brief (emotion reads quickly)
- Long shots: Need more time (complexity to absorb)

**Scene Ripper Current State:**

- No automatic edit point detection
- No transition type classification
- No pacing analysis
- Manual clip arrangement in Sequence tab

**Recommendation:**
```python
# Proposed EditingAnalysis for sequences
@dataclass
class EditingAnalysis:
    transitions: list[TransitionInfo]  # Type between each pair
    pacing_curve: list[float]          # Shot durations over time
    rhythm_classification: str          # "regular", "accelerating", "decelerating", "varied"
    suggested_improvements: list[str]   # "Consider longer shot at 0:45"
```

#### Research Insights

**Simplicity Review:**
> EditingAnalysis may be over-engineering. Consider deferring entirely until users explicitly request pacing analysis. The Sequence tab already shows clip durations visually—that may be sufficient.

**If Implementing:**
- Keep `pacing_curve` as simple list of durations
- Calculate `rhythm_classification` from variance and trend
- Avoid `suggested_improvements`—too opinionated and likely unhelpful

**Transition Detection:**
- Would require frame-by-frame analysis at cut points
- Complex to implement reliably
- Consider out of scope for initial implementation

**References:**
- [LearnAboutFilm Editing](https://learnaboutfilm.com/film-language/editing/)

---

### 10. Reading a Movie Scene (Analysis Framework)

**Film Education Framework:**

The LearnAboutFilm guide suggests analyzing scenes by examining:

1. **Visual Composition** - What's in/not in frame, shot size, angle, arrangement, color, lighting
2. **Movement** - Within frame and camera movement, direction semantics
3. **Editing** - Cut points, transitions, pacing
4. **Sound** - Music, effects, diegetic/non-diegetic
5. **Time** - Real time vs compressed vs stretched

**Scene Ripper Integration:**

This framework maps directly to a "Scene Analysis Report" feature that could synthesize all individual clip analyses into a cohesive narrative.

**Recommendation:**
```python
# Proposed SceneReport generator
def generate_scene_report(clips: list[Clip]) -> str:
    """
    Generate a human-readable film analysis report.

    Synthesizes:
    - Cinematography patterns (dominant shot sizes, angles)
    - Visual motifs (recurring colors, compositions)
    - Pacing analysis (shot duration distribution)
    - Audio characteristics (speech density, music presence)
    - Narrative structure suggestions
    """
```

#### Research Insights

**Report Generation Architecture:**
- Single LLM call with all clip metadata as context
- For large sequences (>50 clips), chunk into sections with summaries
- Use structured prompt with clear sections for each analysis type

**Agent-Native Requirement:**
> Reports must be accessible to agents. Add a tool: `generate_analysis_report(sequence_id: str) -> str`

**Template System:**
- Keep simple: Markdown output with configurable sections
- Avoid complex templating engines—LLM can handle formatting
- Export formats: Markdown (default), HTML (rendered), PDF (via WeasyPrint if needed)

---

## Implementation Roadmap

### Phase 1: Documentation & UI Enhancements (Low Effort) ✅ COMPLETED

**Goal:** Surface existing analysis in film language terms.

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| Tooltips | Add film education tooltips to analysis results | 2 days | ✅ Done |
| Glossary | In-app glossary of film terms | 1 day | ✅ Done |
| Badge Labels | Rename badges to match film terminology | 1 day | ✅ Done |
| Help System | Link badges to film education resources | 1 day | Deferred (no links per user preference) |
| Agent Tools | `get_film_term_definition`, `search_glossary` | 0.5 day | ✅ Done |

**Files created/modified:**
- `core/film_glossary.py` - New: Film terminology glossary data and helpers
- `core/chat_tools.py` - Add agent tools for glossary access
- `ui/dialogs/glossary_dialog.py` - New: Searchable glossary dialog
- `ui/clip_browser.py` - Add tooltips to cinematography badges
- `ui/tabs/analyze_tab.py` - Add glossary button
- `models/cinematography.py` - Enhanced `get_display_badges()` and added `get_display_badges_formatted()`

#### Research Insights

**Qt Tooltip Best Practices:**
```python
# Qt supports rich HTML in tooltips
widget.setToolTip("""
    <b>Close-Up (CU)</b><br/>
    <i>Head and bit of shoulders</i><br/><br/>
    <b>Effect:</b> Emotion, reaction, intimacy<br/>
    <a href='https://learnaboutfilm.com/film-language/picture/shotsize/'>Learn more</a>
""")
```

**Glossary Dialog Implementation:**
```python
class GlossaryDialog(QDialog):
    """Searchable glossary of film terms with category filtering."""

    # Categories for filtering
    CATEGORIES = [
        "All",
        "Shot Sizes",
        "Camera Angles",
        "Camera Movement",
        "Composition",
        "Lighting",
        "Sound",
        "Editing",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search terms...")
        self.search_input.textChanged.connect(self._filter_terms)

        # Category filter dropdown
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.CATEGORIES)
        self.category_combo.currentTextChanged.connect(self._filter_terms)

        self.term_list = QListWidget()
        self.definition_label = QLabel()  # Brief 1-2 sentence definition
        # ... layout setup

    def _filter_terms(self):
        """Filter by search query AND category."""
        query = self.search_input.text().lower()
        category = self.category_combo.currentText()

        for i in range(self.term_list.count()):
            item = self.term_list.item(i)
            term_data = item.data(Qt.UserRole)
            matches_query = query in item.text().lower()
            matches_category = category == "All" or term_data["category"] == category
            item.setHidden(not (matches_query and matches_category))
```

**Glossary Data Structure:**
```python
FILM_GLOSSARY = {
    "CU": {
        "name": "Close-Up",
        "category": "Shot Sizes",
        "definition": "Head and bit of shoulders. Shows emotion and reaction with intimacy.",
    },
    "dutch_angle": {
        "name": "Dutch Angle",
        "category": "Camera Angles",
        "definition": "Tilted horizon creating unease or disorientation.",
    },
    # ... etc
}
```

**Agent-Native Requirement:**
> Add tool for glossary access: `get_film_term_definition(term: str) -> str`

**Edge Cases:**
- Tooltip links may not be clickable on all platforms—provide "?" button as fallback
- Long definitions should truncate in tooltip, expand in glossary dialog

### Phase 2: Expanded Cinematography Analysis (Medium Effort)

**Goal:** Add missing visual analysis concepts.

| Feature | Description | Effort |
|---------|-------------|--------|
| Dutch Tilt Detection | Add horizon tilt analysis | 2 days |
| Camera Position | Frontal/profile/back detection | 2 days |
| Lens Type Estimation | Wide/normal/telephoto classification | 3 days |
| Light Quality | Hard/soft classification | 2 days |
| Color Temperature | Warm/cool estimation | 1 day |

**Files to create/modify:**
- `models/cinematography.py` - Add new fields
- `core/analysis/cinematography.py` - Update VLM prompts

#### Research Insights

**Data Model Pattern (from kieran-python-reviewer):**
```python
from dataclasses import dataclass, field
from typing import Literal, Optional

# Use Literal for enum-like fields
DutchTiltType = Literal["none", "slight", "moderate", "extreme", "unknown"]
CameraPositionType = Literal["frontal", "three_quarter", "profile", "back", "unknown"]
LensType = Literal["wide", "normal", "telephoto", "unknown"]
LightQualityType = Literal["hard", "soft", "mixed", "unknown"]
ColorTempType = Literal["warm", "neutral", "cool", "unknown"]

@dataclass
class CinematographyAnalysis:
    # ... existing fields ...

    # New fields with defaults for backward compatibility
    dutch_tilt: DutchTiltType = "unknown"
    camera_position: CameraPositionType = "unknown"
    estimated_lens_type: LensType = "unknown"
    light_quality: LightQualityType = "unknown"
    color_temperature: ColorTempType = "unknown"

    @classmethod
    def from_dict(cls, data: dict) -> "CinematographyAnalysis":
        # Use .get() for all new optional fields
        return cls(
            # ... existing fields ...
            dutch_tilt=data.get("dutch_tilt", "unknown"),
            camera_position=data.get("camera_position", "unknown"),
            # ... etc
        )
```

**VLM Prompt Updates:**
```python
# Add to CINEMATOGRAPHY_JSON_SCHEMA
{
    "dutch_tilt": {
        "type": "string",
        "enum": ["none", "slight", "moderate", "extreme", "unknown"],
        "description": "Horizon tilt angle. 'none'=level, 'slight'=5-15°, 'moderate'=15-30°, 'extreme'=30°+"
    },
    "camera_position": {
        "type": "string",
        "enum": ["frontal", "three_quarter", "profile", "back", "unknown"],
        "description": "Camera position relative to subject. 'frontal'=face visible, 'profile'=90° side"
    }
}
```

**Important: Always include "unknown"** as valid enum value—VLMs will hallucinate if forced to choose when uncertain.

**Remove Confidence Scores:**
> Research shows VLM confidence scores are unreliable and inconsistent. Remove `confidence` fields from schema; use binary present/absent instead.

### Phase 3: Audio-Guided Sequencing (Medium Effort)

**Goal:** Use audio to guide clip sequencing and editing decisions.

| Feature | Description | Effort |
|---------|-------------|--------|
| Beat Detection | Extract beat timestamps from music tracks | 2 days |
| Tempo Analysis | Calculate BPM for pacing alignment | 1 day |
| Audio Sync Points | Detect transients/hits for cut points | 2 days |
| Rhythm-Based Sequencing | Align clip cuts to beats | 3 days |
| Audio Waveform Display | Visual waveform in timeline | 2 days |

**Files to create:**
- `core/analysis/audio.py` - Beat/tempo detection
- `core/remix/audio_sync.py` - Audio-driven sequencing algorithms
- `ui/widgets/waveform_view.py` - Audio waveform visualization
- `ui/workers/audio_worker.py` - Background processing

**Dependencies:**
- `librosa` for beat tracking and onset detection
- `madmom` (optional) for more accurate beat detection

#### Research Insights

**Architecture Decision: Nested Dataclass**

Use nested dataclass pattern (matches existing `CinematographyAnalysis`):

```python
@dataclass
class AudioAnalysis:
    """Audio analysis results for a clip."""
    audio_present: bool = False
    has_speech: bool = False
    has_music: bool = False
    silence_percentage: float = 0.0
    loudness_lufs: Optional[float] = None
    peak_dbfs: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AudioAnalysis":
        if data is None:
            return cls()
        return cls(
            audio_present=data.get("audio_present", False),
            has_speech=data.get("has_speech", False),
            # ... etc
        )

    def to_dict(self) -> dict:
        return {
            "audio_present": self.audio_present,
            "has_speech": self.has_speech,
            # ... etc
        }

# In Clip model:
@dataclass
class Clip:
    # ... existing fields ...
    audio_analysis: Optional[AudioAnalysis] = None
```

**Audio Extraction with CancellableFFmpegProcess:**

Apply existing subprocess patterns from learnings:

```python
class CancellableFFmpegProcess:
    """Wrapper for FFmpeg subprocess with clean cancellation."""

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.process: Optional[subprocess.Popen] = None
        self._cancelled = threading.Event()

    def run(self) -> bool:
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            while self.process.poll() is None:
                if self._cancelled.is_set():
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    return False
                time.sleep(0.1)
            return self.process.returncode == 0
        except Exception:
            if self.process:
                self.process.terminate()
            raise

    def cancel(self):
        self._cancelled.set()
```

**Check for Audio Track First:**
```python
def has_audio_track(video_path: Path) -> bool:
    """Use ffprobe to check if video has audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())
```

**YAGNI Checkpoint:**
> Consider implementing only `audio_present`, `has_speech`, and `loudness_lufs` in Phase 3. Defer music detection and mood classification to Phase 3.5 after user feedback.

### Phase 4: Sequence-Level Analysis (High Effort)

**Goal:** Analyze relationships between clips in sequences.

| Feature | Description | Effort |
|---------|-------------|--------|
| Pacing Analysis | Shot duration statistics | 2 days |
| Visual Consistency | Color/lighting variance | 3 days |
| Continuity Warnings | 180°/30° rule checks | 5 days |
| Editing Rhythm | Pattern detection | 3 days |

**Files to create:**
- `core/analysis/sequence.py` - Sequence analysis
- `models/sequence_analysis.py` - Data models
- `ui/tabs/sequence_tab.py` - Integrate analysis results

**Architecture consideration:** This requires comparing multiple clips, which is different from the current single-clip analysis pattern.

#### Research Insights

**Lazy Computation Pattern (from architecture-strategist):**

```python
@dataclass
class Sequence:
    id: str
    clips: list[SequenceClip] = field(default_factory=list)
    _analysis_cache: Optional["SequenceAnalysis"] = field(default=None, repr=False)

    @property
    def analysis(self) -> "SequenceAnalysis":
        """Lazy-computed sequence analysis. NOT persisted to project file."""
        if self._analysis_cache is None:
            self._analysis_cache = analyze_sequence(self.clips)
        return self._analysis_cache

    def add_clip(self, clip: SequenceClip):
        self.clips.append(clip)
        self._analysis_cache = None  # Invalidate

    def remove_clip(self, clip_id: str):
        self.clips = [c for c in self.clips if c.id != clip_id]
        self._analysis_cache = None  # Invalidate

    def to_dict(self) -> dict:
        # DO NOT serialize _analysis_cache
        return {"id": self.id, "clips": [c.to_dict() for c in self.clips]}
```

**Performance Safeguards (from performance-oracle):**

```python
def analyze_sequence(clips: list[SequenceClip], project: Project) -> SequenceAnalysis:
    """Analyze sequence with O(N²) guard for large sequences."""

    # Simple pacing stats are O(N)
    durations = [c.duration_seconds for c in clips]
    pacing = PacingAnalysis(
        average_duration=statistics.mean(durations) if durations else 0,
        variance=statistics.variance(durations) if len(durations) > 1 else 0,
        classification=classify_pacing(durations)
    )

    # Similarity analysis is O(N²) - guard against large sequences
    if len(clips) > 100:
        # Use embeddings + approximate nearest neighbors
        similarity = _analyze_similarity_with_embeddings(clips, project)
    else:
        similarity = _analyze_similarity_pairwise(clips, project)

    return SequenceAnalysis(pacing=pacing, similarity=similarity)
```

**Data Integrity (from data-integrity-guardian):**
```python
def _validate_clip_references(sequence: Sequence, project: Project) -> list[str]:
    """Validate all clip references exist in project."""
    errors = []
    for seq_clip in sequence.clips:
        if seq_clip.source_clip_id not in project.clips_by_id:
            errors.append(f"SequenceClip {seq_clip.id} references missing clip {seq_clip.source_clip_id}")
    return errors
```

**Continuity Warnings: Advisory Mode**
```python
@dataclass
class ContinuityWarning:
    """Soft warning about potential continuity issue."""
    warning_type: str  # "180_degree", "30_degree", "jump_cut"
    clip_pair: tuple[str, str]  # (clip_id_1, clip_id_2)
    severity: str  # "low", "medium", "high"
    explanation: str  # "These shots may cross the 180° line"
    can_be_intentional: bool = True  # Most rule-breaking is valid artistically
```

**Pacing Feedback: Three Modes**

1. **Genre Comparison:**
```python
GENRE_PACING_NORMS = {
    "action": {"avg_shot_duration": 2.5, "variance": "high"},
    "drama": {"avg_shot_duration": 5.0, "variance": "medium"},
    "documentary": {"avg_shot_duration": 8.0, "variance": "high"},
    "music_video": {"avg_shot_duration": 1.5, "variance": "very_high"},
}

def compare_to_genre(sequence: Sequence, genre: str) -> dict:
    """Compare sequence pacing to genre norms."""
    norm = GENRE_PACING_NORMS.get(genre)
    actual_avg = sequence.analysis.average_shot_duration
    return {
        "genre": genre,
        "norm_avg": norm["avg_shot_duration"],
        "actual_avg": actual_avg,
        "difference_percent": (actual_avg - norm["avg_shot_duration"]) / norm["avg_shot_duration"] * 100,
    }
```

2. **Pacing Visualization:**
```python
# Generate data for pacing curve chart
def get_pacing_curve(sequence: Sequence) -> list[dict]:
    """Return shot durations over time for visualization."""
    return [
        {"position": i, "duration": clip.duration_seconds, "clip_id": clip.id}
        for i, clip in enumerate(sequence.clips)
    ]
```

3. **Improvement Suggestions:**
```python
def suggest_pacing_improvements(sequence: Sequence) -> list[str]:
    """Advisory suggestions for pacing (not prescriptive)."""
    suggestions = []
    if sequence.analysis.average_shot_duration > 10:
        suggestions.append("Consider shorter shots to increase energy")
    if sequence.analysis.variance < 0.5:
        suggestions.append("Varying shot lengths could add visual interest")
    return suggestions
```

**Scale Consideration (1000s of clips):**
- Pacing analysis is O(N) - scales fine
- Continuity checking needs embedding-based similarity, not pairwise
- Run in background, show results progressively

### Phase 5: Scene Report Generation (Medium Effort)

**Goal:** Generate human-readable film analysis reports.

| Feature | Description | Effort |
|---------|-------------|--------|
| Report Generator | Synthesize clip analyses | 3 days |
| Template System | Customizable report formats | 2 days |
| Export Formats | Markdown, PDF, HTML | 2 days |
| Integration | Button in Sequence tab | 1 day |

**Files to create:**
- `core/export/scene_report.py` - Report generation
- `ui/dialogs/report_dialog.py` - Configuration UI

#### Research Insights

**Report Generation Architecture:**

```python
def generate_scene_report(
    clips: list[Clip],
    project: Project,
    llm_client: LLMClient,
    sections: list[str] = None
) -> str:
    """Generate film analysis report using LLM synthesis."""

    sections = sections or ["overview", "cinematography", "pacing", "recommendations"]

    # Gather clip metadata
    clip_summaries = [_summarize_clip(c) for c in clips]

    # For large sequences, chunk into sections
    if len(clips) > 50:
        return _generate_chunked_report(clips, project, llm_client, sections)

    # Single LLM call for smaller sequences
    prompt = REPORT_PROMPT_TEMPLATE.format(
        clip_count=len(clips),
        clip_summaries="\n".join(clip_summaries),
        sections=", ".join(sections)
    )

    return llm_client.complete(prompt)


def _summarize_clip(clip: Clip) -> str:
    """Create concise text summary of clip for report context."""
    parts = [f"Clip {clip.id[:8]}:"]
    if clip.cinematography:
        c = clip.cinematography
        parts.append(f"  {c.shot_size} {c.camera_angle}")
    if clip.description:
        parts.append(f"  {clip.description[:100]}")
    return "\n".join(parts)
```

**Agent-Native Requirement (from agent-native-reviewer):**

Add tool for report generation:
```python
@tool
def generate_analysis_report(
    sequence_id: Optional[str] = None,
    clip_ids: Optional[list[str]] = None,
    sections: Optional[list[str]] = None
) -> dict:
    """
    Generate a film analysis report for clips or sequence.

    Args:
        sequence_id: Generate report for entire sequence
        clip_ids: Generate report for specific clips
        sections: Which sections to include (overview, cinematography, pacing, recommendations)

    Returns:
        {"success": True, "report": "markdown content...", "word_count": int}
    """
```

**Export Format Simplification:**
- **Markdown**: Default, always supported
- **HTML**: Render markdown to HTML with CSS styling
- **PDF**: Consider optional (requires WeasyPrint or similar)—may be over-engineering

---

## Agent-Native Audit ⚠️ CRITICAL

**Principle:** Every feature a human filmmaker can access MUST have a corresponding agent tool. The agent should be able to perform the complete filmmaking workflow autonomously.

### Required New Tools

| Tool | Purpose | Phase | Priority |
|------|---------|-------|----------|
| `get_film_term_definition` | Access glossary programmatically | 1 | High |
| `search_glossary` | Find terms by keyword | 1 | High |
| `detect_audio_beats` | Extract beats/tempo from audio | 3 | High |
| `align_sequence_to_audio` | Sync clip cuts to beats | 3 | High |
| `get_sequence_analysis` | Get pacing/continuity metrics | 4 | High |
| `check_continuity_issues` | Detect 180°/30° rule violations | 4 | Medium |
| `generate_analysis_report` | Create film analysis reports | 5 | Medium |

### Existing Tools That Need Updates

| Tool | Update Needed |
|------|---------------|
| `analyze_cinematography` | Return new fields (dutch_tilt, camera_position, lens_type, light_quality, color_temperature) |
| `get_clip_metadata` | Include audio sync data when available |
| `arrange_sequence` | Accept beat-aligned timing options |

### Agent Workflow Enablement

The agent should be able to execute this complete workflow:
```
1. analyze_cinematography(clip_ids)     # Get visual analysis
2. detect_audio_beats(audio_source)      # Get rhythm data
3. arrange_sequence(clips, strategy="beat_aligned")  # Build sequence
4. get_sequence_analysis(sequence_id)    # Check pacing
5. check_continuity_issues(sequence_id)  # Validate continuity
6. generate_analysis_report(sequence_id) # Document the edit
```

---

## Acceptance Criteria

### Documentation Deliverables

- [x] In-app glossary with all film language terms
- [x] Tooltips on all cinematography badges
- [ ] Help links to external film education resources (deferred - user preference)
- [ ] CLAUDE.md updated with film language taxonomy reference
- [x] Agent tool: `get_film_term_definition`
- [x] Agent tool: `search_glossary`

### Analysis Enhancements

- [ ] Dutch tilt detection with confidence score
- [ ] Camera position (frontal/profile/back) detection
- [ ] Lens type estimation from visual characteristics
- [ ] Light quality (hard/soft) classification
- [ ] Color temperature estimation

### Audio-Guided Sequencing

- [ ] Beat detection from audio tracks
- [ ] Tempo (BPM) analysis
- [ ] Onset/transient detection for cut points
- [ ] Align sequence cuts to beats
- [ ] Audio waveform visualization in timeline
- [ ] Agent tool: `detect_audio_beats`
- [ ] Agent tool: `align_sequence_to_audio`

### Sequence Analysis

- [ ] Pacing statistics (average shot duration, variance)
- [ ] Genre pacing comparison (action, drama, documentary, music video)
- [ ] Pacing curve visualization
- [ ] Pacing improvement suggestions (advisory)
- [ ] Continuity warnings (advisory mode - 180°, 30°, jump cuts)
- [ ] Visual consistency metrics
- [ ] Scene report generation
- [ ] Agent tool: `get_sequence_analysis`
- [ ] Agent tool: `check_continuity_issues`
- [ ] Scales to 1000+ clips (embedding-based, not pairwise)

---

## Technical Considerations

### VLM Prompt Engineering

Current cinematography prompts in `core/analysis/cinematography.py` can be extended:

```python
# Example: Adding dutch tilt to frame analysis prompt
CINEMATOGRAPHY_PROMPT_FRAME = """
Analyze this frame for cinematographic properties:

...existing fields...

13. DUTCH TILT (horizon_tilt):
    - "none": Horizon is level
    - "slight": 5-15° tilt
    - "moderate": 15-30° tilt
    - "extreme": 30°+ tilt
    - "unknown": Cannot determine (no clear horizon reference)

14. CAMERA POSITION relative to subject:
    - "frontal": Camera faces subject directly
    - "three_quarter": 45° angle
    - "profile": 90° side view
    - "back": Behind subject
    - "unknown": No clear subject or ambiguous positioning
"""
```

#### Research Insights

**Provider-Specific Prompts (from vlm-provider-prompt-strategies skill):**

```python
# Gemini handles detailed JSON schemas well
CINEMATOGRAPHY_SCHEMA_GEMINI = {
    "type": "object",
    "properties": {
        "shot_size": {"type": "string", "enum": SHOT_SIZE_CHOICES + ["unknown"]},
        # ... detailed schema
    }
}

# GPT-4o prefers simpler enum descriptions
CINEMATOGRAPHY_PROMPT_OPENAI = """
Analyze this frame. For each field, choose from the listed options or "unknown" if uncertain.

shot_size: ELS, VLS, LS, MLS, MS, MCU, CU, BCU, ECU, Insert, unknown
camera_angle: low_angle, eye_level, high_angle, dutch_angle, birds_eye, worms_eye, unknown
...
"""
```

**Critical: Always Include "unknown"**
- VLMs hallucinate when forced to choose from limited options
- "unknown" provides safe fallback for ambiguous frames
- Filter "unknown" values in UI display if desired

**Remove Confidence Scores:**
```python
# DON'T DO THIS - confidence scores are unreliable
"shot_size_confidence": {"type": "number", "minimum": 0, "maximum": 1}

# DO THIS - binary classification only
"shot_size": {"type": "string", "enum": [..., "unknown"]}
```

### JSON Schema Updates

```json
{
  "type": "object",
  "properties": {
    "horizon_tilt": {
      "type": "string",
      "enum": ["none", "slight", "moderate", "extreme"]
    },
    "camera_position": {
      "type": "string",
      "enum": ["frontal", "three_quarter", "profile", "back"]
    }
  }
}
```

### Backward Compatibility

New fields should:
- Have default values (`"unknown"` or `None`)
- Not break existing project files
- Use `from_dict()` pattern with `.get()` for optional fields

#### Research Insights

**Existing Pattern (already correct):**
```python
# From models/cinematography.py - this pattern is correct
@classmethod
def from_dict(cls, data: dict) -> "CinematographyAnalysis":
    return cls(
        shot_size=data.get("shot_size", "unknown"),
        camera_angle=data.get("camera_angle", "unknown"),
        # New fields follow same pattern:
        dutch_tilt=data.get("dutch_tilt", "unknown"),
        camera_position=data.get("camera_position", "unknown"),
    )
```

**Tuple Serialization (from data-integrity-guardian):**

If storing tuples (e.g., axis violation pairs), ensure proper serialization:
```python
# Tuples become lists in JSON - handle conversion
axis_violations: list[tuple[str, str]]

def to_dict(self):
    return {
        "axis_violations": [list(pair) for pair in self.axis_violations]
    }

@classmethod
def from_dict(cls, data):
    return cls(
        axis_violations=[tuple(pair) for pair in data.get("axis_violations", [])]
    )
```

### Performance Considerations ⚠️ CRITICAL: 1000s of Clips

Projects may contain **1000+ clips**. All features must be designed for this scale.

#### Architecture for Large Scale

**1. UI Virtualization:**
```python
# Never load all clips into memory at once
class VirtualizedClipGrid(QAbstractItemView):
    """Only render visible clips, load thumbnails on demand."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.visible_range = (0, 50)  # Only load visible items
        self.thumbnail_cache = LRUCache(maxsize=200)

    def scroll_to(self, index: int):
        # Update visible range and load new thumbnails
        self.visible_range = (index, index + 50)
        self._load_visible_thumbnails()
```

**2. Embedding-Based Similarity (not pairwise):**
```python
# O(N²) pairwise comparison is NOT acceptable at 1000+ clips
# Use embeddings + approximate nearest neighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_similar_clips(clips: list[Clip], target_clip: Clip, k: int = 10) -> list[Clip]:
    """Find k most similar clips using pre-computed embeddings."""
    embeddings = np.array([c.embedding for c in clips])  # Pre-computed
    target_embedding = target_clip.embedding

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors([target_embedding])

    return [clips[i] for i in indices[0]]
```

**3. Background Processing with Progress:**
```python
class LargeScaleAnalysisWorker(CancellableWorker):
    """Process clips in batches with progress updates."""

    BATCH_SIZE = 50

    def run(self):
        total = len(self.clip_ids)
        for i in range(0, total, self.BATCH_SIZE):
            if self._cancel_event.is_set():
                return

            batch = self.clip_ids[i:i + self.BATCH_SIZE]
            self._process_batch(batch)
            self.progress.emit(min(i + self.BATCH_SIZE, total), total)
```

**4. Indexed Search/Filter:**
```python
# Pre-build indices for fast filtering
class ClipIndex:
    """Inverted index for fast clip filtering."""

    def __init__(self, clips: list[Clip]):
        self.by_shot_size: dict[str, set[str]] = defaultdict(set)
        self.by_camera_angle: dict[str, set[str]] = defaultdict(set)
        # ... build indices

    def filter(self, shot_size: str = None, camera_angle: str = None) -> set[str]:
        """Return clip IDs matching all criteria."""
        result = None
        if shot_size:
            result = self.by_shot_size.get(shot_size, set())
        if camera_angle:
            angle_set = self.by_camera_angle.get(camera_angle, set())
            result = result & angle_set if result else angle_set
        return result or set()
```

**5. Lazy Computation:**
- Sequence analysis: Compute on-demand, not on every change
- Pacing metrics: Cache and invalidate only when sequence changes
- Continuity checks: Run in background, show results when ready

#### Thread Safety (from existing patterns)
```python
# Use frozen dataclasses for ThreadPoolExecutor tasks
@dataclass(frozen=True)
class AnalysisTask:
    clip_id: str
    video_path: str  # Use str, not Path, for pickling
    output_dir: str
```

#### FFmpeg Path Security (from learnings)
```python
# ALWAYS use argument arrays, never shell interpolation
subprocess.run(["ffmpeg", "-i", str(video_path), ...], shell=False)
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Film term coverage | 90% of LearnAboutFilm concepts | Taxonomy comparison |
| User understanding | Users can explain what badges mean | User survey |
| Analysis accuracy | 85% agreement with manual labels | Validation dataset |
| Performance (analysis) | <5s per clip for expanded analysis | Benchmarking |
| Performance (scale) | UI responsive with 1000+ clips | Load testing |
| Performance (sequence) | Sequence analysis <10s for 500 clips | Benchmarking |
| Agent parity | 100% of human features have agent tools | Audit |

---

## Dependencies & Risks

### Dependencies

- Gemini API for video mode analysis
- FFmpeg for audio extraction
- Optional: Audio classification model

### Risks

| Risk | Mitigation |
|------|------------|
| VLM hallucination on new fields | Add "unknown" enum value (not confidence scores) |
| Audio model size | Make optional, download on first use |
| Sequence analysis complexity | Start with simple pacing metrics |
| Backward compatibility | Careful schema versioning |

#### Research Insights - Additional Risks

| Risk | Mitigation |
|------|------------|
| QThread signal duplicate delivery | Apply guard flag pattern from learnings |
| Subprocess cleanup on exception | Use try/finally with terminate/wait |
| Over-engineering | Apply YAGNI - defer Phases 3-5 until validated need |
| Agent-native gaps | Add tools alongside each new feature |

**Relevant Learnings to Apply:**

1. **QThread Signal Guard** (`docs/solutions/qthread-signal-duplicate.md`):
   - Use `_finished_emitted` guard flag to prevent duplicate `finished` signals
   - Apply to all new workers (AudioAnalysisWorker, etc.)

2. **Subprocess Cleanup** (`docs/solutions/subprocess-cleanup.md`):
   - Always wrap subprocess calls in try/finally
   - Call `process.terminate()` then `process.wait(timeout=5)` on exception

3. **FFmpeg Path Escaping** (`docs/solutions/ffmpeg-path-security.md`):
   - Never use shell=True with user-provided paths
   - Use argument arrays for all FFmpeg commands

4. **Source ID Sync** (`docs/solutions/source-id-mismatch.md`):
   - When creating clips, ensure `source_id` matches the actual source
   - Validate references on project load

---

## References & Research

### Internal References

- Cinematography analysis: `core/analysis/cinematography.py`
- Data model: `models/cinematography.py`
- Worker pattern: `ui/workers/cinematography_worker.py`
- Settings: `core/settings.py:420-424`

### External References

- LearnAboutFilm Film Language: https://learnaboutfilm.com/film-language/
- LearnAboutFilm Shot Sizes: https://learnaboutfilm.com/film-language/picture/shotsize/
- LearnAboutFilm Composition: https://learnaboutfilm.com/film-language/picture/frame-it-right/
- LearnAboutFilm Camera Movement: https://learnaboutfilm.com/film-language/picture/movement/
- LearnAboutFilm Sound: https://learnaboutfilm.com/film-language/sound/
- LearnAboutFilm Editing: https://learnaboutfilm.com/film-language/editing/

### Related Work

- Rich Cinematography Analysis (completed): `docs/plans/2026-02-01-feat-rich-cinematography-analysis-plan.md`

### Research Sources (from deepening)

- [librosa documentation](https://librosa.org/doc/latest/) - Audio feature extraction
- [pyloudnorm PyPI](https://pypi.org/project/pyloudnorm/) - LUFS measurement
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Qt Rich Text](https://doc.qt.io/qt-6/richtext-html-subset.html) - HTML subset for tooltips

---

## SpecFlow Analysis: Open Questions

The following questions were identified during specification analysis and should be resolved before implementation.

### Priority 1: Critical (Blocks Implementation)

**Q1. Dutch Angle vs Dutch Tilt Clarification**

The existing `dutch_angle` in CAMERA_ANGLES refers to camera orientation relative to subject. The proposed `dutch_tilt` refers to horizon tilt specifically. These are different concepts:
- A shot can be `eye_level` (camera at subject height) but still have a `dutch_tilt` (horizon tilted 15°)
- **Recommendation:** Keep both; `dutch_angle` is camera-to-subject relationship, `dutch_tilt` is frame composition

**Q2. AudioAnalysis Storage Pattern**

Where should audio analysis data live in the data model?
- **Option A:** Add fields directly to Clip model (flat): `clip.has_music`, `clip.audio_loudness`
- **Option B:** Nested dataclass (structured): `clip.audio_analysis.has_music`
- **Recommendation:** Option B (matches cinematography pattern)

**Q3. Music Classification Model Selection**

What audio classification approach should be used?
- **Option A:** Basic librosa features (RMS energy, spectral analysis) - no ML model needed
- **Option B:** PANNs AudioSet classifier (~300MB model)
- **Option C:** VGGish (~150MB model)
- **Recommendation:** Option A for Phase 1, defer ML model to Phase 2

**Q4. Clips Without Audio Track**

What happens when analyzing clips that have no audio?
- **Recommendation:** Set `audio_presence: False` and all other audio fields to `None`/`0`

**Q5. Cross-Source Sequence Handling**

How should sequence analysis handle clips from different sources with different visual characteristics?
- **Recommendation:** Normalize per-source, flag cross-source transitions as intentional, compute within-source consistency separately

### Priority 2: Important (Affects UX)

**Q6. Glossary UI Location**

Where should the in-app glossary be accessible?
- **Options:** Modal dialog from menu, sidebar panel, help button in Analyze tab, floating tooltip expansion
- **Recommendation:** "?" button in Analyze tab header opens modal glossary

**Q7. Pacing Classification Thresholds**

What shot durations define fast/medium/slow?
- **Recommendation:** Fast <2s, Medium 2-5s, Slow >5s (can be genre-adjusted)

**Q8. 180-Degree Rule Detection Approach**

How to detect axis violations without pose estimation?
- **Recommendation:** Heuristic based on similar shots (same subject, opposite facing direction) within N clips; flag as "potential" violation

**Q9. Backward Compatibility for New Fields**

How should existing projects handle new fields?
- **Recommendation:** All new fields default to `None` or `"unknown"`, `from_dict()` uses `.get()` for optional fields

**Q10. Report Generation Model**

Should reports use the same VLM as analysis, or a separate synthesis call?
- **Recommendation:** Single LLM call with all clip metadata as context; may need chunking for large sequences (>50 clips)
