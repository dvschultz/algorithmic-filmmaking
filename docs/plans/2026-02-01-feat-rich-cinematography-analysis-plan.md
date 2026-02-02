# Rich Cinematography Analysis (Retrospective)

> **Status**: Completed (Feb 1, 2026)
>
> This is a retrospective plan documenting the feature after implementation.
> The original design was based on film language terminology from learnaboutfilm.com.

## Goal

Add comprehensive film language analysis to clips using VLM (Gemini), capturing cinematographic properties beyond basic shot type classification. This enables more sophisticated clip filtering, sorting, and sequence assembly based on visual storytelling principles.

## Problem

The existing shot type classification (`core/analysis/shots.py`) only captured 5 basic categories (wide, full, medium, close-up, extreme close-up). This was insufficient for:
- Understanding camera angles and their narrative effects
- Detecting camera movement (pan, tilt, tracking, etc.)
- Analyzing composition (subject position, balance, headroom)
- Understanding lighting style and emotional impact
- Making informed editing decisions based on film language

## Solution

### Approach

Use VLM (Gemini) with structured JSON output to extract 18+ cinematography attributes from clips. Support two analysis modes:
- **Frame mode**: Analyze a single thumbnail (fast, cheap)
- **Video mode**: Analyze the full clip for camera movement detection (Gemini only)

### Shot Size Taxonomy (10 classes)

Based on professional cinematography terminology:

| Code | Name | Description |
|------|------|-------------|
| ELS | Extreme Long Shot | Vast environment, people tiny/absent |
| VLS | Very Long Shot | Full environment with visible people |
| LS | Long Shot | Head to toe, full body |
| MLS | Medium Long Shot | 3/4 body (knees up) |
| MS | Medium Shot | Waist to head |
| MCU | Medium Close-Up | Head and shoulders |
| CU | Close-Up | Face fills frame |
| BCU | Big Close-Up | Face, partial features |
| ECU | Extreme Close-Up | Single feature (eyes, lips) |
| Insert | - | Object detail shot |

### Camera Angle Effects

| Angle | Effect | Usage |
|-------|--------|-------|
| low_angle | power | Heroism, dominance |
| eye_level | neutral | Equal footing |
| high_angle | vulnerability | Diminishment |
| dutch_angle | disorientation | Unease, chaos |
| birds_eye | omniscience | Pattern reveal |
| worms_eye | extreme_power | Maximum dominance |

### Additional Attributes

**Composition**:
- Subject position: left_third, center, right_third, distributed
- Headroom: tight, normal, excessive
- Lead room: tight, normal, excessive (space in direction of gaze)
- Balance: balanced, left_heavy, right_heavy, symmetrical

**Camera Movement** (video mode only):
- static, pan, tilt, track, handheld, crane, arc
- Direction: left, right, up, down, forward, backward, clockwise, counterclockwise

**Lighting**:
- Style: high_key, low_key, natural, dramatic
- Direction: front, three_quarter, side, back, below

**Derived Properties**:
- Emotional intensity: low, medium, high
- Suggested pacing: fast, medium, slow

## Implementation

### Files Created

1. **`core/analysis/cinematography.py`** - VLM-based analysis logic
   - `analyze_cinematography()` - Main entry point with auto mode selection
   - `analyze_cinematography_frame()` - Single thumbnail analysis
   - `analyze_cinematography_video()` - Full clip analysis with movement
   - JSON schema for structured output
   - Response parsing with markdown code block handling
   - Validation and normalization of VLM responses

2. **`models/cinematography.py`** - Data model
   - `CinematographyAnalysis` dataclass with 20+ fields
   - Serialization/deserialization methods
   - `get_display_badges()` for compact UI display
   - `get_simple_shot_type()` for compatibility mapping

3. **`ui/workers/cinematography_worker.py`** - Background processing
   - Thread-safe cancellation with `threading.Event`
   - Configurable parallelism (1-5 concurrent VLM calls)
   - Immutable task pattern for thread pool safety
   - Progress signals for UI updates

### Files Modified

1. **`models/clip.py`** - Added `cinematography: Optional[CinematographyAnalysis]` field

2. **`core/settings.py`** - Added settings:
   - `cinematography_input_mode`: "frame" or "video"
   - `cinematography_model`: VLM model (default: gemini-2.5-flash)
   - `cinematography_batch_parallelism`: 1-5 concurrent calls

3. **`ui/main_window.py`** - Integration:
   - "Rich Analysis" button handler
   - Worker lifecycle management
   - Progress and completion handlers

4. **`ui/clip_browser.py`** - Badge display:
   - `_update_cinematography_badges()` method
   - Badge rendering in clip cards

5. **`ui/tabs/analyze_tab.py`** - UI controls:
   - "Rich Analysis" button in toolbar

## Design Decisions

### Why VLM instead of specialized models?

- Single model handles all attributes (no ensemble of classifiers)
- Structured JSON output ensures consistent schema
- Natural language prompts can be refined without code changes
- Gemini supports video input for movement detection

### Thread-safe parallel processing

The worker uses `ThreadPoolExecutor` inside `QThread` for parallel VLM calls. Key patterns:
- `threading.Event` for cancellation (not boolean flag)
- Frozen dataclass for immutable task objects
- Signals emitted from main QThread, not pool threads

### Graceful fallbacks

- Video mode falls back to frame mode on error or large files
- 20MB video size cap prevents memory spikes
- VLM response validation with defaults for missing/invalid values

## Settings

| Setting | Default | Options |
|---------|---------|---------|
| cinematography_input_mode | frame | frame, video |
| cinematography_model | gemini-2.5-flash | Any Gemini model |
| cinematography_batch_parallelism | 2 | 1-5 |

## Cost Considerations

Using Gemini 2.5 Flash:
- Frame mode: ~$0.001 per clip
- Video mode: ~$0.005 per clip (varies with length)

For 100 clips: ~$0.10-0.50

## Future Enhancements

- [ ] Add cinematography filters to sequence tab sorting
- [ ] Expose cinematography in agent tools
- [ ] Add cinematography to clip details sidebar
- [ ] Support other VLMs (GPT-4V, Claude) for frame mode
- [ ] Cache analysis results to avoid re-processing

## References

- Film language terminology: https://learnaboutfilm.com/
- Gemini video understanding: https://ai.google.dev/gemini-api/docs/vision
