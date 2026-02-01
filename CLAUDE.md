# Scene Ripper - Project Conventions

## Overview

Video scene detection, analysis, and editing application with integrated AI agent support. Built with PySide6, PySceneDetect, FFmpeg, and LiteLLM.

## Technology Stack

- **UI Framework**: PySide6 (Qt 6 for Python)
- **Scene Detection**: PySceneDetect with AdaptiveDetector
- **Video Processing**: FFmpeg (via subprocess)
- **Video Download**: yt-dlp
- **Transcription**: faster-whisper
- **LLM Integration**: LiteLLM (multi-provider)
- **YouTube API**: google-api-python-client
- **Python Version**: 3.11+

## Project Structure

```
algorithmic-filmmaking/
├── main.py                       # GUI entry point
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata, CLI entry point
├── ui/
│   ├── main_window.py            # Main application window
│   ├── chat_panel.py             # Agent chat interface
│   ├── chat_worker.py            # Background LLM operations
│   ├── chat_widgets.py           # Chat UI components
│   ├── theme.py                  # Application theming
│   ├── settings_dialog.py        # Settings UI
│   ├── source_browser.py         # Source file browser
│   ├── source_thumbnail.py       # Source thumbnail widget
│   ├── clip_browser.py           # Clip thumbnail grid
│   ├── video_player.py           # Video preview player
│   ├── project_adapter.py        # Project/UI bridge
│   ├── youtube_search_panel.py   # YouTube search UI
│   ├── youtube_result_thumbnail.py # YouTube result widget
│   ├── tabs/                     # Tab-based workflow
│   │   ├── base_tab.py           # Base class for all tabs
│   │   ├── collect_tab.py        # Source collection
│   │   ├── analyze_tab.py        # Video analysis
│   │   ├── sequence_tab.py       # Sequence editing (card-based)
│   │   ├── cut_tab.py            # Scene cutting
│   │   ├── generate_tab.py       # Generation
│   │   └── render_tab.py         # Final rendering
│   ├── timeline/                 # Timeline components
│   │   ├── timeline_widget.py    # Main timeline widget
│   │   ├── timeline_view.py      # Timeline graphics view
│   │   ├── timeline_scene.py     # Timeline scene
│   │   ├── clip_item.py          # Clip representation
│   │   ├── track_item.py         # Track representation
│   │   └── playhead.py           # Playhead control
│   └── widgets/                  # Custom widgets
│       ├── sorting_card.py       # Card widget for clips
│       ├── sorting_card_grid.py  # Grid layout for cards
│       ├── sorting_parameter_panel.py # Sorting controls
│       ├── timeline_preview.py   # Timeline preview widget
│       └── empty_state.py        # Empty state placeholder
├── core/
│   ├── chat_tools.py             # Agent tool definitions
│   ├── tool_executor.py          # Tool execution engine
│   ├── llm_client.py             # LiteLLM client abstraction
│   ├── gui_state.py              # GUI state tracking for agents
│   ├── project.py                # Project management
│   ├── settings.py               # Settings system
│   ├── scene_detect.py           # PySceneDetect wrapper
│   ├── ffmpeg.py                 # FFmpeg operations
│   ├── thumbnail.py              # Thumbnail generation
│   ├── downloader.py             # Video downloading (yt-dlp)
│   ├── transcription.py          # Whisper transcription
│   ├── youtube_api.py            # YouTube API integration
│   ├── sequence_export.py        # Sequence export
│   ├── edl_export.py             # EDL format export
│   ├── dataset_export.py         # Dataset export for training
│   ├── analysis/
│   │   ├── shots.py              # Shot detection
│   │   └── color.py              # Color analysis
│   └── remix/
│       └── shuffle.py            # Clip shuffling algorithms
├── cli/
│   ├── main.py                   # CLI entry point (Click)
│   ├── commands/
│   │   ├── detect.py             # Scene detection command
│   │   ├── analyze.py            # Analysis command
│   │   ├── transcribe.py         # Transcription command
│   │   ├── youtube.py            # YouTube operations
│   │   ├── export.py             # Export command
│   │   └── project.py            # Project management command
│   └── utils/
│       ├── config.py             # CLI configuration
│       ├── errors.py             # Error handling
│       ├── output.py             # Output formatting
│       └── progress.py           # Progress display
├── models/
│   ├── clip.py                   # Source, Clip data models
│   └── sequence.py               # Sequence model
└── tests/
    ├── test_project.py           # Project tests
    ├── test_settings.py          # Settings tests
    ├── test_youtube_api.py       # YouTube API tests
    └── test_cli.py               # CLI tests
```

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure FFmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# Run the GUI app
python main.py

# Or use the CLI
python -m cli.main --help
```

## Application Flows

The application supports multiple workflows. Users rarely follow a strict linear path—they iterate between tabs as creative needs evolve.

### Primary User Flows

**Linear Flow (First-time / Simple projects):**
```
Collect → Cut → Analyze → Sequence → Render
```

**Iterative Flow (Most common in practice):**
```
Sequence → Collect → Cut → Analyze → Sequence → Render
         ↑__________________________|
```
Users often start in Sequence with existing clips, realize they need more material, collect new sources, process them, and return to sequencing.

**Analysis-First Flow (Research/archival projects):**
```
Collect → Cut → Analyze → (export metadata)
```
Some users only need clip analysis and metadata export without creating a final sequence.

**Quick Assembly Flow (Rough cuts):**
```
Collect → Cut → Sequence → Render
```
Skip analysis entirely for fast rough cuts where metadata isn't needed.

### Tab Navigation Patterns

| From Tab | Common Next Steps |
|----------|-------------------|
| **Collect** | Cut (process new sources), Sequence (if sources already have clips) |
| **Cut** | Analyze (enrich clips), Sequence (quick assembly), Collect (need more sources) |
| **Analyze** | Sequence (use enriched clips), Cut (re-detect with different settings) |
| **Sequence** | Render (finalize), Collect (need more material), Analyze (filter by metadata) |
| **Render** | Sequence (adjust edit), done |

### Data Transformation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT STATE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Source (video file)                                            │
│      │                                                          │
│      ├── file_path, fps, duration                               │
│      └── analyzed: bool                                         │
│          │                                                      │
│          │ scene detection (Cut tab)                            │
│          ▼                                                      │
│  Clip[] (detected scenes)                                       │
│      │                                                          │
│      ├── start_frame, end_frame, source_id                      │
│      ├── thumbnail_path                                         │
│      │                                                          │
│      │ analysis operations (Analyze tab) - all optional         │
│      │    ├── Describe → description: str                       │
│      │    ├── Classify → shot_type: str                         │
│      │    ├── Colors → dominant_colors: list                    │
│      │    ├── Transcribe → transcript: str                      │
│      │    └── Objects → detected_objects: list                  │
│      │                                                          │
│      │ add to sequence (Sequence tab)                           │
│      ▼                                                          │
│  SequenceClip[] (timeline arrangement)                          │
│      │                                                          │
│      ├── references Clip via source_clip_id                     │
│      ├── timeline position, in/out points                       │
│      │                                                          │
│      │ render (Render tab)                                      │
│      ▼                                                          │
│  Output file (MP4, EDL, individual clips)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Background Processing Flow

Heavy operations run in QThread workers to keep the UI responsive:

```
User Action → Worker Started → Progress Signals → Completion → UI Update
                   │                   │
                   │              progress(n, total)
                   │              clip_ready(clip)
                   │              error(message)
                   │                   │
                   └───────────────────┴─→ finished()
```

**Key worker patterns:**
- All workers inherit from `CancellableWorker` (ui/workers/base.py)
- Workers emit signals; main thread updates UI
- User can cancel long-running operations
- Errors are caught and emitted as signals, not raised

### Agent Interaction Flow

The chat agent can trigger any user action programmatically:

```
User Message → LLM Processing → Tool Calls → GUI Updates → Response
                    │               │              │
                    │          ToolExecutor   gui_state sync
                    │               │              │
                    └───────────────┴──────────────┴─→ visible feedback
```

**Agent capabilities mirror user capabilities:**
- Navigate tabs
- Select clips/sources
- Trigger analysis operations
- Modify sequence
- The agent sees current GUI state via `gui_state.py`

## Tab Workflow

The application uses a tab-based workflow where each tab represents a stage:

### Data Model Flow

```
Source (video file)
    ↓ scene detection
Clips (detected scenes with start/end frames)
    ↓ analysis (optional)
Clips with metadata (shot_type, colors, description, transcript)
    ↓ add to sequence
SequenceClips (clips arranged on timeline)
    ↓ render
Output video file
```

### Tab Details

1. **Collect** - Import source videos
   - Add local video files via drag-drop or file browser
   - Search and download from YouTube
   - Sources appear in the source browser sidebar
   - Sources have `analyzed: bool` flag (initially False)

2. **Cut** - Scene detection and clip browsing
   - Select a source and run scene detection (sensitivity 1.0-10.0)
   - Detection creates `Clip` objects with `start_frame`, `end_frame`, `source_id`
   - Clips appear in the clip browser grid with thumbnails
   - After detection, `source.analyzed = True`
   - Can preview clips in the video player

3. **Analyze** - Enrich clips with metadata
   - **Describe**: Send frame/video to VLM for natural language description
   - **Classify**: Detect shot type (close-up, medium, wide, etc.)
   - **Detect Objects**: Run YOLO object detection
   - **Colors**: Extract dominant color palette
   - **Transcribe**: Run Whisper speech-to-text
   - All analysis updates clip metadata fields

4. **Sequence** - Arrange clips into a sequence
   - Card-based sorting interface
   - Drag-drop clips to reorder
   - Filter/sort by metadata (shot type, duration, color, etc.)
   - Selected clips are added to the timeline sequence
   - Creates `SequenceClip` objects referencing source clips

5. **Render** - Export final video
   - Export sequence as video file (MP4)
   - Export as EDL for external editors
   - Export clips individually
   - Configure quality, resolution, codec settings

### Key Data Relationships

```python
# Source contains metadata about a video file
Source:
    id: str (UUID)
    file_path: Path
    fps: float
    duration_seconds: float
    analyzed: bool  # True after scene detection

# Clip is a detected scene within a source
Clip:
    id: str (UUID)
    source_id: str  # References Source.id
    start_frame: int
    end_frame: int
    thumbnail_path: Path
    # Analysis metadata (optional):
    shot_type: str
    dominant_colors: list[str]
    description: str
    transcript: str

# SequenceClip is a clip placed on the timeline
SequenceClip:
    id: str (UUID)
    source_clip_id: str  # References Clip.id
    source_id: str       # References Source.id
    start_frame: int     # Position on timeline
    in_point: int        # Clip start (usually same as source clip)
    out_point: int       # Clip end
```

### Project State Management

The `Project` class (`core/project.py`) is the single source of truth:
- **Always use Project methods** to modify state (`add_source()`, `add_clips()`, etc.)
- **Never directly append** to `project.sources` or `project.clips` lists
- Project methods invalidate caches and notify GUI observers
- Cached properties: `sources_by_id`, `clips_by_id`, `clips_by_source`

All tabs inherit from `ui/tabs/base_tab.py` which provides common functionality.

## Agent/LLM Integration

The application includes an AI agent that can assist with video editing tasks.

### Key Components

- **`core/chat_tools.py`** - Defines all available agent tools (select clips, analyze video, etc.)
- **`core/tool_executor.py`** - Executes tool calls from the LLM
- **`core/llm_client.py`** - LiteLLM abstraction for multiple providers
- **`core/gui_state.py`** - Tracks GUI state for agent context awareness
- **`ui/chat_panel.py`** - Chat interface widget
- **`ui/chat_worker.py`** - Background worker for LLM calls

### Tool Executor Pattern

Agent tools are defined in `chat_tools.py` and executed through `ToolExecutor`:

```python
# Tools return results with success flag
return {"success": True, "result": data}

# GUI state is tracked for agent context
gui_state.update_selection(items)
```

## CLI Usage

```bash
# Scene detection
python -m cli.main detect <video_path> [--threshold 3.0]

# Video analysis
python -m cli.main analyze <video_path>

# Transcription
python -m cli.main transcribe <video_path>

# YouTube search
python -m cli.main youtube search "query"

# Export
python -m cli.main export <project> --format edl
```

## YouTube Integration

- **`core/youtube_api.py`** - YouTube Data API v3 access
- **`core/downloader.py`** - yt-dlp wrapper for video downloads
- **`ui/youtube_search_panel.py`** - Search and results UI

Requires a YouTube API key configured in settings.

## Project Management

Projects are managed through `core/project.py`:

- Projects save/load all sources, clips, sequences, and settings
- JSON-based project file format
- Supports auto-save and recovery

## Key Patterns

### Scene Detection
Use `AdaptiveDetector` for dynamic footage (default). Threshold range: 1.0 (sensitive) to 10.0 (less sensitive).

### Background Processing
Heavy operations (detection, thumbnails, LLM calls) run in `QThread` workers to keep UI responsive.

### FFmpeg Safety
Always use argument arrays, never shell interpolation. Validate paths before processing.

### User Settings
Always use paths and configuration from `core/settings.py` (loaded via `load_settings()`). Never hardcode paths or use defaults when a user-configurable setting exists:
- `settings.download_dir` for video downloads
- `settings.project_dir` for project files
- `settings.cache_dir` for thumbnails and cache

### GUI State Synchronization
When the agent makes changes, ensure GUI state is synchronized via `core/gui_state.py`. The agent receives context about current selections, active tabs, and visible items.

### Card-Based UI
The sequence tab uses a card-based interface (`ui/widgets/sorting_card.py`) for visual clip arrangement with drag-and-drop support.

### UI Consistency Standards
Maintain consistent widget sizing across the entire application using constants from `ui/theme.py`:

```python
from ui.theme import UISizes

# Available constants:
UISizes.COMBO_BOX_MIN_HEIGHT    # 28 - Use for ALL combo boxes
UISizes.LINE_EDIT_MIN_HEIGHT    # 28 - Use for text inputs
UISizes.BUTTON_MIN_HEIGHT       # 28 - Use for buttons
UISizes.FORM_LABEL_WIDTH        # 140 - Standard label width
UISizes.FORM_LABEL_WIDTH_NARROW # 120 - Compact layouts
UISizes.FORM_LABEL_WIDTH_WIDE   # 180 - Wide labels
UISizes.COMBO_BOX_MIN_WIDTH     # 200 - Minimum combo width
UISizes.COMBO_BOX_MIN_WIDTH_WIDE # 300 - Wide combo boxes
```

**Dropdowns (QComboBox)**
- Always set minimum height: `combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)`
- Apply to ALL combo boxes throughout the app for visual consistency

**Labels in form layouts**
- Use fixed width for alignment: `label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)`

**Scrollable content**
- Wrap long form content in `QScrollArea` with `setWidgetResizable(True)`
- Use `setFrameShape(QScrollArea.NoFrame)` for seamless appearance

```python
# Standard combo box setup
combo = QComboBox()
combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)

# For form layouts with multiple fields
label = QLabel("Field Name:")
label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_project.py

# Run with verbose output
pytest tests/ -v
```

### Bug Fixes: Prove It Pattern

When given a bug or error report, the first step is to spawn a subagent to write a test that reproduces the issue. Only proceed once reproduction is confirmed.

**Test level hierarchy** - Reproduce at the lowest level that can capture the bug:

1. **Unit test** - Pure logic bugs, isolated functions (lives in `tests/`)
2. **Integration test** - Component interactions, API boundaries (lives in `tests/`)

**For every bug fix:**

1. **Reproduce with subagent** - Spawn a subagent to write a test that demonstrates the bug. The test should *fail* before the fix.
2. **Fix** - Implement the fix.
3. **Confirm** - The test now *passes*, proving the fix works.

If the bug is truly environment-specific or transient, document why a test isn't feasible rather than skipping silently.

## Common Commands

```bash
# Run the GUI app
python main.py

# CLI help
python -m cli.main --help

# Check FFmpeg
ffmpeg -version

# Scene detection test
python -c "from core.scene_detect import SceneDetector; print(SceneDetector)"
```
