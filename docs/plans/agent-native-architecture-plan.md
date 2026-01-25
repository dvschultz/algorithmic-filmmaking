# Agent-Native Architecture Plan

**Date**: 2026-01-25
**Status**: Planning
**Goal**: Make Scene Ripper fully agent-accessible so that any action a user can perform via UI can also be performed programmatically by an agent.

---

## Current State Assessment

### Score: 0/18 capabilities are agent-accessible

All functionality is currently locked behind the Qt UI event loop. Agents cannot invoke any operations or access any data without human interaction.

### Orphan UI Actions (18 total)

| Action | Location | Status |
|--------|----------|--------|
| Import local video | `main_window.py:1094-1103` | ORPHAN |
| Import from URL | `main_window.py:1134-1144` | ORPHAN |
| Download video (YouTube/Vimeo) | `main_window.py:1146-1158` | ORPHAN |
| YouTube search | `main_window.py:1185-1214` | ORPHAN |
| Bulk download from search | `main_window.py:1229-1255` | ORPHAN |
| Scene detection | `main_window.py:1306-1341` | ORPHAN |
| Color extraction | `main_window.py:897-919` | ORPHAN |
| Shot type classification | `main_window.py:936-958` | ORPHAN |
| Transcription | `main_window.py:975-1030` | ORPHAN |
| Export clips | `main_window.py:1815-1921` | ORPHAN |
| Export dataset (JSON) | `main_window.py:1829-1876` | ORPHAN |
| Export sequence (video) | `main_window.py:1930-1998` | ORPHAN |
| Export EDL | `main_window.py:2020-2065` | ORPHAN |
| Save project | `main_window.py:2239-2282` | ORPHAN |
| Load project | `main_window.py:2284-2399` | ORPHAN |
| Add clip to timeline | `main_window.py:1634-1642` | ORPHAN |
| Generate (shuffle) | `generate_tab.py` | ORPHAN |
| Settings management | `main_window.py:695-721` | ORPHAN |

### What's Already Good

- **Clean separation of concerns**: Core logic in `core/`, models in `models/`, UI in `ui/`
- **Good data model serialization**: All models have `to_dict()`/`from_dict()` methods
- **Project files are human-readable JSON**: Located at user-chosen paths
- **Export functions are pure**: `export_dataset()`, `export_edl()` don't depend on UI state
- **Scene detection is self-contained**: `SceneDetector` class can be used standalone
- **FFmpeg operations are modular**: `FFmpegProcessor` wraps all FFmpeg calls cleanly

---

## Implementation Plan

### Phase 1: CLI Interface (High Impact, Medium Effort)

**Goal**: Create a command-line interface that exposes all core operations.

#### Commands to Implement

```bash
# Scene Detection
scene_ripper detect <video> --sensitivity 3.0 --output project.json
scene_ripper detect <video> --min-scene-length 0.5 --output project.json

# Analysis
scene_ripper analyze colors <project.json>
scene_ripper analyze shots <project.json>
scene_ripper transcribe <project.json> --model small.en --language en

# Export
scene_ripper export clips <project.json> --output-dir ./clips/
scene_ripper export dataset <project.json> --output metadata.json
scene_ripper export edl <project.json> --output timeline.edl
scene_ripper export video <project.json> --clips 1,2,3 --output final.mp4

# YouTube
scene_ripper search youtube "<query>" --max-results 25 --output results.json
scene_ripper download <url> --output-dir ./downloads/
scene_ripper download --from-file urls.txt --output-dir ./downloads/

# Project Management
scene_ripper project info <project.json>
scene_ripper project list-clips <project.json>
scene_ripper project add-to-sequence <project.json> --clips 1,2,3
```

#### Technical Approach

1. Create `cli/` directory with command modules
2. Use `click` library for CLI framework
3. Entry point: `python -m scene_ripper <command>` or `scene-ripper <command>`
4. Each command calls existing `core/` functions directly
5. Output formats: JSON (default), plain text (--format text)

#### Files to Create

```
cli/
├── __init__.py
├── main.py          # Click group and entry point
├── detect.py        # scene_ripper detect
├── analyze.py       # scene_ripper analyze colors/shots
├── transcribe.py    # scene_ripper transcribe
├── export.py        # scene_ripper export clips/dataset/edl/video
├── youtube.py       # scene_ripper search/download
└── project.py       # scene_ripper project info/list/add
```

#### Dependencies to Add

```
click>=8.0
```

---

### Phase 2: State Management Extraction (High Impact, High Effort)

**Goal**: Move application state from `MainWindow` to an independent `Project` class.

#### Current State Variables in MainWindow

```python
self.sources: dict[str, Source] = {}
self.clips: list[Clip] = []
self.clips_by_id: dict[str, Clip] = {}
self.clips_by_source: dict[str, list[Clip]] = {}
self.sequence: list[Clip] = []
self._analysis_queue: list[tuple[str, Clip]] = []
```

#### New Project Class

```python
# core/project.py (new)

@dataclass
class Project:
    """Application state independent of UI."""

    path: Optional[Path] = None
    sources: dict[str, Source] = field(default_factory=dict)
    clips: list[Clip] = field(default_factory=list)
    sequence: list[Clip] = field(default_factory=list)

    # Computed indexes
    @property
    def clips_by_id(self) -> dict[str, Clip]: ...

    @property
    def clips_by_source(self) -> dict[str, list[Clip]]: ...

    # Operations
    def add_source(self, source: Source) -> None: ...
    def add_clips(self, clips: list[Clip]) -> None: ...
    def add_to_sequence(self, clip_ids: list[str]) -> None: ...

    # Serialization
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Project": ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> "Project": ...
```

#### Migration Steps

1. Create `core/project.py` with `Project` class
2. Add signals to `Project` for state changes (using Qt signals or custom observer)
3. Update `MainWindow` to use `Project` instance
4. UI components observe `Project` changes via signals
5. CLI instantiates `Project` directly without UI

---

### Phase 3: Environment Variable Support (Medium Impact, Low Effort)

**Goal**: Allow configuration via environment variables for agent/CI workflows.

#### Variables to Support

| Variable | Purpose | Fallback |
|----------|---------|----------|
| `YOUTUBE_API_KEY` | YouTube Data API access | Keyring lookup |
| `SCENE_RIPPER_CACHE_DIR` | Thumbnail cache location | `~/.cache/scene-ripper` |
| `SCENE_RIPPER_DOWNLOAD_DIR` | Default download location | Platform videos dir |
| `SCENE_RIPPER_EXPORT_DIR` | Default export location | Platform videos dir |
| `SCENE_RIPPER_CONFIG` | Config file path | `~/.config/scene-ripper/config.json` |

#### Implementation

Update `core/settings.py`:

```python
def _get_youtube_api_key() -> str:
    """Get API key from environment or keyring."""
    # Environment variable takes precedence
    env_key = os.environ.get("YOUTUBE_API_KEY")
    if env_key:
        return env_key
    # Fall back to keyring
    return _get_api_key_from_keyring()
```

---

### Phase 4: JSON-Based Settings (Medium Impact, Medium Effort)

**Goal**: Remove Qt dependency from settings module.

#### Current Issue

`load_settings()` and `save_settings()` use `QSettings`, which requires a running `QApplication`. This prevents agents from reading/modifying settings.

#### Solution

1. Create `~/.config/scene-ripper/config.json` as primary settings store
2. Settings module reads JSON directly (no Qt dependency)
3. Migrate existing `QSettings` data on first run
4. Keep `QSettings` as fallback for backward compatibility

#### New Settings Structure

```json
{
  "paths": {
    "thumbnail_cache_dir": "~/.cache/scene-ripper/thumbnails",
    "download_dir": "~/Movies/Scene Ripper Downloads",
    "export_dir": "~/Movies"
  },
  "detection": {
    "default_sensitivity": 3.0,
    "min_scene_length_seconds": 0.5
  },
  "analysis": {
    "auto_analyze_colors": true,
    "auto_classify_shots": true,
    "auto_transcribe": true
  },
  "transcription": {
    "model": "small.en",
    "language": "en"
  },
  "export": {
    "quality": "medium",
    "resolution": "original",
    "fps": "original"
  },
  "appearance": {
    "theme_preference": "system"
  },
  "youtube": {
    "results_count": 25,
    "parallel_downloads": 2
  }
}
```

---

### Phase 5: MCP Server (Future Enhancement)

**Goal**: Create an MCP (Model Context Protocol) server for direct tool integration with AI agents.

#### Tools to Expose

```python
# MCP Tools

@mcp_tool
def detect_scenes(video_path: str, sensitivity: float = 3.0) -> dict:
    """Detect scenes in a video file."""
    ...

@mcp_tool
def export_clips(project_path: str, output_dir: str, clip_ids: list[str] = None) -> list[str]:
    """Export clips from a project."""
    ...

@mcp_tool
def search_youtube(query: str, max_results: int = 25) -> list[dict]:
    """Search YouTube for videos."""
    ...

@mcp_tool
def download_video(url: str, output_dir: str = None) -> str:
    """Download a video from URL."""
    ...

@mcp_tool
def analyze_clips(project_path: str, analysis_type: str) -> dict:
    """Run analysis (colors, shots, transcription) on project clips."""
    ...

@mcp_tool
def get_project_info(project_path: str) -> dict:
    """Get information about a project."""
    ...
```

#### Implementation

1. Depends on Phase 1 (CLI) being complete
2. Create `mcp/` directory with server implementation
3. MCP tools call CLI commands or core functions directly
4. Support both stdio and HTTP transports

---

## Success Criteria

### Parity Test

For each UI action, verify there's a CLI/API equivalent:

- [ ] Import video → `scene_ripper detect <video>`
- [ ] Scene detection → `scene_ripper detect`
- [ ] Color analysis → `scene_ripper analyze colors`
- [ ] Shot classification → `scene_ripper analyze shots`
- [ ] Transcription → `scene_ripper transcribe`
- [ ] Export clips → `scene_ripper export clips`
- [ ] Export dataset → `scene_ripper export dataset`
- [ ] Export EDL → `scene_ripper export edl`
- [ ] Export video → `scene_ripper export video`
- [ ] YouTube search → `scene_ripper search youtube`
- [ ] Download video → `scene_ripper download`
- [ ] Save project → Automatic on CLI operations
- [ ] Load project → All commands accept `<project.json>`

### Emergence Test

Can the agent compose tools for unanticipated requests?

**Test 1**: "Find Soviet animation from the 1980s, download 3 videos, detect scenes, and export all close-up shots"
```bash
scene_ripper search youtube "Soviet animation 1980s" --max-results 10 --output results.json
scene_ripper download --from-file <(jq -r '.[0:3] | .[].url' results.json) --output-dir ./soviet/
scene_ripper detect ./soviet/*.mp4 --output soviet-project.json
scene_ripper analyze shots soviet-project.json
scene_ripper export clips soviet-project.json --filter "shot_type=close-up" --output-dir ./closeups/
```

**Test 2**: "Create a 2-minute supercut of all dialogue scenes from my project"
```bash
scene_ripper transcribe project.json
scene_ripper export clips project.json --filter "has_speech=true" --max-duration 120 --output-dir ./dialogue/
scene_ripper export video project.json --clips-from ./dialogue/ --output supercut.mp4
```

**Test 3**: "Compare scene detection at different sensitivities"
```bash
for sens in 2.0 3.0 5.0; do
  scene_ripper detect video.mp4 --sensitivity $sens --output "project-sens-${sens}.json"
  echo "Sensitivity $sens: $(jq '.clips | length' project-sens-${sens}.json) clips"
done
```

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: CLI | 2-3 days | None |
| Phase 2: State Management | 3-4 days | None (can parallel with Phase 1) |
| Phase 3: Environment Variables | 0.5 day | None |
| Phase 4: JSON Settings | 1-2 days | Phase 3 |
| Phase 5: MCP Server | 2-3 days | Phase 1 |

**Total**: ~10-12 days for full agent-native transformation

---

## References

- [Agent-Native Architecture Guide](https://every.to/guides/agent-native)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [Click Documentation](https://click.palletsprojects.com/)
