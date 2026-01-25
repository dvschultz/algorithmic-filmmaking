---
title: "feat: CLI Interface (Agent-Native Phase 1)"
type: feat
date: 2026-01-25
priority: P1
depends_on: []
---

# feat: CLI Interface (Agent-Native Phase 1)

## Overview

Create a command-line interface using Click that exposes all core operations, enabling programmatic control of Scene Ripper without the Qt GUI. This is Phase 1 of the [Agent-Native Architecture Plan](./agent-native-architecture-plan.md).

**Goal:** Any action a user can perform via UI can also be performed by an agent via CLI.

## Problem Statement

All 18 Scene Ripper capabilities are currently locked behind the Qt UI event loop. Agents cannot:
- Detect scenes in videos without launching a GUI
- Run analysis pipelines in CI/CD or scheduled jobs
- Compose operations via shell scripts
- Integrate Scene Ripper into larger automated workflows

The core logic already exists in `core/` with clean separation from UI—it just needs a CLI entry point.

## Proposed Solution

Add a `cli/` package with Click-based commands that call existing `core/` functions directly. The CLI will:

1. Mirror all GUI operations with equivalent commands
2. Support JSON output for machine consumption (`--json`)
3. Use environment variables and config files (not QSettings)
4. Show progress on stderr (keep stdout clean for piping)
5. Follow Unix conventions for exit codes and composability

## Technical Approach

### Directory Structure

```
cli/
├── __init__.py
├── main.py              # Click app entry point, command groups
├── commands/
│   ├── __init__.py
│   ├── detect.py        # scene_ripper detect
│   ├── analyze.py       # scene_ripper analyze colors/shots
│   ├── transcribe.py    # scene_ripper transcribe
│   ├── export.py        # scene_ripper export clips/dataset/edl/video
│   ├── youtube.py       # scene_ripper search/download
│   └── project.py       # scene_ripper project info/list/add
└── utils/
    ├── __init__.py
    ├── output.py        # JSON/text output formatting
    ├── progress.py      # Progress bar utilities
    └── config.py        # CLI-specific config (env vars, TOML)
```

### Entry Point

```python
# cli/main.py
import click
from cli.commands import detect, analyze, transcribe, export, youtube, project

@click.group()
@click.version_option()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
@click.pass_context
def cli(ctx, output_json):
    """Scene Ripper - Video scene detection and analysis."""
    ctx.ensure_object(dict)
    ctx.obj['json'] = output_json

cli.add_command(detect.detect)
cli.add_command(analyze.analyze)
cli.add_command(transcribe.transcribe)
cli.add_command(export.export)
cli.add_command(youtube.search)
cli.add_command(youtube.download)
cli.add_command(project.project)

if __name__ == '__main__':
    cli()
```

### Command Mapping

| GUI Action | CLI Command | Core Function |
|------------|-------------|---------------|
| Import + detect | `scene_ripper detect <video>` | `SceneDetector.detect_scenes()` |
| Color analysis | `scene_ripper analyze colors <project>` | `extract_dominant_colors()` |
| Shot classification | `scene_ripper analyze shots <project>` | `classify_shot_type()` |
| Transcription | `scene_ripper transcribe <project>` | `transcribe_clip()` |
| Export clips | `scene_ripper export clips <project>` | `FFmpegProcessor.extract_clip()` |
| Export dataset | `scene_ripper export dataset <project>` | `export_dataset()` |
| Export EDL | `scene_ripper export edl <project>` | `export_edl()` |
| Export video | `scene_ripper export video <project>` | `export_sequence()` |
| YouTube search | `scene_ripper search <query>` | YouTube Data API |
| Download video | `scene_ripper download <url>` | `VideoDownloader.download()` |
| Project info | `scene_ripper project info <project>` | `load_project()` |
| List clips | `scene_ripper project list-clips <project>` | `load_project()` |
| Add to sequence | `scene_ripper project add-to-sequence <project>` | Project mutation |

### Settings Architecture (Qt-Free)

Create `cli/utils/config.py` that reads settings without Qt dependency:

```python
# cli/utils/config.py
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

def get_config_dir() -> Path:
    """Get config directory following XDG spec on Linux."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home()))
    else:  # macOS/Linux
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return base / 'scene-ripper'

def get_config_path() -> Path:
    return get_config_dir() / 'config.json'

@dataclass
class CLIConfig:
    """CLI configuration, independent of Qt."""
    # Paths
    download_dir: Optional[Path] = None
    export_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    # Detection defaults
    default_sensitivity: float = 3.0
    min_scene_length_seconds: float = 0.5

    # Transcription defaults
    transcription_model: str = "small.en"
    transcription_language: str = "en"

    # Export defaults
    export_quality: str = "medium"  # high, medium, low
    export_resolution: str = "original"  # original, 1080p, 720p, 480p

    # YouTube
    youtube_api_key: Optional[str] = None
    youtube_results_count: int = 25

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load config with priority: env vars > config file > defaults."""
        config = cls()

        # Load from config file if exists
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

        # Override with environment variables
        if api_key := os.environ.get('YOUTUBE_API_KEY'):
            config.youtube_api_key = api_key
        if cache_dir := os.environ.get('SCENE_RIPPER_CACHE_DIR'):
            config.cache_dir = Path(cache_dir)
        if download_dir := os.environ.get('SCENE_RIPPER_DOWNLOAD_DIR'):
            config.download_dir = Path(download_dir)
        if export_dir := os.environ.get('SCENE_RIPPER_EXPORT_DIR'):
            config.export_dir = Path(export_dir)

        return config
```

### Exit Codes

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Operation completed |
| 1 | General error | Unexpected exception |
| 2 | Usage error | Invalid arguments |
| 3 | File not found | Input video/project missing |
| 4 | Dependency missing | FFmpeg/faster-whisper not installed |
| 5 | Network error | YouTube API/download failure |
| 6 | Permission error | Cannot write to output path |
| 7 | Validation error | Invalid sensitivity value |

```python
# cli/utils/errors.py
import sys
from enum import IntEnum

class ExitCode(IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 2
    FILE_NOT_FOUND = 3
    DEPENDENCY_MISSING = 4
    NETWORK_ERROR = 5
    PERMISSION_ERROR = 6
    VALIDATION_ERROR = 7

def exit_with(code: ExitCode, message: str = None):
    if message:
        click.echo(message, err=True)
    sys.exit(code)
```

### Progress Display

Progress goes to stderr so stdout stays clean for piping:

```python
# cli/utils/progress.py
import click
import sys

def create_progress_callback(label: str):
    """Create a progress callback compatible with core/ functions."""
    if not sys.stderr.isatty():
        # Non-interactive: just log status changes
        last_status = [None]
        def callback(progress: float, status: str):
            if status != last_status[0]:
                click.echo(f"{label}: {status}", err=True)
                last_status[0] = status
        return callback
    else:
        # Interactive: use progress bar
        bar = click.progressbar(length=100, label=label, file=sys.stderr)
        bar.__enter__()
        def callback(progress: float, status: str):
            bar.update(int(progress * 100) - bar.pos)
        return callback
```

### Output Formatting

```python
# cli/utils/output.py
import json
import click
from typing import Any
from dataclasses import asdict, is_dataclass

def output_result(data: Any, as_json: bool = False):
    """Output data in requested format."""
    if as_json:
        if is_dataclass(data):
            data = asdict(data)
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                click.echo(f"{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                click.echo(item)
        else:
            click.echo(str(data))
```

### Example Command Implementation

```python
# cli/commands/detect.py
import click
from pathlib import Path
from core.scene_detect import SceneDetector, DetectionConfig
from core.project import save_project
from models.clip import Source, Clip
from cli.utils.output import output_result
from cli.utils.progress import create_progress_callback
from cli.utils.errors import ExitCode, exit_with
from cli.utils.config import CLIConfig

@click.command()
@click.argument('video', type=click.Path(exists=True, path_type=Path))
@click.option('--sensitivity', '-s', default=3.0,
              help='Detection sensitivity (1.0=sensitive, 10.0=less sensitive)')
@click.option('--min-scene-length', '-m', default=0.5,
              help='Minimum scene length in seconds')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output project file path')
@click.option('--force', '-f', is_flag=True,
              help='Overwrite existing output file')
@click.pass_context
def detect(ctx, video: Path, sensitivity: float, min_scene_length: float,
           output: Path, force: bool):
    """Detect scenes in a video file.

    Creates a project file with detected clips.

    Examples:

        scene_ripper detect movie.mp4

        scene_ripper detect movie.mp4 -s 5.0 -o my_project.json
    """
    # Validate sensitivity range
    if not 1.0 <= sensitivity <= 10.0:
        exit_with(ExitCode.VALIDATION_ERROR,
                  f"Sensitivity must be between 1.0 and 10.0, got {sensitivity}")

    # Determine output path
    if output is None:
        output = video.with_suffix('.json')

    # Check for existing file
    if output.exists() and not force:
        exit_with(ExitCode.VALIDATION_ERROR,
                  f"Output file exists: {output}. Use --force to overwrite.")

    # Create detector and config
    config = DetectionConfig(
        threshold=sensitivity,
        min_scene_length=int(min_scene_length * 30),  # Convert to frames
        use_adaptive=True
    )

    detector = SceneDetector()
    progress = create_progress_callback("Detecting scenes")

    try:
        # Run detection
        scenes = detector.detect_scenes_with_progress(
            str(video),
            config=config,
            progress_callback=progress
        )

        # Create source and clips
        source = Source(path=video.resolve())
        clips = [
            Clip(
                source_path=video.resolve(),
                start_time=scene[0],
                end_time=scene[1],
                source_id=source.id
            )
            for scene in scenes
        ]

        # Save project
        save_project(
            path=output,
            sources={source.id: source},
            clips=clips,
            sequence=[]
        )

        # Output result
        result = {
            "video": str(video),
            "output": str(output),
            "clips_detected": len(clips),
            "sensitivity": sensitivity
        }
        output_result(result, as_json=ctx.obj.get('json', False))

    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Video not found: {video}")
    except Exception as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Detection failed: {e}")
```

## Acceptance Criteria

### Functional Requirements

- [x] All 13 commands implemented and tested
- [x] `--json` flag produces valid, parseable JSON for all commands
- [x] Progress displayed on stderr, results on stdout
- [x] Commands are composable via pipes
- [x] Exit codes match specification

### Non-Functional Requirements

- [x] No PySide6/Qt imports in CLI modules
- [x] Works on headless Linux servers
- [ ] Ctrl+C cleanly terminates with partial progress saved
- [x] CLI starts in <1 second (lazy load heavy dependencies)

### Quality Gates

- [ ] `mypy cli/` passes with no errors
- [x] Unit tests for each command
- [ ] Integration test: full pipeline from detect to export

## Implementation Phases

### Phase 1a: Foundation (First)

Create CLI infrastructure:

1. `cli/__init__.py` - Package init
2. `cli/main.py` - Click app with command groups
3. `cli/utils/config.py` - Qt-free settings loader
4. `cli/utils/errors.py` - Exit codes and error handling
5. `cli/utils/output.py` - JSON/text output formatting
6. `cli/utils/progress.py` - Progress bar utilities
7. Update `requirements.txt` to add `click>=8.0`
8. Add entry point to `pyproject.toml` or `setup.py`

**Files to create:**
- `cli/__init__.py`
- `cli/main.py`
- `cli/utils/__init__.py`
- `cli/utils/config.py`
- `cli/utils/errors.py`
- `cli/utils/output.py`
- `cli/utils/progress.py`

### Phase 1b: Core Commands

Implement scene detection and project management:

1. `cli/commands/__init__.py`
2. `cli/commands/detect.py` - Scene detection
3. `cli/commands/project.py` - Project info, list-clips, add-to-sequence

**Files to create:**
- `cli/commands/__init__.py`
- `cli/commands/detect.py`
- `cli/commands/project.py`

### Phase 1c: Analysis Commands

Implement analysis operations:

1. `cli/commands/analyze.py` - Color extraction, shot classification
2. `cli/commands/transcribe.py` - Whisper transcription

**Files to create:**
- `cli/commands/analyze.py`
- `cli/commands/transcribe.py`

### Phase 1d: Export Commands

Implement export operations:

1. `cli/commands/export.py` - clips, dataset, edl, video

**Files to create:**
- `cli/commands/export.py`

### Phase 1e: YouTube Commands

Implement YouTube integration:

1. `cli/commands/youtube.py` - search, download

**Files to create:**
- `cli/commands/youtube.py`

### Phase 1f: Polish

1. Shell completion support
2. Man page / documentation
3. Comprehensive tests

## Success Metrics

### Parity Test

Each UI action has a working CLI equivalent:

- [x] Import video -> `scene_ripper detect <video>`
- [x] Scene detection -> `scene_ripper detect`
- [x] Color analysis -> `scene_ripper analyze colors`
- [x] Shot classification -> `scene_ripper analyze shots`
- [x] Transcription -> `scene_ripper transcribe`
- [x] Export clips -> `scene_ripper export clips`
- [x] Export dataset -> `scene_ripper export dataset`
- [x] Export EDL -> `scene_ripper export edl`
- [x] Export video -> `scene_ripper export video`
- [x] YouTube search -> `scene_ripper search`
- [x] Download video -> `scene_ripper download`
- [x] Project info -> `scene_ripper project info`
- [x] List clips -> `scene_ripper project list-clips`

### Emergence Test

Can agents compose tools for unanticipated requests?

**Test 1:** "Find Soviet animation, download 3 videos, detect scenes, export close-ups"
```bash
scene_ripper search "Soviet animation 1980s" --json | \
  jq -r '.results[:3][].url' | \
  xargs -I {} scene_ripper download {} --output-dir ./soviet/

for video in ./soviet/*.mp4; do
  scene_ripper detect "$video" -o "${video%.mp4}.json"
  scene_ripper analyze shots "${video%.mp4}.json"
done

scene_ripper export clips soviet/*.json --filter "shot_type=close-up" --output-dir ./closeups/
```

**Test 2:** "Create supercut of dialogue scenes"
```bash
scene_ripper transcribe project.json
scene_ripper project list-clips project.json --json | \
  jq -r '.clips[] | select(.transcript != null) | .id' | \
  xargs scene_ripper project add-to-sequence project.json
scene_ripper export video project.json -o supercut.mp4
```

## Dependencies to Add

```
# requirements.txt additions
click>=8.0
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Core functions have hidden Qt dependencies | CLI won't work headless | Audit all imports in core/, create shims if needed |
| Long operations block without progress | Poor UX | Use progress callbacks throughout |
| Model downloads fail silently | Analysis commands unusable | Add `--check-deps` command, clear error messages |
| Path handling differs from GUI | Project files incompatible | Use same path resolution logic from `core/project.py` |

## Open Questions (Resolved)

1. **Re-running detect on existing project?** -> Error by default, `--force` to overwrite, future: `--merge`
2. **Progress for non-TTY?** -> Log status changes to stderr without progress bars
3. **Interrupted operations?** -> Save progress per-clip, skip completed on re-run
4. **Partial failures?** -> Continue, report at end, exit code 1 if any failures
5. **Clip selection syntax?** -> Positional IDs, `--all`, `--filter` for criteria

## References

### Internal

- [Agent-Native Architecture Plan](./agent-native-architecture-plan.md)
- [Subprocess Cleanup Pattern](../solutions/reliability-issues/subprocess-cleanup-on-exception.md)
- [FFmpeg Path Escaping](../solutions/security-issues/ffmpeg-path-escaping-20260124.md)
- Core modules: `core/scene_detect.py`, `core/project.py`, `core/dataset_export.py`

### External

- [Click Documentation](https://click.palletsprojects.com/en/8.1.x/)
- [XDG Base Directory Spec](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
