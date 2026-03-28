# Scene Ripper - Project Conventions

## Overview

Video scene detection, analysis, and algorithmic editing application with 16 sequencer algorithms, integrated AI agent, and MCP server. Dependencies auto-install on demand via `core/feature_registry.py`.

## Technology Stack

- **UI**: PySide6 (Qt 6), mpv (video playback)
- **Scene Detection**: PySceneDetect (AdaptiveDetector)
- **Video Processing**: FFmpeg (subprocess, argument arrays only)
- **Video Download**: yt-dlp
- **Transcription**: faster-whisper, lightning-whisper-mlx (Apple Silicon)
- **Vision/ML**: YOLO (objects), InsightFace (faces), DINOv2/CLIP (embeddings), PaddleOCR (text), mlx-vlm (local VLM)
- **Audio**: librosa (analysis), Demucs (stem separation)
- **LLM**: LiteLLM (multi-provider)
- **YouTube**: google-api-python-client
- **MCP Server**: `scene_ripper_mcp/` (3 files)
- **Python**: 3.11+ | **Deps**: `requirements-core.txt` (always), `requirements-optional.txt` (heavy ML)

## Project Structure

```
main.py                          # GUI entry point
pyproject.toml                   # Metadata, CLI + MCP entry points
ui/                  (19 files)  # Main window, chat, player, browser, theme
  tabs/              (8 tabs)    # collect, cut, analyze, frames, sequence, generate, render
  dialogs/           (12 files)  # Algorithm-specific config dialogs
  workers/           (14 files)  # QThread workers (base.py = CancellableWorker)
  widgets/           (16 files)  # Cards, grids, timeline preview, empty states
  timeline/          (7 files)   # Timeline widget, tracks, clips, playhead
core/                (34 files)  # Business logic, FFmpeg, settings, project, LLM
  analysis/          (15 files)  # Color, shots, brightness, volume, embeddings, OCR, faces, cinematography
  remix/             (14 files)  # Sequencer algorithms (one file per algorithm)
models/              (7 files)   # Source, Clip, Frame, SequenceClip, Sequence, CinematographyAnalysis, Plan
cli/                 (15 files)  # Click CLI with detect, analyze, transcribe, youtube, export commands
scene_ripper_mcp/    (3 files)   # MCP server for external agent access
tests/               (56 files)  # 1153 tests
docs/user-guide/                 # End-user documentation
```

## Running the App

```bash
pip install -r requirements-core.txt  # Core deps (optional ML deps install on demand)
python main.py                        # GUI
python -m cli.main --help             # CLI
scene-ripper-mcp                      # MCP server (after pip install -e .[mcp])
```

## Data Model

```
Source → scene detection → Clip[] → analysis → enriched Clip[]
                                                      ↓
Frame[] (extracted images)                    add to sequence
                                                      ↓
                                              SequenceClip[] → render → output
```

**Source** (`models/clip.py`): `id`, `file_path`, `fps`, `duration_seconds`, `width`, `height`, `analyzed`, `color_profile`

**Clip** (`models/clip.py`): `id`, `source_id`, `start_frame`, `end_frame`, `name`, `disabled`, `thumbnail_path`, `dominant_colors`, `shot_type`, `transcript`, `tags`, `notes`, `object_labels`, `detected_objects`, `face_embeddings`, `person_count`, `description`, `description_model`, `extracted_texts`, `cinematography`, `average_brightness`, `rms_volume`, `embedding`, `first_frame_embedding`, `last_frame_embedding`, `embedding_model`

**Frame** (`models/frame.py`): `id`, `file_path`, `source_id`, `clip_id`, `frame_number`, `width`, `height`, analysis fields mirror Clip

**SequenceClip** (`models/sequence.py`): `id`, `source_clip_id`, `source_id`, `frame_id`, `track_index`, `start_frame`, `in_point`, `out_point`, `hold_frames`, `hflip`, `vflip`, `reverse`, `prerendered_path`

**Project** (`core/project.py`): Single source of truth. Always use `add_source()`, `add_clips()`, etc. — never append directly. Methods invalidate caches (`sources_by_id`, `clips_by_id`, `clips_by_source`) and notify observers.

## Tabs

| Tab | Purpose |
|-----|---------|
| **Collect** | Import local videos, search/download from YouTube |
| **Cut** | Scene detection (sensitivity 1.0-10.0), clip browsing |
| **Analyze** | Describe, classify shots, detect objects/faces, OCR, colors, transcribe, cinematography |
| **Frames** | Extract and browse individual frames from clips |
| **Sequence** | Card-based sorting with 16 algorithms, drag-drop reorder, filter by metadata |
| **Generate** | Generation tools |
| **Render** | Export as MP4, EDL, SRT, individual clips, dataset bundles |

Sequencer algorithm reference: see `.claude/rules/sequencer-algorithms.md` (loads automatically when editing remix/dialog files).

## Key Patterns

### Background Workers
All workers inherit `CancellableWorker` (`ui/workers/base.py`). Workers emit `progress(n, total)`, `clip_ready(clip)`, `error(message)`, `finished()`. Main thread updates UI. User can cancel.

### Feature Registry
`core/feature_registry.py` maps features to binary/package dependencies. Call `check_feature(name)` to test availability, `install_for_feature(name)` to auto-install. The UI shows install prompts when deps are missing.

### FFmpeg Safety
Always use argument arrays, never shell interpolation. Validate paths before processing.

### User Settings
Always use `core/settings.py` (`load_settings()`). Key paths: `settings.download_dir`, `settings.project_dir`, `settings.cache_dir`, `settings.export_dir`. Never hardcode paths.

### Agent/LLM Integration
- `core/chat_tools.py` — tool definitions
- `core/tool_executor.py` — execution engine
- `core/llm_client.py` — LiteLLM abstraction
- `core/gui_state.py` — GUI state tracking for agent context
- `core/plan_controller.py` — multi-step plan execution
- `ui/chat_panel.py` / `ui/chat_worker.py` — chat UI and background worker

Agent capabilities mirror user capabilities (navigate tabs, select clips, trigger analysis, modify sequence). Tools return `{"success": True/False, "result": data}`.

### MCP Server
`scene_ripper_mcp/server.py` exposes project operations to external agents. Entry point: `scene-ripper-mcp` (defined in `pyproject.toml`).

### Card-Based UI
Sequence tab uses `ui/widgets/sorting_card.py` / `sorting_card_grid.py` for visual clip arrangement with drag-drop.

### UI Consistency
See `.claude/rules/ui-consistency.md` (loads automatically when editing ui/ files).

## Testing

56 test files, 1153 tests. Run: `pytest tests/` or `pytest tests/test_specific.py -v`.

### Bug Fixes: Prove It Pattern

1. **Reproduce** — Spawn a subagent to write a test that fails before the fix
2. **Fix** — Implement the fix
3. **Confirm** — Test passes, proving the fix works

If environment-specific, document why a test isn't feasible.

## Debugging

### Output-First Investigation
- Verify export/output logic **first** before blaming upstream (algorithms, generators)
- Trace data flow **backwards** from incorrect output
- Add logging to the exact function producing wrong output before investigating upstream

### Common Bug Patterns
- **Sequence overwrite**: Dialog workflows may have sequences overwritten by generic handlers
- **Empty API responses**: LLM APIs can return `None` content without exceptions — always validate
- **Worker state**: QThread workers may have stale references — check signal connections and lifecycle

### Bug Report Format
Confirm before investigating: (1) **Symptom** — what happens, (2) **Expected** — what should happen, (3) **Success criteria** — how to verify the fix. Ask if unclear.

## Building & Releasing

**Always use GitHub Actions CI for release builds** — never build locally for distribution. Local builds use ad-hoc code signing and skip notarization, so users will get Gatekeeper warnings.

```bash
# Trigger macOS CI build (proper signing + notarization):
gh workflow run build-macos.yml -f version=X.Y.Z

# Or push a tag to trigger automatically:
git tag vX.Y.Z && git push origin vX.Y.Z

# Monitor build:
gh run list --workflow=build-macos.yml --limit=1
gh run watch <run-id>
```

CI handles: Apple Developer code signing, notarization, Sparkle auto-updater appcast, DMG creation, smoke tests, and uploading assets to the GitHub release. The local `packaging/macos/build.sh` script is for development testing only.

## Common Commands

```bash
python main.py                        # GUI
python -m cli.main --help             # CLI
scene-ripper-mcp                      # MCP server
pytest tests/ -x -q                   # Tests (fast)
pytest tests/ -v                      # Tests (verbose)
```
