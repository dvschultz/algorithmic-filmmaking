# Scene Ripper - Project Conventions

## Overview

Video scene detection, analysis, and algorithmic editing application with 23 sequencer algorithms, integrated AI agent, and MCP server. Dependencies auto-install on demand via `core/feature_registry.py`.

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
- **MCP Server**: `scene_ripper_mcp/` (server + tools/ + schemas/ + jobs/, stdio and HTTP transports)
- **Python**: 3.11+ | **Deps**: `requirements-core.txt` (always), `requirements-optional.txt` (heavy ML), `requirements.txt` (full source-dev set)

## Project Structure

```
main.py                          # GUI entry point
pyproject.toml                   # Metadata, CLI + MCP entry points
ui/                  (23 files)  # Main window, chat, player, browser, theme, algorithm_config
  models/            (3 files)   # Shared clip/frame item models owned by ProjectSignalAdapter (docs/architecture/library-models.md)
  tabs/              (6 wired)   # collect, cut, analyze, frames, sequence, render (generate_tab.py exists but is unmounted stub)
  dialogs/           (21 files)  # Algorithm-specific config dialogs, recipe inspect/regenerate
  workers/           (74 files)  # QThread workers (base.py = CancellableWorker + is_transient_provider_error)
  widgets/           (24 files)  # Cards, grids, timeline preview, A/B sequence comparison, empty states
  timeline/          (8 files)   # Timeline widget, tracks, clips, playhead
  session_history.py            # Qt actions projecting shared project history
core/                (71 files)  # Business logic, FFmpeg, settings, project, LLM, runtime_supervisor (managed workers), runtime_families (isolation seam)
  runtime_worker/    (6 files)   # Worker package run by the managed interpreter (stdlib-only at import; calls.py allowlists engine calls)
  analysis/          (18 files)  # Color, shots, brightness, volume, embeddings, OCR, faces, cinematography, gaze
  remix/             (30 files)  # Sequencer algorithm definitions; engine.py/registry.py hold the Qt-free algorithm registry (23 algorithms labeled in ui/algorithm_config.py)
  spine/             (30 files)  # GUI-agnostic shared tool implementations (no PySide6/mpv/av imports); runtime.py = native runtime profiles
models/              (11 files)  # Source, Clip, Frame, SequenceClip, Sequence, SequenceRecipe, AudioSource, CinematographyAnalysis, SequenceAnalysis, Plan
cli/                 (22 files)  # Click CLI with detect, analyze, transcribe, youtube, export, sequence, runtime commands
scene_ripper_mcp/                # MCP server for external agent access
  tools/                         #   Tool registrations (project / clips / sequence / analyze / export / youtube / jobs / runtime)
  schemas/                       #   Pydantic input schemas for MCP tools
  jobs/                          #   SQLite-backed jobs framework (store, runtime, per-project mutex)
  security.py, auth_snapshot.py  #   Auth/permission gating for headless callers
tests/              (404 files)  # ~6100 tests
docs/user-guide/                 # End-user documentation
docs/solutions/                  # Documented solutions (bugs, best practices), YAML frontmatter (module, tags, problem_type)
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

**Source** (`models/clip.py`): `id`, `file_path`, `fps`, `duration_seconds`, `width`, `height`, `cut`, `has_analysis`, `color_profile`

**Clip** (`models/clip.py`): `id`, `source_id`, `start_frame`, `end_frame`, `name`, `disabled`, `thumbnail_path`, `dominant_colors`, `shot_type`, `transcript`, `tags`, `notes`, `object_labels`, `detected_objects`, `face_embeddings`, `person_count`, `description`, `description_model`, `extracted_texts`, `cinematography`, `average_brightness`, `rms_volume`, `embedding`, `first_frame_embedding`, `last_frame_embedding`, `embedding_model`

**Frame** (`models/frame.py`): `id`, `file_path`, `source_id`, `clip_id`, `frame_number`, `width`, `height`, analysis fields mirror Clip

**SequenceRecipe** (`models/recipe.py`): `algorithm`, `algorithm_version`, `parameters`, `seed`, `inputs` (RecipeInput: clip/source ids, frame span, fps, analysis identities), `realized` (RealizedEntry: clip-relative range, transforms, rationale, provider output), `parent_id`, `provider_outputs`. Stored on `Sequence.recipe`; reconstruction replays `realized` without provider calls.

**SequenceClip** (`models/sequence.py`): `id`, `source_clip_id`, `source_id`, `frame_id`, `track_index`, `start_frame`, `in_point`, `out_point`, `hold_frames`, `hflip`, `vflip`, `reverse`, `prerendered_path`

**AudioSource** (`models/audio_source.py`): Imported audio files (music, podcast, voiceover). Not cut into clips and never appear in sequencer output — feed audio-consuming tools like Staccato and transcription.

**SequenceAnalysis** (`models/sequence_analysis.py`): Sequence-level metrics (pacing, continuity, visual consistency) computed across multiple clips. **Not persisted** — cached in memory, invalidated when sequence changes. Includes `GENRE_PACING_NORMS` for comparison.

**Project** (`core/project.py`): Single source of truth. Always use `add_source()`, `add_clips()`, `add_audio_source()`, etc. — never append directly. Methods invalidate caches (`sources_by_id`, `clips_by_id`, `clips_by_source`) and notify observers.

## Tabs

Six tabs are wired in `ui/main_window.py`: Collect, Cut, Analyze, Frames, Sequence, Render. `ui/tabs/generate_tab.py` exists but is **not mounted** — treat it as a stub.

| Tab | Purpose |
|-----|---------|
| **Collect** | Import local videos and audio, search/download from YouTube and Internet Archive |
| **Cut** | Scene detection (sensitivity 1.0-10.0), clip browsing |
| **Analyze** | Describe, classify shots, detect objects/faces, OCR, colors, transcribe, cinematography, gaze |
| **Frames** | Extract and browse individual frames from clips |
| **Sequence** | Card-based sorting with 23 algorithms, drag-drop reorder, filter by metadata |
| **Render** | Export as MP4, EDL, SRT, individual clips, dataset bundles |

Sequencer algorithm reference: see `.claude/rules/sequencer-algorithms.md` (loads automatically when editing remix/dialog files).

## Key Patterns

### Background Workers
All workers inherit `CancellableWorker` (`ui/workers/base.py`). Workers emit `progress(n, total)`, `clip_ready(clip)`, `error(message)`, `finished()`. Main thread updates UI. User can cancel. VLM/LLM workers use `is_transient_provider_error()` from `ui/workers/base.py` to retry transient 429/5xx/network failures with exponential backoff; `summarize_clip_errors()` builds the user-facing batch error summary.

### Undo Commands
Clip enable/disable and manual sequence insertion/removal use Qt-free commands in `core/commands/` through `Project.session`. Call `Project.set_clips_disabled()` or `toggle_clips_disabled()`; the Edit menu and chat share this history through `ui/session_history.py`. Use `Project.insert_sequence_clips()` / `remove_from_sequence()` for timeline membership edits, and `reorder_sequence()` / `update_sequence_clip()` for order, timing, track and transform edits. Drag previews must remain detached until release. Model refreshes must not emit edit signals or call `mark_dirty()`. Sequence creation/deletion and settings use `add_sequence(activate=True)`, `remove_sequence()`, `rename_sequence()` and `update_sequence_metadata()`. Other editorial actions are still being migrated. Keep legacy mutations on `mark_dirty()` even when already dirty so undo cannot hide unrelated unsaved edits. See `docs/architecture/project-sessions.md`.

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
`scene_ripper_mcp/server.py` exposes project operations to external agents. Entry point: `scene-ripper-mcp` (defined in `pyproject.toml`). Supports stdio and HTTP transports (`--transport stdio` / `--transport http --port 8765`). Long-running ops (scene detection, analysis, downloads) are split into `start_*` / `get_job_status` / `get_job_result` / `cancel_job` tools backed by the `scene_ripper_mcp/jobs/` framework (SQLite store, per-project mutex, ThreadPoolExecutor runtime). Tool input shapes are pinned by Pydantic schemas in `scene_ripper_mcp/schemas/`. Permission gating lives in `security.py` and `auth_snapshot.py`. Caller-facing reference: `docs/user-guide/headless-mcp.md`.

### Spine Layering
`core/spine/` is the GUI-agnostic shared layer below both `core/chat_tools.py` (the GUI agent) and `scene_ripper_mcp/tools/*` (the MCP server). Both surfaces import from spine; neither imports the other. Spine modules MUST NOT import PySide6, mpv, av, faster_whisper, paddleocr, or mlx_vlm at module top level — `tests/test_spine_imports.py` is the boundary test that enforces this. Heavy or GUI-bound deps go inside function bodies (lazy import). New project-only tool implementations belong in `core/spine/<topic>.py`; the chat-tools and MCP wrappers are thin delegations.

### Card-Based UI
Sequence tab uses `ui/widgets/sorting_card.py` / `sorting_card_grid.py` for visual clip arrangement with drag-drop.

### UI Consistency
See `.claude/rules/ui-consistency.md` (loads automatically when editing ui/ files).

## Testing

404 test files, ~6100 tests in `tests/`. Plus a separate MCP suite under `scene_ripper_mcp/tests/`. Run: `pytest tests/`, `pytest scene_ripper_mcp/tests/`, or `pytest tests/test_specific.py -v`.

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
