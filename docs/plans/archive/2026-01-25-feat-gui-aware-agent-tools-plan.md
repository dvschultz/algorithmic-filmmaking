---
title: "feat: GUI-Aware Agent Tools"
type: feat
date: 2026-01-25
priority: 1
---

# GUI-Aware Agent Tools

## Overview

Address the primary gap in agent-native architecture: CLI-based tools that don't update the live GUI project. This plan implements GUI-aware versions of critical tools that trigger the same workers used by the GUI, ensuring the agent and user share a synchronized workspace.

**Problem**: Current `detect_scenes`, `analyze_colors`, `analyze_shots`, and `transcribe` tools operate via CLI and write to files. The live GUI project is not updated, breaking the shared workspace principle.

**Solution**: Create new GUI-state-modifying tools that trigger the same QThread workers used by the GUI, with proper signal handling to update the live project.

## Acceptance Criteria

### Phase 1: Import Tool
- [x] `import_video(path: str)` - Import local video file to library
- [x] `select_source(source_id: str)` - Select a source as current

### Phase 2: GUI-Aware Detection
- [x] `detect_scenes_live(source_id: str, sensitivity: float)` - Detect scenes with live GUI update
- [x] Wait for worker completion before returning result
- [x] Update project with detected clips and metadata

### Phase 3: GUI-Aware Analysis
- [x] `analyze_colors_live(clip_ids: list[str])` - Extract colors with live GUI update
- [x] `analyze_shots_live(clip_ids: list[str])` - Classify shots with live GUI update
- [x] `transcribe_live(clip_ids: list[str])` - Transcribe with live GUI update
- [x] `analyze_all_live(clip_ids: list[str])` - Run all analysis sequentially

## Technical Design

### Architecture Pattern

GUI-aware tools follow a synchronous-async bridge pattern:

```
Agent Thread                    Main Thread (Qt)
     │                               │
     ├─ gui_tool_requested ─────────>│
     │                               ├─ Start QThread Worker
     │                               │      │
     │   (agent waits on Event)      │      ├─ Worker runs
     │                               │      │
     │                               │<─────┴─ Worker finished signal
     │                               ├─ Collect results
     │<─────── set_gui_tool_result ──┤
     │                               │
     ├─ Return result to LLM         │
```

### Key Implementation Details

#### 1. Worker Completion Waiting

GUI-aware tools that start workers need to wait for completion. The main window must:
1. Store the tool_call_id when starting the worker
2. Connect worker's `finished` signal to a handler that calls `set_gui_tool_result()`
3. The agent thread waits on `_gui_tool_event` (already implemented)

#### 2. Guard Flags

Each worker operation has a guard flag to prevent duplicate signal handling:
```python
self._detection_finished_handled = False
self._color_analysis_finished_handled = False
self._shot_type_finished_handled = False
self._transcription_finished_handled = False
```

New GUI tool handlers must reset these guards before starting workers.

#### 3. Source Reference

Workers need access to `Source` objects, not just IDs. Tools must resolve source_id to Source before triggering workers.

## Implementation Plan

### Phase 1: Import and Select Tools

#### 1.1 `import_video` Tool

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Import a local video file into the project library.",
    requires_project=True,
    modifies_gui_state=True
)
def import_video(main_window, path: str) -> dict:
    """Import a video file to the library.

    Args:
        path: Absolute path to the video file

    Returns:
        Dict with source_id and metadata if successful
    """
    from pathlib import Path

    video_path = Path(path)
    if not video_path.exists():
        return {"success": False, "error": f"File not found: {path}"}

    if not video_path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
        return {"success": False, "error": f"Unsupported video format: {video_path.suffix}"}

    # Check if already in library
    for source in main_window.project.sources:
        if source.file_path == video_path:
            return {
                "success": True,
                "source_id": source.id,
                "message": "Video already in library",
                "filename": source.filename
            }

    # Add to library (reuse existing method)
    main_window._add_video_to_library(video_path)

    # Find the newly added source
    new_source = None
    for source in main_window.project.sources:
        if source.file_path == video_path:
            new_source = source
            break

    if new_source:
        return {
            "success": True,
            "source_id": new_source.id,
            "filename": new_source.filename,
            "message": f"Imported {new_source.filename}"
        }
    else:
        return {"success": False, "error": "Failed to add video to library"}
```

#### 1.2 `select_source` Tool

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Select a video source as the current active source for detection and editing.",
    requires_project=True,
    modifies_gui_state=True
)
def select_source(main_window, source_id: str) -> dict:
    """Select a source as the current active source.

    Args:
        source_id: ID of the source to select

    Returns:
        Dict with success status and source info
    """
    # Find source
    source = None
    for s in main_window.project.sources:
        if s.id == source_id:
            source = s
            break

    if not source:
        return {"success": False, "error": f"Source not found: {source_id}"}

    # Select it
    main_window._select_source(source)

    return {
        "success": True,
        "source_id": source.id,
        "filename": source.filename,
        "duration_seconds": source.duration_seconds,
        "clip_count": len(main_window.project.clips_by_source.get(source.id, []))
    }
```

### Phase 2: GUI-Aware Detection

#### 2.1 `detect_scenes_live` Tool

This is more complex because we need to wait for the worker to complete.

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Detect scenes in a video source. Updates the live project with detected clips. "
                "This may take a while for long videos.",
    requires_project=True,
    modifies_gui_state=True
)
def detect_scenes_live(
    main_window,
    source_id: str,
    sensitivity: float = 3.0
) -> dict:
    """Detect scenes in a source video with live GUI update.

    Args:
        source_id: ID of the source to analyze
        sensitivity: Detection sensitivity (1.0=sensitive, 10.0=less sensitive)

    Returns:
        Dict with detected clip count and IDs
    """
    # Find source
    source = None
    for s in main_window.project.sources:
        if s.id == source_id:
            source = s
            break

    if not source:
        return {"success": False, "error": f"Source not found: {source_id}"}

    # Check if detection already running
    if main_window.detection_worker and main_window.detection_worker.isRunning():
        return {"success": False, "error": "Scene detection already in progress"}

    # Set current source and start detection
    main_window._select_source(source)

    # Store that we're waiting for detection result
    # The actual detection is triggered via _start_detection
    # We need to signal the main_window to start detection and wait
    main_window._pending_detection_tool_result = True
    main_window._start_detection(sensitivity)

    # Return a marker that tells the GUI tool handler to wait for detection
    return {"_wait_for_detection": True, "source_id": source_id}
```

**File**: `ui/main_window.py` - Modify `_on_gui_tool_requested`

Add handling for tools that need to wait for worker completion:

```python
@Slot(str, dict, str)
def _on_gui_tool_requested(self, tool_name: str, args: dict, tool_call_id: str):
    """Execute a GUI-modifying tool on the main thread."""
    # ... existing code ...

    try:
        result = tool.func(**args)

        # Check if tool needs to wait for async worker
        if isinstance(result, dict) and result.get("_wait_for_detection"):
            # Store tool_call_id for when detection completes
            self._pending_detection_tool_call_id = tool_call_id
            # Don't call set_gui_tool_result yet - detection handler will do it
            return

        # ... rest of existing handling ...
```

**File**: `ui/main_window.py` - Modify `_on_detection_finished`

```python
@Slot()
def _on_detection_finished(self, source: Source, clips: list[Clip]):
    # ... existing code ...

    # If agent is waiting for detection result, send it
    if hasattr(self, '_pending_detection_tool_call_id') and self._pending_detection_tool_call_id:
        result = {
            "success": True,
            "source_id": source.id,
            "clip_count": len(clips),
            "clip_ids": [c.id for c in clips],
            "duration_seconds": source.duration_seconds,
            "fps": source.fps
        }
        if self._chat_worker:
            self._chat_worker.set_gui_tool_result(result)
        self._pending_detection_tool_call_id = None
```

### Phase 3: GUI-Aware Analysis Tools

#### 3.1 `analyze_colors_live` Tool

```python
@tools.register(
    description="Extract dominant colors from clips. Updates the live project.",
    requires_project=True,
    modifies_gui_state=True
)
def analyze_colors_live(main_window, clip_ids: list[str]) -> dict:
    """Extract dominant colors from clips with live GUI update.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with analysis results
    """
    # Resolve clips
    clips = []
    for clip_id in clip_ids:
        clip = main_window.project.clips_by_id.get(clip_id)
        if clip:
            clips.append(clip)

    if not clips:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.color_worker and main_window.color_worker.isRunning():
        return {"success": False, "error": "Color analysis already in progress"}

    # Start analysis and mark for waiting
    main_window._pending_color_tool_result = True
    main_window._analyze_clips_colors(clips)  # Need to add this method

    return {"_wait_for_color_analysis": True, "clip_count": len(clips)}
```

Similar pattern for `analyze_shots_live` and `transcribe_live`.

#### 3.2 `analyze_all_live` Tool

```python
@tools.register(
    description="Run all analysis (colors, shots, transcription) on clips sequentially.",
    requires_project=True,
    modifies_gui_state=True
)
def analyze_all_live(main_window, clip_ids: list[str]) -> dict:
    """Run all analysis on clips with live GUI update.

    Runs colors, shots, and transcription sequentially.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with analysis summary
    """
    # Resolve clips
    clips = []
    for clip_id in clip_ids:
        clip = main_window.project.clips_by_id.get(clip_id)
        if clip:
            clips.append(clip)

    if not clips:
        return {"success": False, "error": "No valid clips found"}

    # Use existing analyze_all mechanism
    main_window._analyze_all_pending = ["colors", "shots", "transcribe"]
    main_window._analyze_all_clips = clips
    main_window._pending_analyze_all_tool_result = True
    main_window._start_next_analyze_all_step()

    return {"_wait_for_analyze_all": True, "clip_count": len(clips)}
```

## Signal Flow Diagram

```
Agent calls detect_scenes_live(source_id="abc", sensitivity=3.0)
    │
    ├─> gui_tool_requested signal to MainWindow
    │       │
    │       ├─> _on_gui_tool_requested()
    │       │       │
    │       │       ├─> Calls detect_scenes_live()
    │       │       │       │
    │       │       │       ├─> Validates source exists
    │       │       │       ├─> Calls _start_detection(3.0)
    │       │       │       └─> Returns {_wait_for_detection: True}
    │       │       │
    │       │       ├─> Sees _wait_for_detection flag
    │       │       ├─> Stores tool_call_id
    │       │       └─> Does NOT call set_gui_tool_result yet
    │       │
    │       └─> DetectionWorker starts in background
    │               │
    │               ├─> ... detection runs ...
    │               │
    │               └─> finished signal
    │                       │
    │                       └─> _on_detection_finished()
    │                               │
    │                               ├─> Updates project with clips
    │                               ├─> Starts thumbnail generation
    │                               ├─> Checks _pending_detection_tool_call_id
    │                               └─> Calls set_gui_tool_result(result)
    │
    └─> Agent receives result with clip_count, clip_ids
```

## Testing Plan

### Test 1: Import Video
```
"Import the video at /path/to/test.mp4"
```
Expected: Video appears in library grid, source_id returned

### Test 2: Detection
```
"Detect scenes in the first video with sensitivity 2.5"
```
Expected:
- Progress bar shows detection progress
- Clips appear in Cut tab when done
- Agent receives clip count and IDs

### Test 3: Analysis Pipeline
```
"Analyze colors and shot types for all clips"
```
Expected:
- Colors extracted and visible in UI
- Shot types classified and visible
- Agent receives summary

### Test 4: Full Workflow
```
"Import video.mp4, detect scenes, and analyze all clips"
```
Expected:
- Video imported
- Scenes detected with thumbnails
- All analysis completed
- Project fully populated

## Dependencies

- Existing worker classes: `DetectionWorker`, `ColorAnalysisWorker`, `ShotTypeWorker`, `TranscriptionWorker`
- GUI tool infrastructure: `gui_tool_requested` signal, `set_gui_tool_result()` method
- Project model: `Project.add_source()`, `Project.add_clip()`

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Worker timeout | Add configurable timeout with informative error message |
| Worker crash | Wrap in try/catch, return error to agent |
| Concurrent tool calls | Check worker.isRunning() before starting |
| Memory with large videos | Use existing chunked processing in workers |

## Success Metrics

- Agent can complete full video import → detection → analysis workflow
- GUI stays synchronized throughout
- No project reload needed between agent operations
- Agent-native score improves from 35/50 to ~42/50

## References

- Guard flag pattern: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- Agent-GUI sync: `docs/plans/2026-01-25-feat-agent-gui-bidirectional-sync-plan.md`
- Worker definitions: `ui/main_window.py:72-369`
- Detection trigger: `ui/main_window.py:2035`
- Color analysis trigger: `ui/main_window.py:1405`
