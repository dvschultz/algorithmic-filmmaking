---
title: "Agent-Native Architecture Review"
type: review
date: 2026-01-25
score: 35/50
---

# Agent-Native Architecture Review

## Summary

Scene Ripper demonstrates a solid foundation for agent-native architecture with a well-designed tool registry and executor pattern. However, there are significant gaps in action parity - several GUI-only features lack agent equivalents, and some context is not fully exposed to the agent. The system prompt includes runtime context which is good, but the tool surface is incomplete relative to the full GUI capability set.

**Overall Score: 35/50 - NEEDS WORK**

The foundation is solid, but the disconnect between CLI-based tools and the live GUI project is the primary barrier. The agent can read and modify project state well, but heavy operations (scene detection, analysis) don't integrate with the GUI workflow.

## Capability Map

| UI Action | Agent Tool | Status |
|-----------|------------|--------|
| **Collect Tab** |
| Import video from file | None | **Missing** |
| Import from URL | `download_video` | Working |
| YouTube search | `search_youtube` | Working |
| Bulk download videos | None | **Missing** |
| Cut new videos (batch analyze) | None | **Missing** |
| Select source | None | **Missing** |
| **Cut Tab** |
| Detect scenes | `detect_scenes` (CLI) | **Partial** |
| Set sensitivity | None | **Missing** |
| Select clips | `select_clips` | Working |
| Send to Analyze | None | **Missing** |
| Drag to timeline | `add_to_sequence` | Working |
| **Analyze Tab** |
| Extract colors | `analyze_colors` (CLI) | **Partial** |
| Classify shots | `analyze_shots` (CLI) | **Partial** |
| Transcribe | `transcribe` (CLI) | **Partial** |
| Analyze all | None | **Missing** |
| Clear clips | None | **Missing** |
| **Sequence Tab** |
| Add clip to timeline | `add_to_sequence` | Working |
| Remove from sequence | `remove_from_sequence` | Working |
| Reorder sequence | `reorder_sequence` | Working |
| Clear sequence | `clear_sequence` | Working |
| Get sequence state | `get_sequence_state` | Working |
| Playback control | None | **Missing** |
| Set playhead | None | **Missing** |
| **Render Tab** |
| Export sequence | None | **Missing** |
| Export selected clips | `export_clips` (CLI) | **Partial** |
| Export all clips | `export_clips` (CLI) | **Partial** |
| Export dataset | None | **Missing** |
| Set quality/resolution/fps | None | **Missing** |
| Export EDL | `export_edl` | Working |
| **Project Management** |
| New project | `new_project` | Working |
| Open project | `load_project` | Working |
| Save project | `save_project` | Working |
| Save as | `save_project` (with path) | Working |
| **Clip Metadata** |
| Add tags | `add_tags` | Working |
| Remove tags | `remove_tags` | Working |
| Add note | `add_note` | Working |
| **Navigation** |
| Switch tabs | `navigate_to_tab` | Working |

## Principle Scores

| Principle | Score | Notes |
|-----------|-------|-------|
| Action Parity | 6/10 | Core operations work, but import, analysis, and export have significant gaps |
| Context Parity | 7/10 | Good project context, missing analysis status and preferences |
| Shared Workspace | 5/10 | CLI tools don't update live project; GUI sync is one-directional |
| Primitives over Workflows | 9/10 | Tools are well-designed primitives |
| Dynamic Context Injection | 8/10 | System prompt includes runtime state, could be more comprehensive |

## Critical Issues (Must Fix)

### 1. Missing `import_video` Tool

**Impact**: Agent cannot add local video files to the library. Users can drag-drop or use Import menu, but agent has no equivalent.

**Fix**: Add `import_video(path: str)` tool that calls `_add_video_to_library(Path(path))` on the main window.

### 2. CLI Tools Don't Update Live Project

**Impact**: The `detect_scenes`, `analyze_colors`, `analyze_shots`, and `transcribe` tools operate via CLI and output to files, but do NOT update the live Project in the GUI. After agent calls these tools, the GUI shows no changes until project reload.

**Fix**: Create GUI-state-modifying versions that trigger the appropriate workers (`DetectionWorker`, `ColorAnalysisWorker`, `ShotTypeWorker`, `TranscriptionWorker`).

### 3. Missing Bulk Download Tool

**Impact**: GUI supports downloading multiple selected YouTube videos in parallel, but agent cannot trigger bulk downloads.

**Fix**: Add `download_videos(video_ids: list[str])` or `download_youtube_selection()` tool.

## Warnings (Should Fix)

### 1. Missing Export Sequence Tool

**Impact**: Agent cannot export the sequence as a video file.

**Recommendation**: Add `export_sequence(output_path: str, quality: str, resolution: str, fps: int)` tool.

### 2. Missing Export Dataset Tool

**Impact**: Agent cannot export clip metadata as JSON dataset.

**Recommendation**: Add `export_dataset(output_path: str)` tool.

### 3. Missing Playback Control Tools

**Impact**: Agent cannot play/pause/stop video playback or seek to positions.

**Recommendation**: Add `play_sequence(start_frame: int)`, `stop_playback()`, `seek_to(time_seconds: float)` tools.

### 4. Incomplete Context Injection

**Impact**: System prompt includes project state but misses:
- Available shot type classifications
- Transcription status per clip
- Current selection state
- Export settings

**Recommendation**: Expand `_build_system_prompt()` to include more detailed clip analysis status.

### 5. Missing `analyze_all` Tool

**Impact**: GUI has "Analyze All" button that runs colors, shots, and transcription sequentially. Agent must call three separate tools manually.

**Recommendation**: Add `analyze_all_clips(clip_ids: list[str])` tool that triggers the sequential workflow.

## What's Working Well

- **Tool Registry Pattern**: Clean decorator-based registration with automatic schema generation from type hints
- **GUI/Worker Thread Separation**: Tools properly distinguish between GUI-modifying (main thread) and background operations
- **System Prompt Context**: Includes project state, default paths, and current GUI state
- **Conflict Detection**: Prevents race conditions between agent and GUI workers with wait-retry mechanism
- **Rich Tool Output**: Tools return structured data with success/error status and relevant metadata
- **Agent Action Sync to GUI**: YouTube search and download results automatically update GUI panels
- **Path Validation**: Security-conscious path validation prevents traversal attacks
- **Workflow Automation**: Progress reporting and failure recovery for compound operations

## Recommended Implementation Order

### Priority 1 (High Impact)
1. Add `import_video(path: str)` tool that adds local video to live project
2. Create `detect_scenes_live(source_id: str, sensitivity: float)` tool that updates GUI
3. Create GUI-aware analysis tools that trigger workers instead of CLI

### Priority 2 (Medium Impact)
4. Add `export_sequence` and `export_dataset` tools
5. Add `analyze_all_clips` compound tool
6. Add `download_videos_bulk(video_ids: list[str])` tool

### Priority 3 (Nice to Have)
7. Add playback control tools
8. Expand system prompt context to include analysis status details
9. Add GUI components for tag/note management (reverse parity)
10. Consider file watching or shared state for bidirectional sync

## Statistics

- **Agent-accessible capabilities**: 27/40
- **Working tools**: 25
- **Missing tools**: ~10-12 (depending on grouping)
- **Partial tools** (CLI-only, no GUI update): 4

## Next Steps

Create a new plan document to address Priority 1 items - specifically the disconnect between CLI tools and the live GUI project. This is the single biggest barrier to full agent-native operation.
