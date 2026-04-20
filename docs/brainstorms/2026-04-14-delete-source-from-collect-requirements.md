---
title: "feat: Delete source videos from Collect tab"
type: feat
status: draft
date: 2026-04-14
---

# Delete source videos from Collect tab

## Problem

There is no way to remove a source video from a project once imported. Users accumulate unwanted sources with no cleanup path. The data model supports `Project.remove_source()` but no UI exposes it.

## Requirements

- R1. Right-click context menu on source thumbnails in the Collect tab with a "Delete" option.
- R2. Deleting a source removes: the source itself, all clips derived from it (Cut + Analyze tabs), and all frames extracted from it (Frames tab). Uses existing `Project.remove_source()` cascade.
- R3. **Sequence guard**: If any of the source's clips appear in any sequence (across all sequences in a multi-sequence project), the delete is blocked. Show an error message naming the sequences that contain the clips: "Cannot delete '[source name]': clips are used in [Chromatics, Tempo Shift]. Delete those sequences first."
- R4. Confirmation dialog for all deletions (even unguarded): "Delete '[source name]' and its N clips? This cannot be undone."
- R5. After deletion, all UI tabs update immediately: source removed from Collect grid, clips removed from Cut browser and Analyze browser, frames removed from Frames browser.
- R6. Multi-select delete: if multiple sources are selected, "Delete N Sources" option in context menu. Same guards apply per-source — partially blocked deletions explain which sources can't be deleted and which can.

## Scope Boundaries

- No deletion of individual clips (only entire sources)
- No undo/restore functionality
- No deletion of the source video file on disk — only removes from the project
- No batch "delete all unused sources" feature

## Success Criteria

- Right-click a source in Collect → Delete → confirmation → source and all clips gone from all tabs
- Attempt to delete a source with clips in a sequence → error names the blocking sequences
- Delete works with multi-select (Cmd+click multiple sources → right-click → Delete N Sources)
- Agent tool `delete_clips` (already exists in `core/chat_tools.py`) continues to work alongside this
