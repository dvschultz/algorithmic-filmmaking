---
title: "feat: Delete source videos from Collect tab"
type: feat
status: active
date: 2026-04-14
origin: docs/brainstorms/2026-04-14-delete-source-from-collect-requirements.md
---

# feat: Delete source videos from Collect tab

## Overview

Add right-click "Delete" to source thumbnails in the Collect tab. Deletion cascades to clips and frames but is blocked when the source's clips appear in any sequence. The data model already supports `Project.remove_source()`; this work adds the UI trigger, sequence guard, and cross-tab cleanup wiring.

## Problem Frame

Users accumulate unwanted source videos with no way to remove them. `Project.remove_source()` exists but no UI exposes it. (see origin: `docs/brainstorms/2026-04-14-delete-source-from-collect-requirements.md`)

## Requirements Trace

- R1. Right-click context menu on source thumbnails with "Delete" option
- R2. Cascade: removes source, all clips (Cut + Analyze), all frames (Frames tab)
- R3. Sequence guard: block if clips in any sequence; name the blocking sequences
- R4. Confirmation dialog for all unguarded deletions
- R5. All tabs update immediately after deletion
- R6. Multi-select delete with per-source guards

## Scope Boundaries

- No individual clip deletion (source-level only)
- No undo/restore
- No file-on-disk deletion
- No "delete all unused" batch feature

## Context & Research

### Relevant Code and Patterns

- `core/project.py:838-860` — `Project.remove_source(source_id)`: removes source + clips + frames, fires `"source_removed"` observer. Does NOT scan sequences for dangling SequenceClips.
- `ui/source_thumbnail.py` — No right-click handling; `mousePressEvent` handles left-click only
- `ui/source_browser.py:242-260` — `remove_source(source_id)`: UI removal (grid, thumbnail, selection cleanup)
- `ui/tabs/collect_tab.py:149-152` — `remove_source(source_id)`: delegates to browser + updates UI state
- `ui/main_window.py:750-756` — `_project_adapter.source_removed` signal exists but is NOT connected to any handler
- `ui/clip_browser.py:404-437` — **Pattern to follow**: `contextMenuEvent` on `ClipThumbnail` with `_build_context_menu()`, keyboard accessibility via `keyPressEvent`

### Institutional Learnings

- Sequence state mismatch: never let widgets hold separate Sequence copies. Use property delegation.

## Key Technical Decisions

- **Sequence guard lives in a new Project method**: `Project.source_in_sequences(source_id) -> list[str]` returns sequence names containing the source's clips. Keeps the guard logic in the model, not the UI.
- **Context menu on SourceThumbnail**: Follow the ClipThumbnail pattern — override `contextMenuEvent`, emit `delete_requested` signal, bubble up through SourceBrowser → CollectTab → MainWindow.
- **Main window handler**: New `_on_delete_source_requested(source_id)` does the guard check, shows confirmation or error, calls `project.remove_source()`, then cleans up all tabs.
- **Connect `source_removed` adapter signal**: Wire it to a handler that refreshes Analyze tab lookups (like `_on_source_added` does for additions).

## Implementation Units

- [ ] **Unit 1: Sequence guard method on Project**

  **Goal:** Add `Project.source_in_sequences(source_id)` that returns a list of sequence names where the source's clips appear.

  **Requirements:** R3

  **Dependencies:** None

  **Files:**
  - Modify: `core/project.py`
  - Test: `tests/test_multi_sequence_project.py`

  **Approach:**
  - Iterate `self.sequences`, for each call `seq.get_all_clips()`, check `sc.source_id == source_id`. Return `[seq.name for seq in self.sequences if any(...)]`.

  **Patterns to follow:**
  - Load-time validation in `Project.load()` uses the same `seq.tracks → track.clips → sc.source_id` scan pattern

  **Test scenarios:**
  - Happy path: source with no clips in any sequence → returns empty list
  - Happy path: source with clips in 1 sequence → returns `["Chromatics"]`
  - Happy path: source with clips in 2 sequences → returns both names
  - Edge case: source not in project → returns empty list
  - Edge case: empty sequences (0 clips) → returns empty list

  **Verification:**
  - Method returns correct sequence names for sources in and out of sequences

- [ ] **Unit 2: Context menu on SourceThumbnail + signal chain**

  **Goal:** Add right-click "Delete" to source thumbnails, with signal chain up to CollectTab.

  **Requirements:** R1, R6

  **Dependencies:** None

  **Files:**
  - Modify: `ui/source_thumbnail.py` — add `contextMenuEvent`, `keyPressEvent` (Delete key), emit `delete_requested`
  - Modify: `ui/source_browser.py` — connect thumbnail signal, emit `delete_source_requested` with source_id(s)
  - Modify: `ui/tabs/collect_tab.py` — add `delete_source_requested` signal, connect from browser
  - Test: `tests/test_multi_sequence_tab.py` (extend with model-level deletion tests)

  **Approach:**
  - SourceThumbnail: override `contextMenuEvent` to show QMenu with "Delete [filename]". Emit `delete_requested(source)`.
  - SourceBrowser: connect each thumbnail's `delete_requested`. When fired, check `self.selected_source_ids` — if the source is part of a multi-select, emit `delete_source_requested` with all selected IDs. Otherwise emit with just the one.
  - CollectTab: add `delete_source_requested = Signal(list)` and re-emit from browser.

  **Patterns to follow:**
  - `ui/clip_browser.py:404-437` — `ClipThumbnail.contextMenuEvent` with keyboard accessibility

  **Test scenarios:**
  - Happy path: single source right-click emits delete_requested with 1 source
  - Happy path: multi-select right-click emits delete_requested with all selected sources

  **Verification:**
  - Right-click shows "Delete" menu item; signal chain fires to CollectTab

- [ ] **Unit 3: Main window delete handler + tab cleanup**

  **Goal:** Wire the delete signal to a handler that performs guard check, confirmation, deletion, and UI cleanup.

  **Requirements:** R2, R3, R4, R5, R6

  **Dependencies:** Unit 1, Unit 2

  **Files:**
  - Modify: `ui/main_window.py` — add `_on_delete_source_requested(source_ids)`, connect `source_removed` adapter signal
  - Test: `tests/test_multi_sequence_project.py` (extend with deletion + guard tests)

  **Approach:**
  - Connect `collect_tab.delete_source_requested` to `_on_delete_source_requested(source_ids: list[str])`.
  - For each source_id, call `project.source_in_sequences(source_id)`. Partition into deletable (empty list) and blocked (non-empty).
  - If any blocked: show `QMessageBox.warning` naming the blocked sources and their sequences.
  - If any deletable: show `QMessageBox.question` confirmation ("Delete N sources and M clips?").
  - On confirm: for each deletable source, call `project.remove_source(source_id)` then `collect_tab.remove_source(source_id)`.
  - Connect `_project_adapter.source_removed` to a new `_on_source_removed` handler that refreshes Analyze tab lookups (mirrors `_on_source_added`).

  **Patterns to follow:**
  - Existing `_on_delete_sequence` in `sequence_tab.py` for confirmation dialog pattern
  - Existing `_on_source_added` for adapter signal wiring pattern

  **Test scenarios:**
  - Happy path: delete unguarded source → source, clips, frames removed from project
  - Happy path: delete multiple unguarded sources → all removed
  - Happy path: blocked source → error message names sequences, source not removed
  - Happy path: mixed batch (2 deletable, 1 blocked) → error for blocked, confirmation for deletable
  - Edge case: delete source that was already removed → no error (guard handles gracefully)
  - Integration: after deletion, `project.sources`, `project.clips`, `project.frames` all reflect removal

  **Verification:**
  - Full delete flow works end-to-end: right-click → guard check → confirm → source gone from all tabs
  - Blocked sources show error with correct sequence names

## System-Wide Impact

- **Interaction graph:** `SourceThumbnail.delete_requested` → `SourceBrowser.delete_source_requested` → `CollectTab.delete_source_requested` → `MainWindow._on_delete_source_requested` → `Project.remove_source()` → `ProjectSignalAdapter.source_removed` → `MainWindow._on_source_removed` (refreshes lookups). The `clips_removed` observer already connected handles Cut + Analyze tab cleanup.
- **Error propagation:** Guard failure shows QMessageBox.warning with sequence names. Deletion failure (if source not found) is a no-op.
- **State lifecycle risks:** Low — `Project.remove_source()` already handles cascade. The new guard is read-only.
- **API surface parity:** Agent tool `delete_clips` in `core/chat_tools.py` already exists for clip-level deletion. Source-level deletion via agent could be added later but is out of scope.
- **Unchanged invariants:** `Project.remove_source()` behavior unchanged — we're just adding a UI trigger and a pre-check.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `source_removed` adapter signal not connected | Unit 3 explicitly wires it. Verified by checking Analyze tab lookups update after deletion. |
| Multi-sequence guard misses a sequence | `source_in_sequences` scans all sequences via `project.sequences`, not just active. Tested with clips in multiple sequences. |

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-14-delete-source-from-collect-requirements.md](docs/brainstorms/2026-04-14-delete-source-from-collect-requirements.md)
- Related code: `core/project.py` (remove_source), `ui/clip_browser.py` (context menu pattern), `ui/source_thumbnail.py`, `ui/source_browser.py`
