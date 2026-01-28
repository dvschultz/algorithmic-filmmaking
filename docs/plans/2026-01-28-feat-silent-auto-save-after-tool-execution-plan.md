---
title: Silent Auto-Save After Tool Execution
type: feat
date: 2026-01-28
---

# Silent Auto-Save After Tool Execution

## Overview

Automatically save the project after agent tools modify project state, without prompting the user. This replaces the current unreliable prompt-based approach (rule 7) with deterministic, silent auto-saves.

## Problem Statement

The current implementation relies on LLM memory to prompt users about saving after tool execution (rule 7 in system prompt). This is:

1. **Unreliable** - LLM may forget to ask, especially in long conversations
2. **Disruptive** - Adds unnecessary chat messages asking "Would you like me to save?"
3. **Inconsistent** - "Always save" preference relies on LLM remembering, which it may not

## Proposed Solution

Add infrastructure for silent, automatic project saves after tools that modify project state:

1. Add `modifies_project_state: bool` flag to `ToolDefinition`
2. Mark tools that change project data with this flag
3. Hook into `_on_chat_tool_result` to trigger auto-save
4. Use debouncing to handle rapid consecutive tool calls
5. Remove rule 7 from system prompt

## Technical Approach

### Architecture

```
Tool Execution Complete
         │
         ▼
_on_chat_tool_result(name, result, success)
         │
         ▼
┌────────────────────────────┐
│ Check: tool_registry.get() │
│ → modifies_project_state?  │
└────────────┬───────────────┘
             │
     ┌───────┴───────┐
     │ Yes           │ No
     ▼               ▼
┌──────────┐    ┌──────────┐
│ Check:   │    │ Skip     │
│ success? │    │          │
│ is_dirty?│    └──────────┘
│ has path?│
└────┬─────┘
     │ All True
     ▼
┌──────────────────┐
│ Schedule save    │
│ (debounced 300ms)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ project.save()   │
│ (silent)         │
└──────────────────┘
```

### Implementation Phases

#### Phase 1: Add Flag to ToolDefinition

**File:** `core/chat_tools.py`

Add new flag to the `ToolDefinition` dataclass:

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable
    parameters: dict
    requires_project: bool = True
    modifies_gui_state: bool = False
    modifies_project_state: bool = False  # NEW
```

Update `ToolRegistry.register()` to accept the new parameter.

#### Phase 2: Mark Tools with Flag

**File:** `core/chat_tools.py`

Tools that modify project state (call Project methods that change data):

| Tool | Reason |
|------|--------|
| `detect_scenes` | Calls `project.add_clips()` |
| `detect_scenes_live` | Calls `project.add_clips()` |
| `detect_scenes_all` | Calls `project.replace_source_clips()` |
| `add_to_sequence` | Calls `project.add_to_sequence()` |
| `remove_from_sequence` | Calls `project.remove_from_sequence()` |
| `clear_sequence` | Calls `project.clear_sequence()` |
| `reorder_sequence` | Calls `project.reorder_sequence()` |
| `import_video` | Calls `project.add_source()` |
| `remove_source` | Calls `project.remove_source()` |
| `set_project_name` | Updates `project.metadata.name` |
| `update_clip` | Updates clip metadata |
| `add_tags` | Updates clip metadata |
| `remove_tags` | Updates clip metadata |
| `add_note` | Updates clip metadata |
| `update_clip_transcript` | Updates clip metadata |
| `analyze_colors_live` | Updates clip metadata |
| `analyze_shots_live` | Updates clip metadata |
| `transcribe_live` | Updates clip metadata |
| `classify_content_live` | Updates clip metadata |
| `detect_objects_live` | Updates clip metadata |
| `count_people_live` | Updates clip metadata |
| `describe_content_live` | Updates clip metadata |
| `run_all_analysis_live` | Updates clip metadata |

Tools that should NOT have the flag:
- `download_video` - Downloads file but doesn't modify Project
- `save_project` - Persists state, doesn't modify it
- `load_project` - Replaces project (no path after load from new file)
- `new_project` - Creates unsaved project (no path)
- `select_clips`, `navigate_to_tab`, etc. - GUI-only changes

#### Phase 3: Implement Auto-Save Logic

**File:** `ui/main_window.py`

Add debounced auto-save to `MainWindow`:

```python
# In __init__:
self._auto_save_timer = QTimer(self)
self._auto_save_timer.setSingleShot(True)
self._auto_save_timer.timeout.connect(self._do_auto_save)
self._auto_save_pending = False

# New method:
def _schedule_auto_save(self):
    """Schedule a debounced auto-save."""
    if not self.project or not self.project.path:
        return
    if not self.project.is_dirty:
        return

    # Debounce: restart timer on each call
    self._auto_save_timer.stop()
    self._auto_save_timer.start(300)  # 300ms debounce

def _do_auto_save(self):
    """Execute the auto-save."""
    if not self.project or not self.project.path:
        return
    if not self.project.is_dirty:
        return

    try:
        self.project.save()
        logger.info(f"Auto-saved project to {self.project.path}")
    except Exception as e:
        logger.error(f"Auto-save failed: {e}")
        # Keep project dirty so user can manually save
```

Modify `_on_chat_tool_result`:

```python
def _on_chat_tool_result(self, name: str, result: dict, success: bool):
    """Handle tool execution completion."""
    logger.info(f"Chat tool {name} completed: success={success}")
    if self._current_tool_indicator:
        self._current_tool_indicator.set_complete(success)

    # Auto-save check
    if success:
        from core.chat_tools import tools as tool_registry
        tool_def = tool_registry.get(name)
        if tool_def and tool_def.modifies_project_state:
            self._schedule_auto_save()
```

#### Phase 4: Update System Prompt

**File:** `ui/chat_worker.py`

Remove rule 7 from the IMPORTANT BEHAVIOR RULES section:

```python
# BEFORE:
"""
6. PROJECT NAMING: Before executing any state-modifying tool...
7. AUTO-SAVE OFFERS: After completing a state-modifying tool...
"""

# AFTER:
"""
6. PROJECT NAMING: Before executing any state-modifying tool (detect_scenes, download_video, add_to_sequence, etc.) on an unnamed project (Name: "Untitled Project", Path: Unsaved), FIRST ask the user what they'd like to name the project, then use set_project_name to set it. After naming, suggest they save the project with save_project to enable auto-save.
"""
```

Note: Rule 6 is updated to mention that saving enables auto-save.

## Acceptance Criteria

### Functional Requirements

- [x] Tools with `modifies_project_state=True` trigger auto-save on success
- [x] Auto-save only occurs if project has a path (was saved before)
- [x] Auto-save only occurs if project is dirty
- [x] Rapid consecutive tool calls result in single save (debouncing)
- [x] Auto-save is silent (no chat messages, no dialogs)
- [x] Failed saves keep project marked dirty
- [x] Rule 7 removed from system prompt

### Non-Functional Requirements

- [x] Debounce window is 300ms
- [x] Save failures are logged but don't interrupt workflow
- [x] No duplicate saves from signal delivery issues

## Success Metrics

- Zero user prompts about saving after tool execution
- Project state persisted reliably after state-modifying operations
- No data loss from forgotten saves

## Dependencies & Prerequisites

- Existing `ToolDefinition` dataclass
- Existing `_on_chat_tool_result` signal handler
- Existing `project.save()` method with atomic write

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Duplicate saves from signal issues | Low | Low | Debounce timer handles this |
| Save during active background operation | Low | Medium | `_on_chat_tool_result` only fires after tool completes |
| Partial state saved | Low | High | Project.save() uses atomic temp file + rename |
| User confusion about auto-save | Medium | Low | Update rule 6 to mention auto-save after first save |

## Files to Modify

| File | Changes |
|------|---------|
| `core/chat_tools.py` | Add `modifies_project_state` to ToolDefinition, update register(), mark ~25 tools |
| `ui/main_window.py` | Add `_auto_save_timer`, `_schedule_auto_save()`, `_do_auto_save()`, update `_on_chat_tool_result()` |
| `ui/chat_worker.py` | Remove rule 7, update rule 6 |

## Test Plan

1. **Basic auto-save**: Run `detect_scenes` on saved project → verify file updated
2. **Unsaved project**: Run `detect_scenes` on new project → verify no save attempt
3. **Failed tool**: Run tool that fails → verify no save
4. **Debouncing**: Run 3 tools rapidly → verify only 1 save
5. **Save failure**: Make project file read-only → verify project stays dirty

## References

### Internal References

- Tool execution: `ui/chat_worker.py:350-420`
- Tool result handler: `ui/main_window.py:1542-1546`
- Project save: `core/project.py:719-754`
- ToolDefinition: `core/chat_tools.py:144-152`

### Institutional Learnings

- State duplication issues: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- Duplicate signal delivery: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
