---
name: pyside6-tool-main-window-injection
description: |
  Fix agent tools not receiving main_window parameter in Scene Ripper PySide6 app.
  Use when: (1) main_window is None inside a tool function that needs GUI access,
  (2) tool needs to check worker status (isRunning(), queues), (3) adding a new
  tool that accesses MainWindow attributes. The modifies_gui_state flag controls
  parameter injection routing.
author: Claude Code
version: 1.0.0
date: 2026-01-29
---

# PySide6 Agent Tool main_window Injection

## Problem
Agent tools that need access to `main_window` attributes (like `detection_worker`,
`_analyze_queue`, etc.) receive `None` for `main_window` even when the parameter
is defined, causing the tool's GUI-dependent functionality to silently fail.

## Context / Trigger Conditions
- Tool function has `main_window` parameter but it's always `None`
- Tool needs to check if a Qt worker is running via `worker.isRunning()`
- Tool needs to access GUI state like `_analyze_queue`, `current_source`
- New tool added with `modifies_gui_state=False` that accesses MainWindow

## Solution

The `modifies_gui_state` flag in `@tools.register()` determines how tools are executed:

| Flag Value | Executor | Parameters Injected | Thread |
|------------|----------|---------------------|--------|
| `False` | `ToolExecutor` | `project` only | Background (ChatAgentWorker) |
| `True` | `_on_gui_tool_requested` | `project`, `main_window`, `gui_state` | Main (GUI thread) |

**To get `main_window` injected, set `modifies_gui_state=True`:**

```python
@tools.register(
    description="Tool that needs GUI access",
    requires_project=True,
    modifies_gui_state=True  # Required for main_window injection
)
def my_tool(main_window, project) -> dict:
    # main_window is now available
    is_running = main_window.detection_worker.isRunning()
    ...
```

**Execution flow:**
1. `ChatAgentWorker._execute_tool_calls()` checks `tool_def.modifies_gui_state`
2. If `True`: emits `gui_tool_requested` signal → `MainWindow._on_gui_tool_requested()`
3. GUI handler inspects function signature and injects matching parameters
4. If `False`: executes via `ToolExecutor` which only injects `project`

## Verification
1. Add logging to verify `main_window is not None`
2. Check that `main_window.detection_worker` (or other attribute) is accessible
3. Verify tool runs on main thread (use `QThread.currentThread()` if needed)

## Example

Before (broken - main_window always None):
```python
@tools.register(
    description="Check detection status",
    requires_project=True,
    modifies_gui_state=False  # Wrong!
)
def check_detection_status(main_window, project) -> dict:
    # main_window is None here
    is_running = main_window.detection_worker.isRunning()  # AttributeError!
```

After (working):
```python
@tools.register(
    description="Check detection status",
    requires_project=True,
    modifies_gui_state=True  # Correct
)
def check_detection_status(main_window, project) -> dict:
    # main_window is properly injected
    is_running = main_window.detection_worker.isRunning()  # Works
```

## Notes

- Tools with `modifies_gui_state=True` run on the main GUI thread, so keep them fast
- Read-only access to GUI state still requires `modifies_gui_state=True` for injection
- The flag name is slightly misleading - it controls thread routing, not just write access
- Parameter injection happens in `MainWindow._on_gui_tool_requested()` at lines 1750-1763
- `ToolExecutor` (used for False) only injects `project` - see `tool_executor.py:128-130`

## Related Files
- `ui/main_window.py:1717-1812` - GUI tool handler with parameter injection
- `ui/chat_worker.py:370-378` - Tool routing based on modifies_gui_state
- `core/tool_executor.py:128-130` - ToolExecutor only injects project
- `core/chat_tools.py` - Tool registration decorators
