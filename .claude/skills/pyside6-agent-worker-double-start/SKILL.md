---
name: pyside6-agent-worker-double-start
description: |
  Fix "Worker already in progress" errors when Agent tools trigger background tasks. Use when: (1) An Agent tool returns a `_wait_for_worker` signal, (2) The Agent immediately reports failure because the worker is already running, (3) The tool implementation manually calls the start function.
author: Claude Code
version: 1.0.0
date: 2026-01-27
---

# PySide6 Agent Worker Double-Start Fix

## Problem
In PySide6 applications where an LLM Agent triggers background workers (QThread), a race condition can occur if the tool function *both* starts the worker *and* returns a signal telling the main thread to wait for that worker. The main thread, receiving the signal, attempts to start the worker again, causing a "Worker already running" error.

## Context / Trigger Conditions
- **Architecture**: Agent runs in a background thread; Tools run on the Main Thread via signal/slot.
- **Pattern**: Tool returns `{"_wait_for_worker": "worker_name"}` to pause the agent until the task completes.
- **Symptom**: The agent logs "Executing GUI tool...", and immediately follows with "Error: Task already in progress" even though no other task was running beforehand.

## Solution
The tool function should be **pure** regarding worker execution. It should only prepare and return the *parameters* needed to start the worker. The actual startup logic should live exclusively in the signal handler (e.g., `MainWindow._on_gui_tool_requested`).

### Bad Pattern (Double Start)
```python
# In chat_tools.py
def start_analysis_live(main_window, items):
    # ERROR: Starts worker here
    main_window.start_analysis_worker(items) 
    
    # AND tells main window to start it (or wait for it)
    return {"_wait_for_worker": "analysis"}
```

### Good Pattern (Parameter Passing)
```python
# In chat_tools.py
def start_analysis_live(main_window, items):
    # Check if already running (safety check)
    if main_window.worker and main_window.worker.isRunning():
        return {"success": False, "error": "Already running"}

    # Just return instructions
    return {
        "_wait_for_worker": "analysis",
        "items": items,
        "mode": "fast"
    }

# In main_window.py
def _on_gui_tool_requested(self, tool_name, result):
    if result.get("_wait_for_worker") == "analysis":
        # Start worker ONLY here
        self.start_analysis_worker(result["items"], result["mode"])
```

## Verification
1. Call the tool via the Agent.
2. Observe logs: The worker should start exactly once.
3. The Agent should pause and wait for the `finished` signal without error.

## Notes
- This enforces a cleaner separation of concerns: Tools define *intent*, the UI Controller handles *execution*.
- Always include a safety check (`isRunning()`) in the tool definition to fail fast if a *different* workflow really is using the worker.
