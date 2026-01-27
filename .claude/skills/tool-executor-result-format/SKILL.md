---
name: tool-executor-result-format
description: |
  Fix "KeyError: 'result'" errors in ToolExecutor.format_for_llm. Use when: (1) Integrating new tools into an agent system, (2) Tool execution succeeds but the agent crashes while formatting the response, (3) The error traceback points to `result["result"]`.
author: Claude Code
version: 1.0.0
date: 2026-01-27
---

# Tool Executor Result Format

## Problem
When implementing custom tools or callbacks for an LLM agent, failing to structure the return value exactly as expected by the `ToolExecutor` causes a crash. The executor expects a specific dictionary structure to format the result for the LLM. If a tool (or a callback handling an async tool result) returns a flat dictionary, the executor fails with `KeyError: 'result'` when trying to access the nested result.

## Context / Trigger Conditions
- **Error**: `KeyError: 'result'` in `ToolExecutor.format_for_llm`
- **Architecture**: Custom agent system with `ToolExecutor` class
- **Scenario**: Completing an asynchronous tool execution (e.g., via a signal handler like `_on_agent_task_finished`)

## Solution
Ensure that any function returning a tool result to the agent (especially async completion handlers) wraps the actual data in a `result` key and includes the required metadata fields.

### Incorrect Structure (Causes Crash)
```python
# The agent expects a wrapper, but gets flat data
result = {
    "success": True,
    "count": 5,
    "items": [...]
}
# Crash: result["result"] raises KeyError
```

### Correct Structure
```python
# Properly wrapped result
result = {
    "tool_call_id": original_tool_call_id,  # Required to match request
    "name": tool_name,                      # Required for context
    "success": True,                        # Status flag
    "result": {                             # <--- The actual data wrapper
        "success": True,
        "count": 5,
        "items": [...]
    }
}
```

## Verification
1. Inspect the code passing the result to the agent (e.g., `chat_worker.set_gui_tool_result(result)`).
2. Verify it contains the top-level keys: `tool_call_id`, `name`, `success`, and `result`.
3. Verify the actual payload is nested inside the `result` key.

## Notes
- This pattern is common in systems that need to standardize error handling and metadata (ID, name) separate from the tool's actual return value.
- For synchronous tools executed directly by `ToolExecutor`, the wrapping is often handled automatically. This issue primarily affects **asynchronous/GUI tools** where the result is constructed manually in a callback.