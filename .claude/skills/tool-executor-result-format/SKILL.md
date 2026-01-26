---
name: tool-executor-result-format
description: |
  Fix for accessing tool results from ToolExecutor in Scene Ripper. Use when:
  (1) signal handlers receive empty data from tool execution, (2) tool results
  appear to be missing fields that the tool definitely returns, (3) writing new
  code that processes results from ChatAgentWorker tool execution. The ToolExecutor
  wraps all tool return values in a "result" key, not "data".
author: Claude Code
version: 1.0.0
date: 2026-01-25
---

# ToolExecutor Result Format

## Problem

When writing code that processes tool execution results from `ChatAgentWorker`, the
actual tool return value is not at the top level of the result dict. This causes
handlers to receive empty or wrong data.

## Context / Trigger Conditions

- Writing signal handlers for tool completion (e.g., `youtube_search_completed`)
- Code expects tool return values at top level but gets `None` or wrong data
- Tool clearly returns data (visible in chat) but handler doesn't receive it
- Working with `result` dict from `ToolExecutor.execute()`

## Solution

The `ToolExecutor.execute()` method wraps all tool return values in this structure:

```python
{
    "tool_call_id": "call_xxx",
    "name": "tool_name",
    "success": True,  # or False
    "result": { ... actual tool return value here ... }
}
```

To access the actual tool data:

```python
# WRONG - "data" key doesn't exist
data = result.get("data", result)

# CORRECT - use "result" key
data = result.get("result", result)

# Then access tool-specific fields
query = data.get("query", "")
videos = data.get("results", [])
```

## Verification

After fixing, log statements should show the correct nested data:

```python
logger.info(f"Result keys: {result.keys()}")  # Shows: tool_call_id, name, success, result
logger.info(f"Data keys: {data.keys()}")       # Shows actual tool return fields
```

## Example

For `search_youtube` tool which returns:
```python
{"success": True, "query": "...", "results": [...]}
```

The full result from ToolExecutor is:
```python
{
    "tool_call_id": "call_abc123",
    "name": "search_youtube",
    "success": True,
    "result": {
        "success": True,
        "query": "nature documentary",
        "results": [{"video_id": "...", "title": "..."}]
    }
}
```

Access pattern:
```python
def _emit_gui_sync_signal(self, tool_name: str, args: dict, result: dict):
    if not result.get("success", False):
        return

    # Unwrap the actual tool return value
    data = result.get("result", result)

    if tool_name == "search_youtube":
        videos = data.get("results", [])  # Now correctly gets the list
```

## Notes

- The outer `success` and inner `success` may both exist - check the outer one for
  execution status, inner one is tool-specific
- For error cases, the result has `"error": "message"` instead of `"result"`
- See `core/tool_executor.py` lines 123-128 for the wrapping logic

## Related Files

- `core/tool_executor.py` - Defines the result structure
- `ui/chat_worker.py` - Consumes results in `_emit_gui_sync_signal`
