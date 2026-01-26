---
name: explicit-tool-result-success-flag
description: |
  Ensures agent tool functions return an explicit boolean "success" flag. Use when: (1) tool functions handle exceptions internally, (2) the UI/Agent displays contradictory messages (e.g., success checkmark with error text), (3) the agent incorrectly assumes success when a tool returns an error dictionary. This prevents logic errors in state-aware applications where the absence of a crash is not sufficient to signal success.
author: Claude Code
version: 1.0.0
date: 2026-01-25
---

# Explicit Tool Result Success Flag

## Problem
In agent-based systems, tools often return dictionaries. If a tool encounters an error (e.g., missing API key) but returns a dictionary containing the error message, the calling infrastructure might treat the *function return* as a success. This leads to "mixed signals" where the UI shows a success indicator (✓) but the text content describes a failure.

## Context / Trigger Conditions
- Tool execution status in UI is contradictory (e.g., `✓ search_youtube completed successfully. The YouTube API error indicates...`)
- Agent attempts to proceed with a workflow using data that failed to generate
- Tool functions catch exceptions and return dictionaries without an explicit status flag

## Solution
1. Always include a top-level `"success": bool` key in the return dictionary of all tool functions.
2. In the tool's error paths (exceptions or validation failures), set `"success": False`.
3. In the tool's success paths, set `"success": True`.
4. Update the agent's response-formatting logic to strictly check the `success` flag before displaying status indicators.

## Verification
- Run the tool in a failure condition (e.g., remove API key).
- Verify the UI shows an error indicator (✗) and no generic success text.
- Verify the agent acknowledges the failure and doesn't try to proceed with invalid data.

## Example
**Before (Buggy):**
```python
def search_youtube(query):
    try:
        results = api.search(query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
```

**After (Correct):**
```python
def search_youtube(query):
    try:
        results = api.search(query)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Notes
- This pattern is especially important for long-running or compound tools where the agent needs to know exactly when to stop or retry.
- Consistent return schemas make it easier to write robust response formatters.
