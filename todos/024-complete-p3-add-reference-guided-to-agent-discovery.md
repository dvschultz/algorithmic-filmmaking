---
status: complete
priority: p3
issue_id: "024"
tags: [code-review, agent-native, discovery]
dependencies: []
---

# Add Reference-Guided to Agent Discovery (System Prompt + list_sorting_algorithms)

## Problem Statement

The system prompt in `chat_worker.py` tells the agent to use `generate_remix` for sequence generation and `list_sorting_algorithms` for discovery. Neither mentions `generate_reference_guided`. The agent has no way to discover this tool exists unless the user explicitly asks for reference-guided matching by name.

## Findings

**System prompt:** `ui/chat_worker.py` line 724 says:
> "Use the generate_remix tool to create sequences with any of the 13 sorting algorithms. Use list_sorting_algorithms first to check which algorithms are available."

No mention of `generate_reference_guided`.

**list_sorting_algorithms:** `core/chat_tools.py` lines 3734-3855 returns algorithm descriptions but does not include reference_guided or mention the separate tool.

## Proposed Solutions

### Option A: Add Note to list_sorting_algorithms Output (Recommended)

Add a `reference_guided_note` field to the response:
```python
return {
    "success": True,
    "algorithms": [...],
    "reference_guided_available": True,
    "reference_guided_note": "For matching clips to a reference video's structure, use the generate_reference_guided tool instead of generate_remix."
}
```

Also add a line to the system prompt mentioning the tool.

**Pros:** Agent can discover the feature through normal discovery flow
**Cons:** Minor system prompt change
**Effort:** Small

## Acceptance Criteria

- [ ] Agent can discover reference-guided capability via list_sorting_algorithms
- [ ] System prompt mentions generate_reference_guided for reference-based matching
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 agent-native review | New agent tools must be added to discovery paths (system prompt, listing tools) |
