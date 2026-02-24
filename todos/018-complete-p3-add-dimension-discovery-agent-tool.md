---
status: complete
priority: p3
issue_id: "018"
tags: [code-review, agent-native, discovery]
dependencies: []
---

# Add Dimension Discovery for Agent Before Reference-Guided Remixing

## Problem Statement

The agent has no way to discover which dimensions have data before calling `generate_reference_guided`. The dialog UI checks via `get_active_dimensions_for_clips()` and auto-disables checkboxes for unavailable dimensions, but no agent tool exposes this information. Without it, the agent must guess which weight keys are valid, and will get opaque errors or silent zero-matches if it specifies dimensions that have no analysis data.

## Findings

**Dialog path:** `ui/dialogs/reference_guide_dialog.py` lines 164-166

```python
from core.remix.reference_match import get_active_dimensions_for_clips
all_clip_objects = [clip for clip, _ in clips]
self._available_dimensions = get_active_dimensions_for_clips(all_clip_objects)
```

This result drives which checkboxes are enabled (line 309-311):

```python
is_available = dim_key in self._available_dimensions
checkbox.setEnabled(is_available)
checkbox.setChecked(is_available)
```

**Agent path:** `core/chat_tools.py` lines 3984-3989

The agent tool validates weight keys against a hardcoded set of valid dimensions:

```python
valid_dims = {"color", "brightness", "shot_scale", "audio", "embedding", "movement", "duration"}
invalid = set(weights.keys()) - valid_dims
```

But it does not check whether those dimensions actually have data for the current clips. The agent has no tool to call `get_active_dimensions_for_clips()` before generating.

## Proposed Solutions

### Option A: Add a `get_available_dimensions` Agent Tool (Recommended)
Create a new tool that calls `get_active_dimensions_for_clips()` and returns the set of dimensions with data:

```python
@register_tool(
    name="get_available_dimensions",
    description="Check which analysis dimensions have data for the current clips",
    requires_project=True,
)
def get_available_dimensions(project, main_window):
    clips = main_window.sequence_tab._available_clips
    clip_objects = [clip for clip, _ in clips]
    from core.remix.reference_match import get_active_dimensions_for_clips
    available = get_active_dimensions_for_clips(clip_objects)
    return {"success": True, "available_dimensions": sorted(available)}
```

**Pros:** Clean separation, agent can check before generating
**Cons:** Requires extra tool call
**Effort:** Small

### Option B: Include Dimension Availability in `get_project_state`
Add an `available_dimensions` field to the project state response so the agent always has this info.

**Pros:** No extra tool call needed, always up-to-date
**Cons:** Adds complexity to a general-purpose tool; dimension check only matters for reference-guided remixing
**Effort:** Small

## Acceptance Criteria

- [ ] Agent can discover which dimensions have analysis data before calling `generate_reference_guided`
- [ ] The discovery mechanism uses the same logic as the dialog (`get_active_dimensions_for_clips`)
- [ ] Agent tool description guides the LLM to check dimensions before generating
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Agent tools need discovery parity with dialog UIs for guided workflows |
