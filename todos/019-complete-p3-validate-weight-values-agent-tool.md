---
status: complete
priority: p3
issue_id: "019"
tags: [code-review, security, validation]
dependencies: []
---

# Validate Weight Values in generate_reference_guided Agent Tool

## Problem Statement

The `generate_reference_guided` agent tool validates weight **keys** (ensuring they are valid dimension names) but does not validate weight **values**. Negative weights would silently invert distance semantics (making the algorithm prefer the *least* similar clip instead of the most similar). Non-numeric values (e.g., strings) would cause a `TypeError` deep in the matching algorithm with no helpful error message for the agent.

## Findings

**Location:** `core/chat_tools.py` lines 3983-3989

Current validation only checks keys:

```python
# Validate weights
valid_dims = {"color", "brightness", "shot_scale", "audio", "embedding", "movement", "duration"}
invalid = set(weights.keys()) - valid_dims
if invalid:
    return {
        "success": False,
        "error": f"Invalid dimensions: {invalid}. Valid: {sorted(valid_dims)}"
    }
```

No validation of weight values follows. The `weights` dict is passed directly to `reference_guided_match()` which uses the values as multipliers in distance calculations.

**Impact scenarios:**
- `{"color": -1.0}` -- inverts color matching, selects most dissimilar clips
- `{"color": "high"}` -- `TypeError` when multiplying float by string in distance computation
- `{"color": 999.0}` -- technically works but overwhelms all other dimensions

## Proposed Solutions

### Option A: Add isinstance Check and 0.0-1.0 Range Validation (Recommended)

```python
# Validate weight values
for dim, value in weights.items():
    if not isinstance(value, (int, float)):
        return {
            "success": False,
            "error": f"Weight for '{dim}' must be a number, got {type(value).__name__}"
        }
    if value < 0.0 or value > 1.0:
        return {
            "success": False,
            "error": f"Weight for '{dim}' must be between 0.0 and 1.0, got {value}"
        }
```

**Pros:** Clear error messages, matches dialog UI slider range (0-100 mapped to 0.0-1.0), prevents semantic inversion
**Cons:** Slightly more restrictive than necessary (values >1.0 could be intentional power-user behavior)
**Effort:** Small

### Option B: Validate Type Only, Warn on Out-of-Range
Check `isinstance` but allow any non-negative float, logging a warning for values outside 0.0-1.0.

**Pros:** More permissive for advanced use cases
**Cons:** Still allows semantic inversion with negative values
**Effort:** Small

## Acceptance Criteria

- [ ] Non-numeric weight values produce a clear error message
- [ ] Negative weight values are rejected (or warned)
- [ ] Weight values outside 0.0-1.0 are handled (rejected or clamped)
- [ ] Error messages guide the agent to correct the input
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Agent tool parameter validation should cover both keys and values |
