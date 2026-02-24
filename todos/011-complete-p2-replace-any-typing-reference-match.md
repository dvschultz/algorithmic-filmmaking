---
status: complete
priority: p2
issue_id: "011"
tags: [code-review, type-safety, quality]
dependencies: []
---

# Replace `Any` Typing in reference_match.py

## Problem Statement

Heavy use of `Any` typing in `core/remix/reference_match.py` defeats type safety. Every function that accepts a `Clip` or `Source` object uses `Any` instead of the actual types, making it impossible for type checkers (mypy, pyright) to catch attribute access errors. For example, `clip.cinematography.shot_size` is accessed without any static guarantee that `clip` has a `cinematography` attribute.

## Findings

**Location:** `core/remix/reference_match.py`

The following functions all use `Any` where `Clip` or `Source` types should be used:

- **Line 9**: `from typing import Any, Optional` - `Any` imported for use throughout
- **Line 56**: `def _get_proximity_score(clip: Any) -> float:` - should be `Clip`
- **Lines 68-72**: `def extract_feature_vector(clip: Any, source: Any, ...) -> dict[str, Any]:` - `clip` should be `Clip`, `source` should be `Source`
- **Lines 115-117**: `def compute_normalizers(all_vectors: list[dict[str, Any]], ...) -> dict[str, tuple[float, float]]:` - return dict values are `Any` but are always numeric
- **Lines 210-216**: `def reference_guided_match(reference_clips: list[tuple[Any, Any]], user_clips: list[tuple[Any, Any]], ...) -> list[tuple[Any, Any]]:` - all `Any` should be `Clip` and `Source`
- **Line 262**: `result: list[tuple[Any, Any]] = []` - should be `list[tuple[Clip, Source]]`
- **Lines 298-300**: `def get_active_dimensions_for_clips(clips: list[Any]) -> list[str]:` - should be `list[Clip]`

The `Any` typing also bleeds into the return types, so callers of `reference_guided_match` lose type information on the returned tuples.

**Impact:** A typo like `clip.cinematograhpy` (misspelling) would not be caught by static analysis. Same for accessing non-existent attributes after a model refactor.

## Proposed Solutions

### Option A: TYPE_CHECKING Guard with Proper Annotations (Recommended)
**Pros:** Full type safety, no circular import risk, zero runtime cost
**Cons:** Slightly more boilerplate at top of file
**Effort:** Small
**Risk:** None

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.clip import Clip, Source

def _get_proximity_score(clip: Clip) -> float: ...
def extract_feature_vector(clip: Clip, source: Source, ...) -> dict[str, float | list[float] | str]: ...
def reference_guided_match(
    reference_clips: list[tuple[Clip, Source]],
    user_clips: list[tuple[Clip, Source]],
    ...
) -> list[tuple[Clip, Source]]: ...
```

### Option B: Direct Imports
**Pros:** Simplest approach
**Cons:** May introduce circular import if models ever import from remix
**Effort:** Small
**Risk:** Low (no current circular dependency, but fragile)

### Option C: Protocol-Based Typing
**Pros:** Decouples from concrete model classes
**Cons:** Over-engineered for this use case, more code to maintain
**Effort:** Medium
**Risk:** Low

## Recommended Action

Option A - Use `TYPE_CHECKING` guard with `from __future__ import annotations` for zero-cost type safety.

## Technical Details

**Affected Files:**
- `core/remix/reference_match.py` - Replace all `Any` with proper `Clip`/`Source` types

**Verification:**
1. Run `mypy core/remix/reference_match.py` or `pyright` - should pass with no errors
2. Intentionally misspell an attribute access - type checker should flag it
3. Run `pytest tests/test_reference_match.py` - all tests still pass

## Acceptance Criteria

- [ ] No `Any` type annotations remain in `reference_match.py` (except for truly dynamic values)
- [ ] `Clip` and `Source` types used for all clip/source parameters and return types
- [ ] `TYPE_CHECKING` guard prevents any runtime import overhead
- [ ] All existing tests pass without modification
- [ ] Type checker (mypy/pyright) reports no new errors

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | `Any` typing should be replaced with proper types using TYPE_CHECKING guard to maintain type safety without runtime cost |

## Resources

- PR #58: Reference-Guided Remixing
- Python typing docs: https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
