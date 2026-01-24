---
status: complete
priority: p2
issue_id: "004"
tags: [code-review, dead-code, imports]
dependencies: ["001"]
---

# Remove Unused Imports from edl_export.py

## Problem Statement

The `core/edl_export.py` module imports `Clip` and `SequenceClip` but neither is used in the code. These are dead imports that should be removed.

## Findings

**Location:** `core/edl_export.py` lines 7-8

```python
from models.clip import Clip, Source  # Clip is unused
from models.sequence import Sequence, SequenceClip  # SequenceClip is unused
```

- `Clip` was only referenced in the type hint for the unused `clips` parameter
- `SequenceClip` is never used as a type annotation - the code iterates over clips from `sequence.get_all_clips()` but doesn't annotate them

## Proposed Solutions

### Option A: Remove Unused Imports (Recommended)
**Pros:** Clean code, passes linting
**Cons:** None
**Effort:** Trivial (2 min)
**Risk:** None

```python
from models.clip import Source
from models.sequence import Sequence
```

## Recommended Action

Option A - Remove unused imports

## Technical Details

**Affected Files:**
- `core/edl_export.py` - Update import statements

**Note:** This depends on removing the `clips` parameter first (issue #001), as `Clip` is referenced in that parameter's type hint.

## Acceptance Criteria

- [ ] `Clip` import removed
- [ ] `SequenceClip` import removed
- [ ] No import errors when module is loaded
- [ ] Linting passes

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Remove imports when removing features that use them |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Code Simplicity Reviewer findings
