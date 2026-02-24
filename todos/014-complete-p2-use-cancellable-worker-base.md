---
status: complete
priority: p2
issue_id: "014"
tags: [code-review, conventions, threading]
dependencies: []
---

# Use CancellableWorker Base Class for ReferenceMatchWorker

## Problem Statement

`ReferenceMatchWorker` inherits from `QThread` directly instead of `CancellableWorker` as specified in CLAUDE.md project conventions. The worker uses a bare `self._cancelled = False` boolean flag for cancellation, which is not thread-safe (no memory barrier). Additionally, the algorithm function `reference_guided_match()` has no cancellation check in its inner loop, meaning a cancel request has no effect until the entire matching computation finishes.

## Findings

**Location:** `ui/dialogs/reference_guide_dialog.py` lines 83-131

```python
class ReferenceMatchWorker(QThread):  # Should be CancellableWorker
    """Background worker for reference-guided matching."""

    progress = Signal(str)
    match_ready = Signal(list)
    error = Signal(str)

    def __init__(self, reference_clips, user_clips, weights,
                 allow_repeats, match_reference_timing, parent=None):
        super().__init__(parent)
        # ...
        self._cancelled = False  # Not thread-safe!

    def run(self):
        try:
            matched = reference_guided_match(...)
            if not self._cancelled:
                self.match_ready.emit(matched)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True  # Race condition: no threading.Event
```

**Convention reference:** `ui/workers/base.py` lines 14-24

```python
class CancellableWorker(QThread):
    """Base class for workers that support cancellation.

    Uses threading.Event for thread-safe cancellation, which is essential
    when workers use ThreadPoolExecutor or other multi-threaded patterns.

    Provides:
    - Thread-safe cancellation via threading.Event
    - cancel() method with logging
    - Common logging patterns for worker lifecycle
    """
```

**Algorithm inner loop has no cancellation check:** `core/remix/reference_match.py` lines 264-286

```python
for ref_idx, ref_vec in enumerate(ref_vectors):  # Could be 100s of iterations
    best_distance = float("inf")
    best_user_idx = None

    for user_idx, user_vec in enumerate(user_vectors):  # Nested loop: R*U iterations
        if not allow_repeats and user_idx in used_indices:
            continue
        dist = weighted_distance(...)  # No cancellation check anywhere
        if dist < best_distance:
            best_distance = dist
            best_user_idx = user_idx
    # ...
```

With 200 reference clips and 500 user clips, this is 100,000 distance computations with no opportunity to bail out.

## Proposed Solutions

### Option A: Inherit CancellableWorker + Pass Cancellation Check to Algorithm (Recommended)
**Pros:** Follows conventions, thread-safe cancellation, responsive cancel during long matches
**Cons:** Requires adding a `cancelled_check` parameter to `reference_guided_match()`
**Effort:** Small
**Risk:** Low

```python
# Worker:
class ReferenceMatchWorker(CancellableWorker):
    def run(self):
        try:
            matched = reference_guided_match(
                ...,
                cancelled_check=self._cancel_event.is_set,
            )
            if not self._cancel_event.is_set():
                self.match_ready.emit(matched)
        except CancelledError:
            pass

# Algorithm:
def reference_guided_match(
    ...,
    cancelled_check: Callable[[], bool] | None = None,
) -> list[tuple[Clip, Source]]:
    for ref_idx, ref_vec in enumerate(ref_vectors):
        if cancelled_check and cancelled_check():
            return result  # Return partial results or empty
        # ... matching logic
```

### Option B: Inherit CancellableWorker Only (No Algorithm Change)
**Pros:** Follows conventions for the worker class, simpler change
**Cons:** Cancel still waits for full algorithm completion
**Effort:** Small
**Risk:** Low (but misses the real benefit of cancellation)

### Option C: Keep QThread but Add threading.Event
**Pros:** Thread-safe cancellation without changing base class
**Cons:** Doesn't follow project conventions, reinvents CancellableWorker
**Effort:** Small
**Risk:** Low

## Recommended Action

Option A - Inherit from `CancellableWorker` and pass a cancellation check into the algorithm's inner loop. This follows project conventions and provides responsive cancellation.

## Technical Details

**Affected Files:**
- `ui/dialogs/reference_guide_dialog.py` - Change `ReferenceMatchWorker` to inherit `CancellableWorker`
- `core/remix/reference_match.py` - Add optional `cancelled_check` parameter to `reference_guided_match()`

**Verification:**
1. Start a reference-guided match with many clips
2. Click cancel during matching
3. Verify the operation stops promptly (within ~100ms)
4. Verify no signals emitted after cancellation
5. Run `pytest tests/test_reference_match.py` - all tests pass

## Acceptance Criteria

- [ ] `ReferenceMatchWorker` inherits from `CancellableWorker` instead of `QThread`
- [ ] `self._cancelled` boolean replaced with `self._cancel_event` from base class
- [ ] `reference_guided_match()` accepts optional `cancelled_check` callable
- [ ] Inner matching loop checks cancellation at each reference clip iteration
- [ ] Cancellation during matching returns partial/empty results gracefully
- [ ] All existing tests pass without modification
- [ ] Manual test: cancel mid-match works responsively

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | New workers should always inherit from CancellableWorker; algorithm functions with O(n*m) loops should accept cancellation callbacks |

## Resources

- PR #58: Reference-Guided Remixing
- `ui/workers/base.py` - CancellableWorker base class
- CLAUDE.md: "All workers inherit from CancellableWorker (ui/workers/base.py)"
