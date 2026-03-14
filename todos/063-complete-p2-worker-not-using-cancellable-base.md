---
status: complete
priority: p2
issue_id: "063"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# RoseHobartWorker Doesn't Inherit CancellableWorker

## Problem Statement

`RoseHobartWorker` inherits from `QThread` directly and uses a plain `self._cancelled = False` boolean. All other workers in the codebase inherit from `CancellableWorker` which uses `threading.Event` for thread-safe cancellation. The main thread sets `_cancelled` in `cancel()` while the worker thread reads it in `run()` — not thread-safe without Event or atomic operations.

## Findings

**Performance Oracle (P3)**: Breaks codebase convention. While unlikely to cause visible bugs under GIL, it could fail on more aggressive memory models.

**Learnings Researcher**: Past solution documents QThread guard flag pattern — every finished signal handler that spawns a downstream worker must have a boolean guard with `Qt.UniqueConnection`.

## Proposed Solutions

### Option A: Inherit from CancellableWorker (Recommended)

Change `class RoseHobartWorker(QThread)` to `class RoseHobartWorker(CancellableWorker)` and replace `self._cancelled` checks with `self.is_cancelled()`.

**Pros:** Thread-safe, consistent with codebase
**Cons:** Minor refactor
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/dialogs/rose_hobart_dialog.py` line 51
- Base class: `ui/workers/base.py` — `CancellableWorker`

## Acceptance Criteria

- [ ] RoseHobartWorker inherits CancellableWorker
- [ ] Cancellation uses thread-safe Event
- [ ] All `self._cancelled` replaced with `self.is_cancelled()`

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Performance Oracle + Learnings | Always use CancellableWorker base for thread-safe cancellation |
