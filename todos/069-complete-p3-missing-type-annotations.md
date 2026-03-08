---
status: complete
priority: p3
issue_id: "069"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Missing Type Annotations on Worker and Dialog

## Problem Statement

Several `__init__` methods and function signatures use bare `list` and `dict` without inner types:
- `RoseHobartWorker.__init__`: `clips: list` (should be `list[Clip]`)
- `RoseHobartDialog.__init__`: `clips` and `sources_by_id` lack annotations
- `FaceDetectionWorker.__init__`: `clips: list, sources_by_id: dict`
- `_order_clips` return type too loose: `list` (should be `list[tuple[Clip, Source]]`)

## Findings

**Python Reviewer**: These should use `list[Clip]`, `dict[str, Source]`, etc. The project uses typed models extensively.

## Proposed Solutions

### Option A: Add Specific Type Annotations

**Effort:** Small | **Risk:** Low

## Technical Details

- **Files:** `ui/dialogs/rose_hobart_dialog.py`, `ui/workers/face_detection_worker.py`

## Acceptance Criteria

- [ ] All public signatures have specific inner type annotations
- [ ] mypy/pyright would pass on these files

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer | Use specific generic types, not bare list/dict |
