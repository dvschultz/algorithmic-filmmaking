---
status: complete
priority: p2
issue_id: "061"
tags: [code-review, performance, rose-hobart]
dependencies: []
---

# InsightFace Model Never Unloaded (~600MB Memory Leak)

## Problem Statement

The `unload_model()` function exists in `core/analysis/faces.py` but is never called anywhere. The InsightFace buffalo_l model (~600MB) persists in a module-level global for the entire session once loaded, even if face detection is never used again. The plan document explicitly flagged this as a risk.

## Findings

**Performance Oracle (P1)**: Other analysis modules follow the pattern of unloading after completion. This was listed as a known risk in the plan.

## Proposed Solutions

### Option A: Call unload_model in Finished Handlers (Recommended)

Add `faces.unload_model()` in:
- `_on_pipeline_face_detection_finished` in main_window.py
- After `RoseHobartWorker` finishes in the dialog

**Pros:** Frees 600MB, matches other analysis module patterns
**Cons:** Next face operation triggers re-load (~5 sec)
**Effort:** Small
**Risk:** Low

## Technical Details

- **Files:** `ui/main_window.py` (finished handler), `ui/dialogs/rose_hobart_dialog.py` (dialog close/finish)
- `core/analysis/faces.py` line 283: `unload_model()` already implemented

## Acceptance Criteria

- [ ] Model unloaded after face detection worker completes
- [ ] Model unloaded after Rose Hobart dialog closes
- [ ] Memory returns to pre-load levels after unload

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Performance Oracle | Always unload heavy ML models after pipeline completion |
