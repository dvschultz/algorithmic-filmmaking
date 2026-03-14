---
status: complete
priority: p1
issue_id: "056"
tags: [code-review, performance, ui, rose-hobart]
dependencies: []
---

# UI Thread Blocks on Reference Image Add

## Problem Statement

`RoseHobartDialog._on_add_reference` calls `extract_faces_from_image()` synchronously on the main thread. This function calls `_load_insightface()`, which on first invocation loads a ~600MB ONNX model. Even after model loading, inference takes 200-500ms on CPU. On first use, the UI freezes for 5-15 seconds with no spinner or feedback.

## Findings

**Performance Oracle (Critical)**: The dialog already has a QStackedWidget pattern that could show a loading state. Every other heavy operation in this codebase uses QThread workers.

**Code Simplicity Reviewer**: The `_ReferenceImageWidget` is justified and well-structured, but the synchronous extraction call in the parent dialog is the issue.

## Proposed Solutions

### Option A: Short-lived QThread for Extraction (Recommended)

Move `extract_faces_from_image` to a small QThread or QRunnable. Show a brief "Detecting face..." overlay on the reference widget while processing.

**Pros:** Non-blocking, consistent with codebase patterns
**Cons:** Slightly more complex dialog code
**Effort:** Medium
**Risk:** Low

### Option B: QApplication.processEvents with Spinner

Call `processEvents()` during extraction with a busy cursor. Less clean but simpler.

**Pros:** Quick to implement
**Cons:** Not truly non-blocking, can cause reentrancy issues
**Effort:** Small
**Risk:** Medium

## Technical Details

- **File:** `ui/dialogs/rose_hobart_dialog.py` lines 484-486
- Also consider pre-warming model when dialog opens (OPT-5 from performance review)

## Acceptance Criteria

- [ ] Adding a reference image does not freeze the UI
- [ ] User sees loading feedback while face detection runs
- [ ] First-time model load shows appropriate indication

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Performance Oracle | Always run InsightFace inference off the main thread |
