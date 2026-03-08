---
status: complete
priority: p2
issue_id: "059"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Duplicated Ordering Logic Between Dialog and Chat Tool

## Problem Statement

The clip ordering logic is implemented twice: in `RoseHobartWorker._order_clips` (dialog, ~34 lines) and in `generate_rose_hobart` (chat tool, ~17 lines) with a `_ORDERING_MAP` dict. Both sort `(Clip, Source, confidence)` tuples identically with 6 ordering strategies. This is the largest duplication in the feature.

## Findings

**Code Simplicity Reviewer**: Most significant duplication — ~50 lines across two files. Highest maintenance risk: two independently-evolving sort implementations.

**Python Reviewer**: Confirmed the pattern divergence risk.

## Proposed Solutions

### Option A: Extract Shared Helper in faces.py (Recommended)

Create `order_matched_clips(matched: list, ordering: str) -> list` in `core/analysis/faces.py`. Both the dialog worker and chat tool call this single function.

**Pros:** ~35 LOC saved, eliminates consistency risk
**Cons:** Slightly couples dialog to core module (acceptable)
**Effort:** Small
**Risk:** Low

## Technical Details

- **Files:** `ui/dialogs/rose_hobart_dialog.py` lines 160-194, `core/chat_tools.py` lines 3981-3997

## Acceptance Criteria

- [ ] Single ordering function shared by dialog and chat tool
- [ ] All 6 ordering strategies work identically in both paths
- [ ] Unit test for ordering function added

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Code Simplicity Reviewer | Extract shared helpers when logic appears in both GUI and agent paths |
