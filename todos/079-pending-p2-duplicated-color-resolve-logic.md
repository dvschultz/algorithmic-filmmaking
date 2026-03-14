---
status: pending
priority: p2
issue_id: "079"
tags: [code-review, quality, duplication]
dependencies: []
---

# Duplicated Color-Resolve Logic Between MainWindow and SequenceExporter

## Problem Statement

The same "resolve dominant color for a sequence clip" logic exists in:
1. `SequenceExporter._resolve_sequence_clip_color` in `core/sequence_export.py` lines 329-357
2. `MainWindow._resolve_sequence_clip_bar_color` in `ui/main_window.py`
3. Inline logic in `order_matched_clips._color_key` in `core/analysis/faces.py`

All three extract `dominant_colors[0]`, clamp to 0-255 int RGB. Future changes to color resolution will need updates in 3 places.

## Findings

- **Python Reviewer**: Medium issue #12
- **Code Simplicity Reviewer**: Finding #1

## Proposed Solutions

### Option A: Extract shared utility function
Create `resolve_clip_dominant_color(clip) -> Optional[tuple[int,int,int]]` in a shared location (e.g., `models/clip.py` or `core/analysis/color.py`).

- **Pros**: Single source of truth, eliminates drift
- **Cons**: Need to parameterize how clip data is accessed (direct clip vs tuple lookup)
- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] Dominant color resolution defined in one place
- [ ] All three call sites use the shared function

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | 2 agents flagged independently |
