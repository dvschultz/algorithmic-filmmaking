---
status: complete
priority: p2
issue_id: "013"
tags: [code-review, cleanup, dead-code]
dependencies: []
---

# Remove Dead Code and Unused Features

## Problem Statement

The Phase 2 implementation contains several pieces of dead code, unused classes, and stub implementations that should be removed to reduce confusion and maintenance burden.

**Why it matters:** Dead code creates confusion, increases maintenance burden, and can mislead developers about what's actually functional.

## Findings

**Found by:** code-simplicity-reviewer, pattern-recognition-specialist, architecture-strategist agents

### Dead Code to Remove

| Location | Item | Lines | Reason |
|----------|------|-------|--------|
| `ui/timeline/track_item.py:63-90` | `TrackHeaderItem` class | 27 | Never instantiated |
| `core/remix/shuffle.py:110-128` | `shuffle_clips()` function | 19 | Never called |
| `models/sequence.py:45-46` | `Track.muted`, `Track.locked` fields | 2 | Never used |
| `models/sequence.py:105-110` | `Sequence.remove_track()` | 6 | Never called |
| `ui/timeline/timeline_widget.py:29` | `generate_requested` signal | 1 | Never emitted |

### Fake Implementations

| Location | Item | Issue |
|----------|------|-------|
| `ui/timeline/timeline_widget.py:207-219` | "Similarity" and "Building" algorithms | Listed in dropdown but just do sequential ordering, same as no algorithm |

**Estimated LOC removal:** ~70 lines

## Proposed Solutions

### Option A: Remove all dead code (Recommended)
Delete all identified dead code in a single cleanup commit.
- **Pros:** Clean codebase, no confusion
- **Cons:** None
- **Effort:** Small
- **Risk:** Low

### Option B: Remove dead code, keep algorithm stubs with TODO
Keep the dropdown but mark unimplemented algorithms clearly.
- **Pros:** Documents planned features
- **Cons:** Still creates false expectations in UI
- **Effort:** Small
- **Risk:** Low

## Technical Details

**Affected files:**
- `ui/timeline/track_item.py`
- `core/remix/shuffle.py`
- `models/sequence.py`
- `ui/timeline/timeline_widget.py`

## Acceptance Criteria

- [ ] `TrackHeaderItem` class removed
- [ ] `shuffle_clips()` function removed
- [ ] Unused `muted`, `locked` fields removed from Track
- [ ] `Sequence.remove_track()` removed
- [ ] `generate_requested` signal removed
- [ ] Fake algorithm implementations removed or clearly marked
- [ ] All imports still work
- [ ] App functionality unchanged

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Multiple agents identified same dead code |

## Resources

- PR: Phase 2 Timeline & Composition
