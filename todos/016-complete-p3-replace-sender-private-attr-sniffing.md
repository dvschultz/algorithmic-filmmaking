---
status: complete
priority: p3
issue_id: "016"
tags: [code-review, quality, signals]
dependencies: []
---

# Replace sender() Private Attribute Sniffing in _apply_reference_guide_sequence

## Problem Statement

`_apply_reference_guide_sequence` uses `self.sender()` to reach back into the dialog and read underscore-prefixed private attributes (`_last_ref_source_id`, `_last_weights`, `_last_allow_repeats`, `_last_match_timing`) to persist metadata on the sequence. This is fragile: if the signal connection changes (e.g., queued connection, intermediary slot, or lambda wrapper), `self.sender()` may return `None` or the wrong object, silently dropping metadata.

## Findings

**Location:** `ui/tabs/sequence_tab.py` lines 839-844

```python
# Store dialog config if available (for save/load round-trip)
dialog = self.sender()
if dialog and hasattr(dialog, '_last_ref_source_id'):
    sequence.reference_source_id = dialog._last_ref_source_id
    sequence.dimension_weights = dialog._last_weights
    sequence.allow_repeats = dialog._last_allow_repeats
    sequence.match_reference_timing = dialog._last_match_timing
```

The slot is connected at line 795:

```python
dialog.sequence_ready.connect(self._apply_reference_guide_sequence)
```

The `sequence_ready` signal currently only carries `list` (line 140 of `ui/dialogs/reference_guide_dialog.py`), so the slot has no way to receive metadata except through this `sender()` back-channel.

## Proposed Solutions

### Option A: Emit Richer Signal with Metadata Dict (Recommended)
Change `sequence_ready` to emit both the clip list and a metadata dict:

```python
# In ReferenceGuideDialog
sequence_ready = Signal(list, dict)  # clips, metadata

# When emitting:
metadata = {
    "reference_source_id": self._last_ref_source_id,
    "dimension_weights": self._last_weights,
    "allow_repeats": self._last_allow_repeats,
    "match_reference_timing": self._last_match_timing,
}
self.sequence_ready.emit(matched_clips, metadata)
```

Then the slot receives metadata as a parameter instead of sniffing the sender.

**Pros:** Clean signal contract, no coupling to dialog internals, survives connection type changes
**Cons:** Minor API change to signal signature
**Effort:** Small

### Option B: Pass Config as Named Signal Arguments via Dataclass
Create a small `ReferenceGuideConfig` dataclass and emit it alongside clips.

**Pros:** Strongly typed, extensible
**Cons:** More boilerplate for a small payload
**Effort:** Small

## Acceptance Criteria

- [ ] `_apply_reference_guide_sequence` no longer calls `self.sender()`
- [ ] `_apply_reference_guide_sequence` no longer reads underscore-prefixed private attributes from the dialog
- [ ] Metadata (reference_source_id, weights, allow_repeats, match_reference_timing) is still persisted on the sequence
- [ ] Save/load round-trip for reference_guided sequences still works
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | sender() + private attribute access is a fragile anti-pattern in Qt signal/slot wiring |
