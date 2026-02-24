---
status: complete
priority: p2
issue_id: "013"
tags: [code-review, yagni, simplicity]
dependencies: []
---

# Remove Unimplemented `match_reference_timing` Parameter

## Problem Statement

The `match_reference_timing` parameter is accepted, stored, serialized, and tested throughout the entire stack but **never actually implemented**. The algorithm function `reference_guided_match()` accepts it as a parameter but performs no timing adjustment based on it. This is a classic YAGNI violation: plumbing for a feature that does not exist yet, adding complexity to every layer of the codebase.

## Findings

**The parameter is threaded through 7 files but has zero implementation logic:**

### Algorithm layer
- **`core/remix/reference_match.py` line 215**: `match_reference_timing: bool = False` - accepted but never read in the function body (lines 236-295 contain no reference to it)

### Dialog layer
- **`ui/dialogs/reference_guide_dialog.py` line 96**: Passed to `ReferenceMatchWorker.__init__()`
- **`ui/dialogs/reference_guide_dialog.py` line 104**: Stored as `self.match_reference_timing`
- **`ui/dialogs/reference_guide_dialog.py` line 118**: Passed through to `reference_guided_match()`
- **`ui/dialogs/reference_guide_dialog.py` line 452**: Read from `self.match_timing_check.isChecked()`

### Tab layer
- **`ui/tabs/sequence_tab.py` line 844**: `sequence.match_reference_timing = dialog._last_match_timing`
- **`ui/tabs/sequence_tab.py` line 1265**: `match_reference_timing: bool = False` parameter on `generate_reference_guided()`
- **`ui/tabs/sequence_tab.py` line 1310**: Passed to `reference_guided_match()`
- **`ui/tabs/sequence_tab.py` line 1347**: `sequence.match_reference_timing = match_reference_timing`

### Agent tool layer
- **`core/chat_tools.py` line 3958**: `match_reference_timing: bool = False` parameter on agent tool
- **`core/chat_tools.py` line 3968**: Docstring describes the parameter
- **`core/chat_tools.py` line 4005**: Passed to `sequence_tab.generate_reference_guided()`

### Model layer
- **`models/sequence.py` line 145**: `match_reference_timing: bool = False` field on `Sequence`
- **`models/sequence.py` lines 204-205**: Serialized to JSON when True
- **`models/sequence.py` line 224**: Deserialized from JSON

### Test layer
- **`tests/test_reference_match.py` lines 320, 327, 335, 344, 360**: Tested for persistence round-trip despite no functional behavior

**Total: ~20 lines of plumbing across 7 files for zero functionality.**

## Proposed Solutions

### Option A: Remove Entirely (Recommended)
**Pros:** Eliminates dead parameter from all layers, simplifies every function signature, removes misleading test coverage, follows YAGNI
**Cons:** If/when timing matching is needed, it must be re-added
**Effort:** Small (30 min - mostly deleting code)
**Risk:** None (parameter has no effect)

Steps:
1. Remove parameter from `reference_guided_match()` signature
2. Remove from `ReferenceMatchWorker`
3. Remove from `ReferenceGuideDialog` (remove checkbox widget)
4. Remove from `SequenceTab.generate_reference_guided()` and `_apply_reference_guide_sequence()`
5. Remove from `generate_reference_guided` agent tool
6. Remove from `Sequence` model
7. Remove/update tests
8. Add a comment in `reference_match.py`: `# Future: match_reference_timing could trim output clips to reference durations`

### Option B: Keep but Mark as NotImplemented
**Pros:** Preserves the API for future use
**Cons:** Still clutters 7 files, checkbox misleads users into thinking it works
**Effort:** Small (just add a `raise NotImplementedError` or `logger.warning`)
**Risk:** Low, but still carries dead code weight

### Option C: Implement It Now
**Pros:** Feature is actually useful - trim clips to match reference durations
**Cons:** Scope creep, not requested, adds complexity to the matching algorithm
**Effort:** Medium-Large (requires changes to clip in/out point handling)
**Risk:** Medium (new behavior to test)

## Recommended Action

Option A - Remove the parameter entirely. When the feature is actually needed, add it back with a proper implementation. The parameter currently misleads users via the dialog checkbox and inflates test surface area.

## Technical Details

**Affected Files:**
- `core/remix/reference_match.py` - Remove parameter from `reference_guided_match()`
- `ui/dialogs/reference_guide_dialog.py` - Remove from worker, remove checkbox
- `ui/tabs/sequence_tab.py` - Remove from `generate_reference_guided()` and apply method
- `core/chat_tools.py` - Remove from agent tool
- `models/sequence.py` - Remove field, update serialization
- `tests/test_reference_match.py` - Remove timing-related test assertions

**Verification:**
1. Run `pytest tests/test_reference_match.py` - all tests pass
2. Run `pytest tests/` - full suite passes
3. Verify dialog no longer shows non-functional checkbox
4. Verify saved projects without `match_reference_timing` still load correctly

## Acceptance Criteria

- [ ] `match_reference_timing` parameter removed from `reference_guided_match()` function
- [ ] Parameter removed from `ReferenceMatchWorker` and dialog
- [ ] Parameter removed from `SequenceTab.generate_reference_guided()`
- [ ] Parameter removed from `generate_reference_guided` agent tool
- [ ] Field removed from `Sequence` model (with backward-compatible deserialization)
- [ ] Dialog checkbox removed
- [ ] Tests updated to not reference the parameter
- [ ] All existing tests pass
- [ ] Comment added noting future timing-match feature possibility

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Parameters should not be threaded through the stack until they have actual implementation; follow YAGNI principle |

## Resources

- PR #58: Reference-Guided Remixing
- YAGNI: https://martinfowler.com/bliki/Yagni.html
