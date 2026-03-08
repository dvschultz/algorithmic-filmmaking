---
status: complete
priority: p2
issue_id: "058"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Duplicated Sensitivity Thresholds

## Problem Statement

`core/analysis/faces.py` defines `SENSITIVITY_PRESETS` with lowercase keys. `ui/dialogs/rose_hobart_dialog.py` defines a separate `_SENSITIVITY_MAP` with title-case keys containing identical threshold values. Two sources of truth for the same data — if someone changes a threshold in one place, the other silently diverges.

## Findings

**Python Reviewer (Critical)**: Single source of truth violation. The chat tool correctly imports from `faces.py`.

**Code Simplicity Reviewer**: ~5 LOC saved, eliminates threshold drift risk.

## Proposed Solutions

### Option A: Dialog Derives from SENSITIVITY_PRESETS (Recommended)

```python
from core.analysis.faces import SENSITIVITY_PRESETS
# Combo box displays preset.title(), lookup uses SENSITIVITY_PRESETS[combo.currentText().lower()]
```

**Pros:** Single source of truth, 5 LOC saved
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **Files:** `core/analysis/faces.py` lines 23-27, `ui/dialogs/rose_hobart_dialog.py` lines 35-39

## Acceptance Criteria

- [ ] Only one definition of sensitivity thresholds (in faces.py)
- [ ] Dialog combo box still shows title-case labels

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer + Code Simplicity | Always derive UI display values from the canonical source |
