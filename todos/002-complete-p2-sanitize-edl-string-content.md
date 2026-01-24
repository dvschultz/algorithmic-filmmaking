---
status: complete
priority: p2
issue_id: "002"
tags: [code-review, security, input-validation]
dependencies: []
---

# Sanitize EDL String Content to Prevent Format Injection

## Problem Statement

The `config.title` and `source.filename` values are written directly to the EDL file without sanitization. A malicious filename containing newlines could inject fake EDL entries or corrupt the file format.

## Findings

**Location:** `core/edl_export.py` lines 66-67, 116

```python
# Line 66 - Title written directly
lines.append(f"TITLE: {config.title}")

# Line 116 - Filename written directly
lines.append(f"* FROM CLIP NAME: {source.filename}")
```

**Attack vector:** If a video file has a filename like:
```
video\n\n002  AUX      V     C        00:00:00:00 00:00:10:00.mp4
```

This would inject a fake EDL event line into the output.

**Risk Level:** LOW - Requires user to have maliciously-named file on system

## Proposed Solutions

### Option A: Add Sanitization Function (Recommended)
**Pros:** Defense in depth, matches existing _sanitize_filename pattern
**Cons:** Small amount of additional code
**Effort:** Small (15 min)
**Risk:** None

```python
def _sanitize_edl_string(value: str) -> str:
    """Remove format-breaking characters from EDL field values."""
    return value.replace('\n', ' ').replace('\r', ' ')[:255]

# Usage:
lines.append(f"TITLE: {_sanitize_edl_string(config.title)}")
lines.append(f"* FROM CLIP NAME: {_sanitize_edl_string(source.filename)}")
```

### Option B: Accept Risk
**Pros:** No code change
**Cons:** Leaves potential format injection
**Effort:** None
**Risk:** Low but non-zero

## Recommended Action

Option A - Add sanitization for defense in depth

## Technical Details

**Affected Files:**
- `core/edl_export.py` - Add sanitization function and apply to title/filename

**Characters to remove:**
- Newlines (`\n`, `\r`) - break EDL format
- Consider length limit (255 chars) to prevent oversized fields

## Acceptance Criteria

- [ ] `_sanitize_edl_string()` function added
- [ ] Title field sanitized before writing
- [ ] Filename field sanitized before writing
- [ ] Test with filename containing newlines
- [ ] No change to valid filenames

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from security review | Always sanitize user content before file output |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Security Sentinel review findings
- Existing pattern: `_sanitize_filename()` in main_window.py
