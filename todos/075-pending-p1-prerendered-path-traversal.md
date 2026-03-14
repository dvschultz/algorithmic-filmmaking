---
status: pending
priority: p1
issue_id: "075"
tags: [code-review, security]
dependencies: []
---

# Path Traversal in SequenceClip.from_dict for prerendered_path

## Problem Statement

`SequenceClip.from_dict()` resolves `prerendered_path` relative to `base_path` without any traversal validation. A malicious project file could set `prerendered_path` to `"../../etc/passwd"` or any `..`-containing path. When `(base_path / p).resolve()` is called, it silently normalizes the traversal. This path is later passed to FFmpeg as `-i` input during export.

By contrast, `Source.from_dict()` in `models/clip.py` explicitly validates with `resolved.relative_to(base_path.resolve())` to prevent path escape.

## Findings

- **Python Reviewer**: Identified as Critical issue #2
- **Security Sentinel**: Identified as HIGH finding #1
- **Architecture Strategist**: Identified as Medium risk #4
- **Learnings Researcher**: docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md documents this exact pattern
- Location: `models/sequence.py` lines 99-104

## Proposed Solutions

### Option A: Add relative_to guard (Recommended)
Add the same traversal check that `Source.from_dict` uses.

```python
if prerendered and base_path:
    p = Path(prerendered)
    if not p.is_absolute():
        resolved = (base_path / p).resolve()
        try:
            resolved.relative_to(base_path.resolve())
        except ValueError:
            logger.warning("Path traversal detected in prerendered_path: %s", prerendered)
            prerendered = None
        else:
            prerendered = str(resolved)
```

- **Pros**: Consistent with existing pattern, one-line fix conceptually
- **Cons**: None
- **Effort**: Small
- **Risk**: Low

## Recommended Action

_To be filled during triage_

## Technical Details

- **Affected files**: `models/sequence.py`
- **Components**: SequenceClip deserialization, project load

## Acceptance Criteria

- [ ] `SequenceClip.from_dict` validates prerendered_path stays within base_path
- [ ] Path traversal attempts result in `prerendered_path=None` with warning log
- [ ] Test: malicious `"../../etc/passwd"` path is rejected
- [ ] Test: valid `"transformed_clips/c1_1_0_0.mp4"` path resolves correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Flagged by 3 of 7 review agents |

## Resources

- `Source.from_dict` in `models/clip.py` lines 140-146 (reference implementation)
- `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`
