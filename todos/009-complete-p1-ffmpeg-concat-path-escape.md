---
status: complete
priority: p1
issue_id: "009"
tags: [code-review, security, ffmpeg]
dependencies: []
---

# FFmpeg Concat File Path Escape Vulnerability

## Problem Statement

The FFmpeg concat demuxer file format interprets certain characters specially. While single quotes are escaped, the current implementation does not handle backslashes (Windows), newline characters in filenames, or other special sequences that could cause parsing issues.

**Why it matters:** A maliciously crafted filename could potentially inject additional directives into the concat list file.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/core/sequence_export.py` lines 156-161

```python
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    for path in segment_paths:
        # Escape single quotes in path
        escaped_path = str(path).replace("'", "'\\''")
        f.write(f"file '{escaped_path}'\n")
    concat_file = f.name
```

**Risk Level:** High (though exploitability is low since segment paths are generated internally)

**Found by:** security-sentinel agent

## Proposed Solutions

### Option A: Add comprehensive escaping (Recommended)
Add backslash escaping and newline rejection:
```python
escaped_path = str(path.resolve()).replace("\\", "\\\\").replace("'", "'\\''")
if '\n' in str(path) or '\r' in str(path):
    raise ValueError("Invalid path with newline characters")
```
- **Pros:** Simple, handles all known edge cases
- **Cons:** None significant
- **Effort:** Small
- **Risk:** Low

### Option B: Use absolute path resolution only
Use `path.resolve()` to normalize paths before writing.
- **Pros:** Very simple
- **Cons:** Doesn't handle all edge cases
- **Effort:** Small
- **Risk:** Medium

## Technical Details

**Affected files:**
- `core/sequence_export.py`

## Acceptance Criteria

- [ ] Backslashes are properly escaped in concat file paths
- [ ] Paths with newline characters are rejected with clear error
- [ ] Existing export functionality continues to work
- [ ] Unit test added for path escaping

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Security-sentinel identified this during Phase 2 review |

## Resources

- PR: Phase 2 Timeline & Composition
- FFmpeg concat demuxer documentation
