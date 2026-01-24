---
status: complete
priority: p1
issue_id: "010"
tags: [code-review, security, path-traversal]
dependencies: []
---

# Path Traversal Vulnerability in Clip Export

## Problem Statement

The `source_name` is derived from the original video file's stem (filename without extension). If a video file has a malicious name containing path separators (e.g., `../../../etc/video.mp4`), this could lead to path traversal when constructing output files.

**Why it matters:** Files could be written outside the intended output directory.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/main_window.py` lines 493-533

```python
def _export_clips(self, clips: list[Clip]):
    # ...
    output_path = Path(output_dir)
    source_name = self.current_source.file_path.stem
    # ...
    output_file = output_path / f"{source_name}_scene_{i+1:03d}.mp4"
```

**Risk Level:** Medium - requires user to import a video with maliciously crafted filename

**Found by:** security-sentinel agent

## Proposed Solutions

### Option A: Sanitize filename (Recommended)
Add a sanitization function:
```python
import re
def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized[:100] or "video"

source_name = self._sanitize_filename(self.current_source.file_path.stem)
```
- **Pros:** Comprehensive, handles all dangerous characters
- **Cons:** None significant
- **Effort:** Small
- **Risk:** Low

### Option B: Use stem with path.name
Use `Path(stem).name` to strip any path components:
```python
source_name = Path(self.current_source.file_path.stem).name
```
- **Pros:** Very simple
- **Cons:** Doesn't handle all special characters
- **Effort:** Small
- **Risk:** Medium

## Technical Details

**Affected files:**
- `ui/main_window.py` - `_export_clips()` method
- Also check `_on_sequence_export_click()` for similar patterns

## Acceptance Criteria

- [ ] Source names with path separators are sanitized
- [ ] Source names with special characters are sanitized
- [ ] Export filenames are limited to safe length
- [ ] Existing export functionality continues to work

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Security-sentinel identified path traversal risk |

## Resources

- PR: Phase 2 Timeline & Composition
- OWASP Path Traversal guidance
