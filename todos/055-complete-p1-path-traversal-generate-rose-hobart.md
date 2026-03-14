---
status: complete
priority: p1
issue_id: "055"
tags: [code-review, security, agent-tools, rose-hobart]
dependencies: []
---

# Path Traversal in generate_rose_hobart Agent Tool

## Problem Statement

The `generate_rose_hobart` chat tool accepts `reference_image_path` as a raw string and converts it to a `Path` without using the codebase's existing `_validate_path()` function. The only check is `ref_path.exists()`. This allows the agent to read any file on disk that OpenCV can parse, including files outside the project or home directory. The `_validate_path()` function (used by 12 other tools) enforces no `..` traversal, absolute paths only, and restricts to safe roots ($HOME, /tmp).

## Findings

**Security Sentinel (High)**: An LLM agent (especially if prompt-injected through user-provided content like video descriptions) could craft a path pointing to sensitive files. The existence check alone can be used as a filesystem oracle. If a sensitive file happens to be a valid image format, its pixel data would be processed.

## Proposed Solutions

### Option A: Use _validate_path (Recommended)

Replace lines 3925-3927 in `core/chat_tools.py`:
```python
valid, error, ref_path = _validate_path(reference_image_path, must_exist=True)
if not valid:
    return {"success": False, "error": error}
```

Also add image extension allowlist:
```python
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
if ref_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
    return {"success": False, "error": f"Unsupported image format: {ref_path.suffix}"}
```

**Pros:** One-line fix, matches all other tools in the file
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 3925-3927

## Acceptance Criteria

- [ ] `_validate_path()` called before any file operations
- [ ] Image extension allowlist enforced
- [ ] Path traversal attempts return clear error

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Security Sentinel | Always use _validate_path() for any agent tool that accepts file paths |
