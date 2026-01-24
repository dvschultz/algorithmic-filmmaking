---
status: complete
priority: p2
issue_id: "005"
tags: [code-review, security, downloader]
dependencies: []
---

# Filename Sanitization Incomplete

## Problem Statement

The `_sanitize_filename()` method does not handle several edge cases that could cause issues:
1. Null bytes and control characters
2. Windows reserved names (CON, PRN, NUL, etc.)
3. yt-dlp template injection via `%` character

**Why it matters:** While not immediately exploitable, these gaps could cause filesystem errors or unexpected behavior with malicious video titles.

## Findings

**Location:** `core/downloader.py:222-230`

```python
def _sanitize_filename(self, name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)  # Does NOT remove %
    sanitized = sanitized.strip('. ')
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized or "video"
```

**Issues:**
1. `%` is not removed - could cause yt-dlp template expansion: `%(id)s`
2. Control characters (0x00-0x1F) not removed
3. Windows reserved names not blocked

## Proposed Solutions

### Option A: Enhanced Sanitization (Recommended)
**Pros:** Covers all edge cases
**Cons:** Slightly more code
**Effort:** Small
**Risk:** Low

```python
def _sanitize_filename(self, name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Remove control characters
    sanitized = ''.join(c for c in name if c.isprintable())

    # Remove filesystem-unsafe characters AND % (yt-dlp template char)
    sanitized = re.sub(r'[<>:"/\\|?*%]', '', sanitized)
    sanitized = sanitized.strip('. ')

    # Check for Windows reserved names
    RESERVED = {'CON', 'PRN', 'AUX', 'NUL',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                'COM6', 'COM7', 'COM8', 'COM9',
                'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5',
                'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    if sanitized.upper() in RESERVED:
        sanitized = f"video_{sanitized}"

    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    return sanitized or "video"
```

## Recommended Action

Implement Option A.

## Technical Details

**Affected files:** `core/downloader.py`

## Acceptance Criteria

- [ ] `%` character is removed from filenames
- [ ] Control characters are stripped
- [ ] Windows reserved names are prefixed
- [ ] Existing functionality preserved

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Consider all platforms in filename handling |

## Resources

- [Microsoft Reserved Names](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file)
