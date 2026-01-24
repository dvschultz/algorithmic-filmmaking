---
status: pending
priority: p1
issue_id: "001"
tags: [code-review, security, downloader]
dependencies: []
---

# URL Scheme Validation Missing

## Problem Statement

The URL validation in `VideoDownloader.is_valid_url()` only checks the domain whitelist but does not validate the URL scheme. This allows potentially dangerous URL schemes like `file://`, `javascript://`, or other non-HTTP schemes to pass validation.

**Why it matters:** An attacker could craft a URL like `file://youtube.com/etc/passwd` which would pass the domain check but could potentially be used to access local files or trigger unexpected behavior in yt-dlp.

## Findings

**Location:** `core/downloader.py:54-72`

```python
def is_valid_url(self, url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # MISSING: No scheme validation!
        if host not in self.ALLOWED_DOMAINS:
            return False, f"Domain not supported: {host}"
        return True, ""
```

**Attack vectors:**
- `file://youtube.com/etc/passwd` - File scheme with whitelisted domain
- `javascript://youtube.com/alert` - JavaScript scheme
- URLs with credentials: `https://user:pass@youtube.com@evil.com/path`

## Proposed Solutions

### Option A: Add Scheme Validation (Recommended)
**Pros:** Simple, targeted fix
**Cons:** None
**Effort:** Small
**Risk:** Low

```python
def is_valid_url(self, url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)

        # Validate scheme FIRST
        if parsed.scheme not in ('http', 'https'):
            return False, "Only HTTP/HTTPS URLs are supported"

        host = parsed.netloc.lower()

        # Remove port if present
        if ':' in host:
            host = host.rsplit(':', 1)[0]

        # Remove credentials if present
        if '@' in host:
            host = host.rsplit('@', 1)[-1]

        if not host:
            return False, "Invalid URL format"

        if host not in self.ALLOWED_DOMAINS:
            return False, f"Domain not supported: {host}"

        return True, ""
    except (ValueError, AttributeError) as e:
        return False, f"URL parsing error: {e}"
```

## Recommended Action

Implement Option A - add scheme validation and handle edge cases with credentials/ports.

## Technical Details

**Affected files:** `core/downloader.py`

## Acceptance Criteria

- [ ] Only `http://` and `https://` schemes are accepted
- [ ] URLs with embedded credentials are handled safely
- [ ] URLs with ports work correctly
- [ ] Test: `file://youtube.com/test` is rejected
- [ ] Test: `https://youtube.com/watch?v=xxx` works

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | URL scheme validation is critical for security |

## Resources

- [OWASP URL Validation](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
