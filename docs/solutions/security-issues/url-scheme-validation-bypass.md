---
title: "URL Scheme Validation Bypasses Domain Whitelist"
category: security-issues
tags: [url-validation, scheme-validation, injection-attack, urlparse, python]
module: downloader
symptom: "URLs with dangerous schemes (file://, javascript://) bypass domain whitelist validation"
root_cause: "URL validation only checked netloc against whitelist without validating scheme"
date: 2026-01-24
---

# URL Scheme Validation Bypasses Domain Whitelist

## Problem

When validating URLs against a domain whitelist, checking only the `netloc` (host) allows dangerous URL schemes to bypass security controls.

## Symptom

URLs like `file://youtube.com/etc/passwd` or `javascript://youtube.com/alert(1)` pass domain validation because the host matches the whitelist, but the scheme allows unintended behavior.

## Attack Vectors

```python
# These all have valid whitelisted hosts but dangerous schemes
"file://youtube.com/etc/passwd"       # Local file access
"javascript://youtube.com/alert(1)"   # Script execution
"data://youtube.com/base64,..."       # Data injection

# Credential/port confusion attacks
"https://user:pass@youtube.com@evil.com/path"  # Credential-based host confusion
"https://youtube.com:8080/path"                # Port-based confusion (may work)
```

## Root Cause

```python
# VULNERABLE: Only checks host, not scheme
def is_valid_url(self, url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()  # scheme is ignored!

    if host not in self.ALLOWED_DOMAINS:
        return False, "Domain not supported"
    return True, ""
```

## Solution

Validate scheme first, then sanitize the host by stripping ports and credentials:

```python
def is_valid_url(self, url: str) -> tuple[bool, str]:
    """Check if URL is from an allowed domain with safe scheme."""
    try:
        parsed = urlparse(url)

        # 1. Validate scheme FIRST - only allow HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            return False, "Only HTTP/HTTPS URLs are supported"

        host = parsed.netloc.lower()

        if not host:
            return False, "Invalid URL format"

        # 2. Remove port if present (e.g., youtube.com:443)
        if ":" in host:
            host = host.rsplit(":", 1)[0]

        # 3. Remove credentials if present (e.g., user:pass@youtube.com)
        if "@" in host:
            host = host.rsplit("@", 1)[-1]

        # 4. Check against whitelist
        if host not in self.ALLOWED_DOMAINS:
            return False, f"Domain not supported: {host}"

        return True, ""

    except (ValueError, AttributeError) as e:
        return False, f"URL parsing error: {e}"
```

## Key Points

1. **Scheme validation first** - Reject non-HTTP(S) before any other checks
2. **Port stripping** - Use `rsplit(":", 1)[0]` to handle `host:port`
3. **Credential stripping** - Use `rsplit("@", 1)[-1]` to handle `user:pass@host`
4. **Specific exceptions** - Catch `ValueError`/`AttributeError`, not bare `except`

## Prevention

- Always validate URL scheme explicitly
- Strip ports and credentials before host comparison
- Use allowlists, not blocklists for URL validation
- Consider using a dedicated URL validation library for complex cases

## References

- [OWASP Input Validation Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
- [Python urlparse documentation](https://docs.python.org/3/library/urllib.parse.html)
