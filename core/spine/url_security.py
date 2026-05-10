"""URL validation — single source of truth for video-download URLs.

Replaces the inline whitelist on ``core.downloader.VideoDownloader`` so both
the GUI agent (via ``core/chat_tools.py``) and the MCP server use the same
scheme-first, host-anchored validator.

Validation order:
1. Scheme: only ``http`` or ``https``.
2. Strip credentials and ports from the netloc.
3. Host whitelist with leading-dot anchor — ``host == 'youtube.com'`` or
   ``host.endswith('.youtube.com')`` matches; ``evil.notyoutube.com`` does not.
"""

from __future__ import annotations

from typing import Tuple
from urllib.parse import urlparse

# Bare-domain whitelist. Both ``host == domain`` (apex) and
# ``host.endswith('.' + domain)`` (subdomain) match.
ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "archive.org",
    }
)


def validate_url(url: str) -> Tuple[bool, str]:
    """Validate a download URL.

    Returns ``(is_valid, error_message)``. Only ``http`` and ``https`` schemes
    are accepted. The host must match an allowed domain at the apex or as a
    subdomain (leading-dot anchor — bare suffix matches like ``notyoutube.com``
    are rejected).
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty"

    try:
        parsed = urlparse(url)
    except (ValueError, AttributeError) as exc:
        return False, f"URL parsing error: {exc}"

    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP/HTTPS URLs are supported"

    netloc = parsed.netloc
    if not netloc:
        return False, "Invalid URL format"

    # Strip credentials (``user:pass@host``).
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]

    # Reject non-default ports — defends against SSRF-style targeting of
    # internal services on hosts that share a name with allowed CDNs.
    # Default ports (80 for http, 443 for https) are accepted, as is the
    # absence of an explicit port.
    default_port = {"http": 80, "https": 443}.get(parsed.scheme)
    if ":" in netloc:
        host_part, _, port_part = netloc.rpartition(":")
        if port_part and not port_part.isdigit():
            return False, "Invalid URL format"
        if port_part and int(port_part) != default_port:
            return False, f"Non-default port not allowed: {port_part}"
        netloc = host_part

    host = netloc.lower()
    if not host:
        return False, "Invalid URL format"

    for domain in ALLOWED_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True, ""

    # Internet Archive serves files from per-CDN subdomains like
    # ``ia800.us.archive.org``; allow them even though they do not end in
    # ``.archive.org`` directly.
    if host.endswith(".us.archive.org"):
        return True, ""

    return False, (
        f"Domain not supported: {host}. "
        "Supported: YouTube, Vimeo, Internet Archive"
    )
