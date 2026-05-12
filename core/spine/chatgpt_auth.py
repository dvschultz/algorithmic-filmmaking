"""ChatGPT subscription auth — spine-safe state layer.

This module owns the in-memory shape of the "Sign in with ChatGPT" feature.
It reads ``Settings.auth_mode`` and the keyring-backed OAuth token blob, and
exposes a tiny pure-function API every consumer reads through at call time.

Spine boundary discipline:

- Imports at module top level are stdlib only. No ``PySide6``, ``litellm``,
  ``httpx``, ``av``, ``mpv``, ``faster_whisper``, ``paddleocr``, ``mlx_vlm``.
  This module must be safely importable from headless processes (MCP server,
  CLI). The boundary is enforced by ``tests/test_spine_imports.py``.
- The OAuth flow itself (HTTP, PKCE, loopback listener) lives in a sibling
  ``core/spine/chatgpt_oauth_flow.py`` module that this module does NOT
  import — keeping the active-state read path free of any network-library
  dependency. ``chatgpt_oauth_flow`` is added in U3.

Why a single shared module instead of caching auth state in each consumer:
the chat worker is recreated per message, analysis workers re-resolve
credentials per call, and the MCP server reloads on each tool invocation —
all consumers re-read this module at call time. AE5 (mid-conversation auth
switch without app restart) holds because nothing caches mode at
construction time.

Refresh-contention locks (``REFRESH_LOCK`` / ``REFRESH_LOCK_ASYNC``) are
declared here so concurrent LLM calls coordinate on a single refresh
attempt per process. The cross-process contract is GUI-only-refresh
(documented in the plan): the GUI process holds the writer role; the MCP
server treats keyring as read-only and surfaces ``TokenExpiredError`` to
its callers rather than refreshing.

The OAuth client_id, endpoints, and scopes below are placeholder strings
until U3 fills them in from the open-source Codex CLI flow. They are
exposed at module scope so ``tests/test_chatgpt_auth.py`` can assert
HTTPS-only schemes — catching ``http://`` transcription mistakes at
merge time.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Tuple


# OAuth endpoint constants
#
# Placeholder HTTPS URLs — U3 replaces these with the real values borrowed
# from Codex CLI's open-source flow. The HTTPS-prefix assertion in
# tests/test_chatgpt_auth.py runs against whatever value is here, so a
# regression to http:// will fail at merge time regardless of U3's order.
CLIENT_ID: str = "PLACEHOLDER_UNTIL_U3"
AUTHORIZATION_ENDPOINT: str = "https://auth.openai.com/oauth/authorize"
TOKEN_ENDPOINT: str = "https://auth.openai.com/oauth/token"
CODEX_COMPLETION_ENDPOINT: str = "https://chatgpt.com/backend-api/codex/responses"
REDIRECT_URI_HOST: str = "127.0.0.1"
SCOPES: Tuple[str, ...] = ("openid", "profile", "email", "offline_access")


# Refresh-coordination locks (one per event loop / thread group within this
# process). Acquired before calling the refresh endpoint; the holder
# re-reads the token blob after acquiring to detect a parallel caller that
# already refreshed. Cross-process coordination is documented in the plan
# as GUI-only-refresh; the MCP process does not hold either of these locks.
REFRESH_LOCK: threading.Lock = threading.Lock()
REFRESH_LOCK_ASYNC: asyncio.Lock = asyncio.Lock()


class AuthMode(StrEnum):
    """User's selected authentication mode for LLM features.

    ``StrEnum`` (Python 3.11+) makes member values plain strings — so
    ``AuthMode.API_KEY == "api_key"`` and ``str(AuthMode.API_KEY) ==
    "api_key"`` are both True. This lets callers compare directly with
    the JSON string values stored in ``Settings.auth_mode`` without
    explicit conversion. Matches the repo precedent in
    ``core.update_models`` (UpdateChannel, UpdateAvailability,
    UpdateCapability).
    """

    API_KEY = "api_key"
    SUBSCRIPTION = "subscription"


@dataclass(frozen=True)
class AuthIdentity:
    """Immutable snapshot of the active sign-in identity.

    Returned by ``load_active_auth()`` when the user is signed in with
    ChatGPT subscription mode and a valid token blob is in keyring.
    Carries display-only identity plus the expiry timestamp the routing
    layer uses to decide whether to refresh before an LLM call.
    """

    account_email: str
    expires_at_unix: int

    @property
    def seconds_until_expiry(self) -> int:
        """Remaining lifetime of the access token in seconds.

        Can be negative when the token is already expired; callers should
        prefer ``is_token_expired(identity, leeway_seconds=...)`` for the
        boolean decision instead of comparing this to zero directly.
        """
        return int(self.expires_at_unix - time.time())


def is_token_expired(identity: AuthIdentity, leeway_seconds: int = 60) -> bool:
    """True when the access token is at-or-past its expiry, allowing leeway.

    A small positive ``leeway_seconds`` (default 60s) treats "about to
    expire" as expired so callers refresh before launching a long-running
    LLM call rather than mid-call. Set ``leeway_seconds=0`` to test the
    strict boundary.
    """
    return time.time() + leeway_seconds >= identity.expires_at_unix


def load_active_auth() -> Tuple[AuthMode, Optional[AuthIdentity]]:
    """Read the user's current auth mode and identity at call time.

    Returns:
        ``(AuthMode.API_KEY, None)`` when the user has selected API-key mode.

        ``(AuthMode.SUBSCRIPTION, AuthIdentity(...))`` when subscription mode
        is active and a valid token blob is in keyring.

        ``(AuthMode.SUBSCRIPTION, None)`` when subscription mode is selected
        but no token blob exists — the "re-auth needed" state. The routing
        layer (U6) translates this into ``TokenMissingError`` so the UI
        (U8) can prompt for sign-in.

    Reads ``Settings.auth_mode`` and the keyring blob fresh on every call —
    no caching. This is what makes AE5 (mid-conversation mode switch
    without restart) structurally easy: every consumer naturally picks up
    the latest state without dedicated propagation code.
    """
    # Lazy imports keep the spine boundary intact: settings.py is already
    # spine-safe but importing it at module scope here would couple our
    # import order to settings' module init. Function-scope keeps both
    # modules independently importable in any order.
    from core.settings import (
        get_chatgpt_oauth_token,
        load_settings,
    )

    settings = load_settings()
    if settings.auth_mode != AuthMode.SUBSCRIPTION.value:
        return AuthMode.API_KEY, None

    blob = get_chatgpt_oauth_token()
    if not blob:
        return AuthMode.SUBSCRIPTION, None

    # Extract identity fields with defensive defaults. The blob schema is
    # documented in core/settings.py:set_chatgpt_oauth_token and produced
    # by U3 (OAuth flow). We tolerate missing fields here rather than
    # crashing — a malformed-but-parseable blob still surfaces as
    # "subscription mode, no identity" rather than raising on every call.
    account_email = ""
    raw_email = blob.get("account_email")
    if isinstance(raw_email, str):
        account_email = raw_email

    expires_at_unix = 0
    raw_expiry = blob.get("expires_at_unix")
    if isinstance(raw_expiry, (int, float)):
        expires_at_unix = int(raw_expiry)

    identity = AuthIdentity(
        account_email=account_email,
        expires_at_unix=expires_at_unix,
    )
    return AuthMode.SUBSCRIPTION, identity
