"""Shared classifier for subscription-mode auth errors.

U8's scope is "translate the new exception types from U6 into user-visible
UX." The hard part — defining the types and raising them from the right
places — lives in :mod:`core.codex_client`. This module provides the
one-line bridge workers and the chat UI use to decide which dialog to show.

The classifier returns a stable string category so callers can branch
in a switch-style block without importing every error type. Stable
across the codebase:

  - ``"token_required"`` — user needs to sign in (no token, or token
    cannot be refreshed); show the re-auth prompt
  - ``"token_expired"`` — same UX as token_required, but distinguishes
    the case where a refresh attempt was made and failed
  - ``"quota_exceeded"`` — show the Plus-quota message with the
    "wait or switch to API key" action
  - ``"codex_backend"`` — generic backend failure; show a transient
    "OpenAI's Codex service had a hiccup" message
  - ``"none"`` — not a subscription-auth error; the caller surfaces
    it via its existing error path

The re-auth prompt entry point (R-E1) is the settings dialog focused
on the subscription panel — workers emit the category and the main
window opens settings rather than spawning a separate OAuthWorker.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

# Re-export the error types so callers have a single import point.
from core.codex_client import (
    CodexBackendError,
    CodexError,
    MalformedCodexResponseError,
    QuotaExceededError,
    TokenExpiredError,
    TokenMissingError,
)


class AuthErrorCategory(StrEnum):
    """Stable category strings U5's re-auth dialog branches on.

    StrEnum (Python 3.11+) means ``AuthErrorCategory.TOKEN_REQUIRED ==
    "token_required"`` is True, so any existing string-comparison
    callers keep working unchanged. Matches the precedent set by
    ``core.update_models.UpdateChannel``.
    """

    TOKEN_REQUIRED = "token_required"
    TOKEN_EXPIRED = "token_expired"
    QUOTA_EXCEEDED = "quota_exceeded"
    CODEX_BACKEND = "codex_backend"
    NONE = "none"


# Backwards-compatible module-level constant aliases. Callers that
# import the bare names keep working; new code prefers the enum.
CATEGORY_TOKEN_REQUIRED = AuthErrorCategory.TOKEN_REQUIRED
CATEGORY_TOKEN_EXPIRED = AuthErrorCategory.TOKEN_EXPIRED
CATEGORY_QUOTA_EXCEEDED = AuthErrorCategory.QUOTA_EXCEEDED
CATEGORY_CODEX_BACKEND = AuthErrorCategory.CODEX_BACKEND
CATEGORY_NONE = AuthErrorCategory.NONE


def classify_subscription_error(exc: BaseException) -> str:
    """Map a caught exception to a stable UX category string.

    Returns ``"none"`` for anything that isn't a subscription-auth error,
    so workers can fall through to their existing error-handling path.
    """
    if isinstance(exc, TokenMissingError):
        return CATEGORY_TOKEN_REQUIRED
    if isinstance(exc, TokenExpiredError):
        return CATEGORY_TOKEN_EXPIRED
    if isinstance(exc, QuotaExceededError):
        return CATEGORY_QUOTA_EXCEEDED
    if isinstance(exc, CodexBackendError):
        return CATEGORY_CODEX_BACKEND
    if isinstance(exc, MalformedCodexResponseError):
        return CATEGORY_CODEX_BACKEND
    return CATEGORY_NONE


def user_message_for_category(category: str, raw_message: Optional[str] = None) -> str:
    """Friendly user-facing string for each category.

    ``raw_message`` is the exception's own message; we use it as a hint
    in the codex_backend case but never display it raw for the auth
    categories (those have purpose-built strings the user sees often
    enough that hand-tuning matters).

    Exact wording is intentionally short and actionable. Workers pass
    this string to whatever toast / dialog they own; the main window's
    re-auth dialog uses category branching for actions, not wording.
    """
    if category == CATEGORY_TOKEN_REQUIRED:
        return (
            "ChatGPT sign-in required. Open Settings → API Keys to sign in, "
            "or switch to API-key mode."
        )
    if category == CATEGORY_TOKEN_EXPIRED:
        return (
            "Your ChatGPT sign-in has expired. Sign in again from Settings, "
            "or switch to API-key mode."
        )
    if category == CATEGORY_QUOTA_EXCEEDED:
        return (
            "You've hit your ChatGPT Plus quota for this 3-hour window. "
            "Wait, or switch to API-key mode to keep working."
        )
    if category == CATEGORY_CODEX_BACKEND:
        if raw_message:
            return f"OpenAI's Codex service returned an error: {raw_message}"
        return "OpenAI's Codex service had a hiccup. Try again, or switch to API-key mode."
    return ""


__all__ = [
    "AuthErrorCategory",
    "CATEGORY_TOKEN_REQUIRED",
    "CATEGORY_TOKEN_EXPIRED",
    "CATEGORY_QUOTA_EXCEEDED",
    "CATEGORY_CODEX_BACKEND",
    "CATEGORY_NONE",
    "CodexBackendError",
    "CodexError",
    "MalformedCodexResponseError",
    "QuotaExceededError",
    "TokenExpiredError",
    "TokenMissingError",
    "classify_subscription_error",
    "user_message_for_category",
]
