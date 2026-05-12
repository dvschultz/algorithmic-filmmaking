"""MCP-side auth snapshot helpers for subscription-mode jobs.

The MCP server (``scene_ripper_mcp/server.py``) loads Settings *once* at
startup and stashes them in the lifespan context. That works for paths
and configuration, but it doesn't work for the ChatGPT subscription
OAuth token — the user signs in / out / refreshes via the GUI, and
the MCP server has to honor those changes without restart.

R-M1 (auth snapshot per job): each LLM-routed MCP tool that launches a
long-running job snapshots the current auth state at job start, pins
that snapshot to the job's lifetime, and fails the job loudly if the
snapshot becomes invalid mid-run rather than silently switching modes
between clips. The snapshot is the per-clip iterator's source of truth.

Trust-boundary statement (R-M2):

  The MCP server is trusted to the same degree as the GUI because both
  run as the same OS user and share the same keyring. This is acceptable
  for a local, single-user desktop application where the MCP server is
  launched by the GUI or by the user themselves. If the MCP server is
  ever exposed over a network socket rather than stdio, this decision
  must be revisited and an MCP-level auth layer added before that point.

The MCP test placement note (R-M3): MCP-server tests live in
``scene_ripper_mcp/tests/`` rather than ``tests/`` — that's the
established convention for MCP-scoped behavior (see
``scene_ripper_mcp/tests/test_integration.py``,
``scene_ripper_mcp/tests/test_jobs_runtime.py``). This module's tests
follow the same convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthSnapshot:
    """Frozen snapshot of auth state at the moment a job is launched.

    The snapshot's identity is its ``access_token`` plus its mode — if
    either changes between snapshot time and a per-clip LLM call, the
    job recognises the drift and fails loudly rather than mixing modes.

    ``access_token`` is None in API-key mode (no token to track) and
    when subscription mode is active but no blob is stored (the
    re-auth state).
    """

    auth_mode: str
    access_token: Optional[str]


def take_auth_snapshot() -> AuthSnapshot:
    """Capture the active auth state as a frozen snapshot.

    Reads ``Settings.auth_mode`` and the keyring blob fresh — same
    discipline as :func:`core.spine.chatgpt_auth.load_active_auth`,
    but typed for the snapshot use case. Call this once at the top
    of an MCP job handler; pin the result to the job and check
    against it via :func:`snapshot_still_valid` before each
    subsequent LLM call.
    """
    from core.settings import get_chatgpt_oauth_token, load_settings

    settings = load_settings()
    if settings.auth_mode != "subscription":
        return AuthSnapshot(auth_mode="api_key", access_token=None)
    blob = get_chatgpt_oauth_token()
    if not blob:
        return AuthSnapshot(auth_mode="subscription", access_token=None)
    access_token = blob.get("access_token")
    return AuthSnapshot(
        auth_mode="subscription",
        access_token=access_token if isinstance(access_token, str) else None,
    )


def snapshot_still_valid(snapshot: AuthSnapshot) -> bool:
    """True when the currently active auth state still matches the snapshot.

    Drift cases this catches mid-job:
    - User changes auth_mode in Settings while the job is running
    - User signs out (token removed from keyring)
    - Token was refreshed by another process in subscription mode
      (the access_token changes; the GUI-only-refresh contract from
      R-S5 means the MCP process should fail loudly rather than try
      to refresh on its own)
    """
    current = take_auth_snapshot()
    return (
        current.auth_mode == snapshot.auth_mode
        and current.access_token == snapshot.access_token
    )


__all__ = ["AuthSnapshot", "take_auth_snapshot", "snapshot_still_valid"]
