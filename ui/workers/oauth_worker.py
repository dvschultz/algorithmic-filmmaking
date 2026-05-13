"""Qt worker that drives the Sign in with ChatGPT flow off the UI thread.

Wraps the GUI-free OAuth primitives in ``core/spine/chatgpt_oauth_flow``.
The orchestration sequence:

  1. ``run()`` generates a PKCE pair and the authorization URL
  2. emits ``authorization_url_ready(url)`` — the settings dialog's slot
     opens this in the user's default browser via ``QDesktopServices``
  3. blocks on ``run_loopback_listener`` until the redirect lands or
     timeout fires; checking ``is_cancelled()`` between iterations is
     handled by the underlying listener (a small per-iteration timeout
     plus this worker checking the cancel flag in between).
  4. exchanges the code for a token blob and emits ``auth_complete(blob)``
  5. on any failure path, emits ``auth_failed(category, user_message)``
     exactly once

Signal-duplication discipline (mirrors the prior bug fix in
docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md):

- ``_finished_handled`` is reset to False at the top of every ``run()``
  call so two sequential sign-in attempts on the same worker instance
  fire exactly one terminal signal each
- The handler slot in the settings dialog (U5) connects with
  ``Qt.UniqueConnection`` and guards its own ``_oauth_in_progress``
  flag for the per-handler half of the pattern
- Every emission path goes through ``_emit_terminal()`` which sets the
  flag and short-circuits on a re-entry

Cancellation: ``litellm.completion``-style "non-interruptible during one
blocking HTTP call" applies here too. The token-endpoint POST is bounded
to ~30s by httpx; the loopback listener checks for cancellation between
its short (~1s) waits. The acceptable worst-case freeze on cancel is one
in-flight HTTP call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class OAuthFailureCategory(StrEnum):
    """OAuth failure categories the settings dialog branches on.

    StrEnum (Python 3.11+) so ``category == "cancelled"`` continues to
    work for the existing string-comparison call sites. Matches the
    precedent set by ``core.update_models.UpdateChannel``.
    """

    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    NETWORK = "network"
    REJECTED = "rejected"
    MALFORMED = "malformed"


# Backwards-compatible module-level constant aliases.
AUTH_FAILED_CANCELLED = OAuthFailureCategory.CANCELLED
AUTH_FAILED_TIMEOUT = OAuthFailureCategory.TIMEOUT
AUTH_FAILED_NETWORK = OAuthFailureCategory.NETWORK
AUTH_FAILED_REJECTED = OAuthFailureCategory.REJECTED
AUTH_FAILED_MALFORMED = OAuthFailureCategory.MALFORMED


class OAuthWorker(CancellableWorker):
    """Cancellable worker that performs the Sign in with ChatGPT OAuth flow."""

    # Emitted once when the authorization URL is ready for the browser.
    authorization_url_ready = Signal(str)

    # Emitted exactly once at the end of a successful flow. Payload is
    # the token blob dict (the shape persisted by
    # ``core.settings.set_chatgpt_oauth_token``).
    auth_complete = Signal(dict)

    # Emitted exactly once on any failure. (category, user_message)
    # where category is one of the AUTH_FAILED_* constants.
    auth_failed = Signal(str, str)

    def __init__(self, timeout_seconds: float = 300.0, parent=None):
        super().__init__(parent)
        self._timeout_seconds = timeout_seconds
        self._finished_handled = False

    # -- Internal: terminal-emission guard ----------------------------------

    def _emit_terminal_complete(self, blob: dict) -> None:
        """Emit auth_complete exactly once, even under signal-duplication."""
        if self._finished_handled:
            logger.warning("OAuthWorker: suppressed duplicate auth_complete emission")
            return
        self._finished_handled = True
        self.auth_complete.emit(blob)

    def _emit_terminal_failed(self, category: str, user_message: str) -> None:
        """Emit auth_failed exactly once, even under signal-duplication."""
        if self._finished_handled:
            logger.warning(
                "OAuthWorker: suppressed duplicate auth_failed emission "
                f"(category={category})"
            )
            return
        self._finished_handled = True
        self.auth_failed.emit(category, user_message)

    # -- QThread.run --------------------------------------------------------

    def run(self) -> None:
        """Thread entry point. Runs the async coroutine via asyncio.run."""
        # Reset guard each run so the same worker instance can be reused
        # across sign-in attempts without a stale flag short-circuiting
        # the next attempt's terminal signal.
        self._finished_handled = False

        try:
            asyncio.run(self._async_run())
        except Exception as exc:  # final safety net — should not reach here
            logger.exception("OAuthWorker: unexpected error in run()")
            self._emit_terminal_failed(
                AUTH_FAILED_NETWORK, f"Unexpected sign-in error: {exc}"
            )

    # -- Async orchestration -----------------------------------------------

    async def _async_run(self) -> None:
        from core.spine.chatgpt_oauth_flow import (
            OAuthCancelledError,
            OAuthHTTPError,
            OAuthHostMismatchError,
            OAuthMalformedResponseError,
            OAuthTimeoutError,
            build_authorization_url,
            exchange_code_for_token,
            generate_pkce_pair,
            run_loopback_listener,
        )

        # 1. Build the authorization URL on this thread (cheap) and signal
        #    the dialog so it can open the browser on the main thread.
        try:
            pair = generate_pkce_pair()
            auth_request = build_authorization_url(pair.challenge)
        except Exception as exc:
            logger.exception("OAuthWorker: failed to build authorization URL")
            self._emit_terminal_failed(
                AUTH_FAILED_NETWORK,
                f"Could not start sign-in: {exc}",
            )
            return

        if self.is_cancelled():
            self._emit_terminal_failed(
                AUTH_FAILED_CANCELLED, "Sign-in was cancelled before it started."
            )
            return

        self.authorization_url_ready.emit(auth_request.url)

        # 2. Run the loopback listener. This blocks until a redirect lands,
        #    timeout fires, or the user cancels. We run it in a worker
        #    thread so the asyncio loop can check cancellation between
        #    short waits — each handle_request() call has a 1s ceiling
        #    on its blocking wait set by the listener internally.
        try:
            auth_code = await asyncio.get_running_loop().run_in_executor(
                None,
                self._run_listener_with_cancellation,
                auth_request.port,
                auth_request.state,
            )
        except OAuthCancelledError as exc:
            self._emit_terminal_failed(
                AUTH_FAILED_CANCELLED, _user_message_for_cancellation(exc)
            )
            return
        except OAuthTimeoutError:
            self._emit_terminal_failed(
                AUTH_FAILED_TIMEOUT,
                "Sign-in timed out — the browser window may not have opened.",
            )
            return
        except OAuthHostMismatchError as exc:
            self._emit_terminal_failed(
                AUTH_FAILED_REJECTED,
                f"Sign-in rejected: {exc}",
            )
            return
        except OAuthHTTPError as exc:
            self._emit_terminal_failed(
                AUTH_FAILED_REJECTED,
                f"Sign-in rejected by OpenAI: {exc}",
            )
            return
        except Exception as exc:
            logger.exception("OAuthWorker: loopback listener failed")
            self._emit_terminal_failed(
                AUTH_FAILED_NETWORK, f"Sign-in failed: {exc}"
            )
            return

        if auth_code is None:
            # Cancellation while waiting in the listener thread.
            self._emit_terminal_failed(
                AUTH_FAILED_CANCELLED, "Sign-in was cancelled."
            )
            return

        if self.is_cancelled():
            self._emit_terminal_failed(
                AUTH_FAILED_CANCELLED, "Sign-in was cancelled."
            )
            return

        # 3. Exchange the code for a token blob. httpx call bounded to 30s.
        try:
            blob = await asyncio.get_running_loop().run_in_executor(
                None,
                exchange_code_for_token,
                auth_code.code,
                pair.verifier,
                auth_request.redirect_uri,
            )
        except OAuthMalformedResponseError as exc:
            self._emit_terminal_failed(
                AUTH_FAILED_MALFORMED,
                f"Sign-in response was malformed: {exc}",
            )
            return
        except OAuthHTTPError as exc:
            self._emit_terminal_failed(
                AUTH_FAILED_REJECTED,
                f"Sign-in rejected by OpenAI: {exc}",
            )
            return
        except Exception as exc:
            logger.exception("OAuthWorker: code exchange failed")
            self._emit_terminal_failed(
                AUTH_FAILED_NETWORK,
                f"Could not complete sign-in: {exc}",
            )
            return

        self._emit_terminal_complete(blob.as_dict())

    # -- Loopback-listener cancellation wrapper ----------------------------

    def _run_listener_with_cancellation(
        self, port: int, expected_state: str
    ):
        """Run the loopback listener in short iterations so cancellation
        is observable. Returns None when cancelled; otherwise the
        AuthorizationCode from the listener (or raises an OAuth* error).
        """
        from core.spine.chatgpt_oauth_flow import (
            OAuthTimeoutError,
            run_loopback_listener,
        )

        deadline = time.time() + self._timeout_seconds
        # Poll in 2s slices so the user's cancel is observed within
        # ~2s of pressing the cancel button.
        slice_seconds = 2.0
        while time.time() < deadline:
            if self.is_cancelled():
                return None
            remaining = deadline - time.time()
            this_slice = min(slice_seconds, remaining)
            try:
                return run_loopback_listener(
                    port=port,
                    expected_state=expected_state,
                    timeout_seconds=this_slice,
                )
            except OAuthTimeoutError:
                # This slice expired with no redirect — check cancel,
                # then loop back into another slice.
                continue
        raise OAuthTimeoutError(
            f"Loopback receiver timed out after {self._timeout_seconds:.0f}s"
        )


def _user_message_for_cancellation(exc: Exception) -> str:
    """Map cancellation causes to a user-friendly message."""
    text = str(exc)
    if "state" in text.lower():
        return "Sign-in was rejected — please try again from Scene Ripper."
    return "Sign-in was cancelled."
