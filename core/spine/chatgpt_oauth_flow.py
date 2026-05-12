"""PKCE + loopback OAuth flow against OpenAI's Codex auth endpoint.

Pure-Python, GUI-free implementation of the authorization-code-with-PKCE
flow Codex CLI uses. The flow:

  1. ``generate_pkce_pair()`` → (verifier, challenge)
  2. ``build_authorization_url(challenge)`` → (url, redirect_uri, state, port)
       — state is generated internally via secrets.token_urlsafe per R-S1
       — redirect_uri binds to 127.0.0.1:<port>
  3. caller opens ``url`` in the user's browser
  4. ``run_loopback_listener(port, expected_state, timeout)`` → AuthorizationCode
       — explicit 127.0.0.1 bind, Host header validation, path allowlist (R-S2)
       — single-shot: handler terminates after one redirect
  5. ``exchange_code_for_token(code, verifier, redirect_uri)`` → TokenBlob
  6. ``refresh_token_blob(refresh_token)`` → TokenBlob (later, on 401)

All HTTP traffic flows through ``httpx.Client``. Tests inject
``httpx.MockTransport`` to assert request shape without live calls.

Error surface — all OAuth flow exceptions inherit from ``OAuthError`` so
callers can catch broadly when they need to:

  - ``OAuthCancelledError``   — user closed the browser / app cancelled
  - ``OAuthTimeoutError``     — loopback waited longer than the timeout
  - ``OAuthHTTPError``        — non-2xx from OpenAI; carries status code
  - ``OAuthMalformedResponseError`` — 2xx but expected fields missing
  - ``OAuthHostMismatchError`` — request to loopback had a bad Host header

Bearer-token redaction (R-S4): any ``httpx`` request error caught here is
re-raised as ``OAuthHTTPError`` with the ``Authorization`` header value
scrubbed from the message. Callers can safely ``logger.error(str(exc))``
without leaking tokens to log files.

Spine boundary: ``httpx`` is allowed in spine per
``tests/test_spine_imports.py``. This module does NOT import PySide6, mpv,
or any heavy ML lib. The OAuth-worker Qt wrapper lives in
``ui/workers/oauth_worker.py`` (U4) and consumes this module.

Placeholders: the actual ``client_id`` and exact token-endpoint request
shape come from Codex CLI's open-source repo. The functions below build
the standard OAuth 2.0 + PKCE flow; ``CLIENT_ID`` is read from
``core/spine/chatgpt_auth.py``'s constants (currently a placeholder). A
follow-up session replaces those constants with the real values and
verifies end-to-end against the real endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import logging
import secrets
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

from core.spine.chatgpt_auth import (
    AUTHORIZATION_ENDPOINT,
    CLIENT_ID,
    REDIRECT_URI_HOST,
    SCOPES,
    TOKEN_ENDPOINT,
)
from core.spine.log_redaction import redact_bearer_tokens as _redact_authorization

logger = logging.getLogger(__name__)


# --- Exceptions --------------------------------------------------------------


class OAuthError(Exception):
    """Base class for all OAuth-flow errors raised by this module."""


class OAuthCancelledError(OAuthError):
    """User cancelled the flow (closed browser, app-side cancel)."""


class OAuthTimeoutError(OAuthError):
    """Loopback receiver did not get a redirect within the timeout window."""


class OAuthHTTPError(OAuthError):
    """OpenAI endpoint returned a non-success status (or httpx errored).

    ``status`` is None when the failure happened before a response was
    received (e.g., DNS, connect, TLS). ``body`` is best-effort — empty
    when no response body was readable. The ``__str__`` of this exception
    has bearer tokens redacted via ``_redact_authorization`` so callers
    can safely log it.
    """

    def __init__(self, status: Optional[int], message: str, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class OAuthMalformedResponseError(OAuthError):
    """Token endpoint returned 2xx but a required field is missing."""


class OAuthHostMismatchError(OAuthError):
    """Loopback handler received a request with an unexpected Host header."""


# --- Data shapes -------------------------------------------------------------


@dataclass(frozen=True)
class PKCEPair:
    """Code verifier + S256-derived challenge for the PKCE flow."""

    verifier: str
    challenge: str


@dataclass(frozen=True)
class AuthorizationRequest:
    """Materials the caller needs to drive the rest of the flow.

    The url is what the GUI opens in the user's browser. The redirect_uri
    must match exactly between the authorization request and the token
    exchange. The state must match what the loopback listener observes.
    The port lets the caller pass it to ``run_loopback_listener``.
    """

    url: str
    redirect_uri: str
    state: str
    port: int


@dataclass(frozen=True)
class AuthorizationCode:
    """Result of a successful loopback redirect."""

    code: str
    state: str


@dataclass(frozen=True)
class TokenBlob:
    """Shape of what we persist to keyring after a successful exchange.

    Matches the schema documented in ``core/settings.py:set_chatgpt_oauth_token``.
    ``expires_at_unix`` is computed at exchange time from the endpoint's
    ``expires_in`` field so consumers don't have to know the issuance time.
    """

    access_token: str
    refresh_token: str
    id_token: str
    expires_at_unix: int
    account_email: str

    def as_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "expires_at_unix": self.expires_at_unix,
            "account_email": self.account_email,
        }


# --- PKCE primitives ---------------------------------------------------------


def generate_pkce_pair() -> PKCEPair:
    """Generate a code_verifier (43-128 random chars) + S256 challenge.

    Per RFC 7636: verifier is a URL-safe random string of 43-128 chars;
    challenge is BASE64URL(SHA256(verifier)) with padding stripped.
    """
    # 32 random bytes encodes to 43 url-safe base64 chars (no padding),
    # which is the minimum-allowed verifier length under RFC 7636.
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return PKCEPair(verifier=verifier, challenge=challenge)


# --- Authorization URL -------------------------------------------------------


def _pick_free_loopback_port() -> int:
    """Bind a transient socket on 127.0.0.1:0 to discover a free port.

    Returns immediately after closing the probe socket. There's a small
    TOCTOU window before the listener re-binds, but the worst-case
    failure (port now occupied) surfaces as a clean ``OSError`` from
    ``HTTPServer`` rather than silent breakage.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((REDIRECT_URI_HOST, 0))
        return probe.getsockname()[1]


def build_authorization_url(challenge: str) -> AuthorizationRequest:
    """Construct the URL the user's browser opens to start the OAuth flow.

    Generates ``state`` internally via ``secrets.token_urlsafe(32)`` so
    callers can't accidentally pass a low-entropy value (R-S1). Picks a
    free localhost port for the loopback redirect URI; the caller passes
    that port to ``run_loopback_listener``.
    """
    state = secrets.token_urlsafe(32)
    port = _pick_free_loopback_port()
    redirect_uri = f"http://{REDIRECT_URI_HOST}:{port}/callback"

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    query = urllib.parse.urlencode(params)
    url = f"{AUTHORIZATION_ENDPOINT}?{query}"
    return AuthorizationRequest(
        url=url, redirect_uri=redirect_uri, state=state, port=port
    )


# --- Loopback listener -------------------------------------------------------


# Successful and error response bodies served back to the browser. Static
# HTML so the user sees something coherent after authorizing.
_SUCCESS_PAGE = (
    b"<!DOCTYPE html><html><head><title>Signed in</title></head>"
    b"<body style='font-family: -apple-system, sans-serif; padding: 40px;'>"
    b"<h2>Sign-in complete.</h2>"
    b"<p>You can close this window and return to Scene Ripper.</p>"
    b"</body></html>"
)
_ERROR_PAGE = (
    b"<!DOCTYPE html><html><head><title>Sign-in failed</title></head>"
    b"<body style='font-family: -apple-system, sans-serif; padding: 40px;'>"
    b"<h2>Sign-in did not complete.</h2>"
    b"<p>Return to Scene Ripper for details, then try again.</p>"
    b"</body></html>"
)


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-shot handler that captures one ``?code=&state=`` redirect.

    Captures the result on the server instance so the calling code can
    read it after the server stops. Validates Host header and path
    before parsing query parameters (R-S2).
    """

    server: "_LoopbackServer"  # narrower type than http.server.HTTPServer

    # Silence the default access log — we have our own logging.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        # Validate Host header — defense in depth against a local process
        # sending a request to the listener with a spoofed Host.
        host_header = self.headers.get("Host", "")
        host_only = host_header.split(":", 1)[0]
        if host_only not in ("localhost", "127.0.0.1"):
            self.server.host_mismatch = host_header
            self.send_response(400)
            self.end_headers()
            return

        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/callback":
            # Anything other than the expected callback path is ignored;
            # the loop continues to wait for the real redirect.
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [""])[0]
        state = params.get("state", [""])[0]
        error = params.get("error", [""])[0]

        if error:
            self.server.error_param = error
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_ERROR_PAGE)
            return

        if not code or not state:
            self.send_response(400)
            self.end_headers()
            return

        self.server.result_code = code
        self.server.result_state = state
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SUCCESS_PAGE)


class _LoopbackServer(http.server.HTTPServer):
    """HTTPServer subclass that holds the single captured result."""

    # ``OAuthWorker._run_listener_with_cancellation`` (U4) wraps the
    # listener in short slices to make user cancellation observable, so
    # the server is bound and torn down repeatedly during one flow.
    # ``SO_REUSEADDR`` lets the rebind succeed inside the kernel's
    # TIME_WAIT window instead of failing spuriously a few seconds in.
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.result_code: Optional[str] = None
        self.result_state: Optional[str] = None
        self.error_param: Optional[str] = None
        self.host_mismatch: Optional[str] = None


def run_loopback_listener(
    port: int, expected_state: str, timeout_seconds: float = 300.0
) -> AuthorizationCode:
    """Run a single-shot HTTPS-callback receiver on 127.0.0.1:port.

    Returns when one valid callback is received (matching state) or
    raises an OAuth* exception on error / timeout. The server is closed
    on every exit path including exceptions.
    """
    server = _LoopbackServer((REDIRECT_URI_HOST, port), _LoopbackHandler)
    server.timeout = max(1.0, timeout_seconds)

    try:
        # handle_request() returns when one request has been handled or
        # the timeout fires. Loop because the handler may have served
        # a 404 (unknown path) and we should keep waiting.
        import time

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            remaining = deadline - time.time()
            server.timeout = max(0.1, remaining)
            server.handle_request()
            if server.host_mismatch:
                raise OAuthHostMismatchError(
                    f"Loopback received request with unexpected Host: "
                    f"{server.host_mismatch!r}"
                )
            if server.error_param:
                raise OAuthHTTPError(
                    status=None,
                    message=f"Authorization server returned error: {server.error_param}",
                )
            if server.result_code is not None and server.result_state is not None:
                if server.result_state != expected_state:
                    # CSRF guard — state must match what we generated.
                    logger.warning(
                        "OAuth state mismatch — possible CSRF or stale flow"
                    )
                    raise OAuthCancelledError(
                        "OAuth state parameter did not match — flow rejected"
                    )
                return AuthorizationCode(
                    code=server.result_code, state=server.result_state
                )
        raise OAuthTimeoutError(
            f"Loopback receiver timed out after {timeout_seconds:.0f}s"
        )
    finally:
        server.server_close()


# --- Token exchange ----------------------------------------------------------


def _post_token_endpoint(
    client: httpx.Client, data: dict
) -> dict:
    """POST to TOKEN_ENDPOINT, return parsed JSON, or raise OAuth*Error."""
    try:
        response = client.post(TOKEN_ENDPOINT, data=data, timeout=30.0)
    except httpx.HTTPError as exc:
        raise OAuthHTTPError(
            status=None,
            message=_redact_authorization(f"Network error contacting token endpoint: {exc}"),
        ) from exc

    if response.status_code >= 400:
        try:
            body_text = response.text
        except Exception:
            body_text = ""
        raise OAuthHTTPError(
            status=response.status_code,
            message=f"Token endpoint returned {response.status_code}",
            body=_redact_authorization(body_text),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise OAuthMalformedResponseError(
            f"Token endpoint returned non-JSON response: {exc}"
        ) from exc


def _token_blob_from_response(body: dict) -> TokenBlob:
    """Validate the token-endpoint response and lift it into a TokenBlob.

    The exact key names (``access_token``, ``refresh_token``, ``id_token``,
    ``expires_in``) are standard OAuth 2.0; OpenAI's actual response shape
    is verified during U3's live smoke test (deferred to a follow-up
    session). ``account_email`` comes from the ``email`` claim in the
    id_token when present; otherwise empty (U5's identity fallback chain
    handles display).
    """
    import time

    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OAuthMalformedResponseError(
            "Token endpoint response missing access_token"
        )
    refresh_token = body.get("refresh_token") or ""
    id_token = body.get("id_token") or ""
    expires_in = body.get("expires_in", 0)
    try:
        expires_in_int = int(expires_in)
    except (TypeError, ValueError):
        expires_in_int = 0
    expires_at_unix = int(time.time()) + max(0, expires_in_int)

    # Best-effort email extraction from id_token (unverified — purely for
    # display; the keyring blob is the auth source of truth, not the
    # decoded id_token).
    account_email = body.get("email", "") or ""
    if isinstance(id_token, str) and id_token and not account_email:
        account_email = _try_extract_email_from_id_token(id_token)

    return TokenBlob(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else "",
        id_token=id_token if isinstance(id_token, str) else "",
        expires_at_unix=expires_at_unix,
        account_email=account_email if isinstance(account_email, str) else "",
    )


def _try_extract_email_from_id_token(id_token: str) -> str:
    """Decode the id_token's payload base64 segment and read 'email'.

    No signature verification — purely for display fallback. If anything
    goes wrong, returns empty string.
    """
    try:
        import json as _json

        segments = id_token.split(".")
        if len(segments) < 2:
            return ""
        # Base64-url-decode the payload segment (middle one).
        payload_b64 = segments[1]
        # Add padding back if missing.
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 = payload_b64 + ("=" * padding)
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = _json.loads(payload_json)
        email = payload.get("email")
        return email if isinstance(email, str) else ""
    except Exception:
        return ""


def exchange_code_for_token(
    code: str,
    verifier: str,
    redirect_uri: str,
    *,
    client: Optional[httpx.Client] = None,
) -> TokenBlob:
    """Exchange the authorization code for a token blob.

    The ``client`` parameter lets tests inject ``httpx.MockTransport``;
    production callers pass None and let this function manage its own
    Client.
    """
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": CLIENT_ID,
        "code_verifier": verifier,
    }
    if client is None:
        with httpx.Client() as owned:
            body = _post_token_endpoint(owned, data)
    else:
        body = _post_token_endpoint(client, data)
    return _token_blob_from_response(body)


def refresh_token_blob(
    refresh_token: str,
    *,
    client: Optional[httpx.Client] = None,
) -> TokenBlob:
    """Use a refresh_token to obtain a fresh access token blob."""
    if not refresh_token:
        raise OAuthMalformedResponseError("refresh_token is empty")
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }
    if client is None:
        with httpx.Client() as owned:
            body = _post_token_endpoint(owned, data)
    else:
        body = _post_token_endpoint(client, data)
    blob = _token_blob_from_response(body)
    # Some OAuth servers return a new refresh_token; some don't. If the
    # response didn't carry one, preserve the previous value so the
    # caller's persisted state stays usable.
    if not blob.refresh_token:
        blob = TokenBlob(
            access_token=blob.access_token,
            refresh_token=refresh_token,
            id_token=blob.id_token,
            expires_at_unix=blob.expires_at_unix,
            account_email=blob.account_email,
        )
    return blob
