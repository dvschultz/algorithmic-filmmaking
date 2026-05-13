"""Tests for the PKCE + loopback OAuth flow module.

Uses ``httpx.MockTransport`` everywhere so tests don't touch the network.
The loopback listener is exercised via direct HTTP requests against a
real local server bound to 127.0.0.1.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.parse
from http import HTTPStatus

import httpx
import pytest

from core.spine.chatgpt_oauth_flow import (
    AuthorizationCode,
    AuthorizationRequest,
    OAuthCancelledError,
    OAuthHTTPError,
    OAuthMalformedResponseError,
    OAuthTimeoutError,
    PKCEPair,
    TokenBlob,
    _pick_free_loopback_port,
    _redact_authorization,
    _try_extract_email_from_id_token,
    build_authorization_url,
    exchange_code_for_token,
    generate_pkce_pair,
    refresh_token_blob,
    run_loopback_listener,
)
from core.spine.chatgpt_auth import (
    AUTHORIZATION_ENDPOINT,
    CLIENT_ID,
    REDIRECT_URI_HOST,
    TOKEN_ENDPOINT,
)


# --- PKCE --------------------------------------------------------------------


class TestPKCE:
    def test_generate_pkce_pair_shape(self):
        pair = generate_pkce_pair()
        assert isinstance(pair, PKCEPair)
        # Verifier per RFC 7636: 43-128 chars, URL-safe.
        assert 43 <= len(pair.verifier) <= 128
        # Challenge has no base64 padding.
        assert "=" not in pair.challenge

    def test_s256_transform_holds(self):
        """challenge == BASE64URL(SHA256(verifier)) without padding."""
        pair = generate_pkce_pair()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(pair.verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        assert pair.challenge == expected

    def test_consecutive_calls_return_distinct_pairs(self):
        a = generate_pkce_pair()
        b = generate_pkce_pair()
        assert a.verifier != b.verifier
        assert a.challenge != b.challenge


# --- Authorization URL -------------------------------------------------------


class TestBuildAuthorizationURL:
    def test_url_includes_required_oauth_params(self):
        pair = generate_pkce_pair()
        req = build_authorization_url(pair.challenge)
        assert isinstance(req, AuthorizationRequest)
        parsed = urllib.parse.urlsplit(req.url)
        # Same scheme + host as configured endpoint
        expected_endpoint = urllib.parse.urlsplit(AUTHORIZATION_ENDPOINT)
        assert parsed.scheme == expected_endpoint.scheme
        assert parsed.netloc == expected_endpoint.netloc

        params = urllib.parse.parse_qs(parsed.query)
        assert params["response_type"] == ["code"]
        assert params["client_id"] == [CLIENT_ID]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"] == [pair.challenge]
        assert "state" in params and len(params["state"][0]) >= 32

    def test_redirect_uri_points_at_loopback(self):
        pair = generate_pkce_pair()
        req = build_authorization_url(pair.challenge)
        parsed = urllib.parse.urlsplit(req.redirect_uri)
        assert parsed.scheme == "http"
        assert parsed.hostname == REDIRECT_URI_HOST
        assert parsed.port == req.port
        assert parsed.path == "/callback"

    def test_state_is_generated_internally(self):
        """build_authorization_url must NOT accept a caller-supplied state."""
        # Sanity: signature has just one positional arg (challenge).
        import inspect

        sig = inspect.signature(build_authorization_url)
        params = list(sig.parameters)
        assert params == ["challenge"], (
            f"build_authorization_url should generate state internally, not accept it; "
            f"got params: {params}"
        )

    def test_state_high_entropy(self):
        """The internally-generated state must have at least 256 bits of entropy."""
        pair = generate_pkce_pair()
        states = {build_authorization_url(pair.challenge).state for _ in range(20)}
        # 20 distinct values from a 256-bit generator — collisions are astronomical.
        assert len(states) == 20


# --- Loopback listener -------------------------------------------------------


def _fire_redirect(
    port: int, query: str, *, host_header: str | None = None
) -> None:
    """Open a socket to the loopback listener and send an HTTP/1.1 GET."""
    import socket

    host_line = host_header if host_header is not None else f"127.0.0.1:{port}"
    request = (
        f"GET /callback?{query} HTTP/1.1\r\n"
        f"Host: {host_line}\r\n"
        "Connection: close\r\n\r\n"
    )
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    try:
        s.connect(("127.0.0.1", port))
        s.sendall(request.encode("ascii"))
        s.recv(4096)  # drain
    finally:
        s.close()


class TestLoopbackListener:
    def test_happy_path_returns_authorization_code(self):
        port = _pick_free_loopback_port()
        state = "expected-state-value"

        def deliver():
            time.sleep(0.05)
            _fire_redirect(port, f"code=abc123&state={state}")

        t = threading.Thread(target=deliver)
        t.start()
        try:
            result = run_loopback_listener(port, expected_state=state, timeout_seconds=5)
        finally:
            t.join()
        assert isinstance(result, AuthorizationCode)
        assert result.code == "abc123"
        assert result.state == state

    def test_state_mismatch_raises_cancelled(self):
        port = _pick_free_loopback_port()

        def deliver():
            time.sleep(0.05)
            _fire_redirect(port, "code=abc123&state=attacker-supplied")

        t = threading.Thread(target=deliver)
        t.start()
        try:
            with pytest.raises(OAuthCancelledError):
                run_loopback_listener(port, expected_state="real-state", timeout_seconds=5)
        finally:
            t.join()

    def test_timeout_raises_timeout_error(self):
        port = _pick_free_loopback_port()
        with pytest.raises(OAuthTimeoutError):
            run_loopback_listener(port, expected_state="x", timeout_seconds=0.5)

    def test_error_param_in_redirect_raises_http_error(self):
        port = _pick_free_loopback_port()

        def deliver():
            time.sleep(0.05)
            _fire_redirect(port, "error=access_denied&state=x")

        t = threading.Thread(target=deliver)
        t.start()
        try:
            with pytest.raises(OAuthHTTPError) as excinfo:
                run_loopback_listener(port, expected_state="x", timeout_seconds=5)
        finally:
            t.join()
        assert "access_denied" in str(excinfo.value)


# --- Token endpoint via MockTransport ----------------------------------------


def _make_mock_client(handler) -> httpx.Client:
    """Wrap an httpx.MockTransport around a single handler callable."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


class TestExchangeCodeForToken:
    def test_happy_path_returns_token_blob(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert str(request.url) == TOKEN_ENDPOINT
            body = urllib.parse.parse_qs(request.content.decode("ascii"))
            assert body["grant_type"] == ["authorization_code"]
            assert body["code"] == ["abc123"]
            assert body["code_verifier"] == ["verifier-xyz"]
            return httpx.Response(
                200,
                json={
                    "access_token": "at_real",
                    "refresh_token": "rt_real",
                    "id_token": "",
                    "expires_in": 3600,
                    "email": "derrick@example.com",
                },
            )

        client = _make_mock_client(handler)
        blob = exchange_code_for_token(
            code="abc123",
            verifier="verifier-xyz",
            redirect_uri="http://127.0.0.1:5050/callback",
            client=client,
        )
        assert isinstance(blob, TokenBlob)
        assert blob.access_token == "at_real"
        assert blob.refresh_token == "rt_real"
        assert blob.account_email == "derrick@example.com"
        # expires_at_unix is set to now + expires_in
        assert int(time.time()) <= blob.expires_at_unix <= int(time.time()) + 3700

    def test_4xx_raises_oauth_http_error(self):
        def handler(request):
            return httpx.Response(400, json={"error": "invalid_grant"})

        client = _make_mock_client(handler)
        with pytest.raises(OAuthHTTPError) as excinfo:
            exchange_code_for_token(
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:5050/callback",
                client=client,
            )
        assert excinfo.value.status == 400

    def test_missing_access_token_raises_malformed(self):
        def handler(request):
            return httpx.Response(200, json={"refresh_token": "rt"})

        client = _make_mock_client(handler)
        with pytest.raises(OAuthMalformedResponseError):
            exchange_code_for_token(
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:5050/callback",
                client=client,
            )


class TestRefreshToken:
    def test_happy_path_returns_new_blob(self):
        def handler(request):
            body = urllib.parse.parse_qs(request.content.decode("ascii"))
            assert body["grant_type"] == ["refresh_token"]
            assert body["refresh_token"] == ["rt_old"]
            return httpx.Response(
                200,
                json={
                    "access_token": "at_new",
                    "refresh_token": "rt_new",
                    "expires_in": 1800,
                },
            )

        client = _make_mock_client(handler)
        blob = refresh_token_blob("rt_old", client=client)
        assert blob.access_token == "at_new"
        assert blob.refresh_token == "rt_new"

    def test_response_without_new_refresh_preserves_old(self):
        """If the response omits refresh_token, the prior one is preserved."""

        def handler(request):
            return httpx.Response(
                200,
                json={"access_token": "at_new", "expires_in": 1800},
            )

        client = _make_mock_client(handler)
        blob = refresh_token_blob("rt_keepme", client=client)
        assert blob.refresh_token == "rt_keepme"

    def test_empty_refresh_token_raises(self):
        with pytest.raises(OAuthMalformedResponseError):
            refresh_token_blob("")

    def test_401_raises_oauth_http_error(self):
        def handler(request):
            return httpx.Response(401, json={"error": "invalid_grant"})

        client = _make_mock_client(handler)
        with pytest.raises(OAuthHTTPError) as excinfo:
            refresh_token_blob("rt_old", client=client)
        assert excinfo.value.status == 401


# --- Bearer-token redaction --------------------------------------------------


class TestRedactAuthorization:
    def test_redacts_bearer_token_in_message(self):
        msg = "Failed: Authorization: Bearer sk-abc123xyz789 was rejected"
        redacted = _redact_authorization(msg)
        assert "sk-abc123xyz789" not in redacted
        assert "[REDACTED]" in redacted

    def test_redacts_case_insensitive(self):
        msg = "authorization=Bearer abc.def.ghi"
        redacted = _redact_authorization(msg)
        assert "abc.def.ghi" not in redacted
        assert "[REDACTED]" in redacted

    def test_passes_through_messages_without_tokens(self):
        msg = "Connection refused — endpoint unreachable"
        assert _redact_authorization(msg) == msg


# --- ID token email extraction -----------------------------------------------


class TestIDTokenEmail:
    def _make_id_token(self, payload: dict) -> str:
        """Build a fake unsigned JWT with the given payload."""
        header_b64 = base64.urlsafe_b64encode(
            json.dumps({"alg": "none"}).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        return f"{header_b64}.{payload_b64}.signature_ignored"

    def test_extracts_email_when_present(self):
        token = self._make_id_token({"email": "derrick@example.com", "sub": "user-1"})
        assert _try_extract_email_from_id_token(token) == "derrick@example.com"

    def test_returns_empty_when_email_absent(self):
        token = self._make_id_token({"sub": "user-1"})
        assert _try_extract_email_from_id_token(token) == ""

    def test_returns_empty_for_malformed_token(self):
        assert _try_extract_email_from_id_token("not.a.jwt") == ""
        assert _try_extract_email_from_id_token("") == ""
