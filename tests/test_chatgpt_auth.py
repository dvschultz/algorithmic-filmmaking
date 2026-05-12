"""Tests for the spine-safe ChatGPT subscription auth-state module.

Covers:

- TLS scheme on OAuth endpoint constants (catches accidental ``http://``)
- ``load_active_auth()`` across all three states (API-key, subscription
  with valid blob, subscription with missing blob)
- ``is_token_expired()`` boundary behavior with the default leeway
- The ``AuthMode`` / ``str`` equality contract callers rely on for
  comparing against ``Settings.auth_mode``'s JSON string values

The spine boundary check itself lives in ``tests/test_spine_imports.py``.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from core.spine.chatgpt_auth import (
    AUTHORIZATION_ENDPOINT,
    AuthIdentity,
    AuthMode,
    CODEX_COMPLETION_ENDPOINT,
    REDIRECT_URI_HOST,
    SCOPES,
    TOKEN_ENDPOINT,
    is_token_expired,
    load_active_auth,
)


class TestEndpointConstants:
    """OAuth endpoint constants must use HTTPS (or the loopback IPv4 host)."""

    @pytest.mark.parametrize(
        "constant_name, value",
        [
            ("AUTHORIZATION_ENDPOINT", AUTHORIZATION_ENDPOINT),
            ("TOKEN_ENDPOINT", TOKEN_ENDPOINT),
            ("CODEX_COMPLETION_ENDPOINT", CODEX_COMPLETION_ENDPOINT),
        ],
    )
    def test_endpoint_uses_https(self, constant_name, value):
        """Every external endpoint must start with ``https://`` — no plaintext."""
        assert value.startswith("https://"), (
            f"{constant_name} = {value!r} is not HTTPS. "
            "OAuth tokens and inference requests must transit TLS."
        )

    def test_redirect_uri_host_is_loopback(self):
        """Loopback host is explicitly 127.0.0.1, never 0.0.0.0 / empty."""
        assert REDIRECT_URI_HOST == "127.0.0.1", (
            f"REDIRECT_URI_HOST = {REDIRECT_URI_HOST!r}. Binding to anything "
            "other than 127.0.0.1 exposes the OAuth callback to other local "
            "processes — see docs/solutions/security-issues/url-scheme-validation-bypass.md"
        )

    def test_scopes_is_non_empty_tuple(self):
        """SCOPES must be a tuple of non-empty strings."""
        assert isinstance(SCOPES, tuple)
        assert len(SCOPES) > 0
        for scope in SCOPES:
            assert isinstance(scope, str) and scope


class TestAuthModeEnum:
    """AuthMode is a str-enum so it compares directly with Settings string values."""

    def test_values_are_canonical_strings(self):
        assert AuthMode.API_KEY.value == "api_key"
        assert AuthMode.SUBSCRIPTION.value == "subscription"

    def test_str_comparison_works_for_settings_interop(self):
        """``mode == "api_key"`` returns True when mode is AuthMode.API_KEY."""
        assert AuthMode.API_KEY == "api_key"
        assert AuthMode.SUBSCRIPTION == "subscription"
        assert AuthMode.API_KEY != "subscription"


class TestIsTokenExpired:
    """Boundary checks for ``is_token_expired``."""

    def test_expired_token_returns_true(self):
        identity = AuthIdentity(
            account_email="a@b.com",
            expires_at_unix=int(time.time()) - 1,
        )
        assert is_token_expired(identity) is True

    def test_fresh_token_with_default_leeway_returns_false(self):
        identity = AuthIdentity(
            account_email="a@b.com",
            expires_at_unix=int(time.time()) + 600,
        )
        assert is_token_expired(identity) is False

    def test_near_expiry_within_leeway_returns_true(self):
        """Within the leeway window, the token is treated as expired."""
        identity = AuthIdentity(
            account_email="a@b.com",
            expires_at_unix=int(time.time()) + 30,
        )
        assert is_token_expired(identity, leeway_seconds=60) is True

    def test_strict_boundary_with_zero_leeway(self):
        """leeway_seconds=0 exercises the boundary case directly."""
        identity = AuthIdentity(
            account_email="a@b.com",
            expires_at_unix=int(time.time()) + 5,
        )
        assert is_token_expired(identity, leeway_seconds=0) is False


class TestLoadActiveAuth:
    """``load_active_auth()`` across the three reachable states."""

    def _make_settings(self, auth_mode):
        from core.settings import Settings

        s = Settings()
        s.auth_mode = auth_mode
        return s

    def test_api_key_mode_returns_no_identity(self):
        """API-key mode short-circuits and returns no identity."""
        settings = self._make_settings("api_key")
        with patch("core.settings.load_settings", return_value=settings):
            mode, identity = load_active_auth()
            assert mode == AuthMode.API_KEY
            assert identity is None

    def test_subscription_mode_with_valid_blob_returns_identity(self):
        """Subscription mode + keyring blob -> identity populated from blob."""
        settings = self._make_settings("subscription")
        blob = {
            "access_token": "at_abc",
            "refresh_token": "rt_xyz",
            "id_token": "id_def",
            "expires_at_unix": int(time.time()) + 3600,
            "account_email": "derrick@example.com",
        }
        with patch("core.settings.load_settings", return_value=settings), \
             patch("core.settings.get_chatgpt_oauth_token", return_value=blob):
            mode, identity = load_active_auth()
            assert mode == AuthMode.SUBSCRIPTION
            assert identity is not None
            assert identity.account_email == "derrick@example.com"
            assert identity.expires_at_unix == blob["expires_at_unix"]

    def test_subscription_mode_without_blob_returns_none_identity(self):
        """Subscription mode but no token in keyring -> re-auth-needed state."""
        settings = self._make_settings("subscription")
        with patch("core.settings.load_settings", return_value=settings), \
             patch("core.settings.get_chatgpt_oauth_token", return_value=None):
            mode, identity = load_active_auth()
            assert mode == AuthMode.SUBSCRIPTION
            assert identity is None

    def test_subscription_mode_with_malformed_blob_returns_defensive_identity(self):
        """A non-empty blob missing expected fields parses defensively, not by raising.

        Distinct from empty / None — those are the "no blob, re-auth needed"
        state. A blob with unrecognized fields (e.g., from a future format
        we haven't shipped yet, or a partial write) should yield an identity
        with empty/zero defaults so the routing layer's expiry check fires.
        """
        settings = self._make_settings("subscription")
        with patch("core.settings.load_settings", return_value=settings), \
             patch(
                 "core.settings.get_chatgpt_oauth_token",
                 return_value={"unrecognized_field": "x"},
             ):
            mode, identity = load_active_auth()
            assert mode == AuthMode.SUBSCRIPTION
            assert identity is not None
            assert identity.account_email == ""
            assert identity.expires_at_unix == 0

    def test_subscription_mode_with_empty_blob_returns_none_identity(self):
        """An empty dict from get_chatgpt_oauth_token is equivalent to None."""
        settings = self._make_settings("subscription")
        with patch("core.settings.load_settings", return_value=settings), \
             patch("core.settings.get_chatgpt_oauth_token", return_value={}):
            mode, identity = load_active_auth()
            assert mode == AuthMode.SUBSCRIPTION
            assert identity is None

    def test_load_reads_fresh_state_on_every_call(self):
        """Two consecutive calls reflect a settings change between them.

        Guards the read-at-call-time discipline that makes AE5 work: no
        caching of mode or identity inside load_active_auth.
        """
        settings_first = self._make_settings("api_key")
        settings_second = self._make_settings("subscription")

        with patch("core.settings.load_settings", side_effect=[settings_first, settings_second]), \
             patch("core.settings.get_chatgpt_oauth_token", return_value=None):
            mode1, _ = load_active_auth()
            mode2, _ = load_active_auth()
            assert mode1 == AuthMode.API_KEY
            assert mode2 == AuthMode.SUBSCRIPTION
