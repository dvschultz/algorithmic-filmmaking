"""Tests for the MCP-side auth-snapshot helpers (U10).

Per the R-M3 placement convention, MCP-scoped tests live in
``scene_ripper_mcp/tests/`` rather than ``tests/``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.settings import Settings
from scene_ripper_mcp.auth_snapshot import (
    AuthSnapshot,
    snapshot_still_valid,
    take_auth_snapshot,
)


def _settings(mode: str) -> Settings:
    s = Settings()
    s.auth_mode = mode
    return s


class TestTakeAuthSnapshot:
    def test_api_key_mode_snapshot(self):
        with patch("core.settings.load_settings", return_value=_settings("api_key")):
            snap = take_auth_snapshot()
            assert snap.auth_mode == "api_key"
            assert snap.access_token is None

    def test_subscription_with_token_snapshot(self):
        with patch(
            "core.settings.load_settings", return_value=_settings("subscription")
        ), patch(
            "core.settings.get_chatgpt_oauth_token",
            return_value={"access_token": "at_abc", "refresh_token": "rt"},
        ):
            snap = take_auth_snapshot()
            assert snap.auth_mode == "subscription"
            assert snap.access_token == "at_abc"

    def test_subscription_without_token_snapshot(self):
        with patch(
            "core.settings.load_settings", return_value=_settings("subscription")
        ), patch(
            "core.settings.get_chatgpt_oauth_token",
            return_value=None,
        ):
            snap = take_auth_snapshot()
            assert snap.auth_mode == "subscription"
            assert snap.access_token is None


class TestSnapshotStillValid:
    def test_unchanged_state_remains_valid(self):
        snap = AuthSnapshot(auth_mode="subscription", access_token="at_abc")
        with patch(
            "core.settings.load_settings", return_value=_settings("subscription")
        ), patch(
            "core.settings.get_chatgpt_oauth_token",
            return_value={"access_token": "at_abc"},
        ):
            assert snapshot_still_valid(snap) is True

    def test_mode_change_invalidates_snapshot(self):
        snap = AuthSnapshot(auth_mode="subscription", access_token="at_abc")
        with patch("core.settings.load_settings", return_value=_settings("api_key")):
            assert snapshot_still_valid(snap) is False

    def test_token_rotation_invalidates_snapshot(self):
        snap = AuthSnapshot(auth_mode="subscription", access_token="at_old")
        with patch(
            "core.settings.load_settings", return_value=_settings("subscription")
        ), patch(
            "core.settings.get_chatgpt_oauth_token",
            return_value={"access_token": "at_new"},
        ):
            assert snapshot_still_valid(snap) is False

    def test_signout_invalidates_snapshot(self):
        snap = AuthSnapshot(auth_mode="subscription", access_token="at_abc")
        with patch(
            "core.settings.load_settings", return_value=_settings("subscription")
        ), patch(
            "core.settings.get_chatgpt_oauth_token",
            return_value=None,
        ):
            assert snapshot_still_valid(snap) is False
