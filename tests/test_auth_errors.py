"""Tests for core/auth_errors.classify_subscription_error.

Each category string is part of the workers' branching contract — once
they're wired into U5's re-auth dialog, the category names become a
load-bearing invariant.
"""

from __future__ import annotations

import pytest

from core.auth_errors import (
    CATEGORY_CODEX_BACKEND,
    CATEGORY_NONE,
    CATEGORY_QUOTA_EXCEEDED,
    CATEGORY_TOKEN_EXPIRED,
    CATEGORY_TOKEN_REQUIRED,
    CodexBackendError,
    MalformedCodexResponseError,
    QuotaExceededError,
    TokenExpiredError,
    TokenMissingError,
    classify_subscription_error,
    user_message_for_category,
)


class TestClassifyErrors:
    @pytest.mark.parametrize(
        "exc, expected_category",
        [
            (TokenMissingError("no blob"), CATEGORY_TOKEN_REQUIRED),
            (TokenExpiredError("401"), CATEGORY_TOKEN_EXPIRED),
            (QuotaExceededError("429"), CATEGORY_QUOTA_EXCEEDED),
            (CodexBackendError(status=500, message="upstream"), CATEGORY_CODEX_BACKEND),
            (MalformedCodexResponseError("bad json"), CATEGORY_CODEX_BACKEND),
            (ValueError("unrelated"), CATEGORY_NONE),
            (RuntimeError("unrelated"), CATEGORY_NONE),
        ],
    )
    def test_classification(self, exc, expected_category):
        assert classify_subscription_error(exc) == expected_category


class TestUserMessages:
    @pytest.mark.parametrize(
        "category",
        [
            CATEGORY_TOKEN_REQUIRED,
            CATEGORY_TOKEN_EXPIRED,
            CATEGORY_QUOTA_EXCEEDED,
            CATEGORY_CODEX_BACKEND,
        ],
    )
    def test_returns_non_empty_for_real_categories(self, category):
        msg = user_message_for_category(category)
        assert isinstance(msg, str) and msg

    def test_returns_empty_for_unknown_category(self):
        assert user_message_for_category("garbage") == ""

    def test_codex_backend_includes_raw_when_provided(self):
        msg = user_message_for_category(
            CATEGORY_CODEX_BACKEND, raw_message="503 upstream"
        )
        assert "503 upstream" in msg

    def test_token_required_mentions_settings_and_api_key_path(self):
        msg = user_message_for_category(CATEGORY_TOKEN_REQUIRED)
        assert "Settings" in msg
        assert "API-key" in msg
