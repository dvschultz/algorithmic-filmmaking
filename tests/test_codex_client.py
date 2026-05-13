"""Tests for core/codex_client.py — the Codex backend HTTP client."""

from __future__ import annotations

import urllib.parse

import httpx
import pytest

from core import codex_client
from core.codex_client import (
    CodexBackendError,
    CodexError,
    MalformedCodexResponseError,
    QuotaExceededError,
    TokenExpiredError,
    TokenMissingError,
    _sanitize_for_log,
    coerce_model_for_codex,
    complete,
)


def _mock_client(handler) -> httpx.Client:
    """Build a sync httpx.Client backed by a MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


# --- Model coercion ---------------------------------------------------------


class TestCoerceModelForCodex:
    @pytest.mark.parametrize(
        "input_model, expected",
        [
            ("gemini-2.5-flash", "gpt-4o-mini"),
            ("gemini-2.5-pro", "gpt-4o"),
            ("gemini-3.1-flash-lite-preview", "gpt-4o-mini"),
            ("gemini/gemini-2.5-flash", "gpt-4o-mini"),
            ("claude-sonnet-4-5", "gpt-4o"),
            ("claude-opus-4", "gpt-4o"),
            ("claude-haiku-3.5", "gpt-4o-mini"),
            ("anthropic/claude-sonnet-4", "gpt-4o"),  # claude-sonnet rule wins (more specific)
            ("openrouter/anthropic/claude", "gpt-4o-mini"),
            ("openai/gpt-4o", "gpt-4o"),  # passthrough, strip prefix
            ("gpt-5.2", "gpt-5.2"),  # already an OpenAI model, no change
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("", "gpt-4o-mini"),  # empty falls back to safe default
        ],
    )
    def test_substitutions(self, input_model, expected):
        assert coerce_model_for_codex(input_model) == expected


# --- Errors -----------------------------------------------------------------


class TestErrorTaxonomy:
    def test_all_errors_inherit_from_codex_error(self):
        for exc_cls in (
            TokenMissingError,
            TokenExpiredError,
            QuotaExceededError,
            CodexBackendError,
            MalformedCodexResponseError,
        ):
            assert issubclass(exc_cls, CodexError)

    def test_codex_backend_error_carries_status_and_body(self):
        exc = CodexBackendError(status=502, message="upstream", body="gateway error")
        assert exc.status == 502
        assert exc.body == "gateway error"
        assert "upstream" in str(exc)


# --- Bearer-token redaction -------------------------------------------------


class TestSanitizeForLog:
    def test_redacts_bearer_in_message(self):
        msg = "Failed: Authorization: Bearer sk-abc.def.ghi was rejected"
        cleaned = _sanitize_for_log(msg)
        assert "sk-abc.def.ghi" not in cleaned
        assert "[REDACTED]" in cleaned

    def test_passes_through_messages_without_tokens(self):
        msg = "Connection refused"
        assert _sanitize_for_log(msg) == msg


# --- complete() -------------------------------------------------------------


class TestComplete:
    def _blob(self, access_token: str = "at_real") -> dict:
        return {
            "access_token": access_token,
            "refresh_token": "rt_keep",
            "expires_at_unix": 9999999999,
            "account_email": "user@example.com",
            "id_token": "",
        }

    def test_happy_path_returns_response_dict(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization", "")
            captured["payload"] = request.content
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )

        client = _mock_client(handler)
        response = complete(
            self._blob(),
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "say hi"}],
            client=client,
        )
        assert response["choices"][0]["message"]["content"] == "hello"
        # The bearer goes out untouched
        assert captured["auth"] == "Bearer at_real"
        # Model was coerced
        import json

        sent = json.loads(captured["payload"].decode("utf-8"))
        assert sent["model"] == "gpt-4o-mini"

    def test_401_raises_token_expired(self):
        def handler(request):
            return httpx.Response(401, json={"error": "invalid_token"})

        client = _mock_client(handler)
        with pytest.raises(TokenExpiredError):
            complete(
                self._blob(),
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
                client=client,
            )

    def test_429_raises_quota_exceeded(self):
        def handler(request):
            return httpx.Response(429, json={"error": "rate_limited"})

        client = _mock_client(handler)
        with pytest.raises(QuotaExceededError):
            complete(
                self._blob(),
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
                client=client,
            )

    def test_500_raises_codex_backend_error_with_status(self):
        def handler(request):
            return httpx.Response(500, text="gateway")

        client = _mock_client(handler)
        with pytest.raises(CodexBackendError) as excinfo:
            complete(
                self._blob(),
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
                client=client,
            )
        assert excinfo.value.status == 500

    def test_missing_access_token_raises_token_missing(self):
        with pytest.raises(TokenMissingError):
            complete(
                {"refresh_token": "rt"},  # no access_token
                model="gpt-4o",
                messages=[],
                client=_mock_client(lambda r: httpx.Response(200, json={})),
            )

    def test_passthrough_kwargs_reach_payload(self):
        captured = {}

        def handler(request):
            import json

            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        client = _mock_client(handler)
        complete(
            self._blob(),
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.2,
            response_format={"type": "json_object"},
            client=client,
        )
        assert captured["payload"]["temperature"] == 0.2
        assert captured["payload"]["response_format"] == {"type": "json_object"}


# --- complete_routed (in llm_client.py) -------------------------------------


class TestCompleteRouted:
    """The chokepoint helper in core.llm_client routes by auth_mode."""

    def test_api_key_mode_delegates_to_litellm(self, monkeypatch):
        from core import llm_client

        captured = {}

        def fake_completion(model, messages, api_key=None, **kwargs):
            captured["model"] = model
            captured["api_key"] = api_key
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "from-litellm"}}]}

        monkeypatch.setattr("litellm.completion", fake_completion)

        # Force load_active_auth to return API_KEY mode.
        from core.spine.chatgpt_auth import AuthMode

        monkeypatch.setattr(
            "core.spine.chatgpt_auth.load_active_auth",
            lambda: (AuthMode.API_KEY, None),
        )

        result = llm_client.complete_routed(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            temperature=0.5,
        )
        assert result["choices"][0]["message"]["content"] == "from-litellm"
        assert captured["model"] == "gpt-4o"
        assert captured["api_key"] == "sk-test"
        assert captured["kwargs"]["temperature"] == 0.5

    def test_subscription_mode_no_token_raises_token_missing(self, monkeypatch):
        from core import llm_client

        from core.spine.chatgpt_auth import AuthMode

        monkeypatch.setattr(
            "core.spine.chatgpt_auth.load_active_auth",
            lambda: (AuthMode.SUBSCRIPTION, None),
        )
        monkeypatch.setattr(
            "core.settings.get_chatgpt_oauth_token",
            lambda: None,
        )

        with pytest.raises(TokenMissingError):
            llm_client.complete_routed(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_subscription_mode_with_token_calls_codex(self, monkeypatch):
        from core import llm_client

        from core.spine.chatgpt_auth import AuthMode, AuthIdentity

        identity = AuthIdentity(account_email="a@b.com", expires_at_unix=9999999999)
        monkeypatch.setattr(
            "core.spine.chatgpt_auth.load_active_auth",
            lambda: (AuthMode.SUBSCRIPTION, identity),
        )
        monkeypatch.setattr(
            "core.settings.get_chatgpt_oauth_token",
            lambda: {"access_token": "at_x", "refresh_token": "rt_x"},
        )

        captured = {}

        def fake_complete(blob, *, model, messages, **kwargs):
            captured["blob"] = blob
            captured["model"] = model
            return {"choices": [{"message": {"content": "from-codex"}}]}

        monkeypatch.setattr("core.codex_client.complete", fake_complete)

        result = llm_client.complete_routed(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "from-codex"
        assert captured["blob"]["access_token"] == "at_x"
