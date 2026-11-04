"""Tests for user-facing local LLM fallback failures."""

from types import SimpleNamespace

import pytest


def test_direct_ollama_connection_failure_is_actionable(monkeypatch):
    from core.llm_client import complete_with_local_fallback

    settings = SimpleNamespace(
        llm_api_base="",
        ollama_model="qwen3:8b",
    )

    def fail_completion(**_kwargs):
        raise ConnectionRefusedError("[Errno 61] Connection refused")

    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("litellm.completion", fail_completion)

    with pytest.raises(RuntimeError) as exc_info:
        complete_with_local_fallback(
            model="ollama/qwen3:8b",
            messages=[{"role": "user", "content": "hello"}],
        )

    message = str(exc_info.value)
    assert "Cannot connect to local Ollama at http://localhost:11434" in message
    assert "ollama pull qwen3:8b" in message
    assert "Original error: [Errno 61] Connection refused" in message


def test_missing_cloud_key_plus_ollama_failure_explains_fallback(monkeypatch):
    from core.llm_client import complete_with_local_fallback

    settings = SimpleNamespace(
        llm_api_base="http://localhost:11434",
        ollama_model="qwen3:8b",
    )

    def fail_completion(**_kwargs):
        raise RuntimeError("OllamaException - [Errno 61] Connection refused")

    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.settings.get_gemini_api_key", lambda: "")
    monkeypatch.setattr("litellm.completion", fail_completion)

    with pytest.raises(RuntimeError) as exc_info:
        complete_with_local_fallback(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
        )

    message = str(exc_info.value)
    assert "No API key is configured for gemini/gemini-2.5-flash" in message
    assert "local Ollama fallback" in message
    assert "Cannot connect to local Ollama at http://localhost:11434" in message


def _two_stage_completion(cloud_exc, calls):
    """Fail the cloud call with ``cloud_exc`` and the Ollama fallback with a refusal."""

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"].startswith("ollama/"):
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        raise cloud_exc

    return completion


def _cloud_failure_cases():
    import httpx
    import litellm

    response = httpx.Response(
        status_code=403,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )
    return [
        pytest.param(
            litellm.Timeout(
                message="Request timed out after 600s",
                model="gemini-2.5-flash",
                llm_provider="gemini",
            ),
            "Timeout",
            "Request timed out after 600s",
            id="timeout",
        ),
        pytest.param(
            litellm.PermissionDeniedError(
                message="PERMISSION_DENIED: API key is restricted",
                llm_provider="gemini",
                model="gemini-2.5-flash",
                response=response,
            ),
            "PermissionDeniedError",
            "API key is restricted",
            id="permission-denied",
        ),
    ]


@pytest.mark.parametrize(("cloud_exc", "type_name", "reason"), _cloud_failure_cases())
def test_cloud_network_or_permission_failure_names_provider_and_reason(
    monkeypatch, cloud_exc, type_name, reason
):
    """Issue #164: cloud timeouts and 403s must not escape as raw LiteLLM errors."""
    from core.llm_client import complete_with_local_fallback

    settings = SimpleNamespace(llm_api_base="", ollama_model="qwen3:8b")
    calls: list[str] = []

    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.settings.get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr("litellm.completion", _two_stage_completion(cloud_exc, calls))

    with pytest.raises(RuntimeError) as exc_info:
        complete_with_local_fallback(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert calls == ["gemini/gemini-2.5-flash", "ollama/qwen3:8b"]
    message = str(exc_info.value)
    assert f"The cloud model gemini/gemini-2.5-flash failed with {type_name}" in message
    assert reason in message
    assert "Cannot connect to local Ollama at http://localhost:11434" in message
