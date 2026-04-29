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
