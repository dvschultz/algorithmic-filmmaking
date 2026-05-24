"""Tests for LLM provider model aliases and validation."""

from core.llm_client import (
    ProviderConfig,
    ProviderType,
    normalize_provider_error,
    resolve_model_alias,
    validate_provider_model,
)


def test_resolve_gemini_renamed_model_alias():
    assert resolve_model_alias("gemini", "gemini-3-flash-preview") == "gemini-2.5-flash"


def test_provider_config_uses_resolved_alias_for_litellm_model():
    config = ProviderConfig(
        provider=ProviderType.GEMINI,
        model="gemini-3-flash-preview",
    )

    assert config.to_litellm_model() == "gemini/gemini-2.5-flash"


def test_validate_provider_model_accepts_alias():
    ok, resolved = validate_provider_model("gemini", "gemini-pro")

    assert ok is True
    assert resolved == "gemini-2.5-pro"


def test_validate_provider_model_rejects_unknown_model():
    ok, message = validate_provider_model("gemini", "not-a-real-model")

    assert ok is False
    assert "Unknown gemini model" in message


def test_normalize_gemini_invalid_argument_error():
    message = normalize_provider_error("gemini", RuntimeError("Invalid argument: missing fields"))

    assert "Gemini rejected the request" in message
    assert "selected model" in message or "provider settings" in message
