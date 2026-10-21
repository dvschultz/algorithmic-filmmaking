"""Claude request-parameter rules applied at every LiteLLM seam.

Claude Opus 4.7+ and the 5.x models return a 400 for non-default
temperature/top_p/top_k, and the 5.x models think by default, so a small
max_tokens can be spent before any text is written.
"""

import asyncio
from types import SimpleNamespace

import pytest

from core.llm_client import (
    CLAUDE_MIN_MAX_TOKENS,
    DEFAULT_MODELS,
    LLMClient,
    ProviderConfig,
    ProviderType,
    complete_routed,
    complete_with_local_fallback,
    get_provider_models,
    prepare_claude_kwargs,
)


def _response(content="ok", finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


@pytest.fixture
def api_key_auth(monkeypatch):
    from core.spine.chatgpt_auth import AuthMode

    monkeypatch.setattr(
        "core.spine.chatgpt_auth.load_active_auth", lambda: (AuthMode.API_KEY, None)
    )


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5",
        "anthropic/claude-opus-5-5",
        "openrouter/anthropic/claude-haiku-5.5",
        "bedrock/anthropic.claude-sonnet-5-5",
        "claude-sonnet-4-5-20250929",
    ],
)
def test_claude_models_drop_sampling_params_and_raise_small_caps(model):
    kwargs = prepare_claude_kwargs(
        model, {"temperature": 0.0, "top_p": 0.9, "top_k": 5, "max_tokens": 50}
    )

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["max_tokens"] == CLAUDE_MIN_MAX_TOKENS


def test_claude_keeps_large_caps():
    kwargs = prepare_claude_kwargs("claude-sonnet-5-5", {"max_tokens": 20000})

    assert kwargs["max_tokens"] == 20000


@pytest.mark.parametrize("model", ["gpt-5.4", "gemini/gemini-2.5-flash", "ollama/qwen3:8b"])
def test_other_providers_keep_sampling_params(model):
    kwargs = prepare_claude_kwargs(model, {"temperature": 0.0, "max_tokens": 50})

    assert kwargs == {"temperature": 0.0, "max_tokens": 50}


def test_complete_routed_strips_temperature_for_claude(monkeypatch, api_key_auth):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    complete_routed(
        model="anthropic/claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=100,
    )

    assert "temperature" not in seen
    assert seen["max_tokens"] == CLAUDE_MIN_MAX_TOKENS


def test_complete_routed_raises_on_claude_refusal(monkeypatch, api_key_auth):
    monkeypatch.setattr(
        "litellm.completion",
        lambda **_kwargs: _response(content="", finish_reason="content_filter"),
    )

    with pytest.raises(RuntimeError, match="declined"):
        complete_routed(
            model="anthropic/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_complete_routed_keeps_temperature_for_gemini(monkeypatch, api_key_auth):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    complete_routed(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
    )

    assert seen["temperature"] == 0.0


def test_local_fallback_strips_temperature_for_claude_only(monkeypatch):
    calls = []
    settings = SimpleNamespace(llm_api_base="", ollama_model="qwen3:8b")

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs["model"].startswith("anthropic/"):
            import litellm

            raise litellm.RateLimitError(
                message="rate limited", llm_provider="anthropic", model=kwargs["model"]
            )
        return _response()

    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.settings.get_api_key_for_model", lambda _model: "sk-ant-test")
    monkeypatch.setattr("litellm.completion", fake_completion)

    complete_with_local_fallback(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.8,
    )

    cloud_call, local_call = calls
    assert cloud_call["model"] == "anthropic/claude-sonnet-5-5"
    assert "temperature" not in cloud_call
    assert local_call["model"] == "ollama/qwen3:8b"
    assert local_call["temperature"] == 0.8


def test_chat_client_strips_temperature_for_claude(monkeypatch):
    seen = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    client = LLMClient(
        ProviderConfig(
            provider=ProviderType.ANTHROPIC,
            model="claude-sonnet-5-5",
            api_key="sk-ant-test",
            temperature=0.7,
        )
    )

    asyncio.run(client.chat([{"role": "user", "content": "hi"}]))

    assert seen["model"] == "anthropic/claude-sonnet-5-5"
    assert "temperature" not in seen


def test_anthropic_defaults_are_current_models():
    model_ids = [model_id for model_id, _label in get_provider_models("anthropic")]

    assert DEFAULT_MODELS["anthropic"] == "claude-sonnet-5-5"
    assert model_ids == ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
    assert DEFAULT_MODELS["openrouter"] == "anthropic/claude-sonnet-5.5"


def test_openrouter_claude_keeps_vendor_prefix():
    from core.llm_client import validate_provider_model

    config = ProviderConfig(
        provider=ProviderType.OPENROUTER, model="anthropic/claude-sonnet-5.5"
    )

    assert config.to_litellm_model() == "openrouter/anthropic/claude-sonnet-5.5"
    assert validate_provider_model("openrouter", DEFAULT_MODELS["openrouter"]) == (
        True,
        "anthropic/claude-sonnet-5.5",
    )


def test_chat_stream_raises_on_claude_refusal():
    from ui.chat_worker import ChatAgentWorker

    refusal = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="partial", tool_calls=None),
                finish_reason="content_filter",
            )
        ]
    )

    class FakeClient:
        async def stream_chat(self, messages, tools=None):
            yield refusal

    worker = ChatAgentWorker(
        ProviderConfig(provider=ProviderType.ANTHROPIC, model="claude-sonnet-5-5"), []
    )

    with pytest.raises(RuntimeError, match="declined"):
        asyncio.run(worker._stream_response(FakeClient(), [{"role": "user", "content": "hi"}]))


def test_shot_fingerprint_records_claude_request_rules():
    from core.operations.shots import shot_model_identity

    claude = shot_model_identity({}, "cloud", "claude-sonnet-5-5")
    gemini = shot_model_identity({}, "cloud", "gemini-3.1-flash-lite-preview")
    local = shot_model_identity({}, "local", "claude-sonnet-5-5")

    assert claude["claude_request"] == {"temperature": None, "max_tokens": CLAUDE_MIN_MAX_TOKENS}
    assert gemini == {"runtime": {}, "backend": "cloud", "cloud_model": "gemini-3.1-flash-lite-preview"}
    assert "claude_request" not in local


@pytest.mark.parametrize(
    ("model", "expected"),
    [("claude-sonnet-5-5", CLAUDE_MIN_MAX_TOKENS), ("gemini-3.1-flash-lite-preview", 50)],
)
def test_custom_query_fingerprint_records_sent_max_tokens(model, expected):
    from core.operations.custom_query import CustomQueryOptions, custom_query_runtime

    runtime = custom_query_runtime(CustomQueryOptions(tier="cloud", model=model))

    assert runtime["max_tokens"] == expected
