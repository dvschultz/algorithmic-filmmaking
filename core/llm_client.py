"""Multi-provider LLM client via LiteLLM.

Supports local (Ollama), OpenAI, Anthropic, Gemini, and OpenRouter providers
with a unified interface for streaming chat completions with tool support.
"""

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

_litellm_encoding_patched = False


def patch_litellm_encoding():
    """Pre-cache a fallback encoding so litellm never tries to load tiktoken.

    In frozen (PyInstaller) builds tiktoken's native extension may be missing or
    incomplete.  LiteLLM only uses the encoding for approximate token-count
    logging, so a simple word-split estimator is an acceptable substitute.

    Safe to call multiple times — the patch is applied at most once.

    NOTE: Tested against litellm==1.82.6.  The patch targets internal cache
    variables (_default_encoding, _encoding_cache) that may change in future
    releases — re-verify after upgrading litellm.
    """
    global _litellm_encoding_patched
    if _litellm_encoding_patched:
        return
    _litellm_encoding_patched = True

    try:
        import litellm  # noqa: F401 — must be importable (bundled in app)
    except ImportError:
        return

    # If tiktoken works fine, nothing to patch.
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return
    except (ImportError, ModuleNotFoundError, OSError, AttributeError, ValueError) as exc:
        logger.debug("tiktoken unavailable, will use fallback encoding: %s", exc)

    class _FallbackEncoding:
        """Minimal stand-in for tiktoken.Encoding used by litellm for token counts."""

        name = "cl100k_base"

        @staticmethod
        def encode(text, *, disallowed_special=(), allowed_special="all"):  # noqa: ARG004
            # ~1 token per 4 chars is a reasonable approximation for cl100k.
            # Return a range (supports len/iter) to avoid allocating a list.
            return range(max(1, len(text) // 4))

    fallback = _FallbackEncoding()

    # Inject into litellm's lazy-import caches so _get_encoding() finds it
    # without ever importing tiktoken.
    import sys

    # Layer 1: _lazy_imports cache (used by _get_default_encoding)
    import litellm._lazy_imports as _lazy
    _lazy._default_encoding = fallback

    # Layer 2: main.py module caches (used by _get_encoding and __getattr__)
    import litellm.main as _main
    _main._encoding_cache = fallback
    _main.__dict__["encoding"] = fallback

    # Layer 3: top-level litellm module dict (used by litellm.encoding)
    litellm_mod = sys.modules.get("litellm")
    if litellm_mod is not None:
        litellm_mod.__dict__["encoding"] = fallback

    # Layer 4: default_encoding module (token_counter.py does a direct top-level
    # ``from litellm.litellm_core_utils.default_encoding import encoding``).
    # Pre-populate sys.modules so that import never executes the real module
    # body (which would try to load tiktoken and crash).
    import types
    de_key = "litellm.litellm_core_utils.default_encoding"
    de_mod = sys.modules.get(de_key)
    if de_mod is None:
        de_mod = types.ModuleType(de_key)
        de_mod.__package__ = "litellm.litellm_core_utils"
        sys.modules[de_key] = de_mod
    de_mod.encoding = fallback

    logger.info("Patched litellm encoding with fallback (tiktoken unavailable)")


class ProviderType(Enum):
    """Supported LLM providers."""
    LOCAL = "local"  # Ollama
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


# Default models for each provider (updated January 2026)
DEFAULT_MODELS = {
    "local": "qwen3:8b",
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5
    "gemini": "gemini-2.5-flash",
    "openrouter": "anthropic/claude-sonnet-4-5",
}

# Available models per provider (first is recommended)
PROVIDER_MODELS = {
    "local": [
        ("qwen3:8b", "Qwen3 8B (Recommended)"),
        ("llama3.2:latest", "Llama 3.2"),
        ("mistral:latest", "Mistral"),
        ("gemma2:latest", "Gemma 2"),
        ("phi3:latest", "Phi-3"),
    ],
    "openai": [
        ("gpt-5.2", "GPT-5.2 (Recommended)"),
        ("gpt-5", "GPT-5"),
        ("gpt-5-mini", "GPT-5 Mini"),
        ("o1-mini", "o1-mini"),
    ],
    "anthropic": [
        ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (Recommended)"),
        ("claude-opus-4-5-20251101", "Claude Opus 4.5"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ],
    "gemini": [
        ("gemini-3-flash-preview", "Gemini 3 Flash (Recommended)"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ],
    "openrouter": [
        ("anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5 (Recommended)"),
        ("anthropic/claude-haiku-4-5", "Claude Haiku 4.5"),
        ("openai/gpt-5.2", "GPT-5.2"),
        ("google/gemini-2.5-flash", "Gemini 2.5 Flash"),
    ],
}


def get_provider_models(provider: str) -> list[tuple[str, str]]:
    """Get available models for a provider as (model_id, display_name) tuples.

    Args:
        provider: Provider key (local, openai, anthropic, gemini, openrouter)

    Returns:
        List of (model_id, display_name) tuples
    """
    return PROVIDER_MODELS.get(provider, [])


def get_default_model(provider: str) -> str:
    """Get the default model for a provider.

    Args:
        provider: Provider key (local, openai, anthropic, gemini, openrouter)

    Returns:
        Default model string for the provider
    """
    return DEFAULT_MODELS.get(provider, "qwen3:8b")


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7

    def to_litellm_model(self) -> str:
        """Convert to LiteLLM model string format."""
        prefixes = {
            ProviderType.LOCAL: "ollama/",
            ProviderType.ANTHROPIC: "anthropic/",
            ProviderType.GEMINI: "gemini/",
            ProviderType.OPENROUTER: "openrouter/",
        }
        prefix = prefixes.get(self.provider, "")
        return f"{prefix}{self.model}"

    def get_api_base(self) -> Optional[str]:
        """Get API base URL, with defaults for local provider."""
        if self.api_base:
            return self.api_base
        if self.provider == ProviderType.LOCAL:
            return "http://localhost:11434"
        return None


class LLMEmptyResponseError(Exception):
    """Raised when an LLM returns ``None`` or an empty response body.

    Carries both the prompt that was sent and a human-readable hint so the
    caller can surface a useful error to the user.

    Per a recurring LLM gotcha, providers occasionally return ``None``
    content without raising an exception — always validate before using
    the response.
    """

    def __init__(self, prompt: str, hint: str) -> None:
        self.prompt = prompt
        self.hint = hint
        super().__init__(f"{hint} (prompt: {prompt!r})")


class OllamaUnreachableError(Exception):
    """Raised when an Ollama HTTP call fails to connect or times out.

    Includes a hint pointing the user at ``check_ollama_health()`` and
    ``ollama serve``.
    """

    def __init__(self, message: str, original: Optional[Exception] = None) -> None:
        self.original = original
        full = (
            f"{message} — start Ollama with 'ollama serve' or check "
            f"check_ollama_health()."
        )
        if original is not None:
            full += f" Original error: {original}"
        super().__init__(full)


async def check_ollama_health(api_base: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check if Ollama is running and accessible.

    Args:
        api_base: Ollama API base URL

    Returns:
        Tuple of (is_healthy, error_message)
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_base}/api/tags")
            if response.status_code == 200:
                return True, ""
            return False, f"Ollama returned status {response.status_code}"
    except ImportError:
        return False, "httpx not installed. Please install it with: pip install httpx"
    except Exception as e:
        error_type = type(e).__name__
        if "ConnectError" in error_type or "ConnectionRefused" in str(e):
            return False, (
                "Ollama is not running. Start it with 'ollama serve' "
                "or switch to a cloud provider in settings."
            )
        if "Timeout" in error_type:
            return False, "Ollama connection timed out. It may be overloaded or not responding."
        return False, f"Cannot connect to Ollama: {e}"


def check_ollama_health_sync(api_base: str = "http://localhost:11434") -> tuple[bool, str]:
    """Synchronous wrapper around :func:`check_ollama_health`.

    Useful for sync entry points (Qt dialogs, CLI commands) that don't run
    inside an event loop.

    Behavior:
        - When ``asyncio.run`` can be used (no running loop), the async
          probe is awaited and its result returned verbatim.
        - When a loop is already running (``asyncio.run`` raises
          ``RuntimeError``), this returns ``(True, "")`` — the assumption is
          that the caller is in an async test harness and a downstream
          client call will surface a real error if Ollama is unreachable.
          This branch is the *only* swallowed exception; every other
          exception type propagates.

    Args:
        api_base: Ollama API base URL.

    Returns:
        ``(is_healthy, error_message)`` — same shape as the async helper.
    """
    import asyncio

    try:
        return asyncio.run(check_ollama_health(api_base))
    except RuntimeError:
        # Already inside an event loop — skip the probe.
        return True, ""


class LLMClient:
    """Unified LLM client supporting multiple providers via LiteLLM."""

    def __init__(self, config: ProviderConfig):
        """Initialize the LLM client.

        Args:
            config: Provider configuration
        """
        self.config = config

    def _prepare_cached_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> tuple[list[dict], Optional[list[dict]]]:
        """Add cache_control markers for Anthropic prompt caching.

        Anthropic's prompt caching reduces costs by 90% for cache hits AND
        cached tokens don't count against rate limits. This method adds
        cache_control markers to:
        - The last tool (for cumulative tool schema caching)
        - System messages (for system prompt caching)

        Args:
            messages: Conversation messages
            tools: Optional tool definitions

        Returns:
            Tuple of (cached_messages, cached_tools) with cache markers added
        """
        # Only apply caching for Anthropic provider
        if self.config.provider != ProviderType.ANTHROPIC:
            return messages, tools

        # Cache tools - mark the last tool for cumulative caching
        cached_tools = tools
        if tools:
            cached_tools = copy.deepcopy(tools)
            cached_tools[-1]["cache_control"] = {"type": "ephemeral"}

        # Cache system messages
        cached_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Convert to content block format with cache_control
                cached_messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }]
                })
            else:
                cached_messages.append(msg)

        return cached_messages, cached_tools

    async def stream_chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> AsyncIterator[dict]:
        """Stream chat completion with optional tool support.

        Args:
            messages: Conversation messages in OpenAI format
            tools: Optional list of tool definitions in OpenAI format

        Yields:
            Streaming chunks with delta content and tool calls
        """
        from litellm import acompletion

        # Apply prompt caching for Anthropic (90% cost savings, rate limit exempt)
        cached_messages, cached_tools = self._prepare_cached_request(messages, tools)

        kwargs = {
            "model": self.config.to_litellm_model(),
            "messages": cached_messages,
            "stream": True,
            "temperature": self.config.temperature,
            "timeout": 120,  # Prevent indefinite hangs on slow/unresponsive providers
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        api_base = self.config.get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        if cached_tools:
            kwargs["tools"] = cached_tools

        logger.debug(
            f"LLM request: model={kwargs['model']}, "
            f"messages={len(messages)}, tools={len(tools) if tools else 0}"
        )

        response = await acompletion(**kwargs)
        async for chunk in response:
            yield chunk

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> dict:
        """Non-streaming chat completion.

        Args:
            messages: Conversation messages in OpenAI format
            tools: Optional list of tool definitions in OpenAI format

        Returns:
            Complete response with content and optional tool calls
        """
        from litellm import acompletion

        # Apply prompt caching for Anthropic (90% cost savings, rate limit exempt)
        cached_messages, cached_tools = self._prepare_cached_request(messages, tools)

        kwargs = {
            "model": self.config.to_litellm_model(),
            "messages": cached_messages,
            "stream": False,
            "temperature": self.config.temperature,
            "timeout": 120,  # Prevent indefinite hangs on slow/unresponsive providers
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        api_base = self.config.get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        if cached_tools:
            kwargs["tools"] = cached_tools

        logger.debug(
            f"LLM request (non-streaming): model={kwargs['model']}, "
            f"messages={len(messages)}, tools={len(tools) if tools else 0}"
        )

        response = await acompletion(**kwargs)
        return response


class FallbackLLMClient:
    """LLM client with automatic fallback to secondary providers."""

    def __init__(self, primary: ProviderConfig, fallbacks: list[ProviderConfig]):
        """Initialize with primary and fallback providers.

        Args:
            primary: Primary provider configuration
            fallbacks: List of fallback provider configurations
        """
        self.primary = primary
        self.fallbacks = fallbacks

    async def stream_chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> AsyncIterator[dict]:
        """Stream chat with automatic fallback on failure.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions

        Yields:
            Streaming chunks from the first successful provider
        """
        providers = [self.primary] + self.fallbacks
        last_error = None

        for config in providers:
            chunks_yielded = False
            try:
                client = LLMClient(config)
                async for chunk in client.stream_chat(messages, tools):
                    chunks_yielded = True
                    yield chunk
                return  # Success, exit
            except Exception as e:
                if chunks_yielded:
                    # Don't fall back after partial stream — the caller already
                    # has partial content, switching providers would splice an
                    # unrelated response into the same stream
                    logger.error(
                        f"Provider {config.provider.value} failed mid-stream: {e}"
                    )
                    raise
                logger.warning(
                    f"Provider {config.provider.value} failed: {e}, "
                    f"trying next fallback..."
                )
                last_error = e
                continue

        # All providers failed
        raise last_error or RuntimeError("All LLM providers failed")


def normalize_model_for_litellm(model: str) -> str:
    """Add the LiteLLM provider prefix when missing (gemini/, anthropic/)."""
    if not model:
        return model
    m = model.lower()
    if "gemini" in m and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        return f"gemini/{model}"
    if "claude" in m and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        return f"anthropic/{model}"
    return model


def _ollama_api_base(settings) -> str:
    return getattr(settings, "llm_api_base", "").strip() or "http://localhost:11434"


def _ollama_model_name(model: str) -> str:
    for prefix in ("ollama/", "ollama_chat/"):
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    return model


def _local_llm_unavailable_message(
    *,
    model: str,
    api_base: str,
    original_error: Exception,
    cloud_model: str | None = None,
    cloud_error: Exception | None = None,
    missing_api_key: bool = False,
) -> str:
    """Build a user-facing error for local Ollama failures."""
    reason_prefix = ""
    if missing_api_key and cloud_model:
        reason_prefix = (
            f"No API key is configured for {cloud_model}, so Scene Ripper tried "
            "the local Ollama fallback. "
        )
    elif cloud_error is not None and cloud_model:
        reason_prefix = (
            f"The cloud model {cloud_model} failed with "
            f"{type(cloud_error).__name__}, so Scene Ripper tried the local "
            "Ollama fallback. "
        )

    return (
        f"{reason_prefix}Cannot connect to local Ollama at {api_base} for "
        f"model {model}. Start Ollama, run `ollama pull {_ollama_model_name(model)}`, "
        "or configure a working cloud LLM/API key in Settings. "
        f"Original error: {original_error}"
    )


def complete_with_local_fallback(
    *,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    **kwargs,
):
    """Run litellm.completion against a cloud model, falling back to local Ollama
    when the cloud call fails with auth/quota errors or no API key is configured.

    Resolves the API key from the model name (not `settings.llm_provider`), so a
    Gemini model gets a Gemini key even if the chat provider is Anthropic.

    Fallback model: `settings.ollama_model` (default `qwen3:8b`), routed through
    LiteLLM's `ollama/` provider. The api_base honors `settings.llm_api_base`
    when set, otherwise LiteLLM uses its Ollama default.

    Returns the LiteLLM response object. Raises a user-facing RuntimeError when
    local Ollama is selected or used as fallback but is not reachable.
    """
    import litellm
    from core.settings import get_api_key_for_model, load_settings

    settings = load_settings()
    cloud_model = normalize_model_for_litellm(model)
    api_base = _ollama_api_base(settings)

    # If the caller already targets a local model, just use it directly.
    if cloud_model.startswith(("ollama/", "ollama_chat/")):
        local_kwargs = dict(kwargs)
        local_kwargs.setdefault("api_base", api_base)
        try:
            return litellm.completion(
                model=cloud_model,
                messages=messages,
                temperature=temperature,
                **local_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                _local_llm_unavailable_message(
                    model=cloud_model,
                    api_base=api_base,
                    original_error=exc,
                )
            ) from exc

    api_key = get_api_key_for_model(cloud_model)
    cloud_error: Optional[Exception] = None
    missing_api_key = False

    if api_key:
        try:
            return litellm.completion(
                model=cloud_model,
                messages=messages,
                api_key=api_key,
                temperature=temperature,
                **kwargs,
            )
        except (
            litellm.AuthenticationError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
        ) as exc:
            cloud_error = exc
            logger.warning(
                "Cloud LLM call failed for %s (%s); falling back to local Ollama",
                cloud_model, type(exc).__name__,
            )
    else:
        missing_api_key = True
        logger.warning(
            "No API key configured for %s; using local Ollama fallback",
            cloud_model,
        )

    local_model = f"ollama/{settings.ollama_model}"
    fallback_kwargs = dict(kwargs)
    fallback_kwargs.setdefault("api_base", api_base)

    try:
        return litellm.completion(
            model=local_model,
            messages=messages,
            temperature=temperature,
            **fallback_kwargs,
        )
    except Exception as local_exc:
        raise RuntimeError(
            _local_llm_unavailable_message(
                model=local_model,
                api_base=api_base,
                original_error=local_exc,
                cloud_model=cloud_model,
                cloud_error=cloud_error,
                missing_api_key=missing_api_key,
            )
        ) from local_exc


def complete_with_enum_constraint(
    prompt: str,
    vocabulary: list[str],
    target_length: int,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    frequencies: Optional[dict[str, int]] = None,
    think: bool = False,
) -> list[str]:
    """Run a vocabulary-constrained Ollama completion via JSON-schema enum.

    Ollama supports decode-time vocabulary constraints via its ``format``
    parameter when ``format`` carries a JSON Schema whose values include an
    ``enum``. Note: Ollama has explicitly closed all GBNF PRs (issue
    #6237, PR #1606); ``format``+enum is currently the only constrained-
    decoding API exposed by Ollama. Mask construction is non-parallelized
    in current Ollama builds — latency at large vocabulary sizes can be
    high. The plan documents this as a deferred concern.

    Implementation note — LiteLLM passthrough investigation:
        LiteLLM 1.82.6's Ollama provider DOES forward ``response_format``
        when supplied as ``{"type": "json_schema", "json_schema":
        {"schema": <dict>}}`` (see ``litellm/llms/ollama/chat/transformation.py``
        ``map_openai_params`` — it pops the schema out of ``response_format``
        and writes it into Ollama's ``format`` field). So we COULD reach
        Ollama through LiteLLM. We deliberately take a direct HTTP path
        via ``httpx`` here instead: (a) it avoids LiteLLM's translation
        layer for the most security-/correctness-sensitive call in this
        module (the only one where a stripped/translated parameter would
        silently corrupt output by removing the constraint), (b) it gives
        us full control over the request timeout for this latency-prone
        endpoint, and (c) it makes the failure modes (connection refused,
        timeout, non-200) directly observable for ``OllamaUnreachableError``.

    Args:
        prompt: User-supplied generation prompt. Combined with ``system_prompt``
            (or the default frequency-annotated vocabulary system prompt).
        vocabulary: List of allowed words. Duplicates are deduplicated and
            sorted before being passed to Ollama; the final order is also
            used for deterministic schema generation.
        target_length: Target ``words[]`` length used in the system prompt
            so the model knows roughly how long an output to produce.
            Ollama's enum constraint enforces vocabulary, not length.
        system_prompt: Optional system message. If ``None``, a default
            system prompt is constructed from ``vocabulary`` + (optional)
            ``frequencies``.
        model: Ollama model name (e.g. ``"qwen3:8b"``). Defaults to the
            project's default local model from settings if available.
        api_base: Ollama API base URL. Defaults to ``llm_api_base`` from
            settings or ``http://localhost:11434``.
        temperature: Sampling temperature for the underlying Ollama call.
        timeout: Total HTTP timeout in seconds. Defaults to 120s because
            enum-constrained decoding is slow at corpus scale.
        frequencies: Optional ``{word: count}`` map used in the default
            system prompt's frequency annotation.

    Returns:
        The parsed ``words`` array from Ollama's JSON response — a list of
        strings drawn from ``vocabulary``.

    Raises:
        LLMEmptyResponseError: response body is ``None``/empty, JSON parse
            fails, or the ``words`` field is missing/non-list/empty.
        OllamaUnreachableError: HTTP connection refused, timed out, or
            returned a non-2xx status.
        ValueError: ``vocabulary`` is empty or ``target_length`` < 1.
    """
    import json

    import httpx

    if not vocabulary:
        raise ValueError("complete_with_enum_constraint: vocabulary must be non-empty")
    if target_length < 1:
        raise ValueError(
            f"complete_with_enum_constraint: target_length must be >= 1, got {target_length}"
        )

    # Deduplicate + sort for deterministic schema construction.
    sorted_vocab = sorted(set(vocabulary))

    if model is None or api_base is None:
        try:
            from core.settings import load_settings
            _settings = load_settings()
            if model is None:
                model = getattr(_settings, "ollama_model", None) or "qwen3:8b"
            if api_base is None:
                api_base = _ollama_api_base(_settings)
        except Exception:  # noqa: BLE001 — fall back to hard defaults
            if model is None:
                model = "qwen3:8b"
            if api_base is None:
                api_base = "http://localhost:11434"

    model = _ollama_model_name(model)

    if system_prompt is None:
        if frequencies:
            annotated = ", ".join(
                f"{word} ({frequencies.get(word, 0)})" for word in sorted_vocab
            )
            vocab_section = f"Available vocabulary (frequency in parens): {annotated}."
        else:
            vocab_section = "Available vocabulary: " + ", ".join(sorted_vocab) + "."
        system_prompt = (
            "You compose short word sequences using ONLY the supplied "
            "vocabulary. Aim for about "
            f"{target_length} words. Respond with a single JSON object of the form "
            "{\"words\": [...]} whose entries are drawn from the vocabulary. "
            f"{vocab_section}"
        )

    schema = {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted_vocab,
                },
            }
        },
        "required": ["words"],
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,
        # Disable Ollama's "thinking" mode for thinking-capable models
        # (qwen3 etc.). Thinking-mode tokens are extra eval cost that
        # produces no observable output here — the JSON-schema-enum
        # constraint is the only steering we need, and the model's
        # chain-of-thought dramatically inflates latency at corpus scale.
        # ``think`` is ignored by non-thinking models.
        "think": bool(think),
        "options": {"temperature": float(temperature)},
    }

    url = api_base.rstrip("/") + "/api/chat"

    try:
        # ``with`` block guarantees connection cleanup even on cancel /
        # timeout / unhandled exception (subprocess-cleanup-on-exception
        # learning, adapted to httpx).
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise OllamaUnreachableError(
            f"Ollama call to {url} timed out after {timeout}s — "
            "constrained decoding can be slow at corpus scale",
            original=exc,
        ) from exc
    except httpx.ConnectError as exc:
        raise OllamaUnreachableError(
            f"Cannot connect to Ollama at {api_base}",
            original=exc,
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaUnreachableError(
            f"Ollama HTTP error talking to {url}",
            original=exc,
        ) from exc

    if response.status_code != 200:
        raise OllamaUnreachableError(
            f"Ollama returned status {response.status_code} from {url}: "
            f"{response.text[:200]}",
        )

    # Validate response shape per the LLM-empty-response gotcha.
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMEmptyResponseError(
            prompt=prompt,
            hint=f"Ollama response was not valid JSON: {exc}",
        ) from exc

    content = (body or {}).get("message", {}).get("content")
    if not content:
        raise LLMEmptyResponseError(
            prompt=prompt,
            hint="Ollama returned no content; the model may have produced an "
                 "empty response — retry the prompt or check model availability",
        )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMEmptyResponseError(
            prompt=prompt,
            hint=f"Ollama JSON content failed to parse: {exc}",
        ) from exc

    if not isinstance(parsed, dict) or "words" not in parsed:
        raise LLMEmptyResponseError(
            prompt=prompt,
            hint="Ollama response was missing the 'words' key",
        )

    words = parsed.get("words")
    if not isinstance(words, list) or not words:
        raise LLMEmptyResponseError(
            prompt=prompt,
            hint="Ollama returned an empty or non-list 'words' field",
        )

    # All entries should be strings; the enum constraint should have
    # prevented anything else.
    return [str(w) for w in words]


def create_provider_config_from_settings() -> ProviderConfig:
    """Create a ProviderConfig from current application settings.

    Returns:
        ProviderConfig instance configured from settings
    """
    from core.settings import get_llm_api_key, load_settings

    settings = load_settings()

    return ProviderConfig(
        provider=ProviderType(settings.llm_provider),
        model=settings.llm_model,
        api_key=get_llm_api_key() or None,
        api_base=settings.llm_api_base or None,
        temperature=settings.llm_temperature,
    )
