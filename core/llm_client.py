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
        ("gemini-2.5-flash", "Gemini 2.5 Flash (Recommended)"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
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
            try:
                client = LLMClient(config)
                async for chunk in client.stream_chat(messages, tools):
                    yield chunk
                return  # Success, exit
            except Exception as e:
                logger.warning(
                    f"Provider {config.provider.value} failed: {e}, "
                    f"trying next fallback..."
                )
                last_error = e
                continue

        # All providers failed
        raise last_error or RuntimeError("All LLM providers failed")


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
