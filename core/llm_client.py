"""Multi-provider LLM client via LiteLLM.

Supports local (Ollama), OpenAI, Anthropic, Gemini, and OpenRouter providers
with a unified interface for streaming chat completions with tool support.
"""

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

        kwargs = {
            "model": self.config.to_litellm_model(),
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        api_base = self.config.get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        if tools:
            kwargs["tools"] = tools

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

        kwargs = {
            "model": self.config.to_litellm_model(),
            "messages": messages,
            "stream": False,
            "temperature": self.config.temperature,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        api_base = self.config.get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        if tools:
            kwargs["tools"] = tools

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
