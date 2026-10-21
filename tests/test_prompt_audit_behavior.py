"""Behavior changes from the prompt audit: parameter descriptions reach the
model, and the text-to-tool-call fallback only runs for local models."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.chat_tools import tools as tool_registry
from core.llm_client import ProviderConfig, ProviderType


def _tool_schema(name: str) -> dict:
    for tool in tool_registry.to_openai_format():
        if tool["function"]["name"] == name:
            return tool["function"]["parameters"]
    raise AssertionError(f"{name} is not registered")


def test_docstring_args_become_parameter_descriptions():
    direction = _tool_schema("generate_remix")["properties"]["direction"]

    assert "rainbow" in direction["description"]


def _stream_of(text: str):
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ]
    )

    class FakeClient:
        async def stream_chat(self, messages, tools=None):
            yield chunk

    return FakeClient()


TEXT_TOOL_CALL = json.dumps({"name": "list_sequences", "arguments": {}})


@pytest.mark.parametrize(
    ("provider", "model", "expect_parsed"),
    [
        (ProviderType.LOCAL, "qwen3:8b", True),
        (ProviderType.ANTHROPIC, "claude-sonnet-5-5", False),
        (ProviderType.GEMINI, "gemini-2.5-flash", False),
    ],
)
def test_text_tool_call_fallback_is_local_only(provider, model, expect_parsed):
    from ui.chat_worker import ChatAgentWorker

    worker = ChatAgentWorker(ProviderConfig(provider=provider, model=model), [])

    content, tool_calls = asyncio.run(
        worker._stream_response(_stream_of(TEXT_TOOL_CALL), [{"role": "user", "content": "hi"}])
    )

    if expect_parsed:
        assert [call["function"]["name"] for call in tool_calls] == ["list_sequences"]
    else:
        assert tool_calls == []
        assert content == TEXT_TOOL_CALL
