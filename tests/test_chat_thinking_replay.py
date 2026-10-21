"""Claude thinking blocks survive the chat agent loop.

Claude 5.x binds each thinking block to the system prompt, tools and every
earlier message. These tests capture the request bodies LiteLLM sends and
check that history only ever grows by appending.
"""

import asyncio
import json

import httpx
import pytest

from core.chat_export import _messages_for_export
from core.llm_client import ProviderConfig, ProviderType
from ui.chat_worker import ChatAgentWorker

CLAUDE = ProviderConfig(
    provider=ProviderType.ANTHROPIC, model="claude-sonnet-5-5", api_key="sk-ant-test"
)


def _sse(events: list[dict]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _thinking(index: int, signature: str) -> list[dict]:
    return [
        {"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": signature}},
        {"type": "content_block_stop", "index": index},
    ]


def _text(index: int, text: str) -> list[dict]:
    return [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": index},
    ]


def _tool_use(index: int) -> list[dict]:
    return [
        {"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": f"toolu_{index}", "name": "list_sequences", "input": {}}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": index},
    ]


def _message(blocks: list[dict], stop_reason: str) -> list[dict]:
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [], "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    }
    end = [
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    return [start, *blocks, *end]


@pytest.fixture
def anthropic_wire(monkeypatch):
    """Serve scripted Anthropic streams and record each request body."""
    bodies: list[dict] = []
    responses: list[httpx.Response | list[dict]] = []

    async def fake_send(self, request, **_kwargs):
        bodies.append(json.loads(request.content))
        reply = responses.pop(0)
        if isinstance(reply, httpx.Response):
            reply.request = request
            return reply
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(reply), request=request
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return bodies, responses


def _run(worker: ChatAgentWorker) -> list:
    outcome: list = []
    worker.complete.connect(lambda *args: outcome.append(args))
    worker.error.connect(lambda message: outcome.append(("error", message)))
    asyncio.run(worker._async_run())
    return outcome


def _block_types(message: dict) -> list[str]:
    content = message["content"]
    return [block["type"] for block in content] if isinstance(content, list) else ["str"]


def test_thinking_is_replayed_and_history_only_appends(anthropic_wire):
    bodies, responses = anthropic_wire
    responses += [
        _message(_thinking(0, "SIG_A") + _tool_use(1), "tool_use"),
        _message(_thinking(0, "SIG_B") + _text(1, "Done."), "end_turn"),
        _message(_thinking(0, "SIG_C") + _text(1, "You're welcome."), "end_turn"),
    ]

    first = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "list my sequences"}])
    assert _run(first)[-1][0] == "Done."
    second = ChatAgentWorker(
        CLAUDE,
        first.conversation + [{"role": "user", "content": "thanks"}],
        system_prompt=first.system_prompt,
    )
    assert _run(second)[-1][0] == "You're welcome."

    assert len(bodies) == 3
    assert bodies[0]["system"] == bodies[1]["system"] == bodies[2]["system"]
    assert bodies[0]["tools"] == bodies[2]["tools"]
    for earlier, later in zip(bodies, bodies[1:]):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]

    assistant_turns = [m for m in bodies[2]["messages"] if m["role"] == "assistant"]
    assert [_block_types(m) for m in assistant_turns] == [["thinking", "tool_use"], ["thinking", "text"]]
    assert [m["content"][0]["signature"] for m in assistant_turns] == ["SIG_A", "SIG_B"]
    assert "temperature" not in bodies[2]


def test_turn_context_rides_in_the_user_message(anthropic_wire):
    bodies, responses = anthropic_wire
    responses.append(_message(_text(0, "Hi."), "end_turn"))

    worker = ChatAgentWorker(
        CLAUDE, [{"role": "user", "content": "hello"}], gui_state_context="Active tab: cut"
    )
    _run(worker)

    system = json.dumps(bodies[0]["system"])
    user_text = bodies[0]["messages"][0]["content"][0]["text"]
    assert "Active tab: cut" not in system
    assert "Download directory" not in system
    assert user_text.startswith("<scene_ripper_context>\n")
    assert "Active tab: cut" in user_text
    assert user_text.endswith("</scene_ripper_context>\n\nhello")
    assert worker.conversation[0]["content"] == user_text


def test_thinking_after_output_drops_every_block_so_far(anthropic_wire):
    bodies, responses = anthropic_wire
    # LiteLLM would replay this block ahead of the text, out of place.
    responses.append(_message(_text(0, "Checking.") + _thinking(1, "SIG_LATE") + _tool_use(2), "tool_use"))
    responses.append(_message(_text(0, "Done."), "end_turn"))
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok", "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "SIG_OLD"}]},
        {"role": "user", "content": "list my sequences"},
    ]

    worker = ChatAgentWorker(CLAUDE, history, system_prompt="fixed")
    _run(worker)

    assert not any("thinking_blocks" in m for m in worker.conversation)
    assert "thinking" not in json.dumps(bodies[1]["messages"])


def test_rejected_replay_retries_once_without_thinking(anthropic_wire):
    bodies, responses = anthropic_wire
    rejection = httpx.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages.1.content.0: Invalid `signature` in `thinking` block. "
                "The block is bound to a different conversation.",
            },
        },
    )
    responses += [rejection, _message(_text(0, "Recovered."), "end_turn")]
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok", "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "SIG_OLD"}]},
        {"role": "user", "content": "again"},
    ]

    worker = ChatAgentWorker(CLAUDE, history, system_prompt="fixed")
    outcome = _run(worker)

    assert outcome[-1][0] == "Recovered."
    assert "SIG_OLD" in json.dumps(bodies[0]["messages"])
    assert "thinking" not in json.dumps(bodies[1]["messages"])
    assert not any("thinking_blocks" in m for m in worker.conversation)


def test_other_providers_never_receive_thinking_blocks():
    sent = []
    worker = ChatAgentWorker(
        ProviderConfig(provider=ProviderType.GEMINI, model="gemini-2.5-flash"),
        [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok", "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "S"}]},
            {"role": "user", "content": "again"},
        ],
        system_prompt="fixed",
    )

    async def fake_stream(_client, messages):
        sent.extend(messages)
        return "Fine.", []

    worker._stream_response = fake_stream
    _run(worker)

    assert sent and not any("thinking_blocks" in m for m in sent)


def test_export_hides_turn_context_and_thinking():
    exported = _messages_for_export(
        [
            {"role": "user", "content": "<scene_ripper_context>\nActive tab: cut\n</scene_ripper_context>\n\nhello"},
            {"role": "assistant", "content": "Hi.", "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "S"}]},
        ]
    )

    assert exported == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
    ]
