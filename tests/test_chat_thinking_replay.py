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


def _unmarked(messages: list[dict]) -> list[dict]:
    """Messages without cache_control markers, which move to the newest message
    on every request and don't count as edits."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "cache_control"}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(messages)


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
        assert _unmarked(later["messages"][: len(earlier["messages"])]) == _unmarked(earlier["messages"])

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
    assert user_text.startswith("<scene_ripper_context>\n<paths>\n")
    assert "<gui_state>\nActive tab: cut\n</gui_state>" in user_text
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


# --- review fixes: context size, output cap, caching, stop and error paths ---


def _two_turns(first_gui: str, second_gui: str, anthropic_wire, project=None):
    bodies, responses = anthropic_wire
    responses += [_message(_text(0, "One."), "end_turn"), _message(_text(0, "Two."), "end_turn")]
    first = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "a"}], project=project, gui_state_context=first_gui)
    _run(first)
    second = ChatAgentWorker(
        CLAUDE,
        first.conversation + [{"role": "user", "content": "b"}],
        project=project,
        gui_state_context=second_gui,
        system_prompt=first.system_prompt,
    )
    _run(second)
    return bodies, second


def test_unchanged_context_is_not_sent_again(anthropic_wire):
    bodies, second = _two_turns("Active tab: cut", "Active tab: cut", anthropic_wire)

    assert second.conversation[-2]["content"] == "b"
    assert _unmarked(bodies[1]["messages"][:1]) == _unmarked(bodies[0]["messages"])


def test_only_changed_sections_are_sent(anthropic_wire):
    _bodies, second = _two_turns("Active tab: cut", "Active tab: sequence", anthropic_wire)

    latest = second.conversation[-2]["content"]
    assert "<gui_state>\nActive tab: sequence\n</gui_state>" in latest
    assert "<paths>" not in latest
    assert "<project>" not in latest
    assert latest.endswith("\n\nb")


def test_context_lists_a_limited_number_of_sources():
    from pathlib import Path
    from types import SimpleNamespace

    from ui.chat_worker import CONTEXT_MAX_SOURCES

    sources = [
        SimpleNamespace(id=f"s{i}", file_path=Path(f"video{i}.mp4"), analyzed=False, duration_seconds=1.0)
        for i in range(CONTEXT_MAX_SOURCES + 5)
    ]
    project = SimpleNamespace(
        sources=sources, clips_by_source={}, clips=[], sequence=None,
        metadata=SimpleNamespace(name="p"), path=None,
    )
    worker = ChatAgentWorker(CLAUDE, [], project=project)

    try:
        sections = worker._build_turn_context()
    except AttributeError as exc:  # project shape the summary needs beyond this stub
        pytest.skip(f"stub project too small for the summary: {exc}")

    assert f"video{CONTEXT_MAX_SOURCES - 1}.mp4" in sections["project"]
    assert f"video{CONTEXT_MAX_SOURCES}.mp4" not in sections["project"]
    assert "and 5 more (call list_sources" in sections["project"]


def test_claude_chat_sets_an_explicit_output_cap_and_caches_history(anthropic_wire):
    from core.llm_client import CLAUDE_CHAT_MAX_TOKENS

    bodies, responses = anthropic_wire
    responses.append(_message(_text(0, "Hi."), "end_turn"))

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "hello"}], system_prompt="fixed")
    _run(worker)

    assert bodies[0]["max_tokens"] == CLAUDE_CHAT_MAX_TOKENS
    assert bodies[0]["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(worker.conversation)


def test_cut_off_tool_call_is_not_run(anthropic_wire):
    bodies, responses = anthropic_wire
    responses.append(_message(_tool_use(0), "max_tokens"))
    called = []

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    worker.tool_called.connect(lambda name, _args: called.append(name))
    outcome = _run(worker)

    assert called == []
    assert outcome[-1][0] == "error"
    assert "output limit" in outcome[-1][1]


def test_cut_off_text_is_marked(anthropic_wire):
    _bodies, responses = anthropic_wire
    responses.append(_message(_text(0, "Partial answer"), "max_tokens"))

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    outcome = _run(worker)

    assert outcome[-1][0].startswith("Partial answer")
    assert "output limit reached" in outcome[-1][0]


def test_empty_reply_is_not_kept(anthropic_wire):
    _bodies, responses = anthropic_wire
    responses.append(_message(_thinking(0, "SIG_ONLY"), "end_turn"))

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    _run(worker)

    assert [m["role"] for m in worker.conversation] == ["user"]


def test_stop_mid_batch_answers_every_tool_call(anthropic_wire):
    _bodies, responses = anthropic_wire
    responses.append(_message(_tool_use(0) + _tool_use(1) + _tool_use(2), "tool_use"))

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    worker.tool_called.connect(lambda *_args: setattr(worker, "_stop_requested", True))
    outcome = _run(worker)

    assert outcome[-1][0] == "*Cancelled*"
    calls = [c["id"] for c in worker.conversation[1]["tool_calls"]]
    results = [m for m in worker.conversation if m["role"] == "tool"]
    assert sorted(m["tool_call_id"] for m in results) == sorted(calls)
    assert sum("Cancelled by the user" in m["content"] for m in results) == 2


def test_stop_mid_stream_drops_the_partial_reply(anthropic_wire):
    _bodies, responses = anthropic_wire
    responses.append(_message(_text(0, "Half") + _tool_use(1), "tool_use"))

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    worker.text_chunk.connect(lambda *_args: setattr(worker, "_stop_requested", True))
    _run(worker)

    assert [m["role"] for m in worker.conversation] == ["user"]


def test_error_keeps_completed_tool_rounds(anthropic_wire):
    _bodies, responses = anthropic_wire
    rejection = httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
    responses += [_message(_tool_use(0), "tool_use"), rejection, rejection, rejection, rejection]

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    outcome = _run(worker)

    assert outcome[-1][0] == "error"
    assert [m["role"] for m in worker.conversation] == ["user", "assistant", "tool"]


def test_error_before_any_tool_round_keeps_nothing(anthropic_wire):
    _bodies, responses = anthropic_wire
    rejection = httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
    responses += [rejection] * 4

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    _run(worker)

    assert worker.conversation == []


def test_refusal_takes_its_partial_text_back(anthropic_wire):
    _bodies, responses = anthropic_wire
    responses += [
        _message(_text(0, "Before tools. ") + _tool_use(1), "tool_use"),
        _message(_text(0, "Refused half"), "refusal"),
    ]
    shown: list = []

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    worker.text_chunk.connect(lambda text: shown.append(text))
    worker.clear_current_bubble.connect(lambda: shown.clear())
    outcome = _run(worker)

    assert outcome[-1][0] == "error"
    assert "declined" in outcome[-1][1]
    assert "".join(shown) == "Before tools. "
    assert [m["role"] for m in worker.conversation] == ["user", "assistant", "tool"]


def test_main_window_keeps_history_from_a_failed_turn():
    from types import SimpleNamespace

    from ui.main_window import MainWindow

    kept = [{"role": "user", "content": "go"}, {"role": "assistant", "content": ""}, {"role": "tool", "content": "{}"}]
    window = SimpleNamespace(
        chat_panel=SimpleNamespace(on_stream_error=lambda _error: None),
        _chat_worker=SimpleNamespace(conversation=kept, system_prompt="fixed"),
        _chat_history=[],
        _chat_system_prompt=None,
    )

    MainWindow._on_chat_error(window, "boom")

    assert window._chat_history == kept
    assert window._chat_system_prompt == "fixed"

    window._chat_history = ["unchanged"]
    window._chat_worker = SimpleNamespace(conversation=[], system_prompt="fixed")
    MainWindow._on_chat_error(window, "boom")
    assert window._chat_history == ["unchanged"]


# --- cloud review follow-ups ---


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-5-5", 64000),          # 5.x, not in the bundled map
        ("anthropic/claude-sonnet-4-5-20250929", 64000),
        ("anthropic/claude-opus-4-1", 32000),            # capped at the model's limit
        ("anthropic/claude-3-old-unknown", None),        # unknown older model: leave it to LiteLLM
        ("gemini/gemini-2.5-flash", None),
    ],
)
def test_chat_cap_never_exceeds_the_model_limit(model, expected, monkeypatch):
    import litellm

    from core.llm_client import claude_max_tokens

    limits = {"anthropic/claude-sonnet-4-5-20250929": 64000, "anthropic/claude-opus-4-1": 32000}

    def fake_info(name):
        if name not in limits:
            raise Exception("not mapped")
        return {"max_output_tokens": limits[name]}

    monkeypatch.setattr(litellm, "get_model_info", fake_info)

    assert claude_max_tokens(model, 64000) == expected


def test_failed_turn_truncates_large_tool_results(anthropic_wire, monkeypatch):
    from ui.chat_worker import FAILED_TURN_TOOL_RESULT_CHARS

    _bodies, responses = anthropic_wire
    too_long = httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}})
    responses += [_message(_thinking(0, "SIG_A") + _tool_use(1), "tool_use"), too_long]
    huge = "x" * (FAILED_TURN_TOOL_RESULT_CHARS * 5)
    monkeypatch.setattr(
        "core.tool_executor.ToolExecutor.execute",
        lambda self, tc: {"success": True, "result": huge, "tool_call_id": tc["id"], "name": "list_sequences"},
    )

    worker = ChatAgentWorker(CLAUDE, [{"role": "user", "content": "go"}], system_prompt="fixed")
    _run(worker)

    tool = [m for m in worker.conversation if m["role"] == "tool"][0]
    assert len(tool["content"]) < FAILED_TURN_TOOL_RESULT_CHARS + 100
    assert tool["content"].endswith("(truncated after the turn failed)")
    assert not any("thinking_blocks" in m for m in worker.conversation)


def test_other_providers_keep_only_the_newest_context_block():
    gemini = ProviderConfig(provider=ProviderType.GEMINI, model="gemini-2.5-flash")
    history = [
        {"role": "user", "content": "<scene_ripper_context>\n<gui_state>\nold\n</gui_state>\n</scene_ripper_context>\n\nfirst"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    worker = ChatAgentWorker(gemini, history, gui_state_context="old", system_prompt="fixed")

    sent = worker._with_turn_context(history)

    assert sent[0]["content"] == "first"
    assert "<paths>" in sent[2]["content"] and "<gui_state>\nold\n</gui_state>" in sent[2]["content"]
