"""Tests for ``core/llm_client.complete_with_enum_constraint``.

Covers the Ollama-via-HTTP path for JSON-schema-enum constrained
decoding. Real Ollama is not exercised here (unit tests only); the
real-server integration test for ``compose_with_llm`` lives in
``tests/test_word_llm_composer.py`` and is gated by the
``SCENE_RIPPER_OLLAMA_INTEGRATION`` env var.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fake httpx response objects
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | str):
        self.status_code = status_code
        if isinstance(body, dict):
            self._body = body
            self.text = json.dumps(body)
        else:
            self._body = None
            self.text = body

    def json(self):
        if self._body is None:
            raise ValueError("not valid json")
        return self._body


class _FakeHttpxClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.posted_url = None
        self.posted_json = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):  # noqa: A002 — match httpx kwarg
        self.posted_url = url
        self.posted_json = json
        if self._exc is not None:
            raise self._exc
        return self._response


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def _ok_body(words: list[str]) -> dict:
    return {"message": {"content": json.dumps({"words": words})}}


def test_returns_words_on_happy_path():
    from core.llm_client import complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(200, _ok_body(["the", "sky"])))

    with patch("httpx.Client", return_value=fake):
        out = complete_with_enum_constraint(
            prompt="compose",
            vocabulary=["the", "sky", "is"],
            target_length=2,
            model="qwen3:8b",
            api_base="http://localhost:11434",
        )

    assert out == ["the", "sky"]


def test_payload_carries_schema_with_enum():
    from core.llm_client import complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(200, _ok_body(["a"])))

    with patch("httpx.Client", return_value=fake):
        complete_with_enum_constraint(
            prompt="x",
            vocabulary=["b", "a", "c"],
            target_length=1,
            model="qwen3:8b",
            api_base="http://localhost:11434",
        )

    body = fake.posted_json
    assert body["model"] == "qwen3:8b"
    schema = body["format"]
    assert schema["type"] == "object"
    enum = schema["properties"]["words"]["items"]["enum"]
    # Vocabulary is deduplicated and sorted for deterministic schema construction.
    assert enum == ["a", "b", "c"]
    # System prompt should mention vocabulary; frequencies omitted by default.
    sys_msg = body["messages"][0]
    assert sys_msg["role"] == "system"
    assert "a" in sys_msg["content"] and "b" in sys_msg["content"]


def test_frequency_annotation_in_system_prompt():
    from core.llm_client import complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(200, _ok_body(["the"])))
    with patch("httpx.Client", return_value=fake):
        complete_with_enum_constraint(
            prompt="x",
            vocabulary=["the", "a"],
            target_length=5,
            frequencies={"the": 42, "a": 38},
            model="qwen3:8b",
            api_base="http://localhost:11434",
        )

    sys_content = fake.posted_json["messages"][0]["content"]
    assert "the (42)" in sys_content
    assert "a (38)" in sys_content


def test_url_is_chat_endpoint():
    """``complete_with_enum_constraint`` posts to /api/chat (chat endpoint)."""
    from core.llm_client import complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(200, _ok_body(["x"])))
    with patch("httpx.Client", return_value=fake):
        complete_with_enum_constraint(
            prompt="p",
            vocabulary=["x"],
            target_length=1,
            model="qwen3:8b",
            api_base="http://localhost:11434/",
        )
    assert fake.posted_url == "http://localhost:11434/api/chat"


def test_model_prefix_stripped():
    """``ollama/`` prefix is stripped before posting to Ollama directly."""
    from core.llm_client import complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(200, _ok_body(["x"])))
    with patch("httpx.Client", return_value=fake):
        complete_with_enum_constraint(
            prompt="p",
            vocabulary=["x"],
            target_length=1,
            model="ollama/qwen3:8b",
            api_base="http://localhost:11434",
        )
    assert fake.posted_json["model"] == "qwen3:8b"


# ---------------------------------------------------------------------------
# Error paths — LLMEmptyResponseError
# ---------------------------------------------------------------------------


def test_none_content_raises_empty_response_error():
    from core.llm_client import LLMEmptyResponseError, complete_with_enum_constraint

    fake = _FakeHttpxClient(
        response=_FakeResponse(200, {"message": {"content": None}})
    )
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(LLMEmptyResponseError) as exc:
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )
    assert "no content" in str(exc.value).lower() or "no content" in exc.value.hint.lower()


def test_empty_string_content_raises_empty_response_error():
    from core.llm_client import LLMEmptyResponseError, complete_with_enum_constraint

    fake = _FakeHttpxClient(
        response=_FakeResponse(200, {"message": {"content": ""}})
    )
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(LLMEmptyResponseError):
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )


def test_empty_words_list_raises_empty_response_error():
    from core.llm_client import LLMEmptyResponseError, complete_with_enum_constraint

    fake = _FakeHttpxClient(
        response=_FakeResponse(
            200, {"message": {"content": json.dumps({"words": []})}}
        )
    )
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(LLMEmptyResponseError):
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )


def test_missing_words_key_raises_empty_response_error():
    from core.llm_client import LLMEmptyResponseError, complete_with_enum_constraint

    fake = _FakeHttpxClient(
        response=_FakeResponse(
            200, {"message": {"content": json.dumps({"sentence": "hi"})}}
        )
    )
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(LLMEmptyResponseError):
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )


def test_non_json_content_raises_empty_response_error():
    from core.llm_client import LLMEmptyResponseError, complete_with_enum_constraint

    fake = _FakeHttpxClient(
        response=_FakeResponse(200, {"message": {"content": "not json at all"}})
    )
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(LLMEmptyResponseError):
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )


# ---------------------------------------------------------------------------
# Error paths — OllamaUnreachableError
# ---------------------------------------------------------------------------


def test_connect_error_raises_ollama_unreachable():
    import httpx

    from core.llm_client import OllamaUnreachableError, complete_with_enum_constraint

    fake = _FakeHttpxClient(exc=httpx.ConnectError("connection refused"))
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(OllamaUnreachableError) as exc:
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )
    assert "ollama" in str(exc.value).lower()


def test_timeout_raises_ollama_unreachable_with_hint():
    import httpx

    from core.llm_client import OllamaUnreachableError, complete_with_enum_constraint

    fake = _FakeHttpxClient(exc=httpx.ReadTimeout("slow"))
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(OllamaUnreachableError) as exc:
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )
    assert "timed out" in str(exc.value).lower() or "timeout" in str(exc.value).lower()


def test_non_200_status_raises_ollama_unreachable():
    from core.llm_client import OllamaUnreachableError, complete_with_enum_constraint

    fake = _FakeHttpxClient(response=_FakeResponse(500, "boom"))
    with patch("httpx.Client", return_value=fake):
        with pytest.raises(OllamaUnreachableError) as exc:
            complete_with_enum_constraint(
                prompt="p", vocabulary=["x"], target_length=1,
                model="m", api_base="http://localhost:11434",
            )
    assert "500" in str(exc.value)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_vocabulary_raises_value_error():
    from core.llm_client import complete_with_enum_constraint

    with pytest.raises(ValueError, match="vocabulary"):
        complete_with_enum_constraint(
            prompt="p", vocabulary=[], target_length=1,
            model="m", api_base="http://localhost:11434",
        )


def test_zero_target_length_raises_value_error():
    from core.llm_client import complete_with_enum_constraint

    with pytest.raises(ValueError, match="target_length"):
        complete_with_enum_constraint(
            prompt="p", vocabulary=["x"], target_length=0,
            model="m", api_base="http://localhost:11434",
        )
