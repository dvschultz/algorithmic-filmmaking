"""Tests for secret redaction utilities."""

import json
import logging

from core.chat_export import ChatExportConfig, export_chat_as_json, export_chat_as_markdown
from core.redaction import SecretRedactionFilter, redact_secrets, redact_text


def test_redact_text_removes_common_provider_keys():
    text = (
        "openai=sk-proj-abcdefghijklmnopqrstuvwxyz123456 "
        "anthropic=sk-ant-abcdefghijklmnopqrstuvwxyz123456 "
        "google=AIzaabcdefghijklmnopqrstuvwxyz123456 "
        "replicate=r8_abcdefghijklmnopqrstuvwxyz123456"
    )

    redacted = redact_text(text)

    assert "sk-proj-" not in redacted
    assert "sk-ant-" not in redacted
    assert "AIza" not in redacted
    assert "r8_" not in redacted
    assert redacted.count("[REDACTED]") == 4


def test_redact_text_removes_signed_url_query_secrets():
    text = (
        "https://example.com/video.mp4?token=abc123&x-goog-signature=deadbeef&keep=ok "
        "https://example.com/watch?v=public"
    )

    redacted = redact_text(text)

    assert "abc123" not in redacted
    assert "deadbeef" not in redacted
    assert "keep=ok" in redacted
    assert "v=public" in redacted


def test_redact_secrets_redacts_sensitive_dict_keys():
    payload = {
        "arguments": {
            "api_key": "plain-secret-value",
            "prompt": "keep this",
        }
    }

    assert redact_secrets(payload) == {
        "arguments": {
            "api_key": "[REDACTED]",
            "prompt": "keep this",
        }
    }


def test_secret_redaction_filter_sanitizes_log_record():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%s",
        args=("sk-ant-abcdefghijklmnopqrstuvwxyz123456",),
        exc_info=None,
    )

    assert SecretRedactionFilter().filter(record) is True
    assert record.getMessage() == "token=[REDACTED]"


def test_chat_export_redacts_content_and_tool_arguments(tmp_path):
    config = ChatExportConfig(
        output_dir=tmp_path,
        format="both",
        include_tool_args=True,
    )
    messages = [
        {
            "role": "user",
            "content": "Use sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "example",
                        "arguments": json.dumps(
                            {"youtube_api_key": "AIzaabcdefghijklmnopqrstuvwxyz123456"}
                        ),
                    }
                }
            ],
        },
    ]

    ok_md, md_path = export_chat_as_markdown(messages, config)
    ok_json, json_path = export_chat_as_json(messages, config)

    assert ok_md is True
    assert ok_json is True
    md_text = tmp_path.joinpath(md_path).read_text(encoding="utf-8")
    json_text = tmp_path.joinpath(json_path).read_text(encoding="utf-8")
    assert "sk-proj-" not in md_text
    assert "AIza" not in md_text
    assert "sk-proj-" not in json_text
    assert "AIza" not in json_text
