"""The chat loop separates GUI request tokens from provider history IDs."""

import os
from pathlib import Path
import subprocess
import sys


def test_chat_loop_reused_provider_ids_and_late_replies():
    code = """
import asyncio
from unittest.mock import AsyncMock, patch
from PySide6.QtCore import QCoreApplication
from core.llm_client import ProviderConfig, ProviderType
from ui.chat_worker import ChatAgentWorker
app = QCoreApplication([])
worker = ChatAgentWorker(ProviderConfig(ProviderType.OPENAI, 'test'), [])
call = {'id': 'reused-provider-id', 'function': {
    'name': 'navigate_to_tab', 'arguments': '{"tab_name": "cut"}'}}
worker._stream_response = AsyncMock(side_effect=[('', [call]), ('', [call]), ('done', [])])
worker._build_system_prompt = lambda: 'test'
replies, completed, errors = [], [], []
def reply(name, args, token):
    assert token != 'reused-provider-id'
    if replies:
        assert not worker.set_gui_tool_result(replies[-1])
    result = {'tool_call_id': token, 'name': name, 'success': True, 'result': {'ok': True}}
    replies.append(result)
    assert worker.set_gui_tool_result(result)
    assert not worker.set_gui_tool_result(result)
worker.gui_tool_requested.connect(reply)
worker.complete.connect(lambda *args: completed.append(args))
worker.error.connect(errors.append)
with patch('ui.chat_worker.LLMClient'), patch('ui.chat_worker.get_tool_timeout', return_value=.1):
    asyncio.run(worker._async_run())
assert not errors, errors
assert len(replies) == 2 and replies[0]['tool_call_id'] != replies[1]['tool_call_id']
assert completed[0][0] == 'done'
history = completed[0][1]
assert [m['tool_call_id'] for m in history if m['role'] == 'tool'] == ['reused-provider-id'] * 2
assert not worker.set_gui_tool_result(replies[-1])
worker.stop()
assert not worker.set_gui_tool_result(replies[-1])
# Timeout cancellation carries the transport token, never a reused provider ID.
timed = ChatAgentWorker(ProviderConfig(ProviderType.OPENAI, 'test'), [])
timed._stream_response = AsyncMock(side_effect=[('', [call]), ('done', [])])
timed._build_system_prompt = lambda: 'test'
requested, cancelled = [], []
timed.gui_tool_requested.connect(lambda name, args, token: requested.append((name, token)))
timed.gui_tool_cancelled.connect(lambda name, token: cancelled.append((name, token)))
with patch('ui.chat_worker.LLMClient'), patch('ui.chat_worker.get_tool_timeout', return_value=0):
    asyncio.run(timed._async_run())
assert cancelled == requested and len(cancelled) == 1
assert cancelled[0][1] != 'reused-provider-id'

"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
