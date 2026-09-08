"""Analysis completion carries its original chat request through queued delivery."""

import os
from pathlib import Path
import subprocess
import sys


def test_analysis_delivery_and_all_migrated_handlers():
    code = """
import threading, time
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from ui.workers.gui_tool_reply import GuiToolReply, AgentAnalysisCompletion, gui_reply_scope
from ui.main_window import MainWindow
app = QCoreApplication([])
class Worker(QThread):
    done = Signal()
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
    def run(self):
        assert self.release.wait(3)
        self.done.emit(); self.done.emit()
window = QObject()
window.project = Project.new()
window._chat_worker = Mock(_stop_requested=False)
received = []
owner = threading.get_ident()
def bind(worker, token):
    window.analysis_worker = worker
    reply = GuiToolReply.capture(window, 'colors', token)
    def complete(*, reply):
        received.append((reply.token, threading.get_ident()))
        reply.send(window, {'success': True, 'result': {}})
    with gui_reply_scope(window, reply):
        relay = AgentAnalysisCompletion(window, worker, 'analysis_worker', complete)
    worker.done.connect(relay.completed)
    worker.start()
def drain(worker):
    worker.release.set()
    assert worker.wait(3000)
    for _ in range(10): app.processEvents()
old, new = Worker(), Worker()
bind(old, 'old'); bind(new, 'new')
drain(old)
assert not received and window.analysis_worker is new
drain(new)
assert received == [('new', owner)]
assert window.analysis_worker is None
stale = Worker()
bind(stale, 'stale')
window.project.clear()
drain(stale)
assert received == [('new', owner)]
# Failed analysis dispatch must not clear another operation's pending request.
from unittest.mock import patch
controller = SimpleNamespace(project=Project.new(), _chat_worker=Mock(_stop_requested=False),
    _pending_agent_tool_call_id='unrelated', _pending_agent_tool_name='export')
def start(wait_type, result):
    assert controller._dispatch_gui_reply.token == 'request'
    return False
controller._start_worker_for_tool = start
tool = SimpleNamespace(name='describe', modifies_gui_state=True, modifies_project_state=False,
    func=lambda: {'_wait_for_worker': 'description'})
with patch('core.chat_tools.tools.get', return_value=tool):
    MainWindow._on_gui_tool_requested(controller, 'describe', {}, 'request')
assert controller._pending_agent_tool_call_id == 'unrelated'
assert controller._dispatch_gui_reply is None
assert controller._chat_worker.set_gui_tool_result.call_args.args[0]['tool_call_id'] == 'request'
# A nested GUI event replacing the chat during a synchronous tool must not reply to the new chat.
new_chat = Mock(_stop_requested=False)
def replace_chat():
    controller._chat_worker = new_chat
    return {'ok': True}
tool.func = replace_chat
controller._apply_gui_tool_side_effects = Mock()
with patch('core.chat_tools.tools.get', return_value=tool):
    MainWindow._on_gui_tool_requested(controller, 'describe', {}, 'replaced')
new_chat.set_gui_tool_result.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
