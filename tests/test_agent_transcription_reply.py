"""A transcription request retains ownership while advancing between sources."""

import os
from pathlib import Path
import subprocess
import sys


def test_transcription_chain_preserves_request_and_stops_after_cancel():
    code = """
from types import SimpleNamespace
from unittest.mock import Mock
from core.project import Project
from ui.main_window import MainWindow
from ui.workers.gui_tool_reply import GuiToolReply
chat = Mock(_stop_requested=False)
source = SimpleNamespace(filename='second.mp4')
window = SimpleNamespace(project=Project.new(), _chat_worker=chat,
    _transcription_finished_handled=False, _pending_agent_transcription=True,
    _agent_transcription_source_queue=[(source, []), (source, [])], _agent_transcription_clips=[],
    _agent_transcription_total_sources=3, status_bar=Mock(), progress_bar=Mock(),
    analyze_tab=Mock(), _start_transcription_worker=Mock(), _update_chat_project_state=Mock(),
    _build_agent_analysis_result=lambda *args: {},
    _pending_agent_tool_call_id='unrelated', _pending_agent_tool_name='export')
window._on_agent_transcription_finished = lambda: None
reply = GuiToolReply.capture(window, 'transcribe', 'original')
MainWindow._on_agent_transcription_finished(window, reply=reply)
assert window._start_transcription_worker.call_args.kwargs['agent_reply'] is reply
assert not chat.set_gui_tool_result.called
assert 'source 2/3' in window.status_bar.showMessage.call_args.args[0]
MainWindow._on_agent_transcription_finished(window, reply=reply)
assert 'source 3/3' in window.status_bar.showMessage.call_args.args[0]
MainWindow._on_agent_transcription_finished(window, reply=reply)
assert chat.set_gui_tool_result.call_args.args[0]['tool_call_id'] == 'original'
assert window._pending_agent_tool_call_id == 'unrelated'
assert window._pending_agent_tool_name == 'export'
# Cancel between sources prevents dispatch of the next source and clears this run.
window._transcription_finished_handled = False
window._agent_transcription_source_queue = [(source, [])]
window._pending_agent_transcription = True
window._start_transcription_worker.reset_mock()
chat._stop_requested = True
MainWindow._on_agent_transcription_finished(window, reply=reply)
window._start_transcription_worker.assert_not_called()
assert not window._agent_transcription_source_queue
assert not window._pending_agent_transcription
window._transcription_finished_handled = False
window._agent_transcription_source_queue = [(source, [])]
window._pending_agent_transcription = True
chat._stop_requested = False
chat.is_gui_tool_pending.return_value = False
MainWindow._on_agent_transcription_finished(window, reply=reply)
window._start_transcription_worker.assert_not_called()
assert not window._agent_transcription_source_queue
# Real queued delivery retains manual callback signatures and replacement cleanup.
import time
from unittest.mock import patch
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
app = QCoreApplication([])
class Worker(QThread):
    job_started = Signal(str, str)
    progress = Signal(int, int)
    status = Signal(str)
    transcript_ready = Signal(str, list)
    transcription_completed = Signal()
    error = Signal(str)
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tasks = ()
    def run(self):
        self.transcription_completed.emit()
        self.transcription_completed.emit()
owner = QObject()
owner.project = Project.new()
owner.transcription_worker = None
owner.settings = Mock()
owner.status_bar = Mock()
owner._stop_worker_safely = Mock()
owner._on_transcription_progress = Mock()
owner._on_transcript_ready = Mock()
owner._on_transcription_error = Mock()
completed = []
def on_complete():
    completed.append(True)
    if len(completed) == 1:
        MainWindow._start_transcription_worker(owner, [], source, on_complete)
with patch('ui.main_window.TranscriptionWorker', Worker):
    MainWindow._start_transcription_worker(owner, [], source, on_complete)
    end = time.monotonic() + 5
    while (len(completed) < 2 or owner.transcription_worker is not None) and time.monotonic() < end:
        app.processEvents(); time.sleep(.001)
    assert completed == [True, True]
    assert owner.transcription_worker is None
    owner._chat_worker = Mock(_stop_requested=False)
    agent_reply = GuiToolReply.capture(owner, 'transcribe', 'agent')
    received = []
    def agent_complete(*, reply): received.append(reply)
    MainWindow._start_transcription_worker(owner, [], source, agent_complete, agent_reply=agent_reply)
    end = time.monotonic() + 5
    while owner.transcription_worker is not None and time.monotonic() < end:
        app.processEvents(); time.sleep(.001)
    assert received == [agent_reply]
    assert owner.transcription_worker is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
