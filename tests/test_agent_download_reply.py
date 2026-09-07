"""Download replies survive source-import deferral without borrowing a newer request."""

import os
from pathlib import Path
import subprocess
import sys


def test_download_reply_and_admission_keep_original_request():
    code = """
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import Mock
from core.project import Project
from models.clip import Source
from ui.main_window import MainWindow
from ui.workers.gui_tool_reply import GuiToolReply
project = Project.new()
chat = Mock(_stop_requested=False)
window = SimpleNamespace(project=project, _chat_worker=chat,
    _source_import_queue=SimpleNamespace(pending=True), _deferred_agent_download_results=None,
    progress_bar=Mock(), status_bar=Mock(), collect_tab=Mock(),
    _update_chat_project_state=Mock(), _select_source=Mock(), _source_selection_generation=0,
    _pending_agent_tool_call_id='unrelated', _pending_agent_tool_name='export')
window._on_agent_bulk_download_finished = MethodType(MainWindow._on_agent_bulk_download_finished, window)
reply = GuiToolReply.capture(window, 'download_videos', 'original')
results = [{'success': True, 'url': 'url'}]
window._on_agent_bulk_download_finished(results, reply=reply)
results[0]['success'] = False
assert not chat.set_gui_tool_result.called
window._source_import_queue.pending = False
MainWindow._on_source_imports_drained(window)
response = chat.set_gui_tool_result.call_args.args[0]
assert response['tool_call_id'] == 'original' and response['result']['success_count'] == 1
assert window._pending_agent_tool_call_id == 'unrelated'
# A replaced chat must neither receive the deferred reply nor admit its queued source.
window._source_import_queue.pending = True
window._on_agent_bulk_download_finished(results, reply=reply)
new_chat = Mock(_stop_requested=False)
window._chat_worker = new_chat
window._source_import_queue.pending = False
window.progress_bar.reset_mock()
MainWindow._on_source_imports_drained(window)
new_chat.set_gui_tool_result.assert_not_called()
window.progress_bar.setVisible.assert_not_called()
request = SimpleNamespace(context=(project.session.session_id, None), reply=reply)
source = Source(file_path=Path('download.mp4'))
MainWindow._on_source_import_ready(window, request, source)
assert not project.sources
window._chat_worker = chat
MainWindow._on_source_import_ready(window, request, source)
assert project.sources == [source]
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
