"""Agent imports share background preparation and retain reply ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.project import Project


def test_folder_tool_defers_probing_to_gui_import_queue(tmp_path, monkeypatch):
    from core.chat_tools import import_folder

    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    probe = Mock(side_effect=AssertionError("Probe ran on GUI dispatch thread"))
    monkeypatch.setattr("core.spine.sources.probe_source", probe)
    project = Project.new(name="import")
    result = import_folder(project, SimpleNamespace(project=project), str(tmp_path))
    assert result["_wait_for_worker"] == "source_import"
    assert result["file_paths"] == [str(media)]
    probe.assert_not_called()
    assert not project.sources


def test_video_tool_waits_for_background_publication(tmp_path):
    from core.chat_tools import import_video

    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    window = SimpleNamespace(project=Project.new(name="import"), _add_video_to_library=Mock())
    result = import_video(window, str(media))
    assert result["_wait_for_worker"] == "source_import"
    assert result["file_paths"] == [str(media)]
    window._add_video_to_library.assert_not_called()


def test_agent_import_thread_publication_and_cancellation(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    code = """
from pathlib import Path
import sys, threading, time
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.spine.sources import add_source_if_missing
from models.clip import Source
from ui.workers.gui_tool_reply import GuiToolReply
from ui.workers.gui_tool_cancellation import cancel_gui_tool_work
from ui.workers.source_import_delivery import AgentSourceImport
from ui.workers.source_import_worker import SourceImportQueue
import ui.workers.source_import_worker as module
app = QCoreApplication([])
owner = threading.get_ident()
root = Path(sys.argv[1])
media = root / 'media.mp4'; media.write_bytes(b'video')
alias = root / 'alias.mp4'; alias.hardlink_to(media)
unrelated = root / 'other.mp4'; unrelated.write_bytes(b'other')
started, release = threading.Event(), threading.Event()
probe_threads, delivered = [], []
def probe(path, cancel):
    probe_threads.append(threading.get_ident())
    started.set()
    assert release.wait(3)
    if cancel.is_set():
        return None
    return Source(file_path=path)
module.prepare_source_import = probe
class Chat:
    _stop_requested = False
    token = 'one'
    def is_gui_tool_pending(self, token, name):
        return token == self.token
    def set_gui_tool_result(self, result):
        delivered.append(result); return True
class Window(QObject):
    def __init__(self):
        super().__init__()
        self.project = Project.new(name='import')
        self._chat_worker = Chat()
        self._source_import_queue = SourceImportQueue(self)
        self._source_import_queue.result_ready.connect(self.publish)
    def publish(self, request, source):
        assert threading.get_ident() == owner
        if request.reply is None or request.reply.is_current(self):
            add_source_if_missing(self.project, source)
    def _queue_source_import(self, path, reply=None):
        self._source_import_queue.submit(path, (self.project.session.session_id, None), reply=reply)
def spin(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.001)
    assert predicate()
window = Window()
window._dispatch_gui_reply = GuiToolReply.capture(window, 'import_folder', 'one')
batch = AgentSourceImport(window); batch.start([media, alias])
assert started.wait(2)
assert not delivered and not window.project.sources
release.set()
spin(lambda: not window._source_import_queue.pending)
assert len(delivered) == 1 and delivered[0]['result']['imported_count'] == 1
assert delivered[0]['result']['skipped_count'] == 1
assert all(t != owner for t in probe_threads)
assert not window._active_source_imports
# Expired agent cancellation reaps only its own preparation. UI imports survive.
window.project = Project.new(name='cancel')
window._chat_worker.token = 'two'
window._dispatch_gui_reply = GuiToolReply.capture(window, 'import_folder', 'two')
started.clear(); release.clear()
batch = AgentSourceImport(window); batch.start([media, alias])
assert started.wait(2)
window._queue_source_import(unrelated)
window._chat_worker.token = None
cancel_gui_tool_work(window, name='import_folder', token='two')
release.set()
spin(lambda: not window._source_import_queue.pending)
assert len(delivered) == 1
assert [s.file_path for s in window.project.sources] == [unrelated]
assert not window._active_source_imports
window._source_import_queue.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
