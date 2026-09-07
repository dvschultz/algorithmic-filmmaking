"""Exercise queued metadata preparation and cancellation with a real Qt loop."""

import os
from pathlib import Path
import subprocess
import sys


def test_import_queue_thread_delivery_reset_and_media_change(tmp_path):
    code = """
from pathlib import Path
import sys, time, threading
from PySide6.QtCore import QCoreApplication
from models.clip import Source
from ui.workers.source_import_worker import SourceImportQueue
import ui.workers.source_import_worker as module
app = QCoreApplication([])
root = Path(sys.argv[1])
path = root / 'video.mp4'
path.write_bytes(b'video')
owner = threading.get_ident()
started, release = threading.Event(), threading.Event()
threads = []
def probe(path, cancel_event):
    threads.append(threading.get_ident())
    started.set()
    assert release.wait(3)
    return Source(file_path=path, fps=24)
module.prepare_source_import = probe
queue = SourceImportQueue()
results, errors, drained = [], [], []
queue.result_ready.connect(lambda request, source: results.append((request, source, threading.get_ident())))
queue.failed.connect(lambda request, error: errors.append(error))
queue.drained.connect(lambda: drained.append(True))
def spin(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    assert predicate()
reply = module.GuiToolReply(object(), 'session', 'download_videos', 'token')
queue.submit(path, ('session', 1), reply=reply)
assert started.wait(2)
assert queue.pending and not results
release.set()
spin(lambda: not queue.pending)
assert results[0][1].fps == 24 and results[0][2] == owner
assert results[0][0].reply is reply
assert threads[0] != owner and drained
# Cancel active and queued work; a fresh submission survives the old completion.
results.clear(); started.clear(); release.clear()
queue.submit(path, ('old', 2))
assert started.wait(2)
queue.submit(path, ('old', 3))
queue.cancel_pending()
queue.submit(path, ('new', 4))
release.set()
spin(lambda: not queue.pending)
assert [r[0].context for r in results] == [('new', 4)]
# Media changed after dispatch cannot publish metadata from the previous file.
results.clear(); started.clear(); release.clear()
queue.submit(path, ('new', 5))
assert started.wait(2)
path.write_bytes(b'replaced video')
release.set()
spin(lambda: not queue.pending)
assert not results and errors
# A preparation failure must not strand later imports.
calls = []
def sometimes_fails(path, cancel_event):
    calls.append(path)
    if len(calls) == 1:
        raise RuntimeError('probe failure')
    return Source(file_path=path)
module.prepare_source_import = sometimes_fails
queue.submit(path, ('new', 6))
queue.submit(path, ('new', 7))
spin(lambda: not queue.pending)
assert results[-1][0].context == ('new', 7)
assert errors[-1] == 'probe failure'
# Shutdown retains the active thread and suppresses its queued result.
module.prepare_source_import = probe
results.clear(); started.clear(); release.clear()
queue.submit(path, ('new', 8))
assert started.wait(2)
threading.Timer(.05, release.set).start()
queue.close()
app.processEvents()
assert not results
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_window_import_delivery_guards_and_agent_completion(tmp_path):
    code = """
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from core.project import Project
from models.clip import Source
from ui.main_window import MainWindow
project = Project.new(name='old')
window = SimpleNamespace(project=project, collect_tab=Mock(), status_bar=Mock(),
    _update_chat_project_state=Mock(), _select_source=Mock(),
    _source_selection_generation=4)
source = Source(file_path=Path('video.mp4'))
request = SimpleNamespace(context=(project.session.session_id, 4))
project.clear()
MainWindow._on_source_import_ready(window, request, source)
assert not project.sources
window.collect_tab.add_source.assert_not_called()
# Import remains valid after unrelated selection changes, but cannot steal selection.
request.context = (project.session.session_id, 3)
MainWindow._on_source_import_ready(window, request, source)
assert project.sources == [source]
window._select_source.assert_not_called()
request.context = (project.session.session_id, 4)
MainWindow._on_source_import_ready(window, request, Source(file_path=source.file_path))
window._select_source.assert_called_once_with(source)
window.collect_tab.add_source.assert_called_once_with(source)
# A completed download must wait for source admission before returning to the agent.
window._source_import_queue = SimpleNamespace(pending=True)
window._deferred_agent_download_results = None
results = [{'success': True}]
MainWindow._on_agent_bulk_download_finished(window, results)
assert window._deferred_agent_download_results == (project.session.session_id, results, None)
window._on_agent_bulk_download_finished = Mock()
MainWindow._on_source_imports_drained(window)
window._on_agent_bulk_download_finished.assert_called_once_with(results, reply=None)
window._on_agent_bulk_download_finished.reset_mock()
window._deferred_agent_download_results = ('obsolete-session', results, None)
MainWindow._on_source_imports_drained(window)
window._on_agent_bulk_download_finished.assert_not_called()
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


def test_preparation_cancellation_skips_thumbnail(tmp_path, monkeypatch):
    from threading import Event
    from unittest.mock import Mock
    from core.spine.sources import prepare_source_import
    from models.clip import Source

    cancel = Event()
    path = tmp_path / "video.mp4"

    def probe(path):
        cancel.set()
        return Source(file_path=path)

    thumbnail = Mock()
    monkeypatch.setattr("core.spine.sources.probe_source", probe)
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", thumbnail)
    assert prepare_source_import(path, cancel) is None
    thumbnail.assert_not_called()
