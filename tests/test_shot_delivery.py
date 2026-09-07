"""Shot analysis cannot publish into a replacement frame or project."""

import os
import subprocess
import sys


def test_frame_shot_result_rejects_replaced_project(tmp_path):
    code = r"""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.analysis_target import AnalysisTarget
from models.frame import Frame
from ui.main_window import MainWindow
from ui.workers.shot_type_worker import ShotTypeWorker
from pathlib import Path
import sys
app = QCoreApplication([])
class Window(QObject): pass
window = Window()
image = Path(sys.argv[1]) / 'frame.png'; image.write_bytes(b'image')
original = Project.new(); original.add_frames([Frame(id='same', file_path=image)])
window.project = original
window.clips_by_id = {}
window.settings = SimpleNamespace(local_model_parallelism=1)
window._on_shot_type_progress = Mock(); window._on_shot_type_error = Mock()
window._on_frame_analysis_op_finished = Mock(); window._mark_dirty = Mock()
window._on_shot_type_ready = lambda *args: MainWindow._on_shot_type_ready(window, *args)
with patch.object(ShotTypeWorker, 'start', lambda _: None):
    MainWindow._launch_frame_analysis_worker(window, 'shots',
        [AnalysisTarget.from_frame(original.frames[0])])
replacement = Project.new(); replacement.add_frames([Frame(id='same', file_path=image)])
window.project = replacement
worker = window._frame_shot_worker
if hasattr(worker, 'outcome_ready'):
    from core.operations.shots import ShotTypeOutcome
    worker.outcome_ready.emit(ShotTypeOutcome('same', 'succeeded', 'wide', .9, target_type='frame'))
else:
    worker.shot_type_ready.emit('same', 'wide', .9)
assert replacement.frames[0].shot_type is None, 'late result changed replacement project'
assert original.frames[0].shot_type is None
worker.analysis_completed.emit()
worker.finished.emit()
window._on_frame_analysis_op_finished.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_frame_shot_completion_and_owner_guards(tmp_path):
    code = r"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from core.operations.shots import ShotTypeTask, ShotTypeOutcome
from models.frame import Frame
from ui.workers.shot_type_delivery import ShotTypeDelivery
from ui.workers.gui_tool_reply import GuiToolReply
from ui.workers.gui_tool_mailbox import GuiToolMailbox
import sys, threading, time
app = QCoreApplication([])
image = Path(sys.argv[1]) / 'frame.png'; image.write_bytes(b'image')
class Window(QObject): pass
class Worker(QThread):
    outcome_ready = Signal(object)
    analysis_completed = Signal()
    def is_cancelled(self): return self.cancelled
    cancelled = False
for mode in ['current', 'cancelled', 'session', 'worker', 'frame_run', 'reply', 'path']:
    window = Window(); window.project = Project.new()
    window.project.add_frames([Frame(id='frame', file_path=image)])
    window._frame_analysis_ops = ['shots']
    mailbox = GuiToolMailbox()
    window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=mailbox.is_pending)
    window._dispatch_gui_reply = GuiToolReply.capture(window, 'shots', mailbox.begin('shots'))
    worker = Worker(); worker.tasks = (ShotTypeTask('frame', image, target_type='frame'),)
    window._frame_shot_worker = worker
    window._on_shot_type_error = Mock(); window._on_shot_type_ready = Mock()
    completed = Mock()
    delivery = ShotTypeDelivery(window, worker, worker_attribute='_frame_shot_worker', on_complete=completed)
    if mode == 'cancelled': worker.cancelled = True
    elif mode == 'session': window.project.session.session_id = 'replacement'
    elif mode == 'worker': window._frame_shot_worker = object()
    elif mode == 'frame_run': window._frame_analysis_ops = ['colors']
    elif mode == 'reply': mailbox.wait(0)
    elif mode == 'path': window.project.path = image.parent / 'other.json'
    worker.outcome_ready.emit(ShotTypeOutcome('frame', 'succeeded', 'wide', .9, target_type='frame'))
    worker.analysis_completed.emit()
    completed.assert_not_called()
    worker.finished.emit(); worker.finished.emit()
    if mode == 'current':
        assert window.project.frames[0].shot_type == 'wide'
        completed.assert_called_once()
    else:
        assert window.project.frames[0].shot_type is None
        completed.assert_not_called()
# Actual queued delivery runs on the owner and waits for native termination.
window = Window(); window.project = Project.new()
window.project.add_frames([Frame(id='frame', file_path=image)])
window._on_shot_type_error = Mock(); window._on_shot_type_ready = Mock()
release = threading.Event(); owner = threading.get_ident(); calls = []
class LiveWorker(Worker):
    def run(self):
        self.outcome_ready.emit(ShotTypeOutcome('frame', 'succeeded', 'wide', .9, target_type='frame'))
        self.analysis_completed.emit()
        assert release.wait(5)
worker = LiveWorker(); worker.tasks = (ShotTypeTask('frame', image, target_type='frame'),)
window._frame_shot_worker = worker
delivery = ShotTypeDelivery(window, worker, worker_attribute='_frame_shot_worker',
    on_complete=lambda: calls.append(threading.get_ident()))
worker.start()
try:
    deadline = time.monotonic() + 5
    while window.project.frames[0].shot_type is None and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.001)
    assert window.project.frames[0].shot_type == 'wide'
    assert not calls and worker.isRunning()
finally:
    release.set(); assert worker.wait(5000)
app.processEvents()
assert calls == [owner]
assert not window._active_shot_workers
from ui.main_window import MainWindow
closing = Window(); closing._check_unsaved_changes = lambda: True
closing.status_bar = Mock()
active = Mock(); active.isRunning.return_value = True
closing._active_shot_workers = {active}
event = Mock()
MainWindow.closeEvent(closing, event)
active.cancel.assert_called_once()
event.ignore.assert_called_once(); event.accept.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
