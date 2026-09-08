"""Real Qt lifetime and shared publication checks for combined clip analysis."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["completed", "aborted_early", "aborted_late", "aborted_combined"])
def test_retired_controller_can_release_its_last_window_reference(mode):
    code = r'''
import gc, sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication, QEvent, QObject
from core.project import Project
from core.settings import Settings
from models.clip import Clip, Source
from ui.workers.clip_analysis import ClipAnalysisController

app = QCoreApplication([])
mode = sys.argv[1]
destroyed = []
def retire():
    window = QObject()
    window.destroyed.connect(lambda: destroyed.append('window'))
    window.project = Project.new()
    window.settings = Settings()
    source = Source(file_path=Path('source.mp4'), fps=30)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    window.project.add_source(source)
    window.project.add_clips([clip])
    if mode == 'completed':
        clip.shot_type = 'wide'
        controller = ClipAnalysisController(window, [clip], ['shots'])
        controller.start()
        assert controller.finished
    else:
        from ui.workers.standalone_analysis import start_standalone_analysis

        noop = lambda *args: None
        replace = lambda *args: setattr(window, '_clip_analysis_controller', object())
        window._ensure_analysis_operation_available = lambda *args, **kwargs: True
        window._on_clip_analysis_progress = noop
        window._on_clip_analysis_status = noop
        window._switch_to_tab = noop
        window._gui_state = SimpleNamespace(set_processing=noop)
        window.analyze_tab = SimpleNamespace(
            add_clips=replace if mode == 'aborted_early' else noop,
            set_analyzing=noop,
        )
        window.progress_bar = SimpleNamespace(
            setVisible=noop, setRange=replace if mode in ('aborted_late', 'aborted_combined') else noop,
        )
        if mode == 'aborted_combined':
            from ui.main_window import MainWindow
            window._custom_query_text = None
            window._filter_available_analysis_operations = lambda operations: operations
            window._on_clip_analysis_finished = noop
            assert not MainWindow._run_analysis_pipeline(window, [clip], ['colors'])
        else:
            assert not start_standalone_analysis(window, [clip.id], 'colors')
        window.analyze_tab.add_clips = noop
        window.progress_bar.setRange = noop
retire()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
gc.collect()
app.processEvents()
assert destroyed == ['window'], destroyed
'''
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, mode],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_closed_controller_has_no_unowned_qt_callbacks():
    # This order reproduced a native crash when a controller's singleShot
    # callback outlived its window and the next widget processed Qt events.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_analysis_pipeline.py",
            "tests/test_analyze_tab_auto_include.py",
            "tests/test_boundary_analysis_workflow.py",
            "tests/test_chip_group.py",
            "tests/test_embeddings_dispatch.py",
            "tests/test_exquisite_ocr_ownership.py",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "native_exit_pending",
        "cancel",
        "project",
        "clip",
        "source",
        "media",
        "range",
        "alias",
        "duplicate",
        "partial",
        "force",
        "reset",
        "start_error",
        "reply",
        "session",
        "path",
        "observer_cancel",
    ],
)
def test_clip_pipeline_owns_native_completion(tmp_path, mode):
    code = r"""
import sys, time, threading
from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.settings import Settings
from core.operations.contracts import ColorOutcome, ColorResult
from core.operations.description import DescriptionOutcome
from models.clip import Clip, Source
from ui.workers.clip_analysis import ClipAnalysisController
from ui.workers.color_worker import ColorAnalysisWorker
from ui.workers.description_worker import DescriptionWorker
app = QCoreApplication([])
directory, mode = Path(sys.argv[1]), sys.argv[2]
window = QObject(); window.project = project = Project.new()
window.settings = Settings(); window.settings.description_model_tier = 'local'
if mode == 'reply':
    from types import SimpleNamespace
    from ui.workers.gui_tool_reply import GuiToolReply
    from ui.workers.gui_tool_mailbox import GuiToolMailbox
    mailbox = GuiToolMailbox()
    window._chat_worker = SimpleNamespace(_stop_requested=False,
        is_gui_tool_pending=mailbox.is_pending, set_gui_tool_result=mailbox.submit)
    window._dispatch_gui_reply = GuiToolReply.capture(window, 'analyze_all_live', mailbox.begin('analyze_all_live'))
source = Source(file_path=directory/'video.mp4', fps=30)
source.file_path.write_bytes(b'video')
clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
clip.thumbnail_path = directory/'thumb.jpg'; clip.thumbnail_path.write_bytes(b'image')
project.add_source(source); project.add_clips([clip])
changes=[]
project.add_observer(lambda event,data: changes.append([c.id for c in data]) if event=='clips_updated' else None)
if mode == 'force': clip.dominant_colors = [(9,9,9)]
project.save = Mock(side_effect=AssertionError('implicit save'))
entered, release = threading.Event(), threading.Event()
started = []; reports = []
def color_run(self):
    started.append('colors')
    self.result = ColorResult(self.request.request_id, (ColorOutcome(clip.id, 'succeeded', ((1,2,3),)),))
    self.analysis_completed.emit()
    entered.set(); assert release.wait(10)
def describe_run(self):
    started.append('describe')
    self.result = () if mode == 'partial' else (DescriptionOutcome(clip.id, 'succeeded', 'A scene', 'test'),)
    self.description_completed.emit()
controller = ClipAnalysisController(window, [clip], ['colors','describe'], force_rerun=mode=='force')
if mode == 'observer_cancel':
    project.add_observer(lambda event,data: controller.cancel() if event=='clips_updated' else None)
controller.completed.connect(lambda owner, result: reports.append(result))
def pump_until(predicate):
    deadline = time.monotonic()+5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.002)
    assert predicate(), 'timed out'
with patch.object(ColorAnalysisWorker, 'run', color_run), patch.object(DescriptionWorker, 'run', describe_run):
    if mode == 'start_error':
        with patch.object(ColorAnalysisWorker, 'start', side_effect=RuntimeError('start failed')):
            controller.start()
        pump_until(lambda: bool(reports))
        assert reports[0]['failed'] == [clip.id]
        assert not controller.workers and not window._active_clip_analyses
        sys.exit(0)
    controller.start(); worker = controller.workers['colors']
    assert entered.wait(5)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.002)
        assert started == ['colors'] and not reports
        assert clip.dominant_colors == ([(9,9,9)] if mode=='force' else None)
        if mode == 'cancel': controller.cancel()
        if mode == 'project': window.project = Project.new()
        if mode == 'reply': assert mailbox.wait(0) is None
        if mode == 'session': project.session.session_id = 'replaced'
        if mode == 'path': project.path = directory/'other.json'
        if mode == 'clip': project.clips_by_id[clip.id] = Clip(id=clip.id, source_id=source.id, start_frame=0, end_frame=30)
        if mode == 'source': project.sources_by_id[source.id] = Source(id=source.id, file_path=source.file_path, fps=30)
        if mode == 'media': source.file_path.write_bytes(b'changed video')
        if mode == 'range': clip.end_frame = 20
        if mode == 'alias': window.color_worker = object()
        if mode == 'duplicate':
            worker.finished.emit()
            for _ in range(5): app.processEvents()
            assert started == ['colors'] and not reports
        if mode == 'reset':
            controller.cancel()
            assert window.color_worker is None
            assert controller in window._active_clip_analyses
            assert worker.isRunning()
    finally:
        release.set(); assert worker.wait(5000)
    if mode == 'native_exit_pending':
        # QThread.finished can be delivered before the native thread is fully
        # joined. The receiver must recheck rather than lose completion.
        running_checks = iter([True])
        native_is_running = worker.isRunning
        worker.isRunning = lambda: next(running_checks, native_is_running())
    pump_until(lambda: bool(reports))
    assert not controller.workers and not window._active_clip_analyses
    project.save.assert_not_called()
    if mode in ('success','duplicate','force','native_exit_pending'):
        assert started == ['colors','describe']
        assert clip.dominant_colors == [(1,2,3)] and clip.description == 'A scene'
        assert reports[0]['succeeded'] == [clip.id]
        assert changes == [[clip.id], [clip.id]], 'publication should refresh each operation once'
    elif mode == 'partial':
        assert clip.dominant_colors == [(1,2,3)] and clip.description is None
        assert reports[0]['failed'] == [clip.id]
    else:
        assert clip.dominant_colors == ([(1,2,3)] if mode=='observer_cancel' else None)
        assert clip.description is None
        assert reports[0]['failed'] == [clip.id]
    assert len(reports) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
