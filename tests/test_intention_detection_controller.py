"""Real Qt worker lifetime coverage for intention detection."""

import os
import subprocess
import sys

import pytest


def test_intention_detection_waits_for_native_completion(tmp_path):
    code = """
import threading, time, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
from models.clip import Source, Clip
from ui.main_window import MainWindow
from ui.workers.detection_worker import DetectionWorker
app=QCoreApplication([])
path=Path(sys.argv[1])/'video.mp4'; path.write_bytes(b'video')
class Window(QObject):
    sources_by_id=property(lambda self:self.project.sources_by_id)
Window._start_intention_detection=MainWindow._start_intention_detection
window=Window(); window.project=Project.new()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
workflow.start('sequential',[path],[])
window.settings=SimpleNamespace(default_sensitivity=3)
window.detection_worker=None; window._detection_generation=0
window.collect_tab=Mock(); window.cut_tab=Mock(); window._on_detection_job_started=Mock()
entered=threading.Event(); release=threading.Event(); workers=[]
original=DetectionWorker.run
def run(self):
    workers.append(self)
    original(self)
    entered.set(); assert release.wait(10)
def detect(self,path,progress):
    source=Source(file_path=path,fps=30)
    return source,[Clip(source_id=source.id,start_frame=0,end_frame=30)]
with patch.object(DetectionWorker,'run',run), patch('core.scene_detect.SceneDetector.detect_scenes_with_progress',detect):
    window._start_intention_detection()
    assert entered.wait(5)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.01)
        assert workflow.state==WorkflowState.DETECTING, 'advanced before native completion'
        assert not window.project.clips, 'published before native completion'
    finally:
        release.set()
        for worker in workers: assert worker.wait(5000)
        app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_standalone_completion_cannot_clear_intention_worker(tmp_path):
    code = """
import threading, time, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator
from models.clip import Source, Clip
from ui.main_window import MainWindow
from ui.workers.detection_worker import DetectionWorker
app=QCoreApplication([])
path=Path(sys.argv[1])/'video.mp4'; path.write_bytes(b'video')
class Window(QObject):
    _start_detection=MainWindow._start_detection
    _start_intention_detection=MainWindow._start_intention_detection
    _on_detection_worker_finished=MainWindow._on_detection_worker_finished
window=Window(); window.project=Project.new()
window.current_source=Source(file_path=path,fps=30)
window.project.add_source(window.current_source)
window.settings=SimpleNamespace(default_sensitivity=3,min_scene_length_seconds=.5)
window.detection_worker=None; window._detection_generation=0
for name in ('_on_detection_job_started','_on_detection_progress','_on_guarded_detection_finished',
             '_on_guarded_detection_error','progress_bar','_gui_state','collect_tab','cut_tab'):
    setattr(window,name,Mock())
entered=threading.Event(); release=threading.Event()
original=DetectionWorker.run
def run(self):
    original(self)
    entered.set(); assert release.wait(10)
def detect(self,path,progress):
    source=Source(file_path=path,fps=30)
    return source,[Clip(source_id=source.id,start_frame=0,end_frame=30)]
with patch('core.scene_detect.SceneDetector.detect_scenes_with_progress',detect):
    window._start_detection()
    standalone=window.detection_worker
    assert standalone.wait(5000)
    # Its native finished notification is queued but has not been delivered.
    window.intention_workflow=workflow=IntentionWorkflowCoordinator()
    workflow.start('sequential',[path],[])
    with patch.object(DetectionWorker,'run',run):
        window._start_intention_detection()
        intention=window.detection_worker
        assert window._active_detection_guard is not standalone.guard
        assert entered.wait(5)
        try:
            for _ in range(10): app.processEvents(); time.sleep(.01)
            assert window.detection_worker is intention, 'old completion cleared new worker'
        finally:
            release.set(); assert intention.wait(5000)
            for _ in range(10): app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
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
        "failed_first",
        "all_failed",
        "cancel",
        "plan",
        "project",
        "session",
        "source",
        "clip",
        "media",
        "path",
        "reply",
        "restart",
        "duplicate",
        "observer_cancel",
        "observer_replace",
        "ui_error",
        "reset",
        "close",
    ],
)
def test_detection_run_ownership(tmp_path, mode):
    code = """
import threading, time, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
from models.clip import Source, Clip
from ui.workers.intention_detection import IntentionDetectionController
from ui.workers.detection_worker import DetectionWorker
app=QCoreApplication([])
directory=Path(sys.argv[1]); mode=sys.argv[2]
paths=[directory/'first.mp4', directory/'second.mp4']
for path in paths: path.write_bytes(b'video')
window=QObject(); window.project=project=Project.new()
source=Source(file_path=paths[0],fps=30)
old=Clip(source_id=source.id,start_frame=0,end_frame=30)
project.add_source(source); project.add_clips([old])
assert project.save(directory/'project.json')
saved=(directory/'project.json').read_bytes()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
workflow.start('sequential',paths,[])
window.settings=SimpleNamespace(default_sensitivity=3)
window.detection_worker=None
window.collect_tab=Mock(); window.cut_tab=Mock()
window._dispatch_gui_reply=SimpleNamespace(is_current=lambda _:True)
entered=threading.Event(); release=threading.Event(); workers=[]; computations=[]
threads=[]; owner=threading.get_ident()
project.add_observer(lambda *args:threads.append(threading.get_ident()))
original=DetectionWorker.run
def run(self):
    workers.append(self)
    original(self)
    entered.set(); assert release.wait(10)
def detect(self,path,progress):
    assert threading.get_ident()!=owner
    assert self.config.threshold==3, 'settings changed after submission'
    computations.append(path)
    if mode=='all_failed' or (mode=='failed_first' and path==paths[0]): raise RuntimeError('decoder failed')
    result=Source(file_path=path,fps=24,duration_seconds=10)
    return result,[Clip(source_id=result.id,start_frame=0,end_frame=240)]
with patch.object(DetectionWorker,'run',run), patch('core.scene_detect.SceneDetector.detect_scenes_with_progress',detect):
    controller=IntentionDetectionController(window); controller.start()
    assert entered.wait(5)
    for _ in range(10): app.processEvents(); time.sleep(.005)
    assert workflow.state==WorkflowState.DETECTING
    assert project.clips==[old] and source.fps==30
    assert len(computations)==1
    window.settings.default_sensitivity=9
    if mode=='cancel': workflow.cancel()
    elif mode=='plan': workflow.cancel(); workflow.start('sequential',paths,[])
    elif mode=='project': window.project=Project.new()
    elif mode=='session': project.clear()
    elif mode=='source':
        data=source.to_dict(); project.remove_source(source.id)
        project.add_source(Source.from_dict(data)); project.add_clips([old])
    elif mode=='clip': project.replace_source_clips(source.id,[Clip.from_dict(old.to_dict())])
    elif mode=='media': paths[0].write_bytes(b'changed')
    elif mode=='path': project.path=directory/'moved.json'
    elif mode=='reply': window._dispatch_gui_reply.is_current=lambda _:False
    elif mode=='restart':
        workflow.cancel(); workflow.start('sequential',paths,[])
        window.settings.default_sensitivity=3
        replacement=IntentionDetectionController(window); replacement.start()
    elif mode=='duplicate':
        controller.worker.finished.emit()
        assert controller.worker.isRunning() and project.clips==[old]
    elif mode=='observer_cancel':
        project.add_observer(lambda event,data:workflow.cancel() if event=='clips_added' else None)
    elif mode=='observer_replace':
        replaced=[]
        def replace(event,data):
            if event=='clips_added' and not replaced:
                replaced.append(True)
                project.replace_source_clips(source.id,[Clip.from_dict(data[0].to_dict())])
        project.add_observer(replace)
    elif mode=='ui_error': window.cut_tab.set_source.side_effect=RuntimeError('view failed')
    elif mode=='reset':
        from ui.main_window import MainWindow
        window._source_import_queue=Mock(); window._cancel_download_workers=Mock()
        window._stop_worker_safely=Mock()
        with patch('ui.main_window.stop_chat_workers'):
            MainWindow._stop_all_workers(window)
        forced=any(call.args[0] is controller.worker for call in window._stop_worker_safely.call_args_list)
    elif mode=='close':
        from ui.main_window import MainWindow
        window._check_unsaved_changes=lambda:True
        window.status_bar=Mock(); event=Mock()
        MainWindow.closeEvent(window,event)
        event.ignore.assert_called_once()
    assert controller in window._active_intention_detections
    release.set()
    deadline=time.monotonic()+15
    while window._active_intention_detections and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.005)
    assert not window._active_intention_detections, 'workers not settled'
    assert window.detection_worker is None
    assert all(t==owner for t in threads)
    assert (directory/'project.json').read_bytes()==saved, 'implicit save'
    if mode in ('success','duplicate','restart'):
        assert workflow.state==WorkflowState.THUMBNAILS
        assert len(workflow.get_all_clips())==2
        assert len(project.sources)==2
        assert project.sources_by_id[source.id] is source and source.fps==24
        assert len(project.clips_by_source[source.id])==1
    elif mode in ('failed_first','source','clip','media'):
        assert workflow.state==WorkflowState.THUMBNAILS
        assert len(workflow.get_all_clips())==1
        assert len(project.clips)==2 and source.fps==30
    elif mode=='all_failed':
        assert workflow.state==WorkflowState.ERROR and project.clips==[old]
    elif mode=='observer_cancel':
        assert workflow.state==WorkflowState.CANCELLED
        assert len(computations)==1 and not workflow.get_all_clips()
    elif mode=='observer_replace':
        assert workflow.state==WorkflowState.THUMBNAILS
        assert len(workflow.get_all_clips())==1
        assert all(project.clips_by_id[c.id] is c for c in workflow.get_all_clips())
    elif mode=='ui_error':
        assert workflow.state==WorkflowState.ERROR
        assert len(computations)==1
    else:
        assert not workflow.get_all_clips()
        assert source.fps==30
        assert len(computations)==1
    if mode=='reset': assert not forced, 'retained worker routed through force-stop helper'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
