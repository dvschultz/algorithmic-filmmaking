"""Intention analysis must hold completion until the native worker exits."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("algorithm", ["color", "storyteller"])
def test_analysis_waits_for_native_completion(tmp_path, algorithm):
    code = """
import threading,time,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from models.clip import Clip,Source
from ui.main_window import MainWindow
from ui.workers.color_worker import ColorAnalysisWorker
from ui.workers.description_worker import DescriptionWorker
app=QCoreApplication([]); directory=Path(sys.argv[1]); algorithm=sys.argv[2]
window=QObject(); window.project=project=Project.new()
source=Source(file_path=directory/'video.mp4'); source.file_path.write_bytes(b'video')
clip=Clip(source_id=source.id,start_frame=0,end_frame=30)
project.add_source(source); project.add_clips([clip])
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
workflow.start(algorithm,[source.file_path],[])
workflow.on_detection_completed(source,[clip]); workflow.on_thumbnails_finished()
workflow.on_analysis_finished=Mock(wraps=workflow.on_analysis_finished)
window.settings=SimpleNamespace(color_analysis_parallelism=1,description_parallelism=1,description_model_tier='local')
window._ensure_analysis_operation_available=lambda *a,**k:True
for name in ('_reset_analysis_run_error','_reset_description_run_errors','_on_color_result','_on_color_job_started','_on_color_error','_on_description_error','_on_description_ready'):
    setattr(window,name,Mock())
window.status_bar=Mock(); window._color_run_error=None; window._description_run_error=None
entered=threading.Event(); release=threading.Event()
def run(self):
    if algorithm=='color': self.analysis_completed.emit()
    else: self.description_completed.emit()
    entered.set(); assert release.wait(10)
cls=ColorAnalysisWorker if algorithm=='color' else DescriptionWorker
with patch.object(cls,'run',run):
    MainWindow._start_intention_analysis(window)
    worker=window.color_worker if algorithm=='color' else window.description_worker
    assert entered.wait(5)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.01)
        assert workflow.on_analysis_finished.call_count==0, 'phase advanced before native completion'
        assert workflow.state==WorkflowState.ANALYZING
    finally:
        release.set(); assert worker.wait(5000)
        for _ in range(10): app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), algorithm],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("algorithm", ["color", "shot_type", "storyteller"])
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "cancel",
        "plan",
        "project",
        "session",
        "path",
        "clip",
        "source",
        "media",
        "alias",
        "duplicate",
        "observer_cancel",
        "empty",
        "gate_clip",
        "saved",
        "receipt",
        "failed",
        "ui_error",
        "reset",
        "close",
        "start_error",
    ],
)
def test_real_analysis_worker_ownership(tmp_path, algorithm, mode):
    code = """
import threading,time,sys
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from core.operations.contracts import ColorOutcome,ColorResult
from core.operations.shots import ShotTypeOutcome
from core.operations.description import DescriptionOutcome
from models.clip import Clip,Source
from ui.main_window import MainWindow
from ui.workers.color_worker import ColorAnalysisWorker
from ui.workers.shot_type_worker import ShotTypeWorker
from ui.workers.description_worker import DescriptionWorker
app=QCoreApplication([]); directory=Path(sys.argv[1]); algorithm,mode=sys.argv[2:]
window=QObject(); window.project=project=Project.new()
source=Source(file_path=directory/'video.mp4'); source.file_path.write_bytes(b'video')
image=directory/'image.jpg'; image.write_bytes(b'image')
clips=[Clip(source_id=source.id,start_frame=i*30,end_frame=(i+1)*30,thumbnail_path=image) for i in range(2)]
project.add_source(source); project.add_clips(clips)
if mode in ('saved','receipt'):
    assert project.save(directory/'project.json')
    before=(directory/'project.json').read_bytes()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
workflow.start(algorithm,[source.file_path],[])
workflow.on_detection_completed(source,clips); workflow.on_thumbnails_finished()
field={'color':'dominant_colors','shot_type':'shot_type','storyteller':'description'}[algorithm]
window.settings=SimpleNamespace(color_analysis_parallelism=1,local_model_parallelism=1,description_parallelism=1,description_model_tier='local')
def gate(*a,**k):
    if mode=='gate_clip': project.replace_source_clips(source.id,[replace(c) for c in clips])
    return True
window._ensure_analysis_operation_available=Mock(side_effect=gate)
window._update_window_title=Mock(); window._on_shot_type_ready=Mock(); window._on_description_ready=Mock()
window.status_bar=Mock(); window._on_color_job_started=Mock()
entered=threading.Event(); release=threading.Event(); thread_ids=[]
def observer(event,*args):
    if event=='clips_updated':
        thread_ids.append(threading.get_ident())
        if mode=='observer_cancel': workflow.cancel()
project.add_observer(observer)
cls={'color':ColorAnalysisWorker,'shot_type':ShotTypeWorker,'storyteller':DescriptionWorker}[algorithm]
attribute={'color':'color_worker','shot_type':'shot_type_worker','storyteller':'description_worker'}[algorithm]
original=cls.run
def run(self):
    original(self); entered.set(); assert release.wait(10)
def colors(request,**kwargs):
    outcomes=tuple(ColorOutcome(t.target_id,'failed' if mode=='failed' else 'succeeded',((10,20,30),)) for t in request.targets)
    for i,o in enumerate(outcomes): kwargs['progress_callback'](i+1,len(outcomes),o)
    return ColorResult(request.request_id,outcomes)
def analyze(tasks,options,**kwargs):
    status='failed' if mode=='failed' else 'succeeded'
    outcomes=tuple(ShotTypeOutcome(t.clip_id,status,'wide',.9) if algorithm=='shot_type' else DescriptionOutcome(t.clip_id,status,'A person walking.','test-model') for t in tasks)
    for i,o in enumerate(outcomes):
        kwargs['on_outcome'](o)
        if 'progress' in kwargs: kwargs['progress'](i+1,len(outcomes))
    return outcomes
if mode=='empty':
    for c in clips: setattr(c,field,[] if algorithm=='color' else '')
with patch.object(cls,'run',run), patch('ui.workers.color_worker.compute_colors',colors), patch('ui.workers.shot_type_worker.run_shot_types',analyze), patch('ui.workers.description_worker.run_description',analyze), patch('core.jobs.gui_shots.run_shot_types',analyze), patch('core.jobs.gui_description.run_description',analyze), patch('core.analysis.description.is_model_loaded',return_value=True), (patch.object(cls,'start',side_effect=RuntimeError('start failed')) if mode=='start_error' else nullcontext()):
    MainWindow._start_intention_analysis(window)
    if mode=='start_error':
        assert workflow.state==WorkflowState.ERROR
        assert not window._active_intention_analyses and getattr(window,attribute,None) is None
        sys.exit(0)
    if mode=='empty':
        assert workflow.state==WorkflowState.BUILDING
        assert not getattr(window,'_active_intention_analyses',set())
        window._ensure_analysis_operation_available.assert_not_called()
        sys.exit(0)
    if mode=='gate_clip' and algorithm!='color':
        assert workflow.state==WorkflowState.ERROR
        assert not getattr(window,'_active_intention_analyses',set())
        sys.exit(0)
    worker=getattr(window,attribute); controller=window._intention_analysis
    assert entered.wait(8)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.005)
        assert workflow.state==WorkflowState.ANALYZING
        assert all(getattr(c,field) is None for c in clips)
        if mode=='cancel': workflow.cancel()
        elif mode=='plan': workflow.cancel(); workflow.start(algorithm,[source.file_path],[])
        elif mode=='project': window.project=Project.new()
        elif mode=='session': project.clear()
        elif mode=='path': project.path=directory/'changed.json'
        elif mode=='clip': project.replace_source_clips(source.id,[replace(c) for c in clips])
        elif mode=='source': project.sources_by_id[source.id]=replace(source)
        elif mode=='media': source.file_path.write_bytes(b'changed media')
        elif mode=='alias': setattr(window,attribute,Mock())
        elif mode=='receipt' and algorithm!='color':
            worker.result=tuple(replace(o,confidence=.1) if algorithm=='shot_type' else replace(o,description='Tampered') for o in worker.result)
        elif mode=='ui_error':
            for name in ('_update_window_title','_on_shot_type_ready','_on_description_ready'):
                getattr(window,name).side_effect=RuntimeError('view failed')
        elif mode=='reset':
            window._source_import_queue=Mock(); window._cancel_download_workers=Mock()
            window._stop_worker_safely=Mock()
            with patch('ui.main_window.stop_chat_workers'): MainWindow._stop_all_workers(window)
            assert not any(call.args[0] is worker for call in window._stop_worker_safely.call_args_list)
            assert controller in window._active_intention_analyses
        elif mode=='close':
            window._check_unsaved_changes=lambda:True; event=Mock()
            MainWindow.closeEvent(window,event)
            event.ignore.assert_called_once()
            assert controller in window._active_intention_analyses
        elif mode=='duplicate':
            worker.finished.emit(); app.processEvents()
            MainWindow._start_intention_analysis(window)
            assert window._intention_analysis is controller and controller.worker is worker
    finally:
        release.set(); assert worker.wait(5000)
        for _ in range(20): app.processEvents(); time.sleep(.005)
    assert not window._active_intention_analyses
    assert all(t==threading.get_ident() for t in thread_ids)
    assert not (directory/'changed.json').exists()
    if mode in ('success','duplicate','saved') or (mode in ('gate_clip','receipt') and algorithm=='color'):
        assert workflow.state==WorkflowState.BUILDING, (workflow.state,controller.errors)
        assert all(getattr(c,field) is not None for c in clips)
    elif mode=='observer_cancel':
        assert workflow.state==WorkflowState.CANCELLED
        if algorithm!='color': assert getattr(clips[1],field) is None
    elif mode=='ui_error':
        assert workflow.state==WorkflowState.ERROR
    else:
        assert all(getattr(c,field) is None for c in clips), mode
        assert workflow.state!=WorkflowState.BUILDING
    if mode=='alias': assert getattr(window,attribute) is not None
    else: assert getattr(window,attribute,None) is None
    if mode in ('saved','receipt'):
        assert (directory/'project.json').read_bytes()==before, 'GUI analysis saved implicitly'
        if mode=='saved':
            if algorithm!='color': assert len(project.metadata.job_results)==2
            assert project.save()
            assert (directory/'project.json').read_bytes()!=before
        project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), algorithm, mode],
        capture_output=True,
        text=True,
        timeout=35,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
