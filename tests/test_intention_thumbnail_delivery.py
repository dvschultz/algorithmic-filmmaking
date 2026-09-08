"""Intention thumbnail cancellation and native completion integration."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "mode",
    [
        "cancel",
        "reentrant",
        "success",
        "duplicate",
        "plan",
        "project",
        "session",
        "path",
        "clip",
        "source",
        "media",
        "range",
        "reply",
        "alias",
        "restart",
        "saved",
        "start_error",
        "stale_dispatch",
        "failed",
        "partial",
        "missing",
        "ready_error",
        "sync_error",
        "observer_cancel",
        "observer_replace",
        "close",
        "reset",
        "prior",
        "linear",
        "cancel_other",
    ],
)
def test_intention_thumbnail_ownership(tmp_path, mode):
    code = """
import sys,threading,time
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from models.clip import Clip,Source
from ui.main_window import MainWindow
from ui.workers.thumbnail_worker import ThumbnailWorker
app=QCoreApplication([]); directory=Path(sys.argv[1]); mode=sys.argv[2]
window=QObject(); window.project=project=Project.new()
source=Source(file_path=directory/'video.mp4'); source.file_path.write_bytes(b'video')
clip=Clip(source_id=source.id,start_frame=0,end_frame=30)
clips=[clip]
if mode=='partial': clips.append(Clip(source_id=source.id,start_frame=30,end_frame=60))
elif mode=='linear': clips.extend(Clip(source_id=source.id,start_frame=i*30,end_frame=(i+1)*30) for i in range(1,32))
project.add_source(source); project.add_clips(clips)
if mode=='saved':
    assert project.save(directory/'project.json'); before=project.path.read_bytes()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
def begin():
    workflow.start('shuffle',[source.file_path],[])
    workflow.on_detection_completed(source,clips)
begin(); original_plan=workflow.plan
window.settings=SimpleNamespace(thumbnail_cache_dir=directory)
window.thumbnail_worker=None
window._dispatch_gui_reply=SimpleNamespace(is_current=lambda _:True)
window._on_thumbnail_ready=Mock(); window.analyze_tab=Mock(); window.cut_tab=Mock()
window.sources_by_id=project.sources_by_id; window.clips_by_id=project.clips_by_id
window._sync_cut_tab_clip_browser=Mock()
window._sync_intention_workflow_ui=lambda *a,**k:MainWindow._sync_intention_workflow_ui(window,*a,**k)
if hasattr(MainWindow,'_on_intention_thumbnails_finished'):
    # Keep the regression harness usable against the pre-migration implementation.
    window._thumbnail_generation=0; window._thumbnails_finished_handled=False
    window._on_intention_thumbnails_finished=lambda *a:MainWindow._on_intention_thumbnails_finished(window,*a)
if mode=='reentrant':
    def restart(*a): workflow.cancel(); begin()
    window.analyze_tab.set_lookups.side_effect=restart
elif mode=='sync_error': window.analyze_tab.set_lookups.side_effect=RuntimeError('sync failed')
elif mode=='ready_error': window._on_thumbnail_ready.side_effect=RuntimeError('view failed')
elif mode=='stale_dispatch': project.replace_source_clips(source.id,[replace(clip)])
observer_threads=[]
def observer(event,*args):
    if event=='clips_updated':
        observer_threads.append(threading.get_ident())
        if mode=='observer_cancel': workflow.cancel()
        elif mode=='observer_replace': project.replace_source_clips(source.id,[replace(clip,thumbnail_path=None)])
project.add_observer(observer)
entered=threading.Event(); release=threading.Event(); original=ThumbnailWorker.run
owner_thread=threading.get_ident(); owner_stat_calls=[]; original_stat=Path.stat
def counted_stat(path,*args,**kwargs):
    if threading.get_ident()==owner_thread: owner_stat_calls.append(path)
    return original_stat(path,*args,**kwargs)
def run(self):
    original(self)
    if mode=='missing': self.result=()
    entered.set(); assert release.wait(10)
def generate(self,**kwargs):
    if mode=='failed' or (mode=='partial' and kwargs['start_seconds']>0): raise RuntimeError('extraction failed')
    path=kwargs['output_path']; path.write_bytes(b'thumbnail'); return path
with patch.object(ThumbnailWorker,'run',run), patch('core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail',generate), (patch.object(ThumbnailWorker,'start',side_effect=RuntimeError('start failed')) if mode=='start_error' else nullcontext()), (patch('pathlib.Path.stat',counted_stat) if mode=='linear' else nullcontext()):
    MainWindow._start_intention_thumbnails(window)
    if mode in ('start_error','stale_dispatch'):
        assert workflow.state==WorkflowState.ERROR
        assert not getattr(window,'_active_thumbnail_workers',set())
        assert not getattr(window,'_active_intention_thumbnails',set())
        assert window.thumbnail_worker is None
        sys.exit(0)
    worker=window.thumbnail_worker; assert entered.wait(5)
    replacement=None
    try:
        assert workflow.state==WorkflowState.THUMBNAILS
        if mode=='plan': workflow.cancel(); begin()
        elif mode=='project': window.project=Project.new()
        elif mode=='session': project.clear()
        elif mode=='path': project.path=directory/'changed.json'
        elif mode=='clip': project.replace_source_clips(source.id,[replace(clip)])
        elif mode=='source': project.sources_by_id[source.id]=replace(source)
        elif mode=='media': source.file_path.write_bytes(b'changed media')
        elif mode=='range': clip.start_frame=3
        elif mode=='reply': window._dispatch_gui_reply.is_current=lambda _:False
        elif mode=='alias': window.thumbnail_worker=Mock()
        elif mode=='cancel_other':
            other=Mock(); other.isRunning.return_value=True; window.thumbnail_worker=other
            MainWindow._on_intention_import_cancelled(window)
            other.cancel.assert_not_called()
        elif mode=='prior':
            clip.thumbnail_path=directory/'manual.jpg'; clip.thumbnail_path.write_bytes(b'manual')
        elif mode=='duplicate':
            worker.finished.emit(); app.processEvents()
            MainWindow._start_intention_thumbnails(window)
            assert window.thumbnail_worker is worker
        elif mode=='close':
            window._check_unsaved_changes=lambda:True; window.status_bar=Mock(); event=Mock()
            MainWindow.closeEvent(window,event); event.ignore.assert_called_once()
        elif mode=='reset':
            window._source_import_queue=Mock(); window._cancel_download_workers=Mock()
            window._stop_worker_safely=Mock()
            with patch('ui.main_window.stop_chat_workers'): MainWindow._stop_all_workers(window)
            assert not any(call.args[0] is worker for call in window._stop_worker_safely.call_args_list)
        for _ in range(10): app.processEvents(); time.sleep(.005)
        if mode=='cancel':
            workflow.cancel()
            assert worker.is_cancelled(), 'workflow cancellation did not reach thumbnail worker'
        elif mode=='restart':
            workflow.cancel(); begin(); MainWindow._start_intention_thumbnails(window)
            replacement=window.thumbnail_worker
            assert replacement is not worker
        assert worker in window._active_thumbnail_workers, 'running worker was released'
    finally:
        release.set(); assert worker.wait(5000)
        if replacement is not None: assert replacement.wait(5000)
        for _ in range(15): app.processEvents(); time.sleep(.005)
    assert not window._active_thumbnail_workers
    assert not window._active_intention_thumbnails
    assert all(t==threading.get_ident() for t in observer_threads)
    if mode=='reentrant':
        assert workflow.plan is not original_plan
        assert workflow.state==WorkflowState.THUMBNAILS, 'old completion advanced replacement plan'
        window.cut_tab.set_source.assert_not_called()
        window._sync_cut_tab_clip_browser.assert_not_called()
    elif mode in ('success','duplicate','restart','saved','prior','linear'):
        assert workflow.state==WorkflowState.BUILDING, workflow.state
        assert clip.thumbnail_path.is_file()
        if mode=='prior': assert clip.thumbnail_path.name=='manual.jpg'
    elif mode=='plan': assert workflow.state==WorkflowState.THUMBNAILS
    elif mode in ('clip','source','media','range','failed','partial','missing','ready_error','sync_error','observer_replace'):
        assert workflow.state==WorkflowState.ERROR, workflow.state
        if mode=='partial': assert clip.thumbnail_path.is_file() and clips[1].thumbnail_path is None
    else: assert workflow.state==WorkflowState.CANCELLED, workflow.state
    if mode in ('clip','source','media','range','project','session','path','reply','alias','plan','close','reset','cancel_other'):
        assert clip.thumbnail_path is None, 'stale result applied'
        window._on_thumbnail_ready.assert_not_called()
    if mode in ('alias','cancel_other'): assert window.thumbnail_worker is not None
    else: assert window.thumbnail_worker is None
    if mode=='observer_replace':
        assert project.clips_by_id[clip.id] is not clip
        assert project.clips_by_id[clip.id].thumbnail_path is None
        window._on_thumbnail_ready.assert_not_called()
    if mode=='linear':
        assert len(owner_stat_calls)<=16*len(clips), ('thumbnail delivery rescans the batch for each result',len(owner_stat_calls))
    if mode=='saved':
        assert project.path.read_bytes()==before, 'implicit project save'
        assert project.save() and project.path.read_bytes()!=before
        project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
