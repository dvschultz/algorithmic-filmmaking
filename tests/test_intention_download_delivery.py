"""Real worker and coordinator coverage for intention download delivery."""

import os
import subprocess
import sys

import pytest


def test_download_phase_waits_for_native_completion(tmp_path):
    code = """
import threading,time,sys
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from core.downloader import DownloadResult
from core.jobs.store import JobStore
from ui.main_window import MainWindow
from ui.workers.download_workers import URLBulkDownloadWorker
app=QCoreApplication([]); directory=Path(sys.argv[1])
class Window(QObject):
    _start_intention_downloads=MainWindow._start_intention_downloads
    _validate_download_directory=MainWindow._validate_download_directory
window=Window(); window.project=Project.new()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
urls=['https://youtube.com/one']; workflow.start('shuffle',[],urls)
window.settings=SimpleNamespace(download_dir=directory)
window._download_deliveries={}; window._active_download_workers=set()
window._ensure_video_download_available=lambda:True
window.collect_tab=Mock()
entered=threading.Event(); release=threading.Event()
original=URLBulkDownloadWorker.run
def run(self):
    original(self); entered.set(); assert release.wait(10)
def download(request,**kwargs):
    path=directory/'video.mp4'; path.write_bytes(b'video')
    return DownloadResult(success=True,file_path=path,duration=3)
with patch.object(URLBulkDownloadWorker,'run',run), patch('core.operations.downloads.run_download',download), patch('ui.workers.download_workers.open_download_store',lambda:JobStore(directory/'jobs.db')):
    window._start_intention_downloads(urls)
    worker=window.url_bulk_download_worker
    assert entered.wait(5)
    try:
        for _ in range(10): app.processEvents(); time.sleep(.01)
        assert workflow.state==WorkflowState.DOWNLOADING, 'advanced before native download completion'
        assert len(window.project.sources)==1
    finally:
        release.set(); assert worker.wait(5000)
        for _ in range(10): app.processEvents()
    assert workflow.state==WorkflowState.DETECTING
    assert not window._active_download_workers and not window._active_intention_downloads
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
    ["denied", "plan", "project", "directory_cancel", "directory_plan", "start_error"],
)
def test_download_dispatch_rechecks_gates(tmp_path, mode):
    code = """
import sys
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from ui.main_window import MainWindow
from ui.workers.download_workers import URLBulkDownloadWorker
app=QCoreApplication([]); directory=Path(sys.argv[1]); mode=sys.argv[2]
window=QObject(); window.project=Project.new()
window.intention_workflow=workflow=IntentionWorkflowCoordinator()
urls=['https://youtube.com/first']; workflow.start('shuffle',[],urls)
window.settings=SimpleNamespace(download_dir=directory)
window._download_deliveries={}; window._active_download_workers=set(); window.url_bulk_download_worker=None
window.collect_tab=Mock()
def gate():
    if mode=='plan': workflow.cancel(); workflow.start('shuffle',[],urls)
    elif mode=='project': window.project=Project.new()
    return mode!='denied'
def validate(path):
    if mode=='directory_plan': workflow.cancel(); workflow.start('shuffle',[],urls)
    return None if mode=='directory_cancel' else path
window._ensure_video_download_available=gate; window._validate_download_directory=validate
with patch.object(URLBulkDownloadWorker,'start',side_effect=RuntimeError('start failed')) as start:
    MainWindow._start_intention_downloads(window,urls)
    assert start.call_count==(1 if mode=='start_error' else 0)
assert not window._active_download_workers
assert not getattr(window,'_active_intention_downloads',set())
assert window.url_bulk_download_worker is None
if mode in ('denied','start_error'): assert workflow.state==WorkflowState.ERROR
elif mode=='directory_cancel': assert workflow.state==WorkflowState.CANCELLED
else: assert workflow.state==WorkflowState.DOWNLOADING
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
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
        "partial",
        "all_failed",
        "cancel",
        "cancel_late",
        "plan",
        "project",
        "session",
        "path",
        "reply",
        "media",
        "source",
        "late_remove",
        "alias",
        "duplicate",
        "restart",
        "observer_cancel",
        "ui_error",
        "reset",
        "close",
    ],
)
def test_download_run_ownership(tmp_path, mode):
    code = """
import threading,time,sys
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,QObject
from core.project import Project
from core.intention_workflow import IntentionWorkflowCoordinator,WorkflowState
from core.downloader import DownloadResult
from core.jobs.store import JobStore
from models.clip import Source
from ui.workers.download_workers import URLBulkDownloadWorker
from ui.workers.intention_run import IntentionRun
from ui.workers.intention_download import IntentionDownloadDelivery
app=QCoreApplication([]); directory=Path(sys.argv[1]); mode=sys.argv[2]
urls=['https://youtube.com/first','https://youtube.com/second']
first=directory/'first.mp4'; first.write_bytes(b'video')
window=QObject(); window.project=project=Project.new()
source=Source(file_path=first,fps=24,color_profile='grayscale'); project.add_source(source)
assert project.save(directory/'project.json'); saved=project.path.read_bytes()
window.intention_workflow=workflow=IntentionWorkflowCoordinator(); workflow.start('shuffle',[],urls)
progress_events=[]; workflow.progress_updated.connect(progress_events.append)
window._download_deliveries={}; window._active_download_workers=set()
window._dispatch_gui_reply=SimpleNamespace(is_current=lambda _:True)
window.collect_tab=Mock()
entered=threading.Event(); release=threading.Event(); owner=threading.get_ident(); threads=[]
project.add_observer(lambda *args:threads.append(threading.get_ident()))
original=URLBulkDownloadWorker.run
def run(self):
    original(self); entered.set(); assert release.wait(10)
def download(request,**kwargs):
    assert threading.get_ident()!=owner
    if mode=='all_failed' or (mode=='partial' and request.url==urls[1]):
        return DownloadResult(success=False,error='provider denied')
    path=first if mode=='alias' else directory/(request.url.rsplit('/',1)[-1]+'.mp4')
    if not path.exists(): path.write_bytes(b'video')
    return DownloadResult(success=True,file_path=path,duration=3)
def launch():
    worker=URLBulkDownloadWorker(urls,directory)
    window.url_bulk_download_worker=worker
    delivery=IntentionDownloadDelivery(window,worker,IntentionRun.capture(window))
    worker.start(); return worker,delivery
with patch.object(URLBulkDownloadWorker,'run',run), patch('core.operations.downloads.run_download',download), patch('ui.workers.download_workers.open_download_store',lambda:JobStore(directory/'jobs.db')):
    worker,delivery=launch(); assert entered.wait(5)
    if mode=='cancel': workflow.cancel()
    elif mode=='plan': workflow.cancel(); workflow.start('shuffle',[],urls)
    elif mode=='project': window.project=Project.new()
    elif mode=='session': project.clear()
    elif mode=='path': project.path=directory/'other.json'
    elif mode=='reply': window._dispatch_gui_reply.is_current=lambda _:False
    elif mode=='media': first.write_bytes(b'changed')
    elif mode=='source': project.remove_source(source.id)
    elif mode=='observer_cancel':
        project.add_observer(lambda event,data:workflow.cancel() if event=='source_added' else None)
    elif mode=='ui_error': window.collect_tab.add_source.side_effect=RuntimeError('view failed')
    elif mode=='reset':
        from ui.main_window import MainWindow
        window._source_import_queue=Mock(); window._stop_worker_safely=Mock()
        window._cancel_download_workers=MainWindow._cancel_download_workers.__get__(window)
        with patch('ui.main_window.stop_chat_workers'): MainWindow._stop_all_workers(window)
    elif mode=='close':
        from ui.main_window import MainWindow
        window._check_unsaved_changes=lambda:True; window.status_bar=Mock(); event=Mock()
        MainWindow.closeEvent(window,event); event.ignore.assert_called_once()
    for _ in range(10): app.processEvents(); time.sleep(.005)
    if mode in ('success','partial','all_failed','alias','duplicate','restart','cancel_late','late_remove'):
        assert workflow.state==WorkflowState.DOWNLOADING, 'premature phase advance'
        assert any('Processed' in event.message for event in progress_events), 'download progress was lost'
    if mode=='cancel_late': workflow.cancel()
    elif mode=='late_remove': project.remove_source(source.id)
    elif mode=='duplicate': worker.finished.emit(); assert worker in window._active_download_workers
    elif mode=='restart': workflow.cancel(); workflow.start('shuffle',[],urls); replacement,_=launch()
    assert delivery in window._active_intention_downloads
    release.set()
    deadline=time.monotonic()+15
    while window._active_download_workers and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.005)
    assert not window._active_download_workers and not window._active_intention_downloads
    assert window.url_bulk_download_worker is None
    assert all(thread==owner for thread in threads)
    assert (directory/'project.json').read_bytes()==saved, 'implicit project save'
    assert source.fps==24 and source.color_profile=='grayscale'
    if mode in ('success','duplicate','restart'):
        assert workflow.state==WorkflowState.DETECTING
        assert workflow.get_sources_to_detect()==[first,directory/'second.mp4']
        assert len(project.sources)==2 and project.sources_by_id[source.id] is source
    elif mode in ('partial','media','source','late_remove'):
        assert workflow.state==WorkflowState.DETECTING
        assert len(workflow.get_sources_to_detect())==1
        assert len(workflow._sources_failed)==1
    elif mode=='alias':
        assert workflow.state==WorkflowState.DETECTING
        assert workflow.get_sources_to_detect()==[first] and len(project.sources)==1
    elif mode=='all_failed':
        assert workflow.state==WorkflowState.ERROR and len(workflow._sources_failed)==2
    elif mode=='ui_error': assert workflow.state==WorkflowState.ERROR
    else:
        assert not workflow.get_sources_to_detect()
        assert workflow.state in (WorkflowState.CANCELLED,WorkflowState.DOWNLOADING)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
