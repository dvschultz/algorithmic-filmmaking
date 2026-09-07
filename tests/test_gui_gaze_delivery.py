"""Queued scalar and empty gaze observations require their exact receipts."""

import os
import subprocess
import sys


def test_queued_gaze_receipts_and_empty_results():
    code = r"""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.gaze import GazeTask, GazeOutcome
from core.jobs.gui_results import GuiResultReceipt
from core.jobs.commits import canonical_json
from ui.workers.gaze_delivery import GazeDelivery
from tests.test_description_operations import project_with_thumbnails
app=QCoreApplication([])
class Worker(QThread):
    gaze_ready=Signal(str,float,float,str)
    observation_ready=Signal(object)
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.observation_ready.emit(self.outcome)
        self.observation_ready.emit(self.outcome)
        if self.outcome.category is not None:
            self.gaze_ready.emit(self.outcome.clip_id,self.outcome.yaw,self.outcome.pitch,self.outcome.category)
window=QObject()
with TemporaryDirectory() as root:
    for empty in (False,True):
      for mode in ('current','tampered','save_as','cancel','worker','session','edit'):
        project=project_with_thumbnails(Path(root),1)
        clip=project.clips[0];source=project.sources[0]
        clip.gaze_yaw=15.;clip.gaze_pitch=2.;clip.gaze_category='looking_right'
        assert project.save(Path(root)/'project.json')
        worker=Worker();worker.cancelled=False
        worker.tasks=(GazeTask(clip.id,clip.source_id,source.file_path,clip.start_frame,clip.end_frame,source.fps),)
        worker.outcome=GazeOutcome(clip.id,'succeeded',code='no_gaze_detected') if empty else GazeOutcome(clip.id,'succeeded',2.,1.,'at_camera')
        receipt=GuiResultReceipt('a'*64,'b'*64,canonical_json(asdict(worker.outcome)))
        worker.cache=SimpleNamespace(path=project.path.resolve(),results={clip.id:receipt})
        window.project=project;window._gaze_worker=worker
        window._on_gaze_error=Mock();window._on_gaze_ready=Mock()
        delivery=GazeDelivery(window,worker)
        if mode=='tampered': worker.cache.results[clip.id]=GuiResultReceipt('a'*64,'b'*64,'{}')
        if mode=='save_as': assert project.save(Path(root)/'copy.json')
        if mode=='cancel': worker.cancelled=True
        if mode=='worker': window._gaze_worker=object()
        if mode=='session': project.clear()
        if mode=='edit': clip.gaze_yaw=99.
        before=(clip.gaze_yaw,clip.gaze_pitch,clip.gaze_category)
        worker.start();assert worker.wait(5000);app.processEvents()
        expected=(None,None,None) if empty else (2.,1.,'at_camera')
        assert (clip.gaze_yaw,clip.gaze_pitch,clip.gaze_category)==(expected if mode=='current' else before)
        assert bool(project.metadata.job_results)==(mode=='current')
        if mode=='current': window._on_gaze_ready.assert_called_once_with(clip.id,*expected)
        else: window._on_gaze_ready.assert_not_called()
        if mode in ('tampered','save_as','edit'): window._on_gaze_error.assert_called_once()
        else: window._on_gaze_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
