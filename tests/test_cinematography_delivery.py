"""Exercise real queued Qt delivery against changing launch context."""

import os
import subprocess
import sys


def test_queued_cinematography_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.cinematography import CinematographyTask
from models.cinematography import CinematographyAnalysis
from models.frame import Frame
from ui.workers.cinematography_delivery import CinematographyDelivery
app = QCoreApplication([])
class Worker(QThread):
    clip_completed = Signal(str, object)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.clip_completed.emit('c-0', CinematographyAnalysis(shot_size='CU'))
        self.clip_completed.emit('c-0', CinematographyAnalysis(shot_size='CU'))
window = QObject()
with TemporaryDirectory() as directory:
    for target_type in ('clip', 'frame'):
        for mode in ('current', 'worker', 'session', 'project', 'edit', 'cancel', 'pipeline', 'reply', 'receipt', 'payload', 'save_as'):
            window._analysis_run = None
            window.project = project_with_thumbnails(Path(directory), 1)
            clip = window.project.clips[0]
            source = window.project.sources[0]
            if target_type == 'frame':
                target = Frame(id=clip.id, file_path=clip.thumbnail_path)
                window.project.add_frames([target])
            else: target = clip
            task = CinematographyTask(clip.id, clip.thumbnail_path,
                source.file_path if source.file_path.exists() else None,
                clip.start_frame, clip.end_frame, source.fps, target_type=target_type)
            worker = Worker(task)
            if mode in ('receipt', 'payload', 'save_as'):
                from dataclasses import asdict
                import json
                from core.jobs.gui_results import GuiResultReceipt
                from core.operations.cinematography import CinematographyOutcome
                window.project.save(Path(directory) / 'project.json')
                result = CinematographyOutcome(clip.id, 'succeeded', json.dumps(CinematographyAnalysis(shot_size='CU').to_dict(), sort_keys=True, allow_nan=False))
                payload = json.dumps(asdict(result))
                if mode == 'payload': payload = payload.replace('CU', 'MS')
                worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={clip.id: GuiResultReceipt('a'*64, 'b'*64, payload)})
                if mode == 'save_as': window.project.save(Path(directory) / 'copy.json')
            window.cinematography_worker = worker
            window._on_cinematography_clip_ready = Mock()
            window._on_cinematography_error = Mock()
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = CinematographyDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.cinematography_worker = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.shot_type = 'wide'
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode in ('current', 'receipt'):
                assert target.shot_type == 'close-up'
                assert target.cinematography.shot_size == 'CU'
                assert bool(window.project.metadata.job_results) == (mode == 'receipt')
                if target_type == 'clip': window._on_cinematography_clip_ready.assert_called_once()
                else:
                    assert clip.cinematography is None
                    window._on_cinematography_clip_ready.assert_not_called()
            else:
                window._on_cinematography_clip_ready.assert_not_called()
                assert target.cinematography is None
            if mode in ('edit', 'payload', 'save_as'): window._on_cinematography_error.assert_called_once()
            else: window._on_cinematography_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_queued_verified_reuse_and_failure_delivery():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.cinematography import CinematographyApplication, CinematographyOptions, cinematography_task, run_cinematography
from models.cinematography import CinematographyAnalysis
from tests.test_description_operations import project_with_thumbnails
from ui.workers.cinematography_delivery import CinematographyDelivery
app = QCoreApplication([])
class Worker(QThread):
    outcome_ready = Signal(object)
    def is_cancelled(self): return False
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.outcome_ready.emit(self.outcome)
with TemporaryDirectory() as directory:
    for mode in ('reuse', 'failure', 'tampered'):
        window = QObject()
        project = window.project = project_with_thumbnails(Path(directory), 1)
        options = CinematographyOptions('cloud', 'frame', 'model', 'local')
        task = cinematography_task(project.clips[0], project.sources[0])
        with patch('core.analysis.cinematography.analyze_cinematography', return_value=CinematographyAnalysis(shot_size='CU', analysis_model='model')):
            first = run_cinematography((task,), options)[0]
        assert CinematographyApplication(project, (task,), options).apply(project, first)
        assert project.save(Path(directory) / 'project.json')
        task = cinematography_task(project.clips[0], project.sources[0], skip_existing=mode != 'failure')
        if mode == 'failure':
            with patch('core.analysis.cinematography.analyze_cinematography', side_effect=ValueError('Invalid answer')):
                outcome = run_cinematography((task,), options)[0]
        else: outcome = run_cinematography((task,), options)[0]
        worker = Worker()
        worker.tasks = (task,)
        worker.options = options
        worker.outcome = outcome
        worker.cache = SimpleNamespace(path=project.path.resolve(), results={}, transient_outcomes={task.clip_id: asdict(outcome)})
        if mode == 'tampered': worker.cache.transient_outcomes[task.clip_id]['record_json'] = '{}'
        window.cinematography_worker = worker
        window._on_cinematography_clip_ready = Mock()
        window._on_cinematography_error = Mock()
        delivery = CinematographyDelivery(window, worker)
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        assert project.clips[0].cinematography.shot_size == 'CU'
        assert not project.metadata.job_results
        if mode == 'reuse': window._on_cinematography_clip_ready.assert_called_once()
        else: window._on_cinematography_clip_ready.assert_not_called()
        if mode == 'failure': assert project.clips[0].analysis_records['cinematography'].state == 'failed'
        if mode == 'tampered': window._on_cinematography_error.assert_called_once()
        else: window._on_cinematography_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
