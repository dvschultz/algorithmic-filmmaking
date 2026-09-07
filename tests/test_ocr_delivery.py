"""Real queued OCR delivery cannot cross worker, project, or target boundaries."""

import os
import subprocess
import sys


def test_queued_ocr_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from threading import get_ident
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.ocr import OcrTask, OcrText, OcrOutcome
from models.clip import ExtractedText
from models.frame import Frame
from ui.workers.ocr_delivery import OcrDelivery
app = QCoreApplication([])
owner = get_ident()
class Worker(QThread):
    outcome_ready = Signal(object)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        task = self.tasks[0]
        result = OcrOutcome(task.clip_id, 'succeeded', task.target_type,
            (OcrText(0, 'SIGN', .9, 'vlm'),))
        self.outcome_ready.emit(result)
        self.outcome_ready.emit(result)
window = QObject()
with TemporaryDirectory() as directory:
    for kind in ('clip', 'frame'):
        for mode in ('current', 'worker', 'session', 'project', 'edit', 'cancel',
                     'pipeline', 'reply', 'range', 'media', 'replaced'):
            window._analysis_run = None
            window.project = project_with_thumbnails(Path(directory), 1)
            clip = window.project.clips[0]
            if kind == 'frame':
                target = Frame(id=clip.id, file_path=clip.thumbnail_path)
                window.project.add_frames([target])
                task = OcrTask(clip.id, target.file_path, 'frame')
            else:
                target = clip
                task = OcrTask.from_clip(clip, window.project.sources[0])
            worker = Worker(task)
            window.text_extraction_worker = worker
            window._on_text_extraction_error = Mock()
            window.analyze_tab = SimpleNamespace(update_clip_extracted_text=Mock(
                side_effect=lambda *a: get_ident() == owner or (_ for _ in ()).throw(AssertionError('wrong thread'))))
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = OcrDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.text_extraction_worker = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.extracted_texts = [ExtractedText(0, 'EDIT', 1, 'vlm')]
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            if mode == 'range':
                if kind == 'clip': target.end_frame += 10
                else: target.frame_number = 99
            if mode == 'media': task.path.write_bytes(b'changed')
            if mode == 'replaced':
                from copy import deepcopy
                if kind == 'clip': window.project.clips_by_id[clip.id] = deepcopy(target)
                else: window.project.frames_by_id[clip.id] = deepcopy(target)
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode == 'current':
                assert target.extracted_texts is not None, (kind, window._on_text_extraction_error.call_args_list, delivery.application.bindings)
                assert target.extracted_texts[0].text == 'SIGN'
                if kind == 'frame': assert clip.extracted_texts is None
                else: window.analyze_tab.update_clip_extracted_text.assert_called_once()
            elif mode == 'edit': assert target.extracted_texts[0].text == 'EDIT'
            else: assert target.extracted_texts is None
            if mode in ('edit', 'range', 'media', 'replaced'):
                window._on_text_extraction_error.assert_called_once()
            else: window._on_text_extraction_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_frame_launcher_completion_is_owned_and_once():
    code = r"""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from ui.main_window import MainWindow
from ui.workers.text_extraction_worker import TextExtractionWorker
app = QCoreApplication([])
for mode in ('current', 'project', 'session', 'worker', 'cancel', 'reply', 'frame_run'):
    window = QObject()
    window.project = Project.new()
    window.settings = SimpleNamespace(text_extraction_method='vlm', text_extraction_vlm_model='test')
    window._on_text_extraction_progress = Mock()
    window._on_text_extraction_error = Mock()
    window._on_frame_analysis_op_finished = Mock()
    window._frame_analysis_ops = ['extract_text']
    window._dispatch_gui_reply = SimpleNamespace(is_current=lambda _: True)
    with patch.object(TextExtractionWorker, 'start', lambda worker: None):
        MainWindow._launch_frame_analysis_worker(window, 'extract_text', [])
    worker = window._frame_text_worker
    if mode == 'project': window.project = Project.new()
    if mode == 'session': window.project.clear()
    if mode == 'worker': window._frame_text_worker = object()
    if mode == 'cancel': worker.cancel()
    if mode == 'reply': window._dispatch_gui_reply.is_current = lambda _: False
    if mode == 'frame_run': window._frame_analysis_ops = ['colors']
    worker.extraction_completed.emit({})
    worker.extraction_completed.emit({})
    if mode == 'current': window._on_frame_analysis_op_finished.assert_called_once_with('extract_text')
    else: window._on_frame_analysis_op_finished.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
