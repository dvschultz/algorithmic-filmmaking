"""Exercise real queued Qt delivery against changing launch context."""

import os
import subprocess
import sys


def test_worker_delivers_verified_empty_and_failed_records(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.settings import Settings
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.object_detection_worker import ObjectDetectionWorker
    from ui.workers.object_detection_delivery import ObjectDetectionDelivery

    app = QApplication.instance() or QApplication([])
    window = QObject()
    assert window.thread() == app.thread()
    window.project = project_with_thumbnails(tmp_path, 1)
    assert window.project.save(tmp_path / "project.json")
    window._on_object_detection_error = Mock()
    monkeypatch.setattr("core.settings.load_settings", lambda: Settings(cache_dir=tmp_path / "cache"))
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.detection.detect_objects", provider)
    deliveries = []
    try:
        for attempt in ("compute", "reuse", "fail"):
            if attempt == "fail":
                provider.side_effect = RuntimeError("provider failed")
            worker = ObjectDetectionWorker(window.project.clips, project=window.project, skip_existing=attempt != "fail")
            window.detection_worker_yolo = worker
            deliveries.append(ObjectDetectionDelivery(window, worker))
            worker.run()
            assert window.project.clips[0].analysis_records["detect_objects"].state == ("failed" if attempt == "fail" else "succeeded")
        assert provider.call_count == 2
        assert window.project.clips[0].detected_objects == []
        assert window.project.clips[0].person_count == 0
        window._on_object_detection_error.assert_not_called()
    finally:
        window.project.close_writer()


def test_queued_object_detection_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.object_detection import ObjectDetectionTask, ObjectDetectionOptions
from models.frame import Frame
from ui.workers.object_detection_delivery import ObjectDetectionDelivery
app = QCoreApplication([])
class Worker(QThread):
    objects_ready = Signal(str, list, int)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False; self.options = ObjectDetectionOptions()
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.objects_ready.emit('c-0', [{'label': 'person', 'confidence': .9, 'bbox': [0, 0, 1, 1]}], 1)
        self.objects_ready.emit('c-0', [{'label': 'person', 'confidence': .9, 'bbox': [0, 0, 1, 1]}], 1)
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
            task = ObjectDetectionTask(clip.id, clip.thumbnail_path,
                target_type=target_type)
            worker = Worker(task)
            if mode in ('receipt', 'payload', 'save_as'):
                from dataclasses import asdict
                import json
                from core.jobs.gui_results import GuiResultReceipt
                from core.operations.object_detection import ObjectDetectionOutcome, DetectedObject
                window.project.save(Path(directory) / 'project.json')
                result = ObjectDetectionOutcome(clip.id, 'succeeded', (DetectedObject('person', .9, (0, 0, 1, 1)),), 1)
                payload = json.dumps(asdict(result))
                if mode == 'payload': payload = payload.replace('person', 'car')
                worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={clip.id: GuiResultReceipt('a'*64, 'b'*64, payload)})
                if mode == 'save_as': window.project.save(Path(directory) / 'copy.json')
            window.detection_worker_yolo = worker
            window._on_object_detection_error = Mock()
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = ObjectDetectionDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.detection_worker_yolo = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.person_count = 99
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode in ('current', 'receipt'):
                assert target.person_count == 1
                assert target.detected_objects[0]['label'] == 'person'
                assert bool(window.project.metadata.job_results) == (mode == 'receipt')
                if target_type == 'frame': assert clip.person_count is None
            else:
                assert target.person_count == (99 if mode == 'edit' else None)
            if mode in ('edit', 'payload', 'save_as'): window._on_object_detection_error.assert_called_once()
            else: window._on_object_detection_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
