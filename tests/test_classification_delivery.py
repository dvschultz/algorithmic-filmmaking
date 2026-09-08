"""Exercise real queued Qt delivery against changing launch context."""

import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize("kind", ["clip", "frame"])
def test_worker_delivers_records_for_reuse_and_failure(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.analysis_target import AnalysisTarget
    from core.settings import Settings
    from models.frame import Frame
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.classification_worker import ClassificationWorker
    from ui.workers.classification_delivery import ClassificationDelivery

    app = QApplication.instance() or QApplication([])
    window = QObject()
    assert window.thread() == app.thread()
    window.project = project_with_thumbnails(tmp_path, 1)
    target = window.project.clips[0]
    if kind == "frame":
        target = Frame(id="frame", file_path=target.thumbnail_path)
        window.project.add_frames([target])
    assert window.project.save(tmp_path / "project.json")
    window._on_classification_error = Mock()
    window.analyze_tab = SimpleNamespace()
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: Settings(cache_dir=tmp_path / "cache")
    )
    provider = Mock(return_value=[("person", 0.9)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    deliveries = []
    try:
        for attempt in ("compute", "reuse", "fail"):
            if attempt == "fail":
                provider.side_effect = RuntimeError("provider failed")
            worker = ClassificationWorker(
                window.project.clips,
                project=window.project,
                analysis_targets=[AnalysisTarget.from_frame(target)]
                if kind == "frame"
                else None,
                skip_existing=attempt != "fail",
            )
            window.classification_worker = worker
            deliveries.append(ClassificationDelivery(window, worker))
            worker.run()
            assert target.analysis_records["classify"].state == (
                "failed" if attempt == "fail" else "succeeded"
            )
            assert (
                worker.result[0].status
                == {"compute": "succeeded", "reuse": "skipped", "fail": "failed"}[
                    attempt
                ]
            )
            if attempt == "reuse":
                assert worker.result[0].labels == ()  # No invented confidence scores.
                assert worker.result[0].label_names == ["person"]
        assert provider.call_count == 2
        assert target.object_labels == ["person"]
        window._on_classification_error.assert_not_called()
    finally:
        window.project.close_writer()


def test_queued_classification_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.classification import ClassificationTask
from models.frame import Frame
from ui.workers.classification_delivery import ClassificationDelivery
app = QCoreApplication([])
class Worker(QThread):
    labels_ready = Signal(str, list)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.labels_ready.emit('c-0', [('person', .9)])
        self.labels_ready.emit('c-0', [('person', .9)])
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
            task = ClassificationTask(clip.id, clip.thumbnail_path,
                target_type=target_type)
            worker = Worker(task)
            if mode in ('receipt', 'payload', 'save_as'):
                from dataclasses import asdict
                import json
                from core.jobs.gui_results import GuiResultReceipt
                from core.operations.classification import ClassificationOutcome
                window.project.save(Path(directory) / 'project.json')
                result = ClassificationOutcome(clip.id, 'succeeded', (('person', .9),))
                payload = json.dumps(asdict(result))
                if mode == 'payload': payload = payload.replace('person', 'car')
                worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={clip.id: GuiResultReceipt('a'*64, 'b'*64, payload)})
                if mode == 'save_as': window.project.save(Path(directory) / 'copy.json')
            window.classification_worker = worker
            window._on_classification_error = Mock()
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = ClassificationDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.classification_worker = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.object_labels = ['user edit']
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode in ('current', 'receipt'):
                assert target.object_labels == ['person']
                assert bool(window.project.metadata.job_results) == (mode == 'receipt')
                if target_type == 'frame': assert clip.object_labels is None
            else:
                assert target.object_labels == (['user edit'] if mode == 'edit' else None)
            if mode in ('edit', 'payload', 'save_as'): window._on_classification_error.assert_called_once()
            else: window._on_classification_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
