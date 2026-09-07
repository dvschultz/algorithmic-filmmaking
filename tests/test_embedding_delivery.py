"""Real Qt queued embedding publication respects the launch context."""

import os
import subprocess
import sys


def test_queued_embeddings_are_applied_once_to_current_targets():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.embeddings import EmbeddingTask, EmbeddingOutcome
from ui.workers.embedding_delivery import EmbeddingDelivery
from tests.test_description_operations import project_with_thumbnails
app = QCoreApplication([])
class Worker(QThread):
    outcome_ready = Signal(object)
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.outcome_ready.emit(self.outcome)
window = QObject()
with TemporaryDirectory() as root:
    for mode in ('current', 'worker', 'project', 'session', 'cancel', 'edit', 'image'):
        project = project_with_thumbnails(Path(root), 1)
        clip = project.clips[0]
        worker = Worker(); worker.cancelled = False
        worker.tasks = (EmbeddingTask(clip.id, clip.thumbnail_path),)
        worker.outcome = EmbeddingOutcome.from_vector(clip.id, [0.1] * 768)
        window.project = project; window._embeddings_worker = worker
        window._on_embeddings_error = Mock(); window._on_embedding_ready = Mock()
        delivery = EmbeddingDelivery(window, worker)
        if mode == 'worker': window._embeddings_worker = object()
        if mode == 'project': window.project = project_with_thumbnails(Path(root), 1)
        if mode == 'session': project.clear()
        if mode == 'cancel': worker.cancelled = True
        if mode == 'edit': clip.embedding = [0.2] * 768
        if mode == 'image': clip.thumbnail_path.write_bytes(b'replaced')
        previous = clip.embedding
        worker.start(); assert worker.wait(5000); app.processEvents()
        assert clip.embedding == ([0.1] * 768 if mode == 'current' else previous)
        if mode == 'current': window._on_embedding_ready.assert_called_once_with(clip.id)
        else: window._on_embedding_ready.assert_not_called()
        if mode in ('edit', 'image'): window._on_embeddings_error.assert_called_once()
        else: window._on_embeddings_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
