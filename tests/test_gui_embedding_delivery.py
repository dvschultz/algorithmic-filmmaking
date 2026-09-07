"""Real queued embedding publication validates saved-project receipts."""

import os
import subprocess
import sys


def test_queued_embedding_receipts_are_required():
    code = r"""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.embeddings import EmbeddingTask, EmbeddingOutcome
from core.jobs.gui_results import GuiResultReceipt
from core.jobs.commits import canonical_json
from ui.workers.embedding_delivery import EmbeddingDelivery
from tests.test_description_operations import project_with_thumbnails
app=QCoreApplication([])
class Worker(QThread):
    outcome_ready=Signal(object)
    def is_cancelled(self): return False
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.outcome_ready.emit(self.outcome)
window=QObject()
with TemporaryDirectory() as root:
    for mode in ('current','tampered','save_as'):
        project=project_with_thumbnails(Path(root),1)
        assert project.save(Path(root)/'project.json')
        clip=project.clips[0]
        worker=Worker()
        worker.tasks=(EmbeddingTask(clip.id,clip.thumbnail_path),)
        worker.outcome=EmbeddingOutcome.from_vector(clip.id,[.1]*768)
        receipt=GuiResultReceipt('a'*64,'b'*64,canonical_json(asdict(worker.outcome)))
        worker.cache=SimpleNamespace(path=project.path.resolve(),results={clip.id:receipt})
        window.project=project;window._embeddings_worker=worker
        window._on_embeddings_error=Mock();window._on_embedding_ready=Mock()
        delivery=EmbeddingDelivery(window,worker)
        if mode=='tampered': worker.cache.results[clip.id]=GuiResultReceipt('a'*64,'b'*64,'{}')
        if mode=='save_as': assert project.save(Path(root)/'copy.json')
        worker.start();assert worker.wait(5000);app.processEvents()
        assert clip.embedding==([.1]*768 if mode=='current' else None)
        assert bool(project.metadata.job_results)==(mode=='current')
        if mode=='current':
            window._on_embedding_ready.assert_called_once_with(clip.id)
            window._on_embeddings_error.assert_not_called()
        else:
            window._on_embedding_ready.assert_not_called()
            window._on_embeddings_error.assert_called_once()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
