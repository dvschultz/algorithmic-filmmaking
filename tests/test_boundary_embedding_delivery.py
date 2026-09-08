"""Real queued boundary delivery must remain on its launching project."""

import os
import subprocess
import sys


def test_real_queued_boundary_delivery_is_owned():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from tests.test_description_operations import project_with_thumbnails
from ui.workers.boundary_embedding_worker import BoundaryEmbeddingWorker
from ui.workers.embedding_delivery import BoundaryEmbeddingDelivery
app = QCoreApplication([])
owners = []
with TemporaryDirectory() as directory:
    root = Path(directory)
    with patch('core.settings.load_settings', lambda: SimpleNamespace(cache_dir=root)), patch('core.analysis.embeddings.extract_boundary_embeddings', return_value=([.1]*768, [.2]*768)), patch('core.analysis.embeddings.unload_model'):
        for mode in ('current', 'save_as', 'edit', 'cancel', 'project', 'pipeline'):
            window = QObject()
            project = project_with_thumbnails(root, 1)
            project.save(root / (mode + '.json'))
            window.project = project
            window._analysis_run = None
            worker = BoundaryEmbeddingWorker(project.clips, project=project)
            window._boundary_embeddings_worker = worker
            window._on_boundary_embeddings_error = Mock()
            window._on_embedding_ready = Mock()
            delivery = BoundaryEmbeddingDelivery(window, worker, pipeline=True)
            owners.append((window, worker, delivery))
            worker.start(); assert worker.wait(10000)
            assert worker.job_status == 'completed'
            assert project.clips[0].first_frame_embedding is None
            if mode == 'save_as': project.save(root / 'copy.json')
            if mode == 'edit': project.clips[0].last_frame_embedding = [3.] * 768
            if mode == 'cancel': worker.cancel()
            if mode == 'project': window.project = project_with_thumbnails(root, 1)
            if mode == 'pipeline': window._analysis_run = object()
            app.processEvents()
            if mode == 'current':
                assert project.clips[0].first_frame_embedding == [.1]*768
                assert project.clips[0].last_frame_embedding == [.2]*768
                assert len(project.metadata.job_results) == 1
                window._on_embedding_ready.assert_called_once()
                assert project.save()
                from core.jobs.store import JobStore
                store = JobStore(root / 'jobs.db')
                try:
                    assert all(store.get_result(rid)['committed'] for rid in project.metadata.job_results)
                finally:
                    store.close()
            else:
                assert project.clips[0].first_frame_embedding is None
                assert not project.metadata.job_results
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
