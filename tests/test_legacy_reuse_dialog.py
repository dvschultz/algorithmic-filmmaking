"""Real queued legacy decisions remain detached until owner publication."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("operation", ["colors", "embeddings", "brightness", "volume", "classify", "detect_objects", "boundary_embeddings", "gaze", "shots", "extract_text", "describe", "cinematography", "transcribe"])
def test_dialog_publication_and_cancellation(tmp_path, operation):
    code = r'''
import sys
import threading
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget
from core.analysis_records import AnalysisFingerprints
from core.analysis_model_identity import DINOV2_TAG
from tests.test_description_operations import project_with_thumbnails
from ui.dialogs.legacy_reuse_dialog import LegacyReuseDialog

app = QApplication([])
owner_thread = threading.get_ident()
original = AnalysisFingerprints.identity
for mode in ('current', 'trim', 'project', 'cancel'):
    owner = QWidget()
    owner.project = project_with_thumbnails(Path(sys.argv[1]), 1)
    project = owner.project
    clip = project.clips[0]
    clip.dominant_colors = [(1, 2, 3)]
    clip.embedding = [0.1] * 768
    clip.embedding_model = DINOV2_TAG
    clip.average_brightness = clip.rms_volume = 0.0
    clip.object_labels = clip.detected_objects = []
    clip.person_count = 0
    clip.extracted_texts = []
    clip.transcript = []
    clip.description = "A person walking"
    clip.shot_type = "wide shot"
    if sys.argv[2] == "cinematography":
        from models.cinematography import CinematographyAnalysis
        clip.cinematography = CinematographyAnalysis()
        clip.shot_type = "medium"
    clip.gaze_yaw = clip.gaze_pitch = 0.0
    clip.gaze_category = "at_camera"
    clip.first_frame_embedding = [0.2] * 768
    clip.last_frame_embedding = [0.3] * 768
    operation = sys.argv[2]
    dialog = LegacyReuseDialog(owner, [clip.id])
    dialog.operation.setCurrentIndex(dialog.operation.findData(operation))
    entered, release = threading.Event(), threading.Event()
    def identity(*args, **kwargs):
        assert threading.get_ident() != owner_thread
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    recorded = []
    original_record = project.record_analysis
    def record(*args, **kwargs):
        recorded.append(threading.get_ident())
        return original_record(*args, **kwargs)
    loop = QEventLoop()
    with patch.object(AnalysisFingerprints, 'identity', identity), patch.object(project, 'record_analysis', record):
        dialog.start_reuse()
        assert entered.wait(5)
        assert operation not in clip.analysis_records
        if mode == 'trim': clip.start_frame += 1
        if mode == 'project': owner.project = object()
        if mode == 'cancel':
            dialog.reject()
            assert dialog.closing
            assert dialog.worker.isRunning()
        dialog.worker.finished.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        release.set()
        loop.exec()
        assert dialog.worker.wait(5000)
        app.processEvents()
    if mode == 'current':
        assert recorded == [owner_thread], recorded
        assert clip.analysis_records[operation].legacy_reuse
        assert clip.analysis_records[operation].provenance == 'unknown'
        assert 'Accepted 1 of 1' in dialog.status.text()
    else:
        assert not recorded
        assert operation not in clip.analysis_records
    dialog.close()
    owner.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), operation],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
