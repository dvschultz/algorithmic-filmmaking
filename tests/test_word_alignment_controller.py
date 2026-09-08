"""The word picker's real queued worker publishes through its project owner."""

import os
import subprocess
import sys

import pytest


def test_controller_records_owned_results_and_rejects_late_changes():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.analysis.alignment import ALIGNMENT_MODEL
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.analysis_record import AnalysisRecord
from tests.test_spine_analyze import _build_project
from ui.dialogs._word_source_picker import WordAlignmentController
from ui.workers.forced_alignment_worker import ForcedAlignmentWorker
original_run = ForcedAlignmentWorker.run
def run_duplicate(self):
    original_run(self)
    if self.result:
        self.outcome_ready.emit(self.result[0])
app = QCoreApplication([])
retained = []
with TemporaryDirectory() as directory:
    for mode in ('current', 'edit', 'cancel', 'record', 'session', 'save_as', 'parent_project'):
        folder = Path(directory) / mode
        folder.mkdir()
        project = _build_project(folder, 1)
        clip = project.clips[0]
        clip.transcript = [TranscriptSegment(0, 1, 'hello', language='en')]
        assert project.save(folder / 'project.json')
        parent = QObject()
        parent._project = project
        ctrl = WordAlignmentController([(clip, project.sources[0])], parent=parent, project=project)
        retained.append((parent, ctrl))
        completed = Mock()
        errors = Mock()
        ctrl.completed.connect(completed)
        ctrl.error.connect(errors)
        wav = folder / 'alignment.wav'
        wav.write_bytes(b'audio')
        def align(*a, **kw):
            kw['on_execution']({'backend': 'ctc', 'model': ALIGNMENT_MODEL, 'revision': 'r1'})
            return [WordTimestamp(0, 1, 'hello', .9)]
        with patch.object(ForcedAlignmentWorker, 'run', run_duplicate), patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=folder)), patch('core.operations.alignment_records.alignment_model_revision', return_value='r1'), patch('ui.workers.forced_alignment_worker.ForcedAlignmentWorker._prepare', return_value=True), patch('core.analysis.alignment.extract_audio_to_wav', return_value=wav), patch('core.analysis.alignment.align_words', side_effect=align):
            ctrl.start([clip], project.sources_by_id)
            worker = ctrl._worker
            assert worker.wait(10000)
            assert ctrl.is_running()  # Queued publication still owns the worker.
            assert clip.transcript[0].words is None
            if mode == 'edit': clip.transcript[0].text = 'manual'
            if mode == 'cancel': ctrl.cancel()
            if mode == 'record': clip.analysis_records['align_words'] = AnalysisRecord.legacy({})
            if mode == 'session': project.clear()
            if mode == 'save_as': project.path = folder / 'other.json'
            if mode == 'parent_project': parent._project = Project.new()
            app.processEvents()
            assert not ctrl.is_running()
            assert ctrl._worker is None
            completed.assert_called_once()
            if mode == 'current':
                errors.assert_not_called()
                assert clip.transcript[0].words[0].text == 'hello'
                assert clip.analysis_records['align_words'].provenance == 'verified'
                assert len(project.metadata.job_results) == 1
                assert project.save()
            else:
                assert clip.transcript[0].words is None
                assert not project.metadata.job_results
        project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "dialog_name",
    [
        "word_sequencer_dialog.WordSequencerDialog",
        "word_llm_composer_dialog.WordLLMComposerDialog",
    ],
)
def test_dialog_reject_waits_for_alignment_completion(dialog_name):
    code = r"""
import importlib, sys
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtWidgets import QApplication
app = QApplication([])
module, name = sys.argv[1].split('.')
dialog = getattr(importlib.import_module('ui.dialogs.' + module), name)(clips=[])
controller = SimpleNamespace(is_running=lambda: True, cancel=Mock())
dialog._alignment_ctrl = controller
dialog.show()
dialog.reject()
assert dialog.isVisible()
assert dialog._alignment_ctrl is controller
controller.cancel.assert_called_once()
dialog._on_alignment_completed()
assert not dialog.isVisible()
assert dialog._alignment_ctrl is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, dialog_name],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
