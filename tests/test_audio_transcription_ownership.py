"""Standalone audio transcription retains the request's input and owner."""

import os
import subprocess
import sys
from unittest.mock import patch

from core.project import Project
from models.audio_source import AudioSource
from tests.test_audio_transcription_flow import _capture_signals
from ui.workers.audio_transcribe_worker import AudioTranscribeWorker


def test_audio_worker_accepts_launcher_project_and_snapshots_input(tmp_path):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio")
    audio = AudioSource(id="original", file_path=path)
    project = Project.new()
    project.add_audio_source(audio)
    worker = AudioTranscribeWorker(audio, project=project)
    transcripts, errors, _, finished = _capture_signals(worker)
    audio.id = "edited"
    audio.file_path = tmp_path / "other.wav"
    with patch("core.transcription.transcribe_video", return_value=[]) as provider:
        worker.run()
    assert provider.call_args.args[0] == path
    assert transcripts == [("original", [])]
    assert errors == []
    assert finished == [True]


def test_precancelled_audio_does_not_start_provider(tmp_path):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio")
    worker = AudioTranscribeWorker(AudioSource(file_path=path))
    worker.cancel()
    with patch("core.transcription.transcribe_video", return_value=[]) as provider:
        worker.run()
    provider.assert_not_called()


def test_real_launcher_queued_delivery_is_owned():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from ui.main_window import MainWindow
from models.audio_source import AudioSource
from core.transcription import TranscriptSegment
app = QCoreApplication([])
owners = []
with TemporaryDirectory() as directory:
    path = Path(directory) / 'audio.wav'
    for mode in ('current', 'project', 'session', 'edit', 'media', 'cancel', 'removed', 'save_as'):
        path.write_bytes(b'audio')
        window = QObject(); owners.append(window)
        window.project = Project.new()
        audio = AudioSource(id='same-id', file_path=path)
        window.project.add_audio_source(audio)
        window.settings = SimpleNamespace(transcription_model='small.en', transcription_language='en',
            transcription_backend='auto', transcription_segmentation_mode='backend', transcription_segment_max_seconds=12)
        window._active_audio_transcribes = set()
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window._on_audio_transcribe_error = Mock()
        window._update_chat_project_state = Mock()
        window._on_audio_transcript_ready = lambda *a: MainWindow._on_audio_transcript_ready(window, *a)
        with patch('ui.workers.audio_transcribe_worker.AudioTranscribeWorker.start'):
            MainWindow._on_audio_transcribe_requested(window, audio.id)
            MainWindow._on_audio_transcribe_requested(window, audio.id)
        assert len(window._active_audio_transcribes) == 1
        worker = next(iter(window._active_audio_transcribes))
        if mode == 'project':
            window.project = Project.new()
            window.project.add_audio_source(AudioSource(id=audio.id, file_path=path))
        if mode == 'session': window.project.clear()
        if mode == 'edit': audio.transcript = [TranscriptSegment(0, 1, 'manual', .9)]
        if mode == 'cancel': worker.cancel()
        if mode == 'removed': window.project.remove_audio_source(audio.id)
        if mode == 'save_as': window.project.path = Path(directory) / 'different.json'
        # Emit genuine queued signals from QThread, including a duplicate result.
        def run():
            worker.transcript_ready.emit(audio.id, [])
            worker.transcript_ready.emit(audio.id, [])
            worker.finished_signal.emit()
        worker.run = run
        if mode == 'media': path.write_bytes(b'changed-media')
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        if mode == 'current':
            assert audio.transcript == [], window._on_audio_transcribe_error.call_args_list
            window._update_chat_project_state.assert_called_once()
        elif mode == 'edit': assert audio.transcript[0].text == 'manual'
        else: assert audio.transcript is None
        if mode == 'project': assert window.project.audio_sources[0].transcript is None
        assert not window._active_audio_transcribes
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_close_waits_for_audio_worker_before_tearing_down_window(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow

    monkeypatch.delenv("SCENE_RIPPER_STARTUP_SMOKE_TEST", raising=False)
    worker = Mock()
    worker.isRunning.return_value = True
    window = SimpleNamespace(
        _check_unsaved_changes=lambda: True,
        _active_audio_transcribes={worker},
        status_bar=SimpleNamespace(showMessage=Mock()),
        _source_import_queue=Mock(),
    )
    event = Mock()
    MainWindow.closeEvent(window, event)
    worker.cancel.assert_called_once()
    worker.terminate.assert_not_called()
    window._source_import_queue.close.assert_not_called()
    event.ignore.assert_called_once()
    event.accept.assert_not_called()


def test_silent_audio_is_complete_on_card_and_agent_surfaces(tmp_path):
    from core.spine.audio_sources import get_audio_source, list_audio_sources

    audio = AudioSource(id="silent", file_path=tmp_path / "silent.wav", transcript=[])
    project = Project.new()
    project.add_audio_source(audio)
    assert list_audio_sources(project)["audio_sources"][0]["transcribed"] is True
    assert get_audio_source(project, audio.id)["audio_source"]["transcript"] == []
    code = r"""
from pathlib import Path
from PySide6.QtWidgets import QApplication, QPushButton
from models.audio_source import AudioSource
from ui.widgets.audio_library_list import AudioLibraryList
app = QApplication([])
row = AudioLibraryList()
row.set_sources([AudioSource(file_path=Path('silent.wav'), transcript=[])])
buttons = row.findChildren(QPushButton)
assert any(b.text() == 'Transcribed' and not b.isEnabled() for b in buttons)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
