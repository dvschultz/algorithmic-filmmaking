"""Audio import ownership survives queued delivery and project replacement."""

import os
import subprocess
import sys


def test_old_project_pending_import_does_not_block_new_project():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from ui.main_window import MainWindow
app = QCoreApplication([])
with TemporaryDirectory() as directory:
    path = Path(directory) / 'audio.wav'; path.write_bytes(b'audio')
    window = QObject()
    window.project = Project.new()
    window._active_audio_imports = set()
    window.status_bar = SimpleNamespace(showMessage=Mock())
    with patch('ui.workers.audio_import_worker.AudioImportWorker.start'):
        MainWindow._on_audio_files_added(window, [path])
        MainWindow._on_audio_files_added(window, [path])
        assert len(window._active_audio_imports) == 1
        window.project = Project.new()
        MainWindow._on_audio_files_added(window, [path])
        assert len(window._active_audio_imports) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_queued_import_is_owned_and_deduplicated():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.audio_source import AudioSource
from ui.main_window import MainWindow
app = QCoreApplication([])
owners = []
with TemporaryDirectory() as directory:
    path = Path(directory) / 'audio.wav'
    for mode in ('current', 'project', 'session', 'save_as', 'cancel', 'media', 'duplicate'):
        path.write_bytes(b'audio')
        window = QObject(); owners.append(window)
        window.project = Project.new()
        window._active_audio_imports = set()
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window._update_chat_project_state = Mock()
        window._on_audio_imported = lambda audio: MainWindow._on_audio_imported(window, audio)
        window._on_audio_import_error = Mock()
        with patch('ui.workers.audio_import_worker.AudioImportWorker.start'):
            MainWindow._on_audio_files_added(window, [path])
        worker = next(iter(window._active_audio_imports))
        with patch('core.ffmpeg.FFmpegProcessor') as processor:
            processor.return_value.ffprobe_available = True
            processor.return_value.get_audio_info.return_value = dict(duration=10, sample_rate=48000, channels=2)
            worker.start(); assert worker.wait(5000)
        assert worker in window._active_audio_imports
        if mode == 'project': window.project = Project.new()
        if mode == 'session': window.project.clear()
        if mode == 'save_as': window.project.path = Path(directory) / 'different.sceneripper'
        if mode == 'cancel': worker.cancel()
        if mode == 'media': path.write_bytes(b'changed')
        if mode == 'duplicate': window.project.add_audio_source(AudioSource(id='existing', file_path=path))
        app.processEvents()
        assert not window._active_audio_imports
        assert len(window.project.audio_sources) == (1 if mode in ('current', 'duplicate') else 0)
        if mode == 'duplicate': assert window.project.audio_sources[0].id == 'existing'
        if mode == 'current': window._update_chat_project_state.assert_called_once()
        else: window._update_chat_project_state.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
