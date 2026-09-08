"""Native audio reuse dispatch preserves owner/reply and cancellation boundaries."""

import os
import subprocess
import sys


def test_legacy_audio_agent_dispatch_and_owner_publication():
    code = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject, QThread
from core.project import Project
from core.settings import Settings
from core.chat_tools import tools
from models.audio_source import AudioSource
from ui.main_window import MainWindow
from ui.workers.gui_tool_cancellation import cancel_gui_tool_work
from ui.workers.legacy_audio_reuse_worker import LegacyAudioReuseWorker

app = QCoreApplication([])
owners = []
tool = tools.get('accept_legacy_audio_transcript')
assert tool.modifies_gui_state and tool.modifies_project_state
with TemporaryDirectory() as directory:
    for mode in ('current', 'reply', 'project', 'edit', 'cancel'):
        window = QObject(); owners.append(window)
        window.project = Project.new()
        path = Path(directory) / 'audio.wav'; path.write_bytes(b'audio')
        audio = AudioSource(id='audio', file_path=path, duration_seconds=2, sample_rate=48000, channels=2, transcript=[])
        window.project.add_audio_source(audio)
        window.settings = Settings(transcription_backend='groq', transcription_cloud_model='original')
        window._active_audio_transcribes = set()
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window._update_window_title = Mock()
        window._on_audio_transcript_ready = Mock()
        window._on_audio_transcribe_error = Mock()
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=lambda *a: True, set_gui_tool_result=Mock(return_value=True))
        requester = window._chat_worker
        window._on_audio_legacy_reuse_requested = lambda aid, **kw: MainWindow._on_audio_legacy_reuse_requested(window, aid, **kw)
        window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
        with patch.object(LegacyAudioReuseWorker, 'start'):
            MainWindow._on_gui_tool_requested(window, 'accept_legacy_audio_transcript', {'audio_source_id': audio.id}, 'request')
        requester.set_gui_tool_result.assert_not_called()
        worker = next(iter(window._active_audio_transcribes))
        window.settings.transcription_cloud_model = 'changed'
        assert 'transcribe' not in audio.analysis_records
        if mode == 'reply':
            window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=lambda *a: True, set_gui_tool_result=Mock(return_value=True))
        elif mode == 'project':
            window.project = Project.new()
        elif mode == 'edit':
            audio.duration_seconds = 3
        elif mode == 'cancel':
            with patch.object(worker, 'isRunning', return_value=True):
                cancel_gui_tool_work(window, name='accept_legacy_audio_transcript', token='request')
            assert worker.is_cancelled()
        QThread.start(worker)
        assert worker.wait(5000)
        assert 'transcribe' not in audio.analysis_records
        for _ in range(15): app.processEvents()
        assert not window._active_audio_transcribes
        if mode == 'current':
            record = audio.analysis_records['transcribe']
            assert record.legacy_reuse and record.provenance == 'unknown'
            assert record.identity.to_dict()['parameters']['model'] == 'original'
            assert requester.set_gui_tool_result.call_count == 1
            result = requester.set_gui_tool_result.call_args.args[0]
            assert result['name'] == 'accept_legacy_audio_transcript' and result['tool_call_id'] == 'request'
            assert result['result']['provenance'] == 'unknown'
            assert result['result']['saved'] is False
        else:
            assert 'transcribe' not in audio.analysis_records
'''
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "HF_HUB_OFFLINE": "1"}, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
