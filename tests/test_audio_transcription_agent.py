"""GUI audio transcription replies remain bound to the initiating request."""

import os
import subprocess
import sys


def test_agent_audio_transcription_real_dispatch_and_queued_reply():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.audio_source import AudioSource
from core.transcription_models import TranscriptSegment
from core.chat_tools import tools
from ui.main_window import MainWindow
app = QCoreApplication([])
owners = []
tool = tools.get('transcribe_audio_source')
assert tool.modifies_gui_state and tool.modifies_project_state
with TemporaryDirectory() as directory:
    media = Path(directory) / 'voice.wav'; media.write_bytes(b'audio')
    for mode in ('current', 'reply', 'project', 'edit', 'cancel', 'failure'):
        window = QObject(); owners.append(window)
        window.project = Project.new()
        audio = AudioSource(id='audio', file_path=media)
        window.project.add_audio_source(audio)
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=lambda *a: True, set_gui_tool_result=Mock(return_value=True))
        requester = window._chat_worker
        window._active_audio_transcribes = set()
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window.settings = SimpleNamespace(transcription_model='small.en', transcription_language='en', transcription_backend='faster-whisper', transcription_segmentation_mode='backend', transcription_segment_max_seconds=12)
        window._on_audio_transcript_ready = Mock()
        window._on_audio_transcribe_error = Mock()
        window._on_audio_transcribe_requested = lambda aid: MainWindow._on_audio_transcribe_requested(window, aid)
        window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
        with patch('ui.workers.audio_transcribe_worker.AudioTranscribeWorker.start'):
            MainWindow._on_gui_tool_requested(window, 'transcribe_audio_source', {'audio_source_id': 'audio'}, 'request')
        worker = next(iter(window._active_audio_transcribes))
        requester.set_gui_tool_result.assert_not_called()
        with patch('core.transcription.transcribe_video', return_value=[], side_effect=RuntimeError('provider failed') if mode == 'failure' else None):
            worker.start(); assert worker.wait(10000)
        if mode == 'reply': window._chat_worker = object()
        if mode == 'project': window.project = Project.new()
        if mode == 'edit': audio.transcript = [TranscriptSegment(0, 1, 'manual')]
        if mode == 'cancel': worker.cancel()
        app.processEvents()
        assert not window._active_audio_transcribes
        if mode in ('reply', 'project'):
            requester.set_gui_tool_result.assert_not_called()
            assert audio.transcript is None
        else:
            requester.set_gui_tool_result.assert_called_once()
            result = requester.set_gui_tool_result.call_args.args[0]
            assert result['tool_call_id'] == 'request'
            assert result['name'] == 'transcribe_audio_source'
            assert result['success'] == (mode == 'current'), result
            if mode == 'current':
                assert audio.transcript == []
                assert result['result']['segment_count'] == 0
            if mode == 'edit': assert audio.transcript[0].text == 'manual'
        if mode == 'current':
            immediate = tool.func(window, 'audio')
            assert immediate['result']['status'] == 'skipped'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
