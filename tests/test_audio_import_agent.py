"""Audio import dispatch must not probe on the agent or GUI thread."""

import os
import subprocess
import sys


def test_audio_import_agent_dispatch_and_owned_completion():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.chat_tools import tools
from models.audio_source import AudioSource
from ui.main_window import MainWindow
app = QCoreApplication([])
owners = []
tool = tools.get('import_audio_source')
assert tool.modifies_gui_state and tool.modifies_project_state
with TemporaryDirectory() as directory:
    media = Path(directory) / 'voice.wav'; media.write_bytes(b'audio')
    for mode in ('current', 'reply', 'expired', 'project', 'save_as', 'cancel', 'failure', 'apply_failure', 'duplicate'):
        window = QObject(); owners.append(window)
        window.project = Project.new(); window.project.path = Path(directory) / 'project.sceneripper'
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=Mock(return_value=True), set_gui_tool_result=Mock(return_value=True))
        requester = window._chat_worker
        window._active_audio_imports = set()
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window._on_audio_imported = Mock()
        window._on_audio_import_error = Mock()
        window._apply_gui_tool_side_effects = Mock()
        window._on_audio_files_added = lambda paths: MainWindow._on_audio_files_added(window, paths)
        window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
        with patch('core.ffmpeg.FFmpegProcessor') as probe, patch('ui.workers.audio_import_worker.AudioImportWorker.start'):
            MainWindow._on_gui_tool_requested(window, 'import_audio_source', {'file_path': 'voice.wav'}, 'request')
            probe.assert_not_called()
        worker = next(iter(window._active_audio_imports))
        assert worker.task.path == media.resolve()
        requester.set_gui_tool_result.assert_not_called()
        # An overlapping request must not wait forever or start another worker.
        with patch('ui.workers.audio_import_worker.AudioImportWorker.start'):
            MainWindow._on_gui_tool_requested(window, 'import_audio_source', {'file_path': 'voice.wav'}, 'second')
        assert len(window._active_audio_imports) == 1
        assert requester.set_gui_tool_result.call_args.args[0]['success'] is False
        requester.set_gui_tool_result.reset_mock()
        with patch('core.ffmpeg.FFmpegProcessor') as probe:
            probe.return_value.ffprobe_available = True
            probe.return_value.get_audio_info.return_value = dict(duration=10, sample_rate=48000, channels=2)
            if mode == 'failure': probe.return_value.get_audio_info.side_effect = RuntimeError('probe failed')
            worker.start(); assert worker.wait(10000)
        if mode == 'reply': window._chat_worker = object()
        if mode == 'expired': requester.is_gui_tool_pending.return_value = False
        if mode == 'project': window.project = Project.new()
        if mode == 'save_as': window.project.path = Path(directory) / 'other.sceneripper'
        if mode == 'cancel': worker.cancel()
        if mode == 'duplicate': window.project.add_audio_source(AudioSource(id='existing', file_path=media))
        if mode == 'apply_failure': window.project.add_audio_source = Mock(side_effect=ValueError('publication failed'))
        app.processEvents()
        assert not window._active_audio_imports
        assert len(window.project.audio_sources) == (1 if mode in ('current', 'duplicate') else 0)
        if mode in ('reply', 'expired', 'project'):
            requester.set_gui_tool_result.assert_not_called()
        else:
            requester.set_gui_tool_result.assert_called_once()
            result = requester.set_gui_tool_result.call_args.args[0]
            assert result['name'] == 'import_audio_source' and result['tool_call_id'] == 'request'
            assert result['success'] == (mode in ('current', 'duplicate')), result
            if result['success']:
                audio = window.project.audio_sources[0]
                assert result['result']['audio_source_id'] == audio.id
                assert result['result']['filename'] == 'voice.wav'
                assert result['result']['duration'] == audio.duration_seconds
                with patch('core.ffmpeg.FFmpegProcessor') as probe:
                    immediate = tool.func(window, 'voice.wav')
                    probe.assert_not_called()
                assert immediate['audio_source_id'] == audio.id
        if mode == 'current': window._on_audio_imported.assert_called_once()
        else: window._on_audio_imported.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
