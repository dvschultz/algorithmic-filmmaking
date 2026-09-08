"""Native agent dispatch waits for guarded legacy decision publication."""

import os
import subprocess
import sys


def test_agent_legacy_reuse_dispatch_and_reply_ownership():
    code = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.chat_tools import tools
from core.analysis_model_identity import DINOV2_TAG
from ui.main_window import MainWindow
from tests.test_description_operations import project_with_thumbnails

app = QCoreApplication([])
owners = []
tool = tools.get('accept_legacy_analysis')
assert tool.modifies_gui_state and tool.modifies_project_state
with TemporaryDirectory() as directory:
    for operation in ('colors', 'embeddings', 'brightness', 'volume', 'classify', 'detect_objects', 'boundary_embeddings', 'gaze', 'shots'):
        for mode in ('current', 'reply', 'project', 'edit', 'cancel'):
            window = QObject(); owners.append(window)
            window.project = project_with_thumbnails(Path(directory), 1)
            clip = window.project.clips[0]
            clip.dominant_colors = [(1, 2, 3)]
            clip.embedding, clip.embedding_model = [0.1] * 768, DINOV2_TAG
            clip.average_brightness = clip.rms_volume = 0.0
            clip.object_labels = clip.detected_objects = []
            clip.person_count = 0
            clip.shot_type = "wide shot"
            clip.gaze_yaw = clip.gaze_pitch = 0.0
            clip.gaze_category = "at_camera"
            clip.first_frame_embedding = [0.2] * 768
            clip.last_frame_embedding = [0.3] * 768
            window._chat_worker = SimpleNamespace(_stop_requested=False,
                is_gui_tool_pending=lambda *a: True, set_gui_tool_result=Mock(return_value=True))
            requester = window._chat_worker
            window._update_window_title = Mock()
            window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
            with patch('ui.workers.legacy_reuse_worker.LegacyReuseWorker.start'):
                MainWindow._on_gui_tool_requested(window, 'accept_legacy_analysis',
                    {'operation': operation, 'clip_ids': [clip.id]}, 'request')
            requester.set_gui_tool_result.assert_not_called()
            worker = next(iter(window._active_legacy_reuses))
            worker.start(); assert worker.wait(10000)
            assert operation not in clip.analysis_records
            if mode == 'reply': window._chat_worker = object()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': clip.start_frame += 1
            if mode == 'cancel': worker.cancel()
            app.processEvents()
            assert not window._active_legacy_reuses
            if mode in ('reply', 'project'):
                requester.set_gui_tool_result.assert_not_called()
            else:
                requester.set_gui_tool_result.assert_called_once()
                result = requester.set_gui_tool_result.call_args.args[0]
                assert result['tool_call_id'] == 'request'
                assert result['name'] == 'accept_legacy_analysis'
                assert result['success'] == (mode == 'current'), result
            if mode == 'current':
                assert clip.analysis_records[operation].legacy_reuse
                assert clip.analysis_records[operation].provenance == 'unknown'
                assert result['result']['accepted'] == [clip.id]
                assert result['result']['saved'] is False
            else:
                assert operation not in clip.analysis_records
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
