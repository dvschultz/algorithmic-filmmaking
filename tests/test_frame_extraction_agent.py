"""The registered extraction tool reaches the shared worker and its requester."""

from types import SimpleNamespace
from unittest.mock import Mock
import os
import subprocess
import sys


def test_extraction_marker_dispatches_options_and_clip():
    from ui.main_window import MainWindow

    launcher = Mock(return_value=True)
    window = SimpleNamespace(_on_extract_frames_requested=launcher)
    assert MainWindow._start_worker_for_tool(
        window,
        "extract_frames",
        {
            "_source_id": "source",
            "_clip_id": "clip",
            "_mode": "interval",
            "_interval": 5,
        },
    )
    launcher.assert_called_once_with("source", "interval", 5, clip_id="clip")


def test_real_agent_dispatch_queued_result_and_original_reply():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.clip import Source, Clip
from core.chat_tools import tools, get_tool_timeout
from ui.main_window import MainWindow
app = QCoreApplication([])
owners = []
assert get_tool_timeout('extract_frames') == 1200
with TemporaryDirectory() as directory, patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=Path(directory))):
    root = Path(directory)
    media = root / 'video.mp4'; media.write_bytes(b'video')
    for mode in ('current', 'reply', 'project', 'edit', 'cancel', 'failure', 'receipt'):
        window = QObject(); owners.append(window)
        window.project = Project.new()
        project = window.project
        source = Source(id='source', file_path=media)
        project.add_source(source)
        clip = Clip(id='clip', source_id='source', start_frame=5, end_frame=15)
        project.add_clips([clip])
        project.save(root / f'{mode}.sceneripper')
        window.sources_by_id = project.sources_by_id
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=lambda *a: True, set_gui_tool_result=Mock(return_value=True))
        requester = window._chat_worker
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window.settings = SimpleNamespace(cache_dir=root)
        window._on_frames_extracted = Mock()
        window._on_extract_frames_requested = lambda *a, **kw: MainWindow._on_extract_frames_requested(window, *a, **kw)
        window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
        with patch('ui.workers.frame_extraction_worker.FrameExtractionWorker.start'):
            MainWindow._on_gui_tool_requested(window, 'extract_frames', {'source_id': 'source', 'clip_id': 'clip', 'interval': 5}, 'request')
        worker = window._frame_extraction_worker
        requester.set_gui_tool_result.assert_not_called()
        assert tools.get('extract_frames').func(project, window, 'source')['success'] is False
        def provider(path, output, fps, **kwargs):
            assert kwargs['start_frame'] == 5 and kwargs['end_frame'] == 15
            if mode == 'failure': raise RuntimeError('provider failed')
            output.mkdir(parents=True)
            image = output / 'frame_000005.png'
            Image.new('RGB', (20, 12)).save(image)
            return [image]
        with patch('core.ffmpeg.extract_frames_batch', side_effect=provider):
            worker.start(); assert worker.wait(10000)
        if mode == 'reply': window._chat_worker = object()
        if mode == 'project': window.project = Project.new()
        if mode == 'edit': clip.start_frame = 7
        if mode == 'cancel': worker.cancel()
        if mode == 'receipt': worker.cache.recorded.task['interval'] = 1
        app.processEvents()
        assert window._frame_extraction_worker is None
        if mode in ('reply', 'project'):
            requester.set_gui_tool_result.assert_not_called()
            assert not project.frames
        else:
            requester.set_gui_tool_result.assert_called_once()
            result = requester.set_gui_tool_result.call_args.args[0]
            assert result['tool_call_id'] == 'request' and result['name'] == 'extract_frames'
            assert result['success'] == (mode == 'current'), result
            assert len(project.frames) == (1 if mode == 'current' else 0)
            if mode == 'current':
                assert result['result']['frame_ids'] == [project.frames[0].id]
                assert project.frames[0].frame_number == 5
                assert project.metadata.job_results
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
