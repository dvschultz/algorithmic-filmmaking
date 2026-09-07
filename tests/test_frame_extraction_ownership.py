"""Exercise the real launcher and queued QThread completion."""

import os
import subprocess
import sys


def test_close_waits_for_frame_extraction_before_teardown(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow

    monkeypatch.delenv("SCENE_RIPPER_STARTUP_SMOKE_TEST", raising=False)
    worker = Mock()
    worker.isRunning.return_value = True
    window = SimpleNamespace(
        _check_unsaved_changes=lambda: True,
        _frame_extraction_worker=worker,
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


def test_frame_launcher_preserves_owner_and_native_lifetime():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.clip import Source
from ui.main_window import MainWindow
app = QCoreApplication([])
owners = []
with TemporaryDirectory() as directory:
    root = Path(directory)
    for mode in ('current', 'project', 'session', 'edit', 'media', 'cancel', 'save_as', 'error'):
        path = root / 'video.mp4'; path.write_bytes(b'video')
        window = QObject(); owners.append(window)
        window.project = Project.new()
        source = Source(id='source', file_path=path, fps=30)
        window.project.add_source(source)
        window.sources_by_id = window.project.sources_by_id
        window.settings = SimpleNamespace(cache_dir=root)
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window.frames_tab = SimpleNamespace(update_frame_browser=Mock())
        window._on_frames_extracted = lambda *args: MainWindow._on_frames_extracted(window, *args)
        with patch('ui.workers.frame_extraction_worker.FrameExtractionWorker.start'):
            MainWindow._on_extract_frames_requested(window, source.id, 'interval', 5)
            worker = window._frame_extraction_worker
            MainWindow._on_extract_frames_requested(window, source.id, 'interval', 5)
            assert window._frame_extraction_worker is worker
        if mode == 'project': window.project = Project.new()
        if mode == 'session': window.project.clear()
        if mode == 'edit': source.fps = 24
        if mode == 'save_as': window.project.path = root / 'elsewhere.sceneripper'
        if mode == 'cancel': worker.cancel()
        if mode == 'media': path.write_bytes(b'replacement')
        def provider(video, output, fps, **kwargs):
            assert video == path and fps == 30
            if mode == 'error': raise RuntimeError('test failure')
            output.mkdir(parents=True)
            image = output / 'frame_000005.png'
            Image.new('RGB', (20, 12)).save(image)
            return [image]
        with patch('core.ffmpeg.extract_frames_batch', side_effect=provider):
            worker.start(); assert worker.wait(5000)
        assert window._frame_extraction_worker is worker
        assert not window.project.frames  # owner has not processed queued results
        app.processEvents()
        assert window._frame_extraction_worker is None
        if mode == 'current':
            assert [frame.frame_number for frame in window.project.frames] == [5]
            window.frames_tab.update_frame_browser.assert_called_once()
        else:
            assert not window.project.frames
            window.frames_tab.update_frame_browser.assert_not_called()
        if mode == 'error':
            assert 'test failure' in window.status_bar.showMessage.call_args.args[0]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
