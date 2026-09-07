"""Desktop and agent image imports share off-thread work and owner delivery."""

import os
import subprocess
import sys


def test_close_waits_for_image_import_before_teardown(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow

    monkeypatch.delenv("SCENE_RIPPER_STARTUP_SMOKE_TEST", raising=False)
    worker = Mock()
    worker.isRunning.return_value = True
    window = SimpleNamespace(
        _check_unsaved_changes=lambda: True,
        _image_import_worker=worker,
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


def test_real_image_import_dispatch_and_queued_ownership():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from threading import get_ident
from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.chat_tools import tools
from ui.main_window import MainWindow
app = QCoreApplication([])
owner = get_ident(); owners = []
with TemporaryDirectory() as directory:
    folder = Path(directory).resolve()
    media = folder / 'image.png'; Image.new('RGB', (400, 300), 'red').save(media)
    bad = folder / 'bad.png'; bad.write_bytes(b'invalid')
    for mode in ('manual', 'agent', 'partial', 'reply', 'expired', 'project', 'save_as', 'cancel', 'failure', 'apply_failure'):
        window = QObject(); owners.append(window)
        window.project = Project.new()
        if mode != 'manual': window.project.save(folder / (mode + '.sceneripper'))
        original = window.project
        window._image_import_worker = None
        window.settings = SimpleNamespace(cache_dir=folder)
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window.frames_tab = SimpleNamespace(update_frame_browser=Mock(side_effect=lambda: get_ident() == owner or (_ for _ in ()).throw(AssertionError('wrong thread'))))
        window._update_chat_project_state = Mock()
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=Mock(return_value=True), set_gui_tool_result=Mock(return_value=True))
        requester = window._chat_worker
        window._on_import_images_requested = lambda paths, **kw: MainWindow._on_import_images_requested(window, paths, **kw)
        window._start_worker_for_tool = lambda kind, result: MainWindow._start_worker_for_tool(window, kind, result)
        paths = [str(bad)] if mode == 'failure' else [str(media), str(bad)] if mode == 'partial' else [str(media)]
        with patch('ui.workers.image_import_worker.ImageImportWorker.start'), patch('PIL.Image.open') as decode:
            if mode == 'manual':
                assert window._on_import_images_requested(paths)
            else:
                MainWindow._on_gui_tool_requested(window, 'import_frames', {'file_paths': paths}, 'request')
            decode.assert_not_called()
        worker = window._image_import_worker
        assert worker is not None
        requester.set_gui_tool_result.assert_not_called()
        assert not window._on_import_images_requested(paths)
        worker.start(); assert worker.wait(10000)
        assert not original.frames and window._image_import_worker is worker
        if mode == 'reply': window._chat_worker = object()
        if mode == 'expired': requester.is_gui_tool_pending.return_value = False
        if mode == 'project': window.project = Project.new()
        if mode == 'save_as': original.path = folder / 'different.sceneripper'
        if mode == 'cancel': worker.cancel()
        if mode == 'apply_failure': original.add_frames = Mock(side_effect=ValueError('publication failed'))
        app.processEvents()
        assert window._image_import_worker is None
        success = mode in ('manual', 'agent', 'partial')
        assert len(original.frames) == int(success)
        if success:
            frame = original.frames[0]
            assert (frame.width, frame.height) == (400, 300)
            assert frame.thumbnail_path.is_file()
            assert (frame.file_path == media) == (mode == 'manual')
            if mode != 'manual': assert frame.file_path.read_bytes() == media.read_bytes()
            window.frames_tab.update_frame_browser.assert_called_once()
            window._update_chat_project_state.assert_called_once()
        else:
            window.frames_tab.update_frame_browser.assert_not_called()
        if mode in ('manual', 'reply', 'expired', 'project'):
            requester.set_gui_tool_result.assert_not_called()
        else:
            requester.set_gui_tool_result.assert_called_once()
            result = requester.set_gui_tool_result.call_args.args[0]
            assert result['name'] == 'import_frames' and result['tool_call_id'] == 'request'
            assert result['success'] == success, result
            if success:
                assert result['result']['imported_count'] == 1
                assert result['result']['frame_ids'] == [original.frames[0].id]
                if mode == 'partial': assert len(result['result']['errors']) == 1
        original.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
