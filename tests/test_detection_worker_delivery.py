"""Verify the desktop adapter invokes the shared operation on its worker thread."""

from pathlib import Path
import subprocess
import sys


def test_detection_worker_snapshots_settings_and_suppresses_cancelled_result():
    code = '''
import threading
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from core.scene_detect import DetectionConfig
from core.project import Project
from models.clip import Source
from ui.main_window import DetectionWorker

app = QCoreApplication([])
owner = threading.get_ident()
source = Source(file_path=Path('video.mp4'))
project = Project.new(name='delivery')
project.add_source(source)
for outcome in ("completed", "cancelled", "pre_cancelled", "cancel_error", "failed"):
    cancelled = outcome in ("cancelled", "pre_cancelled", "cancel_error")
    config = DetectionConfig(threshold=4)
    worker = DetectionWorker(Path('video.mp4'), config, project=project)
    config.threshold = 9
    loop = QEventLoop()
    received, errors = [], []
    started = []
    worker.job_started.connect(lambda task, persistence: started.append((task, persistence)))
    guarded = []
    worker.result_ready.connect(lambda guard, source, clips: guarded.append((guard, threading.get_ident())))
    worker.detection_completed.connect(lambda source, clips: received.append(source))
    worker.error.connect(errors.append)
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    class Detector:
        def __init__(self, config):
            assert threading.get_ident() != owner
            assert config.threshold == 4

        def detect_scenes_with_progress(self, path, progress):
            if outcome == "failed":
                raise RuntimeError("decoder failed")
            if cancelled:
                worker.cancel()
            if outcome == "cancel_error":
                raise RuntimeError("decoder stopped during cancellation")
            return source, []

    if outcome == 'pre_cancelled':
        worker.cancel()
    with patch('core.scene_detect.SceneDetector', Detector):
        timer.start(5000)
        worker.start()
        loop.exec()
        assert worker.wait(5000)
    assert started == [(worker.task_id, "session_only")], started
    assert worker.job_status == ("cancelled" if cancelled else outcome), worker.job_status
    assert worker._runtime is None
    assert worker.operation.session_id == project.session.session_id
    assert received == ([source] if outcome == "completed" else []), received
    assert guarded == ([(worker.guard, owner)] if outcome == "completed" else []), guarded
    if outcome == "failed":
        assert len(errors) == 1 and "decoder failed" in errors[0], errors
    else:
        assert errors == [], errors
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_desktop_publication_uses_guarded_target_for_both_entry_points():
    code = '''
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from core.project import Project
from core.operations.detection import DetectionGuard
from models.clip import Source, Clip
from ui.main_window import MainWindow

class Controller:
    _on_guarded_detection_error = MainWindow._on_guarded_detection_error
    _on_guarded_intention_detection_error = MainWindow._on_guarded_intention_detection_error
    _on_guarded_detection_finished = MainWindow._on_guarded_detection_finished
    _on_detection_finished = MainWindow._on_detection_finished
    _on_guarded_intention_detection_finished = MainWindow._on_guarded_intention_detection_finished
    _on_intention_detection_completed = MainWindow._on_intention_detection_completed
    sources_by_id = property(lambda self: self.project.sources_by_id)
    clips_by_id = property(lambda self: self.project.clips_by_id)
    clips = property(lambda self: self.project.clips)
    sources = property(lambda self: self.project.sources)

for intention in (False, True):
    controller = Controller()
    controller.project = Project.new(name='original')
    target = Source(file_path=Path('target.mp4'))
    selected = Source(file_path=Path('selected.mp4'))
    for source in (target, selected):
        controller.project.add_source(source)
        controller.project.add_clips([Clip(source_id=source.id, start_frame=0, end_frame=30)])
    controller.current_source = selected
    guard = DetectionGuard.capture(controller.project, target.file_path)
    controller._active_detection_guard = guard
    controller._detection_finished_handled = False
    controller._detection_generation = 1
    controller.intention_workflow = None
    controller.detection_worker = None
    controller.thumbnail_worker = None
    controller.settings = SimpleNamespace(thumbnail_cache_dir=Path('/tmp'))
    for name in ('_gui_state', 'collect_tab', 'cut_tab', 'status_bar',
                 '_stop_worker_safely', '_update_window_title', '_on_thumbnail_progress',
                 '_on_thumbnail_ready', '_on_thumbnails_finished', '_on_detection_error',
                 '_on_intention_detection_error'):
        setattr(controller, name, Mock())
    controller.analyze_tab = Mock()
    controller.analyze_tab.remove_orphaned_clips.return_value = 0
    result = Source(file_path=target.file_path, fps=24, duration_seconds=10)
    clips = [Clip(source_id=result.id, start_frame=0, end_frame=240)]

    def deliver():
        if intention:
            controller._on_guarded_intention_detection_finished(guard, result, clips, 1)
        else:
            controller._on_guarded_detection_finished(guard, result, clips)

    def error():
        if intention:
            controller._on_guarded_intention_detection_error(guard, "old error", 1)
        else:
            controller._on_guarded_detection_error(guard, "old error")

    # Neither an obsolete task nor another session may consume the result.
    controller._active_detection_guard = object()
    deliver()
    error()
    assert len(controller.project.clips_by_source[target.id]) == 1
    assert target.fps == 30
    controller._active_detection_guard = guard
    original = controller.project
    controller.project = Project.new(name='other')
    deliver()
    error()
    assert not controller.project.sources
    controller.project = original
    with patch('ui.main_window.ThumbnailWorker'):
        deliver()
        deliver()  # duplicate delivery is a no-op
        error()  # late error after successful delivery is also ignored
    assert controller.current_source is selected
    assert target.fps == 24 and target.analyzed
    assert selected.fps == 30 and not selected.analyzed
    assert len(controller.project.sources) == 2
    assert controller.project.clips_by_source[target.id] == clips
    assert len(controller.project.clips_by_source[selected.id]) == 1
    assert all(c.source_id == target.id for c in clips)
    controller._on_detection_error.assert_not_called()
    controller._on_intention_detection_error.assert_not_called()
    if not intention:
        controller._thumbnails_finished_handled = False
        controller._pending_agent_detection = True
        controller._pending_agent_tool_call_id = 'call'
        controller._pending_agent_tool_name = 'detect_scenes_live'
        for name in ('_chat_worker', 'render_tab', 'progress_bar', '_update_chat_project_state',
                     '_sync_cut_tab_clip_browser', '_refresh_sequence_tab_clips', '_start_next_analysis'):
            setattr(controller, name, Mock())
        controller._build_agent_detected_clip_summary = lambda clips: []
        with patch('ui.main_window.QTimer'):
            MainWindow._on_thumbnails_finished(controller)
        response = controller._chat_worker.set_gui_tool_result.call_args.args[0]['result']
        assert response['source_id'] == target.id, response
        assert response['clip_ids'] == [c.id for c in clips], response
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
