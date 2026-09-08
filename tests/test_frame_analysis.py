"""Frame workflows use actual worker outcomes and native thread completion."""

import os
import subprocess
import sys

import pytest

from core.operations.frame_analysis import FrameAnalysisPlan


def test_ordered_plan_rejects_duplicate_completion_and_tracks_each_frame():
    plan = FrameAnalysisPlan(["one", "two", "one"], ["shots", "classify", "shots"])
    assert plan.begin_next() == "shots"
    with pytest.raises(RuntimeError):
        plan.begin_next()
    assert plan.finish("shots", {"one": "succeeded", "two": "failed"})
    assert not plan.finish("shots", {"one": "succeeded", "two": "succeeded"})
    assert plan.begin_next() == "classify"
    assert plan.finish("classify", {"one": "skipped", "two": "succeeded"})
    assert plan.begin_next() is None
    assert plan.successful_ids() == ("one",)


def test_cancelled_plan_never_starts_later_operations():
    plan = FrameAnalysisPlan(["one"], ["shots", "classify"])
    assert plan.begin_next() == "shots"
    plan.cancel()
    assert plan.finish("shots", {"one": "succeeded"})
    assert plan.begin_next() is None
    assert plan.successful_ids() == ()


@pytest.mark.parametrize("operations", [[], ["transcribe"], ["unknown"]])
def test_invalid_frame_plan_fails_before_dispatch(operations):
    with pytest.raises(ValueError):
        FrameAnalysisPlan(["one"], operations)


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "failure",
        "all_failed",
        "cancel",
        "project",
        "session",
        "path",
        "frame",
        "media",
        "reply",
        "restart",
        "duplicate",
        "prepopulated",
        "missing_image",
        "startup_error",
        "forged",
        "settings",
        "cancel_on_publish",
        "replace_on_publish",
        "cancel_on_mark",
    ],
)
def test_frame_controller_uses_real_workers_and_preserves_editor_state(tmp_path, mode):
    code = r"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock, patch
from threading import Event
import sys, time
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.settings import Settings
from models.frame import Frame
from models.cinematography import CinematographyAnalysis
from ui.workers.frame_analysis import FrameAnalysisController, create_frame_analysis_worker
from core.jobs.store import JobStore
app = QCoreApplication([])
directory = Path(sys.argv[1]); mode = sys.argv[2]
class Window(QObject): pass
window = Window(); project = Project.new(); window.project = project
count = 2 if mode in ('failure', 'cancel_on_publish', 'replace_on_publish', 'cancel_on_mark') else 1
for i in range(count):
    image = directory / f'{i}.png'; image.write_bytes(b'image')
    project.add_frames([Frame(id=str(i), file_path=image)])
if mode == 'prepopulated': project.frames[0].shot_type = 'manual'
assert project.save(directory / 'project.json')
before = project.path.read_bytes()
window.settings = Settings(cache_dir=directory, shot_classifier_tier='cpu',
    description_model_tier='cloud', description_model_cloud='test', description_input_mode='frame',
    cinematography_tier='cloud', cinematography_model='test', cinematography_input_mode='frame',
    text_extraction_method='vlm', text_extraction_vlm_model='test')
window._dispatch_gui_reply = SimpleNamespace(is_current=lambda _: True)
entered = Event(); release = Event()
blocking = mode in ('cancel', 'project', 'session', 'path', 'frame', 'media', 'reply', 'restart', 'duplicate', 'settings')
def shots(path):
    if blocking:
        entered.set()
        assert release.wait(10)
    if mode == 'all_failed' or (mode == 'failure' and path.name == '1.png'):
        return 'unknown', .1
    return 'wide', .9
shot_provider = Mock(side_effect=shots)
class_provider = Mock(return_value=[('cat', .9)])
ops = ['colors', 'shots', 'classify', 'detect_objects', 'extract_text', 'describe', 'cinematography'] if mode == 'success' else ['shots', 'classify']
if mode in ('all_failed', 'missing_image'): ops = ['shots']
if mode == 'settings': ops = ['shots', 'extract_text', 'describe', 'cinematography']
if mode == 'missing_image': project.frames[0].file_path.unlink()
observed_options = {}
def factory(project, settings, op, targets, **kwargs):
    if mode == 'startup_error' and op == 'shots': raise RuntimeError('startup failed')
    worker, application = create_frame_analysis_worker(project, settings, op, targets, **kwargs)
    if op != 'colors': observed_options[op] = worker.options
    return worker, application
with patch('core.settings.load_settings', lambda: window.settings), \
     patch('core.analysis.color.extract_dominant_colors', return_value=[(255, 0, 0)]), \
     patch('core.analysis.shots.classify_shot_type', shot_provider), \
     patch('core.analysis.classification.classify_frame', class_provider), \
     patch('core.analysis.detection.detect_objects', return_value=[]), \
     patch('core.analysis.ocr.extract_text_from_frame', return_value=('SIGN', .9, 'vlm')), \
     patch('core.analysis.description.describe_frame', return_value=('Generated', 'test')), \
     patch('core.analysis.cinematography.analyze_cinematography', return_value=CinematographyAnalysis(shot_size='CU')), \
     patch('ui.workers.frame_analysis.create_frame_analysis_worker', factory):
    controller = FrameAnalysisController(window, [frame.id for frame in project.frames], ops)
    completions = []; controller.completed.connect(lambda _, result: completions.append(result))
    def observer(event, data):
        if event != 'frames_updated': return
        if mode == 'cancel_on_publish': controller.cancel()
        elif mode == 'replace_on_publish': window.project = Project.new()
        elif mode == 'cancel_on_mark' and data[0].analyzed: controller.cancel()
    if mode in ('cancel_on_publish', 'replace_on_publish', 'cancel_on_mark'): project.add_observer(observer)
    controller.start(); controller.start()  # repeated starts are harmless
    if blocking:
        assert entered.wait(5)
        assert class_provider.call_count == 0
        if mode == 'cancel': controller.cancel()
        elif mode == 'settings':
            window.settings.description_model_cloud = 'changed'
            window.settings.cinematography_model = 'changed'
            window.settings.text_extraction_vlm_model = 'changed'
        elif mode == 'project': window.project = Project.new()
        elif mode == 'session': project.clear()
        elif mode == 'path': assert project.save(directory / 'copy.json')
        elif mode == 'frame': project.frames[0] = Frame(id='0', file_path=directory/'0.png'); project._invalidate_caches()
        elif mode == 'media': (directory/'0.png').write_bytes(b'changed')
        elif mode == 'reply': window._dispatch_gui_reply.is_current = lambda _: False
        elif mode == 'restart':
            replacement = FrameAnalysisController(window, ['0'], ['classify']); replacement.start()
        elif mode == 'duplicate':
            controller.worker.analysis_completed.emit()
            controller.worker.finished.emit()
            assert class_provider.call_count == 0
        release.set()
    if mode == 'forged':
        assert controller.worker.wait(10000)
        controller.worker.result = tuple(replace(outcome, confidence=.5) for outcome in controller.worker.result)
    deadline = time.monotonic() + 15
    while window._active_frame_analyses and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.005)
    assert not window._active_frame_analyses, 'workflow did not settle'
    assert len(completions) == 1, completions
    result = completions[0]
    assert (directory/'project.json').read_bytes() == before, 'analysis implicitly saved the project'
    if mode == 'cancel_on_mark':
        assert project.frames[0].analyzed
        assert not project.frames[1].analyzed, 'second frame marked after cancellation'
        assert result['cancelled']
    elif mode in ('cancel_on_publish', 'replace_on_publish'):
        assert project.frames[0].shot_type == 'wide'
        assert project.frames[1].shot_type is None, 'second frame published after cancellation/replacement'
        assert result['cancelled']
        class_provider.assert_not_called()
    elif mode == 'settings':
        assert observed_options['describe'].model == 'test', observed_options
        assert observed_options['cinematography'].model == 'test', observed_options
        assert observed_options['extract_text'].vlm_model == 'test', observed_options
        assert result['succeeded'] == ['0'], result
    elif mode == 'success':
        assert result['succeeded'] == ['0'], result
        assert project.frames[0].analyzed
        assert project.frames[0].description == 'Generated'
        assert project.frames[0].extracted_texts[0].text == 'SIGN'
        assert len(project.metadata.job_results) == 6
        assert project.save()
    elif mode == 'failure':
        assert result['succeeded'] == ['0'] and result['failed'] == ['1'], result
        assert project.frames[0].analyzed and not project.frames[1].analyzed
        assert project.frames[1].object_labels == ['cat']
    elif mode in ('duplicate', 'prepopulated'):
        assert result['succeeded'] == ['0'], result
        assert class_provider.call_count == 1
        assert shot_provider.call_count == 1
        assert project.frames[0].shot_type == 'wide'
    elif mode == 'restart':
        assert result['cancelled']
        assert project.frames[0].shot_type is None
        assert project.frames[0].object_labels == ['cat']
    else:
        assert not result['succeeded'], result
        if project.frames: assert not project.frames[0].analyzed
        if mode in ('cancel', 'project', 'session', 'path', 'reply'):
            assert result['cancelled']
            class_provider.assert_not_called()
        if mode == 'missing_image':
            assert not project.is_dirty
        if mode == 'all_failed':
            assert project.is_dirty
            assert all(frame.analysis_records['shots'].state == 'failed' for frame in project.frames)
        if mode == 'forged': assert project.frames[0].shot_type is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
