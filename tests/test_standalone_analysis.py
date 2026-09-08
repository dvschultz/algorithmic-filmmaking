"""Standalone requests share clip scheduling without losing explicit options."""

from types import SimpleNamespace
from unittest.mock import Mock
import os
import subprocess
import sys

import pytest

from core.project import Project
from core.settings import Settings
from models.clip import Clip, Source
from ui.main_window import MainWindow


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(
    "method,operation,kwargs",
    [
        ("start_agent_color_analysis", "colors", {}),
        ("start_agent_shot_analysis", "shots", {}),
        ("start_agent_transcription", "transcribe", {}),
        ("start_agent_classification", "classify", {"top_k": 9}),
        (
            "start_agent_object_detection",
            "detect_objects",
            {"confidence": 0.8, "detect_all": False},
        ),
        (
            "start_agent_description",
            "describe",
            {"tier": "cloud", "prompt": "Look closely"},
        ),
    ],
)
def test_public_entry_points_use_shared_controller(
    tmp_path, monkeypatch, method, operation, kwargs, disabled
):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    source.file_path.write_bytes(b"video")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30, disabled=disabled)
    project.add_source(source)
    project.add_clips([clip])
    window = SimpleNamespace(
        project=project,
        settings=Settings(),
        _ensure_analysis_operation_available=Mock(return_value=True),
        analyze_tab=Mock(),
        _switch_to_tab=Mock(),
        _gui_state=Mock(),
        progress_bar=Mock(),
        status_bar=Mock(),
        _on_clip_analysis_progress=Mock(),
        _on_clip_analysis_status=Mock(),
    )
    factory = Mock()
    monkeypatch.setattr("ui.workers.clip_analysis.ClipAnalysisController", factory)
    assert getattr(MainWindow, method)(window, [clip.id], **kwargs)
    factory.assert_called_once()
    assert factory.call_args.args == (window, [clip], [operation])
    assert factory.call_args.kwargs["standalone"] is True
    options = factory.call_args.kwargs["options"]
    for name, value in kwargs.items():
        assert getattr(options, name) == value
    factory.return_value.start.assert_called_once()


@pytest.mark.parametrize("operation", ["classify", "detect_objects", "describe"])
def test_factory_preserves_explicit_options(tmp_path, operation):
    from core.operations.clip_analysis import ClipAnalysisOptions
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    options = ClipAnalysisOptions(
        top_k=9, confidence=0.8, detect_all=False, tier="cloud", prompt="Look closely"
    )
    worker, _ = create_clip_analysis_worker(
        project, Settings(), operation, [clip], options=options
    )
    if operation == "classify":
        assert worker.options.top_k == 9
    elif operation == "detect_objects":
        assert worker.options.confidence == 0.8
        assert not worker.options.detect_all
    else:
        assert worker.options.tier == "cloud"
        assert worker.options.prompt == "Look closely"


@pytest.mark.parametrize(
    "change", ["project", "session", "path", "clip", "source", "media", "run", "reply"]
)
def test_dependency_dialog_cannot_retarget_standalone_request(
    tmp_path, monkeypatch, change
):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4")
    source.file_path.write_bytes(b"video")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    reply = Mock()
    reply.is_current.return_value = True
    window = SimpleNamespace(project=project, _dispatch_gui_reply=reply)

    def gate(*args, **kwargs):
        if change == "project":
            window.project = Project.new()
        elif change == "session":
            project.session.session_id = "new"
        elif change == "path":
            project.path = tmp_path / "new.json"
        elif change == "clip":
            project.clips_by_id[clip.id] = Clip(id=clip.id, source_id=source.id)
        elif change == "source":
            project.sources_by_id[source.id] = Source(
                id=source.id, file_path=source.file_path
            )
        elif change == "media":
            source.file_path.write_bytes(b"new video")
        elif change == "run":
            window._clip_analysis_controller = object()
        else:
            reply.is_current.return_value = False
        return True

    window._ensure_analysis_operation_available = gate
    factory = Mock()
    monkeypatch.setattr("ui.workers.clip_analysis.ClipAnalysisController", factory)
    assert not MainWindow.start_agent_description(window, [clip.id])
    factory.assert_not_called()


@pytest.mark.parametrize("detect_all", [True, False])
def test_object_summaries_keep_zero_results_and_detection_mode(detect_all):
    from core.operations.clip_analysis import ClipAnalysisOptions
    from ui.workers.standalone_analysis import finish_standalone_analysis

    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    clip.detected_objects = [{"label": "person"}, {"label": "car"}]
    clip.person_count = 1
    reply = Mock()
    controller = SimpleNamespace(
        owns_project=lambda: True,
        owns_view=lambda: False,
        clips={clip.id: clip},
        plan=SimpleNamespace(clip_ids=[clip.id], operations=["detect_objects"]),
        options=ClipAnalysisOptions(detect_all=detect_all),
        reply=reply,
    )
    window = SimpleNamespace(
        _update_chat_project_state=Mock(),
        _build_agent_analysis_result=lambda clips, ops, message, extra: extra,
    )
    result = dict(
        succeeded=[clip.id],
        failed=[],
        cancelled=False,
        operations={},
        errors=[],
        analyzed_sources=[],
    )
    finish_standalone_analysis(window, controller, result)
    payload = reply.send.call_args.args[1]["result"]
    assert payload["analyzed_clips"] == 1 and payload["total_people_detected"] == 1
    assert ("object_counts" in payload) is detect_all
    if detect_all:
        assert payload["object_counts"] == {"person": 1, "car": 1}
    clip.detected_objects = []
    clip.person_count = 0
    finish_standalone_analysis(window, controller, result)
    assert reply.send.call_args.args[1]["result"]["analyzed_clips"] == 1


def test_person_only_request_recomputes_unverified_zero_count(tmp_path, monkeypatch):
    import time
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.operations.clip_analysis import ClipAnalysisOptions
    from ui.workers.clip_analysis import ClipAnalysisController
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    app = QApplication.instance() or QApplication([])
    window = QObject()
    window.project = project = Project.new()
    window.settings = Settings()
    source = Source(file_path=tmp_path / "source.mp4")
    source.file_path.write_bytes(b"source")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30, person_count=0)
    clip.thumbnail_path = tmp_path / "thumb.jpg"
    clip.thumbnail_path.write_bytes(b"image")
    project.add_source(source)
    project.add_clips([clip])
    for name in (
        "_gui_state",
        "analyze_tab",
        "progress_bar",
        "status_bar",
        "collect_tab",
        "_update_chat_project_state",
    ):
        setattr(window, name, Mock())
    window._dispatch_gui_reply = Mock()
    window._build_agent_analysis_result = lambda clips, ops, message, extra: extra
    provider = Mock(return_value=0)
    monkeypatch.setattr("core.analysis.detection.count_people", provider)
    factory = Mock(wraps=create_clip_analysis_worker)
    monkeypatch.setattr("ui.workers.clip_analysis.create_clip_analysis_worker", factory)
    controller = ClipAnalysisController(
        window,
        [clip],
        ["detect_objects"],
        standalone=True,
        options=ClipAnalysisOptions(detect_all=False),
    )
    controller.start()
    deadline = time.monotonic() + 10
    while not controller.finished and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    assert controller.finished
    factory.assert_called_once()
    provider.assert_called_once()
    assert clip.analysis_records["detect_objects"].state == "succeeded"
    payload = window._dispatch_gui_reply.send.call_args.args[1]["result"]
    assert payload["success"] and payload["analyzed_clips"] == 1
    assert payload["total_people_detected"] == 0 and "object_counts" not in payload


@pytest.mark.parametrize("operation", ["classify", "describe"])
def test_partial_summary_does_not_count_old_results_on_failed_clips(operation):
    from ui.workers.standalone_analysis import finish_standalone_analysis

    accepted = Clip(source_id="source", start_frame=0, end_frame=30)
    failed = Clip(source_id="source", start_frame=30, end_frame=60)
    for clip in (accepted, failed):
        clip.description = "A description"
        clip.object_labels = [("house", 0.9)]
    reply = Mock()
    controller = SimpleNamespace(
        owns_project=lambda: True,
        owns_view=lambda: False,
        clips={c.id: c for c in (accepted, failed)},
        plan=SimpleNamespace(clip_ids=[accepted.id, failed.id], operations=[operation]),
        reply=reply,
    )
    window = SimpleNamespace(
        _update_chat_project_state=Mock(),
        _build_agent_analysis_result=lambda clips, ops, message, extra: extra,
    )
    finish_standalone_analysis(
        window,
        controller,
        dict(
            succeeded=[accepted.id],
            failed=[failed.id],
            cancelled=False,
            operations={},
            errors=["provider failed"],
            analyzed_sources=[],
        ),
    )
    payload = reply.send.call_args.args[1]["result"]
    assert payload["total_clips"] == 2
    if operation == "classify":
        assert payload["classified_clips"] == 1
        assert payload["sample_labels"] == [
            {"clip_id": accepted.id, "labels": [("house", 0.9)]}
        ]
    else:
        assert payload["described_clips"] == 1 and payload["error_count"] == 1
        assert payload["success"] and payload["last_error"] == "provider failed"


def test_reentrant_ui_setup_does_not_overwrite_new_request(tmp_path, monkeypatch):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    window = SimpleNamespace(
        project=project,
        _ensure_analysis_operation_available=Mock(return_value=True),
        _on_clip_analysis_progress=Mock(),
        _on_clip_analysis_status=Mock(),
        analyze_tab=Mock(),
        _switch_to_tab=Mock(),
        _gui_state=Mock(),
        progress_bar=Mock(),
    )
    replacement = object()
    window.analyze_tab.add_clips.side_effect = lambda *args: setattr(
        window, "_clip_analysis_controller", replacement
    )
    factory = Mock()
    monkeypatch.setattr("ui.workers.clip_analysis.ClipAnalysisController", factory)
    assert not MainWindow.start_agent_description(window, [clip.id])
    factory.return_value.start.assert_not_called()
    window._switch_to_tab.assert_not_called()
    window.analyze_tab.set_analyzing.assert_not_called()
    assert window._clip_analysis_controller is replacement


@pytest.mark.parametrize(
    "mode", ["older_first", "newer_first", "cancel", "path", "project"]
)
def test_overlapping_requests_keep_native_and_reply_ownership(tmp_path, mode):
    code = r"""
import sys,time,threading
from pathlib import Path
from types import MethodType
from unittest.mock import Mock,patch
from PySide6.QtCore import QObject,QCoreApplication
from core.project import Project
from core.settings import Settings
from models.clip import Clip,Source
from core.operations.description import DescriptionOutcome
from core.operations.shots import ShotTypeOutcome
from ui.workers.description_worker import DescriptionWorker
from ui.workers.shot_type_worker import ShotTypeWorker
from ui.workers.gui_tool_reply import GuiToolReply
from ui.workers.gui_tool_cancellation import cancel_gui_tool_work
from ui.main_window import MainWindow
app=QCoreApplication([]); directory=Path(sys.argv[1]); mode=sys.argv[2]
window=QObject(); window.project=project=Project.new(); window.settings=Settings()
source=Source(file_path=directory/'source.mp4',fps=30); source.file_path.write_bytes(b'video')
clip=Clip(source_id=source.id,start_frame=0,end_frame=30)
clip.thumbnail_path=directory/'thumb.jpg'; clip.thumbnail_path.write_bytes(b'thumb')
project.add_source(source); project.add_clips([clip]); project.save=Mock()
window._chat_worker=Mock(_stop_requested=False)
window._ensure_analysis_operation_available=Mock(return_value=True)
for name in ('analyze_tab','progress_bar','status_bar','collect_tab','_gui_state','_switch_to_tab','_update_chat_project_state'):
    setattr(window,name,Mock())
window._pending_agent_tool_call_id='unrelated'
for name in ('_on_clip_analysis_progress','_on_clip_analysis_status','_build_agent_analysis_result','_build_agent_analysis_summary','_build_agent_clip_context'):
    setattr(window,name,MethodType(getattr(MainWindow,name),window))
entered={name:threading.Event() for name in ('describe','shots')}
release={name:threading.Event() for name in entered}
def describe(self):
    assert self.options.tier=='cloud' and self.options.prompt=='Specific prompt'
    self.result=(DescriptionOutcome(clip.id,'succeeded','A description','model'),)
    self.description_completed.emit(); entered['describe'].set(); assert release['describe'].wait(10)
def shots(self):
    self.result=(ShotTypeOutcome(clip.id,'succeeded','wide',.9),)
    self.analysis_completed.emit(); entered['shots'].set(); assert release['shots'].wait(10)
def pump(until):
    end=time.monotonic()+5
    while not until() and time.monotonic()<end:
        app.processEvents(); time.sleep(.002)
    assert until()
with patch.object(DescriptionWorker,'run',describe), patch.object(ShotTypeWorker,'run',shots):
    window._dispatch_gui_reply=GuiToolReply.capture(window,'describe','description-request')
    assert MainWindow.start_agent_description(window,[clip.id],tier='cloud',prompt='Specific prompt')
    first=window._clip_analysis_controller; first_worker=window.description_worker
    assert entered['describe'].wait(5)
    window._dispatch_gui_reply=GuiToolReply.capture(window,'shots','shot-request')
    assert MainWindow.start_agent_shot_analysis(window,[clip.id])
    second=window._clip_analysis_controller; second_worker=window.shot_type_worker
    assert entered['shots'].wait(5)
    assert not MainWindow.start_agent_shot_analysis(window,[clip.id])
    try:
        for _ in range(10): app.processEvents()
        assert clip.description is None and clip.shot_type is None
        window._chat_worker.set_gui_tool_result.assert_not_called()
        if mode=='cancel': cancel_gui_tool_work(window,name='describe',token='description-request')
        if mode=='path': project.path=directory/'other.json'
        if mode=='project': window.project=Project.new()
        first_name='shots' if mode=='newer_first' else 'describe'
        first_owner=second if first_name=='shots' else first
        release[first_name].set()
        pump(lambda: first_owner.finished)
        if mode in ('older_first','newer_first','cancel'):
            window.analyze_tab.set_analyzing.assert_any_call(True,'shots')
            assert not any(call.args==(False,) for call in window.analyze_tab.set_analyzing.call_args_list)
    finally:
        for event in release.values(): event.set()
        assert first_worker.wait(5000) if not first.finished else True
        assert second_worker.wait(5000) if not second.finished else True
    pump(lambda: first.finished and second.finished)
    assert not window._active_clip_analyses and window._clip_analysis_controller is None
    project.save.assert_not_called()
    assert window._pending_agent_tool_call_id=='unrelated'
    replies={call.args[0]['tool_call_id']:call.args[0] for call in window._chat_worker.set_gui_tool_result.call_args_list}
    if mode=='project':
        assert not replies and clip.description is None and clip.shot_type is None
    else:
        assert set(replies)=={'description-request','shot-request'}
        if mode in ('older_first','newer_first'):
            assert clip.description=='A description' and clip.shot_type=='wide'
            assert replies['description-request']['result']['described_clips']==1
            assert replies['description-request']['result']['sample_descriptions'][0]['description']=='A description'
            assert replies['shot-request']['result']['shot_type_summary']=={'wide':1}
        else:
            assert clip.description is None
            assert replies['description-request']['result']['cancelled']
            if mode=='cancel': assert clip.shot_type=='wide'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
