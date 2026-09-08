"""Regression tests for analysis dependency gates in GUI entry points."""

from types import SimpleNamespace

import pytest

from ui.main_window import MainWindow


def test_frame_analysis_skips_blocked_operations(monkeypatch):
    from unittest.mock import Mock

    controller = Mock()
    factory = Mock(return_value=controller)
    monkeypatch.setattr("ui.workers.frame_analysis.FrameAnalysisController", factory)
    harness = SimpleNamespace(
        project=SimpleNamespace(
            session=SimpleNamespace(session_id="session"),
            frames_by_id={"frame-1": object()},
        ),
        progress_bar=Mock(),
        status_bar=Mock(),
        _filter_available_analysis_operations=lambda ops: [
            op for op in ops if op != "shots"
        ],
        _on_frame_analysis_progress=Mock(),
        _on_frame_analysis_finished=Mock(),
    )
    MainWindow._run_frame_analysis(
        harness, [SimpleNamespace(id="frame-1")], ["colors", "shots"]
    )
    factory.assert_called_once_with(harness, ["frame-1"], ["colors"])
    controller.start.assert_called_once()


@pytest.mark.parametrize("change", ["project", "session", "frame"])
def test_frame_gate_does_not_retarget_expired_request(monkeypatch, change):
    from unittest.mock import Mock
    from core.project import Project
    from models.frame import Frame

    factory = Mock()
    monkeypatch.setattr("ui.workers.frame_analysis.FrameAnalysisController", factory)
    project = Project.new()
    project.add_frames([Frame(id="frame-1")])
    harness = SimpleNamespace(
        project=project,
        progress_bar=Mock(),
        status_bar=Mock(),
        _on_frame_analysis_progress=Mock(),
        _on_frame_analysis_finished=Mock(),
    )

    def gate(operations):
        if change == "project":
            harness.project = Project.new()
            harness.project.add_frames([Frame(id="frame-1")])
        elif change == "session":
            project.clear()
            project.add_frames([Frame(id="frame-1")])
        else:
            project.remove_frames(["frame-1"])
            project.add_frames([Frame(id="frame-1")])
        return operations

    harness._filter_available_analysis_operations = gate
    MainWindow._run_frame_analysis(harness, [SimpleNamespace(id="frame-1")], ["shots"])
    factory.assert_not_called()


@pytest.mark.parametrize(
    "method_name,expected_op,kwargs,worker_attr",
    [
        ("start_agent_color_analysis", "colors", {}, "color_worker"),
        ("start_agent_transcription", "transcribe", {}, "transcription_worker"),
        ("start_agent_shot_analysis", "shots", {}, "shot_type_worker"),
        ("start_agent_classification", "classify", {}, "classification_worker"),
        ("start_agent_object_detection", "detect_objects", {}, "detection_worker_yolo"),
        (
            "start_agent_description",
            "describe",
            {"tier": "local"},
            "description_worker",
        ),
    ],
)
def test_single_operation_agent_flows_abort_when_dependency_missing(
    method_name,
    expected_op,
    kwargs,
    worker_attr,
):
    from unittest.mock import Mock
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new()
    source = Source()
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    harness = SimpleNamespace(
        project=project,
        _ensure_analysis_operation_available=Mock(return_value=False),
        analyze_tab=Mock(),
    )
    assert not getattr(MainWindow, method_name)(harness, [clip.id], **kwargs)
    assert harness._ensure_analysis_operation_available.call_args.args[0] == expected_op
    harness.analyze_tab.add_clips.assert_not_called()


def test_intention_shot_analysis_fails_when_dependency_missing():
    from pathlib import Path
    from core.project import Project
    from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
    from models.clip import Source, Clip

    workflow = IntentionWorkflowCoordinator()
    source = Source(id="source", file_path=Path("video.mp4"))
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=30)
    workflow.start("shot_type", [source.file_path], [])
    workflow.on_detection_completed(source, [clip])
    workflow.on_thumbnails_finished()
    results = []
    workflow.workflow_completed.connect(results.append)
    harness = SimpleNamespace(
        intention_workflow=workflow,
        project=Project.new(),
        settings=SimpleNamespace(local_model_parallelism=1),
        shot_type_worker=None,
        _ensure_analysis_operation_available=lambda *_args, **_kwargs: False,
    )
    MainWindow._start_intention_analysis(harness)
    assert workflow.state == WorkflowState.ERROR
    assert results[0].error_message == "Shot analysis is unavailable"
    assert harness.shot_type_worker is None
