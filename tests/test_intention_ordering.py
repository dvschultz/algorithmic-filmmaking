"""Intention transitions must respect prerequisites and cancellation."""

from pathlib import Path

import pytest

from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
from models.clip import Clip, Source


def ready_for_analysis(algorithm="shot_type"):
    workflow = IntentionWorkflowCoordinator()
    source = Source(id="source", file_path=Path("video.mp4"))
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=30)
    workflow.start(algorithm, [source.file_path], [])
    workflow.on_detection_completed(source, [clip])
    workflow.on_thumbnails_finished()
    return workflow, clip


@pytest.mark.parametrize("callback", ["on_thumbnails_finished", "on_analysis_finished"])
def test_out_of_order_completion_cannot_skip_detection(callback):
    workflow = IntentionWorkflowCoordinator()
    workflow.start("shot_type", [Path("video.mp4")], [])
    getattr(workflow, callback)()
    assert workflow.state == WorkflowState.DETECTING


def test_cancel_from_step_completed_does_not_start_next_step():
    workflow, clip = ready_for_analysis()
    clip.shot_type = "wide"
    started = []
    workflow.step_started.connect(lambda name, *_: started.append(name))
    workflow.step_completed.connect(lambda _: workflow.cancel())
    workflow.on_analysis_finished()
    assert workflow.state == WorkflowState.CANCELLED
    assert started == []


def test_metadata_removed_during_transition_blocks_building():
    workflow, clip = ready_for_analysis()
    clip.shot_type = "wide"
    workflow.step_completed.connect(lambda _: setattr(clip, "shot_type", None))
    workflow.on_analysis_finished()
    assert workflow.state == WorkflowState.ERROR


@pytest.mark.parametrize("algorithm", ["color", "shot_type", "storyteller"])
def test_analysis_completion_requires_actual_metadata(algorithm):
    workflow, _ = ready_for_analysis(algorithm)
    results = []
    workflow.workflow_completed.connect(results.append)
    workflow.on_analysis_finished()
    assert workflow.state == WorkflowState.ERROR
    assert len(results) == 1 and not results[0].success


def test_duplicate_build_completion_emits_one_terminal_result():
    workflow, clip = ready_for_analysis("shuffle")
    results = []
    workflow.workflow_completed.connect(results.append)
    workflow.on_building_complete([clip])
    workflow.on_building_complete([clip])
    assert len(results) == 1


def test_failed_workflow_is_not_finalized_as_success(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from core.intention_workflow import WorkflowResult
    from ui.main_window import MainWindow

    scheduled = Mock()
    monkeypatch.setattr("ui.main_window.QTimer.singleShot", scheduled)
    window = SimpleNamespace(
        intention_import_dialog=Mock(),
        _on_intention_workflow_error=Mock(),
        _finalize_intention_workflow=Mock(),
    )
    MainWindow._on_intention_workflow_completed(
        window, WorkflowResult(False, "shot_type", 1, 1, 0, "Missing shot analysis")
    )
    scheduled.assert_not_called()
    window.intention_import_dialog.set_complete.assert_not_called()
    window._on_intention_workflow_error.assert_called_once_with("Missing shot analysis")


@pytest.mark.parametrize("change", ["project", "session", "plan", "workflow"])
def test_delayed_finalization_keeps_original_owner(monkeypatch, change):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from core.project import Project
    from core.intention_workflow import WorkflowResult
    from ui.main_window import MainWindow

    workflow, clip = ready_for_analysis("shuffle")
    workflow.on_building_complete([clip])
    project = Project.new()
    window = SimpleNamespace(
        project=project, intention_workflow=workflow, intention_import_dialog=Mock()
    )
    window._finalize_intention_workflow = (
        lambda *args: MainWindow._finalize_intention_workflow(window, *args)
    )
    scheduled = Mock()
    monkeypatch.setattr("ui.main_window.QTimer.singleShot", scheduled)
    MainWindow._on_intention_workflow_completed(
        window, WorkflowResult(True, "shuffle", 1, 1, 0)
    )
    if change == "project":
        window.project = Project.new()
    elif change == "session":
        project.clear()
    elif change == "plan":
        workflow.start("shuffle", [Path("new.mp4")], [])
    else:
        window.intention_workflow = IntentionWorkflowCoordinator()
    scheduled.call_args.args[1]()
    window.intention_import_dialog.hide.assert_not_called()


def test_partial_downloads_are_counted_once():
    from types import SimpleNamespace

    workflow = IntentionWorkflowCoordinator()
    urls = ["https://example.com/one", "https://example.com/two"]
    workflow.start("shuffle", [Path("local.mp4")], urls)
    success = SimpleNamespace(success=True, file_path=Path("download.mp4"))
    failure = SimpleNamespace(success=False, file_path=None, error="Failed")
    workflow.on_download_video_finished(urls[0], success)
    workflow.on_download_video_finished(urls[0], success)
    workflow.on_download_video_finished(urls[1], failure)
    workflow.on_download_video_finished(urls[1], failure)
    workflow.on_download_all_finished([success, failure])
    assert workflow.get_sources_to_detect() == [Path("download.mp4"), Path("local.mp4")]
    source = Source(id="download", file_path=Path("download.mp4"))
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=30)
    workflow.on_detection_completed(source, [clip])
    workflow.on_detection_completed(source, [clip])
    assert workflow.get_current_source_path() == Path("local.mp4")
    workflow.on_detection_error("Local source failed")
    workflow.on_thumbnails_finished()
    results = []
    workflow.workflow_completed.connect(results.append)
    workflow.on_building_complete([clip])
    assert results[0].sources_processed == 1 and results[0].sources_failed == 2


def test_download_aliases_and_local_input_queue_one_source(tmp_path):
    from types import SimpleNamespace

    media = tmp_path / "video.mp4"
    urls = ["https://example.com/one", "https://example.com/alias"]
    workflow = IntentionWorkflowCoordinator()
    workflow.start("shuffle", [media], urls)
    result = SimpleNamespace(success=True, file_path=media)
    for url in urls:
        workflow.on_download_video_finished(url, result)
    workflow.on_download_all_finished([result, result])
    assert workflow.get_sources_to_detect() == [media]


@pytest.mark.parametrize("algorithm", ["shot_type", "storyteller"])
@pytest.mark.parametrize("change", ["project", "session", "run", "cancel"])
def test_analysis_gate_does_not_retarget_request(algorithm, change):
    from types import SimpleNamespace
    from core.project import Project
    from ui.main_window import MainWindow

    workflow, _ = ready_for_analysis(algorithm)
    project = Project.new()
    results = []
    workflow.workflow_completed.connect(results.append)
    window = SimpleNamespace(
        project=project,
        intention_workflow=workflow,
        settings=SimpleNamespace(description_model_tier="cloud"),
    )

    def gate(*args, **kwargs):
        if change == "project":
            window.project = Project.new()
        elif change == "session":
            project.clear()
        elif change == "run":
            workflow.cancel()
            workflow.start("shuffle", [Path("new.mp4")], [])
        else:
            workflow.cancel()
        return False

    window._ensure_analysis_operation_available = gate
    MainWindow._start_intention_analysis(window)
    assert results == []


@pytest.mark.parametrize("algorithm", ["color", "shot_type", "storyteller", "shuffle"])
def test_successful_plan_runs_each_step_once(algorithm):
    workflow, clip = ready_for_analysis(algorithm)
    if algorithm == "color":
        clip.dominant_colors = []
    elif algorithm == "shot_type":
        clip.shot_type = "wide"
    elif algorithm == "storyteller":
        clip.description = "A person walking."
    workflow.on_analysis_finished()
    assert workflow.state == WorkflowState.BUILDING
    workflow.on_building_complete([clip])
    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan.completed == [step.name for step in workflow.plan.steps]
