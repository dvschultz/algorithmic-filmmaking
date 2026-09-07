"""Regression tests for analysis dependency gates in GUI entry points."""

from types import SimpleNamespace

import pytest

from ui.main_window import MainWindow


def test_pipeline_skips_blocked_operations_and_runs_remaining():
    class Harness:
        def __init__(self):
            self._gui_state = SimpleNamespace(set_processing=lambda *_args: None)
            self.analyze_tab = SimpleNamespace(set_analyzing=lambda *_args: None)
            self.project = SimpleNamespace(
                session=SimpleNamespace(session_id="session")
            )
            self.started = False

        def _filter_available_analysis_operations(self, operations, **_kwargs):
            return [op for op in operations if op != "shots"]

        def _start_next_analysis_phase(self):
            self.started = True

        def _reset_analysis_run_error(self, _op_key):
            pass

    harness = Harness()

    MainWindow._run_analysis_pipeline(
        harness,
        [SimpleNamespace(id="clip-1")],
        ["colors", "shots"],
    )

    assert harness.started is True
    assert harness._analysis_selected_ops == ["colors"]
    assert harness._analysis_pending_phases == ["local"]


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
    class Harness:
        def __init__(self):
            clip = SimpleNamespace(id="clip-1")
            self.project = SimpleNamespace(clips_by_id={"clip-1": clip})
            self.shot_type_worker = None
            self.classification_worker = None
            self.detection_worker_yolo = None
            self.description_worker = None
            self.analyze_tab = SimpleNamespace(
                add_clips=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("should not add clips")
                ),
                set_analyzing=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("should not set analyzing")
                ),
            )
            self.progress_bar = SimpleNamespace(
                setVisible=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("should not show progress")
                ),
                setRange=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("should not set range")
                ),
            )
            self.status_bar = SimpleNamespace(
                showMessage=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("should not show status")
                )
            )

        def _ensure_analysis_operation_available(self, op_key, **_kwargs):
            assert op_key == expected_op
            return False

    harness = Harness()

    started = getattr(MainWindow, method_name)(harness, ["clip-1"], **kwargs)

    assert started is False
    assert getattr(harness, worker_attr) is None


def test_start_agent_transcription_aborts_when_dependency_missing():
    class Harness:
        def _ensure_analysis_operation_available(self, op_key, **_kwargs):
            assert op_key == "transcribe"
            return False

    started = MainWindow.start_agent_transcription(Harness(), ["clip-1"])

    assert started is False


def test_launch_transcription_worker_skips_runtime_broken_backend(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace()
            self.status = []
            self.finished = []

            self.status_bar = SimpleNamespace(
                showMessage=lambda message: self.status.append(message)
            )

        def _on_analysis_phase_worker_finished(self, op_key):
            self.finished.append(op_key)

    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["transcribe"],
    )
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature: (False, ["runtime:broken"]),
    )

    harness = Harness()

    MainWindow._launch_transcription_worker(
        harness, [SimpleNamespace(source_id="source-1")]
    )

    assert harness.finished == ["transcribe"]
    assert harness.status == [
        "Transcription unavailable - install dependencies in Settings > Dependencies"
    ]


def test_intention_shot_analysis_finishes_when_dependency_missing():
    class Workflow:
        def __init__(self):
            self.finished = False

        def get_all_clips(self):
            return [SimpleNamespace(id="clip-1", shot_type=None)]

        def get_algorithm_with_direction(self):
            return "shot_type", None

        def on_analysis_finished(self):
            self.finished = True

    class Harness:
        def __init__(self):
            self.intention_workflow = Workflow()
            self.project = SimpleNamespace(sources_by_id={})
            self.settings = SimpleNamespace(local_model_parallelism=1)
            self.shot_type_worker = None

        def _ensure_analysis_operation_available(self, op_key, **_kwargs):
            assert op_key == "shots"
            return False

    harness = Harness()

    MainWindow._start_intention_analysis(harness)

    assert harness.intention_workflow.finished is True
    assert harness.shot_type_worker is None
