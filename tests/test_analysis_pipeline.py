"""Regression tests for GUI analysis pipeline orchestration logic."""

from types import SimpleNamespace

import pytest

from ui.main_window import MainWindow


def test_phase_advances_from_local_to_cloud_after_color_completion():
    """Pipeline advances to cloud phase once local color op finishes."""

    class Harness:
        def __init__(self):
            self._analysis_pending_phases = ["local", "cloud"]
            self._analysis_selected_ops = ["colors", "describe"]
            self._analysis_clips = [SimpleNamespace(id="clip-1")]
            self._analysis_current_phase = ""
            self._analysis_phase_remaining = 0
            self._analysis_completed_ops = []
            self.launched_ops = []
            self.pipeline_completed = False

        def _launch_analysis_worker(self, op_key, clips):
            self.launched_ops.append(op_key)
            if op_key == "colors":
                MainWindow._on_analysis_phase_worker_finished(self, "colors")

        def _on_analysis_pipeline_complete(self):
            self.pipeline_completed = True

        def _start_next_analysis_phase(self):
            MainWindow._start_next_analysis_phase(self)

    harness = Harness()

    MainWindow._start_next_analysis_phase(harness)

    assert harness.launched_ops == ["colors", "describe"]
    assert harness._analysis_current_phase == "cloud"
    assert harness._analysis_phase_remaining == 1
    assert harness.pipeline_completed is False


class SignalStub:
    """Minimal signal stub with Qt-like connection behavior."""

    def __init__(self):
        self._callbacks = []

    def connect(self, callback, *connection_type):
        # Simulate Qt behavior that rejects lambda + UniqueConnection.
        if connection_type and getattr(callback, "__name__", "") == "<lambda>":
            raise TypeError("unique lambda connections are invalid")
        self._callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


def _build_fake_worker(completion_signal: str, extra_signals: list[str]):
    """Create a fake worker class that emits completion immediately on start()."""

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.finished = SignalStub()
            setattr(self, completion_signal, SignalStub())
            for signal_name in extra_signals:
                setattr(self, signal_name, SignalStub())

        def deleteLater(self):
            return None

        def start(self):
            getattr(self, completion_signal).emit()
            self.finished.emit()

    return FakeWorker


@pytest.mark.parametrize(
    "launcher,worker_cls,completion_signal,extra_signals,expected_op",
    [
        (
            "_launch_colors_worker",
            "ColorAnalysisWorker",
            "analysis_completed",
            ["progress", "color_ready"],
            "colors",
        ),
        (
            "_launch_shots_worker",
            "ShotTypeWorker",
            "analysis_completed",
            ["progress", "shot_type_ready"],
            "shots",
        ),
        (
            "_launch_classification_worker",
            "ClassificationWorker",
            "classification_completed",
            ["progress", "labels_ready"],
            "classify",
        ),
        (
            "_launch_object_detection_worker",
            "ObjectDetectionWorker",
            "detection_completed",
            ["progress", "objects_ready"],
            "detect_objects",
        ),
        (
            "_launch_description_worker",
            "DescriptionWorker",
            "description_completed",
            ["progress", "description_ready", "error"],
            "describe",
        ),
    ],
)
def test_launch_worker_emits_pipeline_completion(
    monkeypatch,
    launcher,
    worker_cls,
    completion_signal,
    extra_signals,
    expected_op,
):
    """Each pipeline launcher wires worker completion to phase completion."""
    monkeypatch.setattr(
        f"ui.main_window.{worker_cls}",
        _build_fake_worker(completion_signal, extra_signals),
    )

    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace(
                color_analysis_parallelism=2,
                local_model_parallelism=1,
                description_model_tier="cloud",
                description_parallelism=2,
            )
            self.project = SimpleNamespace(sources_by_id={})
            self.sources_by_id = {}
            self.finished_ops = []
            self.color_worker = None
            self.shot_type_worker = None
            self.classification_worker = None
            self.detection_worker_yolo = None
            self.description_worker = None

        def _on_color_progress(self, *_args):
            return None

        def _on_color_ready(self, *_args):
            return None

        def _on_shot_type_progress(self, *_args):
            return None

        def _on_shot_type_ready(self, *_args):
            return None

        def _on_classification_progress(self, *_args):
            return None

        def _on_classification_ready(self, *_args):
            return None

        def _on_object_detection_progress(self, *_args):
            return None

        def _on_objects_ready(self, *_args):
            return None

        def _on_description_progress(self, *_args):
            return None

        def _on_description_ready(self, *_args):
            return None

        def _on_description_error(self, *_args):
            return None

        def _on_analysis_phase_worker_finished(self, op_key):
            self.finished_ops.append(op_key)

        # Named pipeline slots (match MainWindow's named methods)
        def _on_pipeline_colors_finished(self):
            self._on_analysis_phase_worker_finished("colors")

        def _on_pipeline_shots_finished(self):
            self._on_analysis_phase_worker_finished("shots")

        def _on_pipeline_classify_finished(self):
            self._on_analysis_phase_worker_finished("classify")

        def _on_pipeline_detect_objects_finished(self):
            self._on_analysis_phase_worker_finished("detect_objects")

        def _on_pipeline_describe_finished(self):
            self._on_analysis_phase_worker_finished("describe")

    harness = Harness()
    getattr(MainWindow, launcher)(harness, [SimpleNamespace(id="clip-1")])

    assert harness.finished_ops == [expected_op]
