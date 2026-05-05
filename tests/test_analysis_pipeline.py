"""Regression tests for GUI analysis pipeline orchestration logic."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.transcription import TranscriptSegment
from models.cinematography import CinematographyAnalysis
from models.clip import ExtractedText, Source
from tests.conftest import make_test_clip
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


def test_pipeline_complete_refreshes_cut_browser_after_analysis():
    """Analysis completion should refresh data-backed clip entries via project.update_clips.

    The single project.update_clips() call replaces the previous duplicate path
    that called update_clips directly on both browsers; both Cut and Analyze
    tabs now refresh through the standard `clips_updated` observer chain.
    """
    clip = make_test_clip("clip-1", source_id="src-1")
    source = Source(id="src-1", file_path=Path("/test/video.mp4"))
    project_updated = []

    class AnalyzeTab:
        clip_browser = SimpleNamespace(update_clips=lambda *_args, **_kwargs: None)

        def set_analyzing(self, _value):
            return None

    class FakeProject:
        def __init__(self):
            self.sources_by_id = {source.id: source}
            self.path = None

        def update_clips(self, clips):
            project_updated.extend(clips)

    class Harness:
        def __init__(self):
            self._analysis_clips = [clip]
            self._analysis_completed_ops = ["describe"]
            self._analysis_selected_ops = ["describe"]
            self._pending_agent_analyze_all = False
            self._chat_worker = None
            self._active_custom_query_text = "query"
            self._gui_state = SimpleNamespace(clear_processing=lambda _name: None)
            self.analyze_tab = AnalyzeTab()
            self.cut_tab = SimpleNamespace(
                clip_browser=SimpleNamespace(
                    update_clips=lambda *_args, **_kwargs: None
                )
            )
            self.progress_bar = SimpleNamespace(setVisible=lambda _value: None)
            self.status_bar = SimpleNamespace(showMessage=lambda _message: None)
            self.project = FakeProject()
            self.collect_tab = SimpleNamespace(update_source_has_analysis=lambda *_args: None)

        def _get_completed_analysis_error_labels(self, _completed):
            return []

        def _update_chat_project_state(self):
            return None

    harness = Harness()

    MainWindow._on_analysis_pipeline_complete(harness)

    # Single project-level update notification (no duplicate per-browser calls).
    assert project_updated == [clip]
    assert source.has_analysis is True


def test_agent_analysis_summary_includes_operation_specific_results():
    clip = make_test_clip(
        "clip-1",
        dominant_colors=[(10, 20, 30), (255, 0, 128)],
        shot_type="close-up",
        object_labels=["eye", "face"],
        detected_objects=[{"label": "person", "confidence": 0.92}],
        person_count=1,
        description="Close-up of an eye",
    )
    clip.face_embeddings = [{"embedding": [0.1, 0.2], "confidence": 0.88, "bbox": [1, 2, 3, 4]}]
    clip.extracted_texts = [
        ExtractedText(frame_number=1, text="LOOK", confidence=0.91, source="vlm")
    ]
    clip.transcript = [
        TranscriptSegment(start_time=0.0, end_time=1.2, text="I see an eye", confidence=0.9)
    ]
    clip.description_model = "qwen3-vl-4b"
    clip.cinematography = CinematographyAnalysis(
        shot_size="CU",
        shot_size_confidence=0.83,
        lighting_style="low_key",
        analysis_model="test-cine",
    )
    clip.custom_queries = [
        {"query": "eye", "match": True, "confidence": 0.93, "model": "qwen3-vl-4b"}
    ]
    clip.gaze_category = "at_camera"
    clip.gaze_yaw = 1.5
    clip.gaze_pitch = -0.5
    clip.embedding = [0.1] * 512
    clip.embedding_model = "dinov2-vit-b-14"
    clip.first_frame_embedding = [0.1] * 512
    clip.last_frame_embedding = [0.2] * 512

    source = Source(id="src-1", file_path=Path("/test/video.mp4"))
    harness = SimpleNamespace(
        project=SimpleNamespace(sources_by_id={source.id: source}),
        _active_custom_query_text="eye",
    )
    harness._build_agent_clip_context = lambda clip: MainWindow._build_agent_clip_context(
        harness, clip
    )
    harness._build_custom_query_agent_summary = (
        lambda clips, query: MainWindow._build_custom_query_agent_summary(
            harness, clips, query
        )
    )
    harness._rgb_to_hex = MainWindow._rgb_to_hex
    harness._truncate_for_agent = MainWindow._truncate_for_agent
    harness._summarize_detections_for_agent = MainWindow._summarize_detections_for_agent
    harness._build_agent_analysis_summary = (
        lambda clips, ops: MainWindow._build_agent_analysis_summary(harness, clips, ops)
    )

    results = MainWindow._build_agent_analysis_summary(
        harness,
        [clip],
        [
            "colors",
            "shots",
            "classify",
            "detect_objects",
            "face_embeddings",
            "extract_text",
            "transcribe",
            "describe",
            "cinematography",
            "custom_query",
            "gaze",
            "embeddings",
        ],
    )

    assert results["colors"]["clips"][0]["dominant_colors_hex"] == ["#0a141e", "#ff0080"]
    assert results["shots"]["distribution"] == {"close-up": 1}
    assert results["classify"]["clips"][0]["labels"] == ["eye", "face"]
    assert results["detect_objects"]["total_people"] == 1
    assert results["detect_objects"]["clips"][0]["objects"] == [
        {"label": "person", "confidence": 0.92}
    ]
    assert results["face_embeddings"]["total_faces"] == 1
    assert results["extract_text"]["clips"][0]["text"] == "LOOK"
    assert results["transcribe"]["clips"][0]["transcript_excerpt"] == "I see an eye"
    assert results["describe"]["clips"][0]["description"] == "Close-up of an eye"
    assert results["describe"]["clips"][0]["model"] == "qwen3-vl-4b"
    assert results["cinematography"]["clips"][0]["cinematography"]["shot_size"] == "CU"
    assert results["custom_query"]["matched_count"] == 1
    assert results["gaze"]["distribution"] == {"at_camera": 1}
    assert results["embeddings"]["clips"][0]["embedding_dimensions"] == 512
    assert results["embeddings"]["clips"][0]["has_boundary_embeddings"] is True
    assert "Do not invent" in results["response_guidance"]


def test_agent_analysis_result_wraps_single_operation_structured_results():
    clip = make_test_clip(
        "clip-1",
        dominant_colors=[(10, 20, 30)],
        description="A clipped result",
    )
    source = Source(id="src-1", file_path=Path("/test/video.mp4"))
    harness = SimpleNamespace(
        project=SimpleNamespace(sources_by_id={source.id: source}),
        _active_custom_query_text=None,
    )
    harness._build_agent_clip_context = lambda clip: MainWindow._build_agent_clip_context(
        harness, clip
    )
    harness._build_custom_query_agent_summary = (
        lambda clips, query: MainWindow._build_custom_query_agent_summary(
            harness, clips, query
        )
    )
    harness._rgb_to_hex = MainWindow._rgb_to_hex
    harness._truncate_for_agent = MainWindow._truncate_for_agent
    harness._summarize_detections_for_agent = MainWindow._summarize_detections_for_agent
    harness._build_agent_analysis_summary = (
        lambda clips, ops: MainWindow._build_agent_analysis_summary(harness, clips, ops)
    )

    result = MainWindow._build_agent_analysis_result(
        harness,
        [clip],
        ["colors"],
        "Extracted colors from 1 clips",
    )

    assert result["success"] is True
    assert result["operations_completed"] == ["colors"]
    assert result["analysis_results"]["colors"]["clips"][0]["dominant_colors_hex"] == [
        "#0a141e"
    ]
    assert "Do not invent" in result["analysis_results"]["response_guidance"]


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
            ["progress", "color_ready", "error"],
            "colors",
        ),
        (
            "_launch_shots_worker",
            "ShotTypeWorker",
            "analysis_completed",
            ["progress", "shot_type_ready", "error"],
            "shots",
        ),
        (
            "_launch_classification_worker",
            "ClassificationWorker",
            "classification_completed",
            ["progress", "labels_ready", "error"],
            "classify",
        ),
        (
            "_launch_object_detection_worker",
            "ObjectDetectionWorker",
            "detection_completed",
            ["progress", "objects_ready", "error"],
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

        def _on_color_error(self, *_args):
            return None

        def _on_shot_type_progress(self, *_args):
            return None

        def _on_shot_type_ready(self, *_args):
            return None

        def _on_shot_type_error(self, *_args):
            return None

        def _on_classification_progress(self, *_args):
            return None

        def _on_classification_ready(self, *_args):
            return None

        def _on_classification_error(self, *_args):
            return None

        def _on_object_detection_progress(self, *_args):
            return None

        def _on_objects_ready(self, *_args):
            return None

        def _on_object_detection_error(self, *_args):
            return None

        def _on_description_progress(self, *_args):
            return None

        def _on_description_ready(self, *_args):
            return None

        def _on_description_error(self, *_args):
            return None

        def _reset_description_run_errors(self):
            self._description_run_error = None
            self._description_run_errors = []

        def _reset_analysis_run_error(self, *_args):
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


def test_shot_type_error_handler_records_reason():
    messages = []

    class Harness:
        def __init__(self):
            self._shot_type_run_error = None
            self._gui_state = SimpleNamespace(set_last_error=lambda value: messages.append(("last_error", value)))
            self.status_bar = SimpleNamespace(showMessage=lambda text, timeout=0: messages.append(("status", text, timeout)))

    harness = Harness()
    MainWindow._on_shot_type_error(harness, "clip-1: torch import failed")

    assert harness._shot_type_run_error == "clip-1: torch import failed"
    assert ("last_error", "Shot type classification error: clip-1: torch import failed") in messages
    assert ("status", "Shot type classification finished with errors", 5000) in messages


def test_description_error_handler_records_first_error():
    messages = []

    class Harness:
        def __init__(self):
            self._description_run_error = None
            self._description_run_errors = []
            self._gui_state = SimpleNamespace(
                set_last_error=lambda value: messages.append(("last_error", value))
            )
            self.status_bar = SimpleNamespace(
                showMessage=lambda text, timeout=0: messages.append(("status", text, timeout))
            )

        def _summarize_description_errors(self):
            return MainWindow._summarize_description_errors(self)

    harness = Harness()
    MainWindow._on_description_error(harness, "clip-1", "401 Unauthorized")

    assert harness._description_run_error == "clip-1: 401 Unauthorized"
    assert (
        "last_error",
        "Description error: clip-1: 401 Unauthorized",
    ) in messages
    assert ("status", "Description generation finished with errors", 5000) in messages


def test_description_error_handler_summarizes_multiple_failures():
    messages = []

    class Harness:
        def __init__(self):
            self._description_run_error = None
            self._description_run_errors = []
            self._gui_state = SimpleNamespace(
                set_last_error=lambda value: messages.append(("last_error", value))
            )
            self.status_bar = SimpleNamespace(
                showMessage=lambda text, timeout=0: messages.append(("status", text, timeout))
            )

        def _summarize_description_errors(self):
            return MainWindow._summarize_description_errors(self)

    harness = Harness()
    MainWindow._on_description_error(harness, "clip-1", "401 Unauthorized")
    MainWindow._on_description_error(harness, "clip-2", "429 Too Many Requests")

    assert "Description failed for 2 clips" in harness._description_run_error
    assert "- clip-1: 401 Unauthorized" in harness._description_run_error
    assert "- clip-2: 429 Too Many Requests" in harness._description_run_error
    assert messages.count(("status", "Description generation finished with errors", 5000)) == 2


def test_analysis_error_summary_dialog_is_aggregated(monkeypatch):
    messages = []

    class Harness:
        _color_run_error = None
        _shot_type_run_error = None
        _classification_run_error = None
        _object_detection_run_error = None
        _text_extraction_run_error = None
        _transcription_run_error = "FFmpeg is required for transcription"
        _description_run_error = "clip-1: 401 Unauthorized"
        _cinematography_run_error = None

        def _get_completed_analysis_error_details(self, completed_ops):
            return MainWindow._get_completed_analysis_error_details(self, completed_ops)

    monkeypatch.setattr(
        "ui.main_window.QMessageBox.warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    harness = Harness()
    MainWindow._show_completed_analysis_error_dialog(
        harness,
        ["transcribe", "describe"],
    )

    assert len(messages) == 1
    title, message = messages[0]
    assert title == "Analysis Finished With Errors"
    assert "Transcription:" in message
    assert "FFmpeg is required for transcription" in message
    assert "Description:" in message
    assert "clip-1: 401 Unauthorized" in message
