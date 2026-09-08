"""Regression tests for GUI analysis pipeline orchestration logic."""

from pathlib import Path
from types import SimpleNamespace


from core.transcription import TranscriptSegment
from models.cinematography import CinematographyAnalysis
from models.clip import ExtractedText, Source
from tests.conftest import make_test_clip
from ui.main_window import MainWindow


def test_color_result_ignores_a_replaced_project(tmp_path):
    from unittest.mock import patch

    from core.operations.colors import ColorApplication, color_request, compute_colors
    from core.project import Project
    from tests.test_spine_analyze import _build_project

    original = _build_project(tmp_path, 1)
    request = color_request(original)
    application = ColorApplication(original, request)
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        result = compute_colors(request)
    harness = SimpleNamespace(project=Project.new())
    MainWindow._on_color_result(harness, application, result)
    assert original.clips[0].dominant_colors is None


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
