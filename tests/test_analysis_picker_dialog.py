"""Tests for disabling already-complete operations in AnalysisPickerDialog."""

import pytest

from tests.conftest import make_test_clip
from tests.analysis_fixtures import verify_clip_analysis


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Settings:
    def __init__(self, selected: list[str] | None = None):
        self.analysis_selected_operations = selected or []


def test_dialog_disables_completed_operations_and_ignores_saved_checks(qapp, tmp_path):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
    verify_clip_analysis(clip, tmp_path)
    settings = _Settings(selected=["colors", "shots"])

    dialog = AnalysisPickerDialog(
        clip_count=1,
        scope_label="selected clips",
        settings=settings,
        clips=[clip],
    )

    colors_cb = dialog._checkboxes["colors"]
    shots_cb = dialog._checkboxes["shots"]

    assert colors_cb.isEnabled() is False
    assert colors_cb.isChecked() is False
    assert shots_cb.isEnabled() is True
    assert shots_cb.isChecked() is True


def test_dialog_select_all_skips_disabled_operations(qapp, tmp_path):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
    verify_clip_analysis(clip, tmp_path)
    settings = _Settings(selected=[])

    dialog = AnalysisPickerDialog(
        clip_count=1,
        scope_label="selected clips",
        settings=settings,
        clips=[clip],
    )

    dialog._select_all()
    assert dialog._checkboxes["colors"].isChecked() is False
    assert dialog._checkboxes["shots"].isChecked() is True


def test_force_rerun_enables_completed_operations(qapp, tmp_path):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
    verify_clip_analysis(clip, tmp_path)
    settings = _Settings(selected=[])

    dialog = AnalysisPickerDialog(
        clip_count=1,
        scope_label="selected clips",
        settings=settings,
        clips=[clip],
    )

    colors_cb = dialog._checkboxes["colors"]
    assert colors_cb.isEnabled() is False

    dialog._force_rerun_cb.setChecked(True)
    assert dialog.force_rerun() is True
    assert colors_cb.isEnabled() is True

    colors_cb.setChecked(True)
    assert dialog.selected_operations() == ["colors"]
    assert dialog._run_btn.isEnabled() is True

    dialog._force_rerun_cb.setChecked(False)
    assert colors_cb.isEnabled() is False
    assert colors_cb.isChecked() is False
    assert dialog._run_btn.isEnabled() is False


def test_dialog_run_disabled_when_every_operation_complete(qapp, tmp_path, monkeypatch):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip(
        "c1",
        dominant_colors=[(1, 2, 3)],
        shot_type="wide",
        transcript_text="hello",
        object_labels=["car"],
        detected_objects=[{"label": "car", "confidence": 0.9}],
        description="desc",
    )
    clip.extracted_texts = []
    clip.face_embeddings = [{"bbox": [0, 0, 50, 50], "embedding": [0.1] * 512, "confidence": 0.9}]
    clip.gaze_category = "at_camera"
    clip.embedding = [0.1] * 768  # DINOv2 visual embedding
    clip.first_frame_embedding = [0.1] * 768
    clip.last_frame_embedding = [0.2] * 768
    clip.custom_queries = [{"query": "test", "match": True, "confidence": 0.9, "model": "test"}]
    from core.settings import Settings
    model_settings = Settings(description_model_tier="cloud", description_model_cloud="gpt-test", description_input_mode="frame", cinematography_tier="cloud", cinematography_model="gpt-test", cinematography_input_mode="frame")
    monkeypatch.setattr("core.settings.load_settings", lambda: model_settings)
    source = verify_clip_analysis(clip, tmp_path, embeddings=True, objects=True, ocr=True, classify=True, shots=True, gaze=True, boundary=True, descriptions=True, cinematography=True)

    settings = _Settings(selected=["colors", "shots", "transcribe"])
    dialog = AnalysisPickerDialog(
        clip_count=1,
        scope_label="selected clips",
        settings=settings,
        clips=[clip],
        sources_by_id={source.id: source},
    )

    # custom_query never auto-completes (each query is unique), all others should be disabled
    for key, cb in dialog._checkboxes.items():
        if key == "custom_query":
            assert cb.isEnabled(), "custom_query should always be enabled"
        else:
            assert not cb.isEnabled(), f"{key} should be disabled when complete"


def test_dialog_keeps_legacy_analysis_available_for_recomputation(qapp):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
    clip.embedding = [0.1] * 768
    dialog = AnalysisPickerDialog(clip_count=1, scope_label="selected clips", settings=_Settings(), clips=[clip])
    assert dialog._checkboxes["colors"].isEnabled()
    assert dialog._checkboxes["embeddings"].isEnabled()
