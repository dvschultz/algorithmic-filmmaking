"""Tests for disabling already-complete operations in AnalysisPickerDialog."""

import pytest

from tests.conftest import make_test_clip


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


def test_dialog_disables_completed_operations_and_ignores_saved_checks(qapp):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
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


def test_dialog_select_all_skips_disabled_operations(qapp):
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog

    clip = make_test_clip("c1", dominant_colors=[(10, 20, 30)])
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


def test_dialog_run_disabled_when_every_operation_complete(qapp):
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
    clip.extracted_texts = [object()]
    clip.cinematography = object()
    clip.face_embeddings = [{"bbox": [0, 0, 50, 50], "embedding": [0.1] * 512, "confidence": 0.9}]
    clip.custom_queries = [{"query": "test", "match": True, "confidence": 0.9, "model": "test"}]

    settings = _Settings(selected=["colors", "shots", "transcribe"])
    dialog = AnalysisPickerDialog(
        clip_count=1,
        scope_label="selected clips",
        settings=settings,
        clips=[clip],
    )

    assert all(not cb.isEnabled() for cb in dialog._checkboxes.values())
    assert dialog._run_btn.isEnabled() is False
