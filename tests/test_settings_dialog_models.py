"""Settings dialog keeps saved model ids that are no longer in its lists."""

from unittest.mock import patch

import pytest

from core.settings import Settings


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_dialog(settings: Settings):
    from ui.settings_dialog import SettingsDialog

    # The dialog probes every feature runtime on build; those probes import
    # insightface/onnx on a QThread, which aborts the process mid-suite.
    with (
        patch("keyring.get_password", return_value=None),
        patch(
            "ui.widgets.dependency_widgets._check_feature_ready_responsive",
            return_value=(True, []),
        ),
    ):
        return SettingsDialog(settings)


def test_vlm_models_list_claude_5_5():
    from ui.settings_dialog import VLM_MODELS

    claude = [m for m in VLM_MODELS if m.startswith("claude-")]
    assert claude == ["claude-sonnet-5-5", "claude-haiku-5-5"]


def test_unknown_saved_chat_model_is_kept(qapp):
    saved = "claude-sonnet-4-5-20250929"
    settings = Settings(anthropic_model=saved)
    dialog = _make_dialog(settings)

    combo = dialog.anthropic_model_combo
    assert combo.currentData() == saved
    assert combo.currentText() == f"{saved} (saved)"

    with patch("keyring.get_password", return_value=None):
        dialog._save_to_settings()
    assert settings.anthropic_model == saved


def test_known_saved_chat_model_is_not_duplicated(qapp):
    dialog = _make_dialog(Settings(anthropic_model="claude-haiku-5-5"))

    combo = dialog.anthropic_model_combo
    count = combo.count()
    dialog._set_model_combo("anthropic", "claude-haiku-5-5")
    assert combo.currentData() == "claude-haiku-5-5"
    assert combo.count() == count


def test_unknown_saved_vlm_models_are_kept(qapp):
    saved = "claude-sonnet-4-6"
    settings = Settings(
        description_model_cloud=saved,
        cinematography_model=saved,
        text_extraction_vlm_model=saved,
        exquisite_corpus_model=saved,
    )
    dialog = _make_dialog(settings)

    # These combos save currentText(), so the label must be the bare id.
    for combo in (
        dialog.vision_cloud_combo,
        dialog.cine_cloud_combo,
        dialog.text_vlm_combo,
        dialog.corpus_model_combo,
    ):
        assert combo.currentText() == saved

    with patch("keyring.get_password", return_value=None):
        dialog._save_to_settings()
    assert settings.description_model_cloud == saved
    assert settings.cinematography_model == saved
    assert settings.text_extraction_vlm_model == saved
    assert settings.exquisite_corpus_model == saved
