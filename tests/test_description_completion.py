"""Completion indicators require current provenance without loading inference runtimes."""

from dataclasses import replace

import pytest

from core.analysis_availability import compute_disabled_operations
from core.operations.description import (
    DescriptionApplication,
    DescriptionOptions,
    description_task,
    run_description,
)
from core.settings import Settings
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def analyzed(tmp_path, monkeypatch):
    settings = Settings(
        description_model_tier="cloud",
        description_model_cloud="gemini-test",
        description_input_mode="frame",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *a, **kw: ("Generated", "gemini-test"),
    )
    task = description_task(project.clips[0], project.sources[0])
    options = DescriptionOptions("cloud", model="gemini-test", input_mode="frame")
    outcome = run_description((task,), options)[0]
    assert DescriptionApplication(project, (task,), options).apply(project, outcome)
    return project, settings


def complete(project):
    return "describe" in compute_disabled_operations(
        project.clips, ["describe"], sources_by_id=project.sources_by_id
    )


def test_verified_default_description_is_complete(analyzed):
    project, _ = analyzed
    assert complete(project)


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "failed",
        "model",
        "input_mode",
        "tier",
        "fps",
        "source",
        "image",
        "text",
        "prompt",
    ],
)
def test_changed_description_stays_available(analyzed, change, tmp_path):
    project, settings = analyzed
    clip = project.clips[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "failed":
        clip.analysis_records["describe"] = replace(
            clip.analysis_records["describe"], state="failed", error="Failed"
        )
    elif change == "model":
        settings.description_model_cloud = "different"
    elif change == "input_mode":
        settings.description_input_mode = "video"
    elif change == "tier":
        settings.description_model_tier = "local"
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "source":
        project.sources[0].file_path = tmp_path / "other.mp4"
    elif change == "image":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "text":
        clip.description = "Edited"
    else:
        from models.analysis_record import AnalysisIdentity

        record = clip.analysis_records["describe"]
        data = record.identity.to_dict()
        data["prompt_sha256"] = "0" * 64
        clip.analysis_records["describe"] = replace(
            record, identity=AnalysisIdentity.from_dict(data)
        )
    assert not complete(project)


def test_unknown_source_binding_cannot_claim_completion(analyzed):
    project, _ = analyzed
    assert "describe" not in compute_disabled_operations(project.clips, ["describe"])


def test_completion_does_not_probe_provider_imports(analyzed, monkeypatch):
    project, _ = analyzed
    monkeypatch.setattr(
        "core.analysis.description.is_mlx_vlm_available",
        lambda: pytest.fail("Provider import probe on UI thread"),
    )
    assert complete(project)


def test_runtime_version_change_invalidates_completion(analyzed, monkeypatch):
    project, _ = analyzed
    monkeypatch.setattr(
        "core.operations.description.model_runtime",
        lambda *args: {"packages": {"changed": "version"}},
    )
    assert not complete(project)


def test_local_completion_uses_only_known_backend(analyzed, monkeypatch):
    project, settings = analyzed
    settings.description_model_tier = "local"
    settings.description_model_local = "moondream-test"
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    monkeypatch.setattr(
        "core.analysis.description._load_local_model", lambda *args: None
    )
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *a, **kw: ("Local text", "moondream-test"),
    )
    task = description_task(project.clips[0], project.sources[0], skip_existing=False)
    options = DescriptionOptions("local", model="moondream-test", input_mode="frame")
    result = run_description((task,), options)[0]
    assert DescriptionApplication(project, (task,), options).apply(project, result)
    monkeypatch.setattr(
        "core.analysis.description.is_mlx_vlm_available",
        lambda: pytest.fail("Import probe on UI thread"),
    )
    monkeypatch.setattr(
        "core.analysis_model_identity.known_mlx_vlm_availability", lambda: False
    )
    assert complete(project)
    monkeypatch.setattr(
        "core.analysis_model_identity.known_mlx_vlm_availability", lambda: None
    )
    assert not complete(project)


@pytest.mark.parametrize("surface", ["picker", "quick_run"])
def test_controls_use_current_source_lookup(analyzed, surface):
    from PySide6.QtWidgets import QApplication
    from ui.dialogs.analysis_picker_dialog import AnalysisPickerDialog
    from ui.tabs.analyze_tab import AnalyzeTab

    app = QApplication.instance() or QApplication([])
    project, settings = analyzed
    if surface == "picker":
        widget = AnalysisPickerDialog(
            1,
            "selected",
            settings,
            clips=project.clips,
            sources_by_id=project.sources_by_id,
        )
        assert not widget._checkboxes["describe"].isEnabled()
        widget.close()
        project.sources[0].fps += 1
        widget = AnalysisPickerDialog(
            1,
            "selected",
            settings,
            clips=project.clips,
            sources_by_id=project.sources_by_id,
        )
        assert widget._checkboxes["describe"].isEnabled()
    else:
        widget = AnalyzeTab()
        widget.set_lookups(project.clips_by_id, project.sources_by_id)
        widget.add_clips([project.clips[0].id])
        row = widget.quick_run_combo.findData("describe")
        assert not widget.quick_run_combo.model().item(row).isEnabled()
        project.sources[0].fps += 1
        widget.set_lookups(project.clips_by_id, project.sources_by_id)
        assert widget.quick_run_combo.model().item(row).isEnabled()
    widget.close()
    app.processEvents()
