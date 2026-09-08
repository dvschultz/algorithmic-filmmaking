"""Face cost estimates use verified completion and requested sampling."""

import pytest
from ui.dialogs.rose_hobart_dialog import RoseHobartDialog
from core.cost_estimates import estimate_sequence_cost
from tests.test_face_records import setup as face_setup, run  # noqa: F401


@pytest.fixture
def analyzed(request):
    project, provider, directory = request.getfixturevalue("face_setup")
    run(project)
    return project, provider, directory


@pytest.mark.parametrize(
    "change",
    ["none", "empty", "legacy", "media", "weights", "interval", "missing_source"],
)
def test_face_estimate_tracks_verified_inputs(analyzed, change):
    project, provider, directory = analyzed
    clip, source = project.clips[0], project.sources[0]
    if change == "empty":
        original = provider.side_effect
        provider.side_effect = lambda **kwargs: original(**kwargs) and []
        run(project, skip=False)
    elif change == "legacy":
        clip.analysis_records.clear()
    elif change == "media":
        source.file_path.write_bytes(b"changed")
    elif change == "weights":
        (directory / "recognition.onnx").write_bytes(b"changed")
    estimates = estimate_sequence_cost(
        "rose_hobart",
        project.clips,
        override_required=["face_embeddings"],
        sources_by_id={} if change == "missing_source" else project.sources_by_id,
        face_sample_interval=0.5 if change == "interval" else 1.0,
    )
    if change in ("none", "empty"):
        assert estimates == []
    else:
        assert len(estimates) == 1
        assert estimates[0].clips_needing == 1


def test_dialog_refreshes_estimate_for_requested_interval(analyzed, monkeypatch):
    from unittest.mock import Mock

    project, _, _ = analyzed
    blocked = Mock(
        side_effect=AssertionError("cost checks must not hash or load models")
    )
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", blocked)
    monkeypatch.setattr("core.analysis.faces._load_insightface", blocked)
    dialog = RoseHobartDialog(project.clips, project.sources_by_id, project=project)
    assert dialog.cost_panel._estimates == []
    dialog.sample_spin.setValue(0.5)
    assert dialog.cost_panel._estimates[0].clips_needing == 1
    dialog.sample_spin.setValue(1.0)
    assert dialog.cost_panel._estimates == []
    blocked.assert_not_called()
    dialog.reject()
