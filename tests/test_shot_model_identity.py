"""Shot runtime identity describes the exact local revision and cloud prompt."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("fallback", [False, True])
def test_siglip_loader_uses_one_pinned_revision(monkeypatch, fallback):
    import sys
    from core.analysis import shots
    from core.analysis_model_identity import SIGLIP_REVISION

    processor = Mock()
    model = Mock()
    if fallback:
        processor.from_pretrained.side_effect = [
            RuntimeError("additional_chat_templates 404"),
            object(),
        ]
    snapshot = Mock(return_value="/model/snapshot")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot)
    )
    monkeypatch.setattr(shots, "_model", None)
    monkeypatch.setattr(shots, "_processor", None)
    monkeypatch.setattr(
        shots, "ensure_classification_runtime_available", lambda: (processor, model)
    )
    shots.load_classification_model()
    assert (
        processor.from_pretrained.call_args_list[0].kwargs["revision"]
        == SIGLIP_REVISION
    )
    if fallback:
        assert snapshot.call_args.kwargs["revision"] == SIGLIP_REVISION
        model.from_pretrained.assert_called_once_with(
            "/model/snapshot", local_files_only=True
        )
    else:
        assert model.from_pretrained.call_args.kwargs["revision"] == SIGLIP_REVISION


def test_shot_runtime_includes_model_revision_and_cloud_prompt():
    from core.analysis_model_identity import SIGLIP_REVISION, SHOT_CLOUD_PROMPT
    from core.jobs.shots import _runtime

    runtime = _runtime()
    assert runtime["revision"] == SIGLIP_REVISION
    assert runtime["cloud"]["prompt"] == SHOT_CLOUD_PROMPT
