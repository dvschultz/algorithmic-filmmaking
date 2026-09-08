"""Per-query status verifies provenance without hashing or loading inference."""

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from core.analysis_availability import custom_query_is_complete
from core.operations.custom_query import custom_query_record_key
from core.settings import Settings
from tests import test_custom_query_analysis_records

setup = test_custom_query_analysis_records.setup
evaluate = test_custom_query_analysis_records.evaluate


@pytest.mark.parametrize("change", ["none", "other_query", "legacy", "range", "fps", "path", "value", "model", "failure"])
def test_query_completion_tracks_verified_negative_results(setup, monkeypatch, tmp_path, change):
    project, _ = setup
    settings = Settings(cache_dir=tmp_path / "cache", description_model_tier="cloud", description_model_cloud="model")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    assert evaluate(project).status == "succeeded"
    clip, source = project.clips[0], project.sources[0]
    if change == "other_query":
        evaluate(project, "cat")
    elif change == "legacy":
        clip.analysis_records.clear()
    elif change == "range":
        clip.end_frame += 1
    elif change == "fps":
        source.fps += 1
    elif change == "path":
        source.file_path = tmp_path / "replacement.mp4"
        source.file_path.write_bytes(b"replacement")
    elif change == "value":
        clip.custom_queries[-1]["match"] = True
    elif change == "model":
        settings = replace(settings, description_model_cloud="other-model")
    elif change == "failure":
        monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", Mock(side_effect=ValueError("invalid answer")))
        assert evaluate(project, reuse=False).status == "failed"
    blocked = Mock(side_effect=AssertionError("status must not hash or import inference"))
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", blocked)
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", blocked)
    assert custom_query_is_complete(clip, source, "person") == (change in ("none", "other_query"))
    assert not custom_query_is_complete(clip, source, "unasked query")
    blocked.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_status_accepts_negative_queries_but_not_legacy_values(setup, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.analyze import get_analysis_status

    project, _ = setup
    settings = Settings(cache_dir=tmp_path / "cache", description_model_tier="cloud", description_model_cloud="model")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    evaluate(project)
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    status = json.loads(await get_analysis_status(str(path)))
    assert status["analysis"]["custom_queries"] == {"analyzed": 1, "pending": 0}
    project.clips[0].analysis_records.pop(custom_query_record_key("person"))
    assert project.save(path)
    status = json.loads(await get_analysis_status(str(path)))
    assert status["analysis"]["custom_queries"] == {"analyzed": 0, "pending": 1}
    project.session.close()
