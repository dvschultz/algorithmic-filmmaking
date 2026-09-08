"""MCP analysis status counts current verified results, including empty output."""

import json
from unittest.mock import Mock

import pytest

from core.project import Project
from core.settings import Settings
from models.clip import Clip, Source
from scene_ripper_mcp.tools.analyze import get_analysis_status
from tests.analysis_fixtures import verify_clip_analysis


@pytest.mark.asyncio
@pytest.mark.parametrize("key,options", [
    ("colors", {}), ("shots", {"shots": True}),
    ("classification", {"classify": True}), ("objects", {"objects": True}),
    ("descriptions", {"descriptions": True}), ("text", {"ocr": True}),
    ("gaze", {"gaze": True}), ("embeddings", {"embeddings": True}),
    ("boundary_embeddings", {"boundary": True}),
])
async def test_mcp_counts_verified_current_results(tmp_path, monkeypatch, key, options):
    settings = Settings(cache_dir=tmp_path / "cache", description_model_tier="cloud", description_model_cloud="gpt-test")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    project = Project(sources=[source], clips=[clip])
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", Mock(side_effect=AssertionError("status must not hash media")))
    result = json.loads(await get_analysis_status(str(path)))
    assert result["success"], result
    assert result["analysis"][key]["analyzed"] == 1
    assert result["analysis"][key]["pending"] == 0
    clip.end_frame += 1
    assert project.save(path)
    stale = json.loads(await get_analysis_status(str(path)))
    assert stale["analysis"][key]["analyzed"] == 0
    assert stale["analysis"][key]["pending"] == 1
    if key == "shots":
        assert stale["analysis"][key]["distribution"] == {}
    project.session.close()


@pytest.mark.asyncio
async def test_mcp_does_not_promote_legacy_fields_to_verified_status(tmp_path):
    clip = Clip(
        source_id="source", start_frame=0, end_frame=30,
        dominant_colors=[(1, 2, 3)], shot_type="wide shot",
        object_labels=["car"], detected_objects=[{"label": "car"}],
        description="legacy", gaze_category="center", embedding=[0.1] * 768,
        custom_queries=[{"query": "person", "match": True}], tags=["keep"], notes="Keep this edit",
    )
    source = Source(id=clip.source_id, file_path=tmp_path / "source.mp4", fps=30)
    source.file_path.write_bytes(b"source")
    project = Project(sources=[source], clips=[clip])
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    result = json.loads(await get_analysis_status(str(path)))
    assert result["success"], result
    for key in ("colors", "shots", "classification", "objects", "descriptions", "gaze", "embeddings", "custom_queries"):
        assert result["analysis"][key]["analyzed"] == 0
    assert result["metadata"] == {"clips_with_tags": 1, "clips_with_notes": 1}
    project.session.close()


@pytest.mark.asyncio
async def test_unknown_custom_query_record_is_preserved_and_pending(tmp_path):
    from models.analysis_record import UnreadableAnalysisRecord

    source = Source(id="source", file_path=tmp_path / "source.mp4", fps=30)
    source.file_path.write_bytes(b"source")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    raw = {"version": 999, "future_data": "preserve me"}
    clip.analysis_records["custom_query:future"] = UnreadableAnalysisRecord(json.dumps(raw))
    project = Project(sources=[source], clips=[clip])
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    result = json.loads(await get_analysis_status(str(path)))
    assert result["success"], result
    assert result["analysis"]["custom_queries"] == {"analyzed": 0, "pending": 1}
    assert json.loads(path.read_text())["clips"][0]["analysis_records"]["custom_query:future"] == raw
    project.session.close()
