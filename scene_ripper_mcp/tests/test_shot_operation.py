"""The legacy synchronous MCP shot tool shares execution and saved-state guards."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from core.project import Project
from models.clip import Clip, Source
from scene_ripper_mcp.tools.analyze import analyze_shots


@pytest.fixture
def project_file(tmp_path):
    image = tmp_path / "thumbnail.png"
    image.write_bytes(b"image")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    project = Project.new()
    project.add_source(
        Source(id="source", file_path=video, fps=24.0, duration_seconds=5.0)
    )
    project.add_clips(
        [
            Clip(
                id="one",
                source_id="source",
                start_frame=0,
                end_frame=24,
                thumbnail_path=image,
                shot_type="close-up",
            ),
            Clip(
                id="two",
                source_id="source",
                start_frame=24,
                end_frame=48,
                thumbnail_path=image,
            ),
            Clip(id="missing", source_id="source", start_frame=48, end_frame=72),
        ]
    )
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    project.close_writer()
    return path, image


def test_legacy_tool_shared_classification_and_response(project_file, monkeypatch):
    from core.operations.shots import run_shot_types

    path, _ = project_file
    compute = Mock(side_effect=[("wide", 0.9), ("unknown", 0.5)])
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    shared = Mock(wraps=run_shot_types)
    monkeypatch.setattr("core.operations.shots.run_shot_types", shared)
    result = json.loads(asyncio.run(analyze_shots(str(path))))
    assert result == {
        "success": True,
        "analyzed_clips": 1,
        "skipped_clips": 2,
        "total_clips": 3,
        "shot_type_distribution": {"wide": 1},
    }
    shared.assert_called_once()
    assert compute.call_count == 2
    saved = json.loads(path.read_text())
    assert saved["clips"][0]["shot_type"] == "wide"
    assert saved["clips"][1].get("shot_type") is None


@pytest.mark.parametrize("failure", ["provider", "media"])
def test_legacy_tool_failure_does_not_save_partial_results(
    project_file, monkeypatch, failure
):
    path, image = project_file
    before = path.read_bytes()
    calls = 0

    def compute(_):
        nonlocal calls
        calls += 1
        if failure == "provider":
            if calls == 1:
                return "wide", 0.9
            raise RuntimeError("provider unavailable")
        image.write_bytes(b"changed image")
        return "wide", 0.9

    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    result = json.loads(asyncio.run(analyze_shots(str(path))))
    assert result["success"] is False
    expected = "provider unavailable" if failure == "provider" else "stale_input"
    assert expected in result["error"]
    assert path.read_bytes() == before
