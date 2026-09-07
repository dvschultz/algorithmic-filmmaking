"""The legacy synchronous MCP shot tool shares execution and saved-state guards."""

import asyncio
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from core.project import Project
from models.clip import Clip, Source
from scene_ripper_mcp.tools.analyze import analyze_shots


@pytest.fixture
def project_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
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
    monkeypatch.setattr("core.jobs.shots.run_shot_types", shared)
    result = json.loads(asyncio.run(analyze_shots(str(path))))
    assert result == {
        "success": True,
        "analyzed_clips": 1,
        "skipped_clips": 2,
        "total_clips": 3,
        "shot_type_distribution": {"wide": 1},
    }
    assert shared.called
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


@pytest.mark.parametrize("failure", ["save", "checkpoint"])
def test_legacy_retry_reuses_inference_after_persistence_failure(
    project_file, monkeypatch, failure
):
    from core.jobs.store import JobStore

    path, _ = project_file
    compute = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    with monkeypatch.context() as patch:
        if failure == "save":
            patch.setattr(
                "core.spine.project_io.save_with_mtime_check",
                Mock(side_effect=OSError("disk full")),
            )
            patch.setattr(
                "core.jobs.commits.save_with_mtime_check",
                Mock(side_effect=OSError("disk full")),
            )
        else:
            patch.setattr(
                JobStore, "checkpoint_results", Mock(side_effect=OSError("checkpoint"))
            )
        first = json.loads(asyncio.run(analyze_shots(str(path))))
    assert first["success"] is False
    compute.side_effect = AssertionError("Must reuse completed inference")
    second = json.loads(asyncio.run(analyze_shots(str(path))))
    assert second == {
        "success": True,
        "analyzed_clips": 2,
        "skipped_clips": 1,
        "total_clips": 3,
        "shot_type_distribution": {"wide": 2},
    }
    assert compute.call_count == 2
    saved = Project.load(path)
    store = JobStore(path.parent / "jobs.db")
    try:
        assert len(saved.metadata.job_results) == 2
        assert all(
            store.get_result(rid)["committed"] for rid in saved.metadata.job_results
        )
    finally:
        store.close()


def test_legacy_provider_retry_reuses_earlier_success(project_file, monkeypatch):
    path, _ = project_file
    before = path.read_bytes()
    compute = Mock(
        side_effect=[("wide", 0.9), RuntimeError("temporary"), ("medium", 0.8)]
    )
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    first = json.loads(asyncio.run(analyze_shots(str(path))))
    assert first["success"] is False
    assert path.read_bytes() == before
    result = json.loads(asyncio.run(analyze_shots(str(path))))
    assert result["success"] is True
    assert result["shot_type_distribution"] == {"wide": 1, "medium": 1}
    assert compute.call_count == 3


def test_legacy_successful_new_request_refreshes_labels(project_file, monkeypatch):
    path, _ = project_file
    compute = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    assert json.loads(asyncio.run(analyze_shots(str(path))))["success"]
    compute.return_value = ("medium", 0.8)
    response = json.loads(asyncio.run(analyze_shots(str(path))))
    assert response["shot_type_distribution"] == {"medium": 2}
    assert compute.call_count == 4
    assert len(Project.load(path).metadata.job_results) == 4


def test_legacy_uses_supplied_mcp_store_without_closing_it(project_file, monkeypatch):
    from core.jobs.store import JobStore

    path, _ = project_file
    store = JobStore(path.parent / "server" / "jobs.db")
    close = Mock(wraps=store.close)
    monkeypatch.setattr(store, "close", close)
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"job_store": store})
    )
    monkeypatch.setattr(
        "core.analysis.shots.classify_shot_type", Mock(return_value=("wide", 0.9))
    )
    try:
        result = json.loads(asyncio.run(analyze_shots(str(path), ctx=ctx)))
        assert result["success"]
        close.assert_not_called()
        saved = Project.load(path)
        assert len(saved.metadata.job_results) == 2
        assert all(
            store.get_result(rid)["committed"] for rid in saved.metadata.job_results
        )
        assert not (path.parent / "jobs.db").exists()
    finally:
        store.close()


def test_legacy_external_project_edit_is_not_overwritten(project_file, monkeypatch):
    path, _ = project_file
    external = None

    def compute(_):
        nonlocal external
        if external is None:
            data = json.loads(path.read_text())
            data["name"] = "external edit"
            path.write_text(json.dumps(data))
            external = path.read_bytes()
        return "wide", 0.9

    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    result = json.loads(asyncio.run(analyze_shots(str(path))))
    assert result["success"] is False
    assert result["error"]["code"] == "project_modified_externally"
    assert path.read_bytes() == external
