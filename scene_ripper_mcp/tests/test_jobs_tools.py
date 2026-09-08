"""Tests for the generic job-management MCP tools.

Covers ``get_job_status``, ``get_job_result``, ``cancel_job``, ``list_jobs``,
and ``purge_old_jobs``. The store/runtime are tested separately in
``test_jobs_store.py`` / ``test_jobs_runtime.py``; these tests focus on
the MCP-tool wrappers and their JSON contract (R28 information-disclosure
discipline, R26 snake_case fields).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scene_ripper_mcp.jobs import JobRuntime, JobStore
from scene_ripper_mcp.jobs.store import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from scene_ripper_mcp.tools.jobs import (
    cancel_job,
    get_job_result,
    get_job_status,
    list_jobs,
    purge_old_jobs,
    start_analyze_clips,
    start_analyze_colors,
    start_describe,
    start_detect_scenes_bulk,
    start_generate_thumbnails,
)
from scene_ripper_mcp.tools.clips import filter_clips, get_clip_metadata, list_clips


@pytest.fixture
def lifespan_ctx(tmp_path):
    """Build a Context-shaped object with the lifespan dict the tools expect."""
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store, max_workers=4)
    request_context = SimpleNamespace(
        lifespan_context={
            "job_store": store,
            "job_runtime": runtime,
        }
    )
    ctx = AsyncMock()
    ctx.request_context = request_context
    yield ctx, store, runtime
    runtime.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["colors", "brightness", "volume", "classify", "detect_objects"])
async def test_explicit_legacy_reuse_job_saves_unknown_provenance(lifespan_ctx, tmp_path, operation):
    from core.project import Project
    from tests.test_description_operations import project_with_thumbnails
    from scene_ripper_mcp.tools.jobs import start_accept_legacy_analysis

    ctx, store, _ = lifespan_ctx
    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].dominant_colors = [(1, 2, 3)]
    project.clips[0].object_labels = project.clips[0].detected_objects = []
    project.clips[0].person_count = 0
    project.clips[0].average_brightness = project.clips[0].rms_volume = 0.0
    path = tmp_path / "legacy.sceneripper"
    assert project.save(path)
    response = json.loads(await start_accept_legacy_analysis(str(path), operation, ctx=ctx))
    assert response["success"], response
    _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
    output = json.loads(await get_job_result(response["task_id"], ctx=ctx))
    assert output["result"]["result"]["accepted"] == ["c-0"]
    loaded = Project.load(path)
    record = loaded.clips[0].analysis_records[operation]
    assert record.legacy_reuse and record.provenance == "unknown"
    loaded.close_writer()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["project", "media"])
async def test_legacy_reuse_rejects_changes_while_queued(lifespan_ctx, tmp_path, change):
    from tests.test_spine_analyze import _build_project
    from scene_ripper_mcp.tools.jobs import start_accept_legacy_analysis

    ctx, store, runtime = lifespan_ctx
    project = _build_project(tmp_path, 1, populate_colors=1)
    path = tmp_path / "legacy.sceneripper"
    assert project.save(path)
    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    try:
        for event in entered:
            def block(progress, cancel, event=event):
                event.set()
                assert release.wait(10)
                return {}
            runtime.submit(kind="blocker", args={}, run=block)
        assert all(event.wait(5) for event in entered)
        response = json.loads(await start_accept_legacy_analysis(str(path), "colors", ctx=ctx))
        assert response["success"], response
        if change == "project":
            project.clips[0].start_frame += 1
            assert project.save()
        else:
            project.sources[0].file_path.write_bytes(b"replacement media")
        expected = path.read_bytes()
        release.set()
        _wait_for_status(store, response["task_id"], STATUS_FAILED)
        assert path.read_bytes() == expected
    finally:
        release.set()


def _wait_for_status(store, task_id, expected, timeout=5.0):
    deadline = time.monotonic() + timeout
    targets = {expected} if isinstance(expected, str) else expected
    while time.monotonic() < deadline:
        status = store.get(task_id).status
        if status in targets:
            return status
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {targets}; last={status!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "failed"])
async def test_terminal_error_exposes_available_output_only_through_result(lifespan_ctx, status):
    ctx, store, runtime = lifespan_ctx
    payload = {"succeeded": ["clip-1"], "unprocessed": ["clip-2"]}
    if status == "failed":
        payload.update(success=False, error="second clip failed")

    def run(progress, cancel):
        if status == "cancelled":
            cancel.set()
        return payload

    task = runtime.submit(kind="partial", args={}, run=run)["task_id"]
    _wait_for_status(store, task, status)
    response = json.loads(await get_job_result(task_id=task, ctx=ctx))
    assert response["success"] is False
    assert response["status"] == status
    assert response["error"]["code"] == f"job_{status}"
    assert response["result"] == payload
    projection = json.loads(await get_job_status(task_id=task, ctx=ctx))
    assert "clip-1" not in json.dumps(projection)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["media", "project", "caller_arguments"])
async def test_color_job_binds_submitted_snapshot(lifespan_ctx, tmp_path, monkeypatch, change):
    from unittest.mock import Mock
    from core.jobs.spec import OperationSpec
    from tests.test_spine_analyze import _build_project

    ctx, store, runtime = lifespan_ctx
    project = _build_project(tmp_path, 2)
    path = tmp_path / "colors.sceneripper"
    assert project.save(path)
    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)
    try:
        for event in entered:
            def block(progress, cancel, event=event):
                event.set()
                assert release.wait(10)
                return {}
            runtime.submit(kind="blocker", args={}, run=block)
        assert all(event.wait(5) for event in entered)
        ids = ["c-0"]
        started = json.loads(await start_analyze_colors(str(path), clip_ids=ids, ctx=ctx))
        assert started["success"], started
        task = started["task_id"]
        operation = OperationSpec.from_json(store.get(task).operation_json)
        assert operation.arguments["clip_ids"] == ["c-0"]
        assert operation.input_revision
        if change == "media":
            (tmp_path / "video.mp4").write_bytes(b"changed media")
        elif change == "project":
            project.metadata.name = "changed project"
            assert project.save(path)
        else:
            ids.append("c-1")
        release.set()
        expected = STATUS_COMPLETED if change == "caller_arguments" else STATUS_FAILED
        _wait_for_status(store, task, expected)
        assert extract.call_count == (1 if change == "caller_arguments" else 0)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_color_job_persists_success_and_reports_each_failed_target(lifespan_ctx, tmp_path):
    from unittest.mock import patch

    from core.project import Project
    from tests.test_spine_analyze import _build_project

    ctx, store, _runtime = lifespan_ctx
    project = _build_project(tmp_path, 2)
    path = tmp_path / "colors.sceneripper"
    assert project.save(path)
    with patch("core.analysis.color.extract_dominant_colors", side_effect=[[], [(1, 2, 3)]]):
        started = json.loads(await start_analyze_colors(
            project_path=str(path), clip_ids=["c-0", "c-1", "missing"], ctx=ctx,
        ))
        assert started["success"], started
        _wait_for_status(store, started["task_id"], STATUS_COMPLETED)
    output = json.loads(await get_job_result(task_id=started["task_id"], ctx=ctx))
    result = output["result"]["result"]
    assert result["succeeded"] == [{"clip_id": "c-1", "color_count": 1}]
    assert [f["code"] for f in result["failed"]] == ["no_colors_extracted", "target_not_found"]
    restored = Project.load(path)
    assert restored.clips[0].dominant_colors is None
    assert restored.clips[1].dominant_colors == [(1, 2, 3)]


class _FakeSettings:
    def __init__(self, thumbnail_cache_dir):
        self.thumbnail_cache_dir = thumbnail_cache_dir


class _FakeThumbnailGenerator:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_clip_thumbnail(
        self,
        video_path,
        start_seconds,
        end_seconds,
        output_path=None,
        width=320,
        height=180,
    ):
        output_path = output_path or self.cache_dir / "generated.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"thumb")
        return output_path


def _make_project_file(tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    project_path = tmp_path / "project.sceneripper"

    project = Project.new(name="project")
    source = Source(
        id="src-1",
        file_path=video,
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    assert project.save(project_path)
    return project_path


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_boundary_embedding_submission_recovers_with_public_job_kind(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from core.jobs.spec import OperationSpec
    from scene_ripper_mcp.tools.jobs import start_generate_boundary_embeddings
    from scene_ripper_mcp.tools.jobs import start_analyze_clips

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    compute = Mock(return_value=([.123456789] * 768, [.987654321] * 768))
    monkeypatch.setattr("core.analysis.embeddings.extract_boundary_embeddings", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    for _ in range(2):
        response = json.loads(await start_analyze_clips(str(path), operations=["boundary_embeddings"], ctx=ctx) if generic else await start_generate_boundary_embeddings(str(path), ctx=ctx))
        assert response["success"], response
        _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
        row = store.get(response["task_id"])
        assert row.kind == ("analyze_clips" if generic else "generate_boundary_embeddings")
        assert OperationSpec.from_json(row.operation_json).kind == row.kind
    saved = Project.load(path)
    assert saved.clips[0].first_frame_embedding == [.123456789] * 768
    assert saved.clips[0].last_frame_embedding == [.987654321] * 768
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    compute.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["media", "project", "caller_arguments"])
async def test_detection_job_binds_snapshot_and_reuses_results(lifespan_ctx, tmp_path, monkeypatch, change):
    from unittest.mock import Mock
    from core.project import Project
    from models.clip import Clip, Source

    ctx, store, runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    source = Source(file_path=tmp_path / "video.mp4")
    compute = Mock(return_value=(source, [Clip(source_id=source.id, start_frame=0, end_frame=30)]))
    monkeypatch.setattr("core.jobs.detection.run_detection", compute)
    monkeypatch.setattr("core.spine.detect._generate_detected_clip_thumbnails", lambda *a, **k: {"generated": [], "failed": [], "skipped": []})
    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    try:
        for event in entered:
            def block(progress, cancel, event=event):
                event.set()
                assert release.wait(10)
                return {}
            runtime.submit(kind="blocker", args={}, run=block)
        assert all(event.wait(5) for event in entered)
        ids = ["src-1"]
        started = json.loads(await start_detect_scenes_bulk(str(path), ids, ctx=ctx))
        assert started["success"], started
        task = started["task_id"]
        assert store.get(task).operation_json
        if change == "media":
            source.file_path.write_bytes(b"changed media")
        elif change == "project":
            project = Project.load(path)
            project.clips[0].notes = "edited while queued"
            assert project.save(path)
        else:
            ids.append("not-submitted")
        release.set()
        expected = STATUS_COMPLETED if change == "caller_arguments" else STATUS_FAILED
        _wait_for_status(store, task, expected)
        assert compute.call_count == (1 if change == "caller_arguments" else 0)
        if change == "caller_arguments":
            first = store.get(task).result["result"]
            assert len(first["succeeded"]) == 1 and not first["failed"]
            retried = json.loads(await start_detect_scenes_bulk(str(path), ["src-1"], ctx=ctx))
            _wait_for_status(store, retried["task_id"], STATUS_COMPLETED)
            assert store.get(retried["task_id"]).result["result"] == first
            assert compute.call_count == 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_get_job_status_unknown_id(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await get_job_status(task_id="nope", ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "job_not_found"


@pytest.mark.asyncio
async def test_get_job_status_excludes_payload(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(
        kind="x",
        args={"secret": "do-not-leak"},
        project_path="/tmp/p.sceneripper",
    )
    out = json.loads(await get_job_status(task_id=row.id, ctx=ctx))
    assert out["success"] is True
    assert out["task_id"] == row.id
    # Sensitive payload columns must NOT appear in get_job_status (R28).
    assert "args_json" not in out
    assert "result_json" not in out
    assert "error" not in out


@pytest.mark.asyncio
async def test_get_job_result_not_terminal(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(kind="x", args={}, status=STATUS_RUNNING)
    store.update_status(row.id, STATUS_RUNNING, result={"private_output": "unfinished"})
    out = json.loads(await get_job_result(task_id=row.id, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "not_terminal"
    assert out["error"]["status"] == STATUS_RUNNING
    assert "private_output" not in json.dumps(out)


@pytest.mark.asyncio
async def test_get_job_result_completed_returns_payload(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx

    def run(progress, cancel):
        return {"value": 99}

    submit = runtime.submit(kind="t", args={}, run=run)
    _wait_for_status(store, submit["task_id"], STATUS_COMPLETED)

    out = json.loads(await get_job_result(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is True
    assert out["result"] == {"value": 99}
    assert out["status"] == STATUS_COMPLETED


@pytest.mark.asyncio
async def test_get_job_result_failed_surfaces_sanitized_error(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx

    def run(progress, cancel):
        raise RuntimeError("kaboom")

    submit = runtime.submit(kind="t", args={}, run=run)
    _wait_for_status(store, submit["task_id"], STATUS_FAILED)

    out = json.loads(await get_job_result(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is False
    assert out["status"] == STATUS_FAILED
    assert out["error"]["code"] == "job_failed"
    assert "kaboom" in out["error"]["message"]
    assert "result" not in out
    # Sanitized — no absolute paths.
    assert "/Users/" not in out["error"]["message"]


@pytest.mark.asyncio
async def test_cancel_job_running(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx
    started = threading.Event()

    def run(progress, cancel):
        started.set()
        for _ in range(200):
            if cancel.is_set():
                return {}
            time.sleep(0.01)
        return {}

    submit = runtime.submit(kind="t", args={}, run=run)
    started.wait(timeout=2.0)

    out = json.loads(await cancel_job(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is True
    assert out["ok"] is True

    _wait_for_status(store, submit["task_id"], STATUS_CANCELLED)


@pytest.mark.asyncio
async def test_cancel_job_already_terminal(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    out = json.loads(await cancel_job(task_id=row.id, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "already_terminal"


@pytest.mark.asyncio
async def test_cancel_job_unknown(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await cancel_job(task_id="not-a-real-id", ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "job_not_found"


@pytest.mark.asyncio
async def test_list_jobs_returns_safe_projection(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    a = store.insert(kind="x", args={"a": 1}, status=STATUS_QUEUED)
    b = store.insert(kind="y", args={"b": 2}, status=STATUS_COMPLETED)
    out = json.loads(await list_jobs(ctx=ctx))
    assert out["success"] is True
    assert out["count"] == 2
    ids = {j["task_id"] for j in out["jobs"]}
    assert ids == {a.id, b.id}
    # Safe projection — no payload columns.
    for j in out["jobs"]:
        assert "args_json" not in j
        assert "result_json" not in j
        assert "error" not in j


@pytest.mark.asyncio
async def test_list_jobs_status_filter(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    store.insert(kind="x", args={}, status=STATUS_QUEUED)
    completed = store.insert(kind="x", args={}, status=STATUS_COMPLETED)

    out = json.loads(
        await list_jobs(status_filter=[STATUS_COMPLETED], ctx=ctx)
    )
    assert out["count"] == 1
    assert out["jobs"][0]["task_id"] == completed.id


@pytest.mark.asyncio
async def test_purge_old_jobs_explicit_only(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    # Old completed row.
    old = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    store.update_status(old.id, STATUS_COMPLETED, terminal=True)
    long_ago = time.time() - 90 * 86400
    import sqlite3

    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ?",
            (long_ago, old.id),
        )
        conn.commit()

    out = json.loads(await purge_old_jobs(days=30, ctx=ctx))
    assert out["success"] is True
    assert out["deleted_count"] == 1


@pytest.mark.asyncio
async def test_purge_old_jobs_negative_days_rejected(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await purge_old_jobs(days=-5, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "invalid_days"


@pytest.mark.asyncio
async def test_start_generate_thumbnails_saves_project_paths(lifespan_ctx, tmp_path, monkeypatch):
    from core.project import load_project

    ctx, store, _runtime = lifespan_ctx
    project_path = _make_project_file(tmp_path)

    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: _FakeSettings(tmp_path / "thumbs"),
    )
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", _FakeThumbnailGenerator)

    out = json.loads(
        await start_generate_thumbnails(project_path=str(project_path), ctx=ctx)
    )

    assert out["success"] is True
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)

    result = json.loads(await get_job_result(task_id=out["task_id"], ctx=ctx))
    assert result["success"] is True
    assert len(result["result"]["result"]["succeeded"]) == 1

    _sources, clips, *_ = load_project(project_path)
    assert clips[0].thumbnail_path is not None
    assert clips[0].thumbnail_path.exists()


@pytest.mark.asyncio
async def test_start_describe_job_saves_description(lifespan_ctx, tmp_path, monkeypatch):
    from core.project import load_project

    ctx, store, _runtime = lifespan_ctx
    project_path = _make_project_file(tmp_path)

    from core.project import Project
    project = Project.load(project_path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *args, **kwargs: ("A person standing in a doorway.", "fake-vlm"),
    )

    out = json.loads(await start_describe(project_path=str(project_path), ctx=ctx))

    assert out["success"] is True
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)

    _sources, clips, *_ = load_project(project_path)
    assert clips[0].description == "A person standing in a doorway."
    assert clips[0].description_model == "fake-vlm"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "media"])
async def test_describe_queue_pins_options_and_media(lifespan_ctx, tmp_path, monkeypatch, change):
    from unittest.mock import Mock
    from core.project import Project
    from core.settings import Settings

    ctx, store, runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    settings = Settings(description_model_tier="cloud", description_model_cloud="gpt-original", description_input_mode="frame")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    compute = Mock(return_value=("Generated", "gpt-original"))
    monkeypatch.setattr("core.analysis.description.describe_frame", compute)
    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    try:
        for event in entered:
            def block(progress, cancel, event=event):
                event.set()
                assert release.wait(10)
                return {}
            runtime.submit(kind="blocker", args={}, run=block)
        assert all(event.wait(5) for event in entered)
        response = json.loads(await start_describe(str(path), ctx=ctx))
        assert response["success"], response
        if change == "model":
            settings.description_model_cloud = "gpt-replacement"
        else:
            thumbnail.write_bytes(b"changed")
        release.set()
        _wait_for_status(store, response["task_id"], STATUS_COMPLETED if change == "model" else STATUS_FAILED)
        if change == "model":
            assert compute.call_args.kwargs["model_name"] == "gpt-original"
        else:
            compute.assert_not_called()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_custom_query_submission_pins_options_and_records_results(lifespan_ctx, tmp_path, monkeypatch):
    from core.project import Project
    from core.settings import Settings
    from scene_ripper_mcp.tools import jobs
    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    settings = Settings(description_model_tier="cloud", description_model_cloud="original")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=(True, 0.9, "original"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", compute)
    ids = ["clip-1"]
    assert await jobs.start_custom_query(str(path), "person", ids, ctx=ctx) == "queued"
    ids.clear()
    settings.description_model_cloud = "replacement"
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["result"]["succeeded"][0]["query"] == "person"
    assert compute.call_args.kwargs["model_name"] == "original"
    saved = Project.load(path)
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
async def test_ocr_force_refresh_replaces_existing_text(lifespan_ctx, tmp_path, monkeypatch):
    from core.project import Project
    from models.clip import ExtractedText
    from scene_ripper_mcp.tools import jobs

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    project.clips[0].extracted_texts = [ExtractedText(0, "OLD", 0.9, "vlm")]
    project.save()
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", provider)
    response = json.loads(await jobs.start_extract_text(str(path), force=True, ctx=ctx))
    assert response["success"], response
    _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
    provider.assert_called_once()
    assert Project.load(path).clips[0].extracted_texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_cinematography_submission_pins_options_and_records_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from core.settings import Settings
    from models.cinematography import CinematographyAnalysis
    from scene_ripper_mcp.tools import jobs

    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    settings = Settings(cinematography_tier="cloud", cinematography_model="original")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=CinematographyAnalysis(shot_size="CU"))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", compute)
    ids = ["clip-1"]
    if generic:
        response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["cinematography"], ctx=ctx)
    else:
        response = await jobs.start_analyze_cinematography(str(path), clip_ids=ids, ctx=ctx)
    assert response == "queued"
    ids.clear()
    settings.cinematography_model = "replacement"
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["success"]
    assert compute.call_args.kwargs["model"] == "original"
    saved = Project.load(path)
    assert saved.clips[0].cinematography.shot_size == "CU"
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_gaze_submission_records_durable_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value={"gaze_yaw": 2.123456, "gaze_pitch": -1.23456, "gaze_category": "at_camera"})
    monkeypatch.setattr("core.analysis.gaze.extract_gaze_from_clip", compute)
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", Mock())
    monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    ids = ["clip-1"]
    if generic:
        response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["gaze"], ctx=ctx)
    else:
        response = await jobs.start_analyze_gaze(str(path), clip_ids=ids, sample_interval=.4, ctx=ctx)
    assert response == "queued"
    ids.clear()
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["success"]
    assert compute.call_args.kwargs["sample_interval"] == (1.0 if generic else .4)
    saved = Project.load(path)
    assert saved.clips[0].gaze_yaw == 2.12
    assert saved.clips[0].gaze_pitch == -1.23
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("public_kind", ["analyze_shots", "analyze_gaze", "detect_faces", "detect_objects", "analyze_classify", "analyze_cinematography", "generate_embeddings", "extract_text"])
async def test_durable_analysis_alias_submits_to_real_runtime(lifespan_ctx, tmp_path, monkeypatch, public_kind):
    from core.project import Project
    from core.jobs.spec import OperationSpec
    from core.settings import Settings
    from models.cinematography import CinematographyAnalysis
    from scene_ripper_mcp.tools import jobs

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    settings = Settings(cinematography_tier="cloud", cinematography_model="test-model")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.analysis.gaze.extract_gaze_from_clip", Mock(return_value={"gaze_yaw": 2., "gaze_pitch": 1., "gaze_category": "at_camera"}))
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", Mock())
    monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    from tests.analysis_fixtures import mock_face_execution
    mock_face_execution(monkeypatch, tmp_path, Mock(return_value=[]))
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", Mock(return_value=[]))
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths]))
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())
    monkeypatch.setattr("core.analysis.detection.detect_objects", Mock(return_value=[]))
    monkeypatch.setattr("core.analysis.classification.classify_frame", Mock(return_value=[("cat", .9)]))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", Mock(return_value=("wide", .9)))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", Mock(return_value=CinematographyAnalysis(shot_size="CU")))
    tool = getattr(jobs, "start_" + public_kind)
    response = json.loads(await tool(str(path), ctx=ctx))
    assert response["success"], response
    _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
    row = store.get(response["task_id"])
    assert row.kind == public_kind
    assert OperationSpec.from_json(row.operation_json).kind == public_kind
    assert len(row.result["result"]["succeeded"]) == 1
    receipts = Project.load(path).metadata.job_results
    assert len(receipts) == 1
    assert all(store.get_result(rid)["committed"] for rid in receipts)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_ocr_submission_reuses_empty_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", provider)
    for _ in range(2):
        response = await jobs.start_analyze_clips(str(path), operations=["extract_text"], ctx=ctx) if generic else await jobs.start_extract_text(str(path), ctx=ctx)
        response = json.loads(response)
        assert response["success"], response
        _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
    provider.assert_called_once()
    saved = Project.load(path)
    assert saved.clips[0].extracted_texts == []
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_embedding_submission_records_durable_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "embedding-thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    compute = Mock(side_effect=lambda paths: [[.123456789] * 768 for _ in paths])
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    for _ in range(2):
        if generic:
            response = await jobs.start_analyze_clips(str(path), operations=["embeddings"], ctx=ctx)
        else:
            response = await jobs.start_generate_embeddings(str(path), ctx=ctx)
        response = json.loads(response)
        assert response["success"], response
        _wait_for_status(store, response["task_id"], STATUS_COMPLETED)
    saved = Project.load(path)
    assert saved.clips[0].embedding == [.123456789] * 768
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    compute.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_face_submission_records_durable_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=[{"embedding": [.123456789] * 512, "confidence": .9, "bbox": [0, 0, 1, 1]}])
    from tests.analysis_fixtures import mock_face_execution
    mock_face_execution(monkeypatch, tmp_path, compute)
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())
    ids = ["clip-1"]
    if generic:
        response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["face_embeddings"], ctx=ctx)
    else:
        response = await jobs.start_detect_faces(str(path), clip_ids=ids, sample_interval=0.4, ctx=ctx)
    assert response == "queued"
    ids.clear()
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["success"]
    assert compute.call_args.kwargs["sample_interval"] == (1.0 if generic else .4)
    saved = Project.load(path)
    assert saved.clips[0].face_embeddings[0]["embedding"][0] == .12346
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_object_detection_submission_records_durable_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}])
    monkeypatch.setattr("core.analysis.detection.detect_objects", compute)
    ids = ["clip-1"]
    if generic:
        response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["detect_objects"], ctx=ctx)
    else:
        response = await jobs.start_detect_objects(str(path), clip_ids=ids, confidence=0.4, ctx=ctx)
    assert response == "queued"
    ids.clear()
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["success"]
    assert compute.call_args.kwargs == {"confidence_threshold": 0.5 if generic else 0.4}
    saved = Project.load(path)
    assert saved.clips[0].detected_objects[0]["label"] == "person"
    assert saved.clips[0].person_count == 1
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_shot_submission_captures_targets_and_recovers_results(
    lifespan_ctx, tmp_path, monkeypatch, generic
):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "shot.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=("wide", .9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    for _ in range(2):
        ids = ["clip-1"]
        if generic:
            response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["shots"], ctx=ctx)
        else:
            response = await jobs.start_analyze_shots(str(path), clip_ids=ids, ctx=ctx)
        assert response == "queued"
        ids.clear()
        result = captured["run"](lambda *_: None, threading.Event())
        assert result["success"]
    compute.assert_called_once()
    saved = Project.load(path)
    assert saved.clips[0].shot_type == "wide"
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
@pytest.mark.parametrize("generic", [False, True])
async def test_classification_submission_records_durable_results(lifespan_ctx, tmp_path, monkeypatch, generic):
    from core.project import Project
    from scene_ripper_mcp.tools import jobs

    ctx, store, _runtime = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    captured = {}
    monkeypatch.setattr(jobs, "_start_job", lambda ctx, **kwargs: captured.update(kwargs) or "queued")
    compute = Mock(return_value=[("person", 0.9)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", compute)
    ids = ["clip-1"]
    if generic:
        response = await jobs.start_analyze_clips(str(path), clip_ids=ids, operations=["classify"], ctx=ctx)
    else:
        response = await jobs.start_analyze_classify(str(path), clip_ids=ids, top_k=3, threshold=0.4, ctx=ctx)
    assert response == "queued"
    ids.clear()
    result = captured["run"](lambda *_: None, threading.Event())
    assert result["success"]
    assert compute.call_args.kwargs == {"top_k": 5 if generic else 3, "threshold": 0.1 if generic else 0.4}
    saved = Project.load(path)
    assert saved.clips[0].object_labels == ["person"]
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
async def test_start_analyze_clips_uses_durable_custom_query(
    lifespan_ctx, tmp_path, monkeypatch
):
    from core.project import Project, load_project

    ctx, store, _runtime = lifespan_ctx
    project_path = _make_project_file(tmp_path)

    project = Project.load(project_path)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"image")
    project.clips[0].thumbnail_path = thumbnail
    assert project.save()
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", lambda **_: (True, 0.91, "fake-vlm"))

    out = json.loads(
        await start_analyze_clips(
            project_path=str(project_path),
            operations=["custom_query"],
            query="person",
            ctx=ctx,
        )
    )

    assert out["success"] is True
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)

    assert len(Project.load(project_path).metadata.job_results) == 1
    _sources, clips, *_ = load_project(project_path)
    assert clips[0].custom_queries == [
        {
            "query": "person",
            "match": True,
            "confidence": 0.91,
            "model": "fake-vlm",
        }
    ]


@pytest.mark.asyncio
async def test_clip_tools_expose_agent_context_parity(tmp_path):
    from core.project import load_project

    project_path = _make_project_file(tmp_path)
    sources, clips, *_ = load_project(project_path)
    clip = clips[0]
    clip.description = "A bright red sign above a doorway."
    clip.description_model = "fake-vlm"
    clip.detected_objects = [{"label": "sign", "confidence": 0.82}]
    clip.person_count = 0
    clip.gaze_yaw = 1.2
    clip.gaze_pitch = -0.4
    clip.gaze_category = "at_camera"
    clip.custom_queries = [
        {
            "query": "red sign",
            "match": True,
            "confidence": 0.88,
            "model": "fake-vlm",
        }
    ]
    from core.project import Project

    project = Project.new(name="project")
    for source in sources:
        project.add_source(source)
    project.add_clips([clip])
    assert project.save(project_path)

    listed = json.loads(await list_clips(project_path=str(project_path)))
    row = listed["clips"][0]
    assert row["description"] == "A bright red sign above a doorway."
    assert row["detected_object_labels"] == ["sign"]
    assert row["gaze"]["category"] == "at_camera"
    assert row["custom_queries"][0]["query"] == "red sign"

    detail = json.loads(
        await get_clip_metadata(project_path=str(project_path), clip_id=clip.id)
    )
    analysis = detail["clip"]["analysis"]
    assert analysis["description"] == "A bright red sign above a doorway."
    assert analysis["detected_objects"][0]["label"] == "sign"
    assert analysis["gaze"]["category"] == "at_camera"

    filtered = json.loads(
        await filter_clips(
            project_path=str(project_path),
            has_description=True,
            has_objects=True,
            custom_query="red sign",
        )
    )
    assert filtered["filtered_count"] == 1


@pytest.mark.asyncio
async def test_download_submission_freezes_url_list(tmp_path, monkeypatch, lifespan_ctx):
    from scene_ripper_mcp.tools import jobs
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    urls = ['https://youtube.com/original']
    assert await jobs.start_download_videos(urls, str(tmp_path), ctx=lifespan_ctx[0]) == 'queued'
    urls[0] = 'https://youtube.com/replaced'
    seen = []
    monkeypatch.setattr('core.jobs.downloads.run_saved_downloads', lambda store, items, *a, **k: seen.extend(items))
    captured['run'](None, threading.Event())
    assert seen == ['https://youtube.com/original']
    assert captured['args']['urls'] == seen


@pytest.mark.asyncio
async def test_download_submission_rejects_replaced_directory(tmp_path, monkeypatch, lifespan_ctx):
    from scene_ripper_mcp.tools import jobs
    target = tmp_path / 'downloads'
    target.mkdir()
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    await jobs.start_download_videos(['https://youtube.com/a'], str(target), ctx=lifespan_ctx[0])
    target.rename(tmp_path / 'old-downloads')
    target.mkdir()
    with pytest.raises(RuntimeError, match='directory changed'):
        captured['run'](None, threading.Event())


@pytest.mark.asyncio
async def test_transcription_submission_freezes_ids_and_rejects_queued_media_changes(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools import jobs
    from core.jobs.commits import StaleJobResult
    from core.project import Project
    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    ids = ['clip-1']
    assert await jobs.start_transcribe(str(path), clip_ids=ids, ctx=ctx) == 'queued'
    ids.clear()
    assert captured['operation'].arguments['clip_ids'] == ['clip-1']
    Project.load(path).sources[0].file_path.write_bytes(b'changed')
    with pytest.raises(StaleJobResult, match='queued'):
        captured['run'](lambda *_: None, threading.Event())


@pytest.mark.asyncio
async def test_transcription_job_saves_receipts(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_transcribe
    from core.project import Project
    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    monkeypatch.setattr('core.transcription.transcribe_clip', lambda **_: [])
    out = json.loads(await start_transcribe(str(path), ctx=ctx))
    assert out['success'] is True
    _wait_for_status(store, out['task_id'], STATUS_COMPLETED)
    project = Project.load(path)
    assert project.clips[0].transcript == []
    assert len(project.metadata.job_results) == 1


@pytest.mark.asyncio
async def test_sync_transcription_uses_server_result_store(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.analyze import transcribe
    from core.project import Project

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    monkeypatch.setattr('core.transcription.transcribe_clip', lambda **_: [])
    result = json.loads(await transcribe(str(path), ctx=ctx))
    assert result['success'] is True
    receipts = Project.load(path).metadata.job_results
    assert len(receipts) == 1
    assert all(store.get_result(result_id)['committed'] for result_id in receipts)


@pytest.mark.asyncio
async def test_analysis_plan_freezes_submission_and_saves_transcription(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools import jobs
    from core.project import Project

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    monkeypatch.setattr('core.transcription.transcribe_clip', lambda **_: [])
    operations = ['transcribe']
    ids = ['clip-1']
    assert await jobs.start_analyze_clips(str(path), operations, ids, ctx=ctx) == 'queued'
    operations.clear()
    ids.clear()
    result = captured['run'](lambda *_: None, threading.Event())
    assert 'transcribe' in result['result']['operations']
    assert captured['operation'].arguments['clip_ids'] == ['clip-1']
    receipts = Project.load(path).metadata.job_results
    assert len(receipts) == 1
    assert all(store.get_result(result_id)['committed'] for result_id in receipts)


@pytest.mark.asyncio
async def test_alignment_job_saves_word_timestamps(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_align_words
    from core.project import Project
    from core.transcription_models import TranscriptSegment, WordTimestamp

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    project.clips[0].transcript = [TranscriptSegment(0, 1, 'hello', language='en')]
    assert project.save()
    wav = tmp_path / 'audio.wav'
    wav.write_bytes(b'fake')
    monkeypatch.setattr('core.feature_registry.check_feature_ready', lambda *_: (True, []))
    monkeypatch.setattr('core.analysis.alignment.extract_audio_to_wav', lambda *a, **k: wav)
    from core.analysis.alignment import ALIGNMENT_MODEL
    monkeypatch.setattr('core.operations.alignment_records.alignment_model_revision', lambda: 'r1')
    def align(*a, **k):
        k['on_execution']({'backend': 'ctc', 'model': ALIGNMENT_MODEL, 'revision': 'r1'})
        return [WordTimestamp(0, 1, 'hello')]
    monkeypatch.setattr('core.analysis.alignment.align_words', align)
    out = json.loads(await start_align_words(str(path), ctx=ctx))
    assert out['success'] is True
    _wait_for_status(store, out['task_id'], STATUS_COMPLETED)
    result = json.loads(await get_job_result(out['task_id'], ctx=ctx))
    assert result['result']['result']['succeeded'] == [{'clip_id': 'clip-1', 'word_count': 1}]
    assert Project.load(path).clips[0].transcript[0].words[0].text == 'hello'
    assert not wav.exists()
    saved = Project.load(path)
    assert len(saved.metadata.job_results) == 1
    assert all(store.get_result(rid)['committed'] for rid in saved.metadata.job_results)


@pytest.mark.asyncio
async def test_alignment_job_freezes_ids_and_rejects_queued_media_change(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools import jobs
    from core.jobs.commits import StaleJobResult
    from core.project import Project
    from core.transcription_models import TranscriptSegment

    ctx, store, _ = lifespan_ctx
    path = _make_project_file(tmp_path)
    project = Project.load(path)
    project.clips[0].transcript = [TranscriptSegment(0, 1, 'hello', language='en')]
    assert project.save()
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    ids = ['clip-1']
    assert await jobs.start_align_words(str(path), ids, ctx=ctx) == 'queued'
    ids.clear()
    assert captured['operation'].arguments['clip_ids'] == ['clip-1']
    project.sources[0].file_path.write_bytes(b'changed')
    with pytest.raises(StaleJobResult, match='queued'):
        captured['run'](lambda *_: None, threading.Event())


@pytest.mark.asyncio
async def test_standalone_audio_job_saves_and_preserves_silence(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_transcribe_audio
    from core.project import Project
    from models.audio_source import AudioSource
    from unittest.mock import Mock

    ctx, store, _ = lifespan_ctx
    media = tmp_path / "audio.wav"
    media.write_bytes(b"audio")
    project = Project.new()
    project.add_audio_source(AudioSource(id="audio", file_path=media))
    path = tmp_path / "audio.sceneripper"
    project.save(path)
    project.close_writer()
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    for status in ("succeeded", "skipped"):
        out = json.loads(await start_transcribe_audio(str(path), "audio", backend="faster-whisper", ctx=ctx))
        assert out["success"], out
        _wait_for_status(store, out["task_id"], STATUS_COMPLETED)
        result = json.loads(await get_job_result(out["task_id"], ctx=ctx))
        assert result["result"]["result"]["status"] == status
    assert provider.call_count == 1
    saved = json.loads(path.read_text())
    assert saved["audio_sources"][0]["transcript"] == []
    assert all(store.get_result(rid)["committed"] for rid in saved["job_results"])
    invalid = json.loads(await start_transcribe_audio(str(path), "missing", ctx=ctx))
    assert invalid["success"] is False


@pytest.mark.asyncio
async def test_frame_extraction_job_saves_and_deduplicates_submission(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_extract_frames
    from core.project import Project
    from models.clip import Source, Clip
    from tests.test_frame_extraction_operations import extract

    ctx, store, _ = lifespan_ctx
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    project = Project.new()
    project.add_source(Source(id="source", file_path=media))
    project.add_clips([Clip(id="clip", source_id="source", start_frame=0, end_frame=15)])
    path = tmp_path / "frames.sceneripper"
    project.save(path)
    project.close_writer()
    provider = Mock(side_effect=extract)
    monkeypatch.setattr("core.ffmpeg.extract_frames_batch", provider)
    out = json.loads(await start_extract_frames(str(path), "source", interval=5,
        clip_id="clip", idempotency_key="one-batch", ctx=ctx))
    assert out["success"], out
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)
    result = json.loads(await get_job_result(out["task_id"], ctx=ctx))
    assert result["result"]["result"]["frame_count"] == 3
    assert [f["frame_number"] for f in json.loads(path.read_text())["frames"]] == [0, 5, 10]
    assert provider.call_args.kwargs["end_frame"] == 15
    repeated = json.loads(await start_extract_frames(str(path), "source", interval=5,
        clip_id="clip", idempotency_key="one-batch", ctx=ctx))
    assert repeated["task_id"] == out["task_id"]
    assert provider.call_count == 1
    for options in ({"interval": 0}, {"mode": "bad"}, {"clip_id": "missing"}):
        invalid = json.loads(await start_extract_frames(str(path), "source", ctx=ctx, **options))
        assert invalid["success"] is False
    saved = json.loads(path.read_text())
    assert all(store.get_result(rid)["committed"] for rid in saved["job_results"])


@pytest.mark.asyncio
async def test_audio_import_job_saves_and_deduplicates(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_import_audio
    from core.project import Project

    ctx, store, _ = lifespan_ctx
    media = tmp_path / "voice.wav"
    media.write_bytes(b"audio")
    project = Project.new()
    path = tmp_path / "audio.sceneripper"
    project.save(path)
    project.close_writer()
    processor = Mock(ffprobe_available=True)
    processor.get_audio_info.return_value = dict(duration=10, sample_rate=48000, channels=2)
    monkeypatch.setattr("core.ffmpeg.FFmpegProcessor", lambda: processor)
    out = json.loads(await start_import_audio(str(path), "voice.wav", idempotency_key="import-one", ctx=ctx))
    assert out["success"], out
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)
    result = json.loads(await get_job_result(out["task_id"], ctx=ctx))
    aid = result["result"]["result"]["audio_source_id"]
    repeated = json.loads(await start_import_audio(str(path), "voice.wav", idempotency_key="import-one", ctx=ctx))
    assert repeated["task_id"] == out["task_id"]
    fresh = json.loads(await start_import_audio(str(path), "voice.wav", ctx=ctx))
    _wait_for_status(store, fresh["task_id"], STATUS_COMPLETED)
    existing = json.loads(await get_job_result(fresh["task_id"], ctx=ctx))
    assert existing["result"]["result"]["status"] == "skipped"
    assert existing["result"]["result"]["audio_source_id"] == aid
    processor.get_audio_info.assert_called_once()
    saved = json.loads(path.read_text())
    assert len(saved["audio_sources"]) == 1
    assert saved["audio_sources"][0]["id"] == aid
    assert all(store.get_result(rid)["committed"] for rid in saved["job_results"])


@pytest.mark.asyncio
@pytest.mark.parametrize("media_path", ["../voice.wav", "/etc/voice.wav", "", "voice.txt"])
async def test_audio_import_rejects_invalid_media_paths_before_submission(lifespan_ctx, tmp_path, monkeypatch, media_path):
    from scene_ripper_mcp.tools import jobs
    from core.project import Project

    ctx, _, _ = lifespan_ctx
    project = Project.new()
    path = tmp_path / "audio.sceneripper"
    project.save(path)
    project.close_writer()
    submit = Mock(return_value=json.dumps({"success": True}))
    monkeypatch.setattr(jobs, "_start_job", submit)
    result = json.loads(await jobs.start_import_audio(str(path), media_path, ctx=ctx))
    assert result["success"] is False
    submit.assert_not_called()


@pytest.mark.asyncio
async def test_image_import_job_preserves_partial_errors_and_submission_identity(lifespan_ctx, tmp_path, monkeypatch):
    from scene_ripper_mcp.tools.jobs import start_import_images
    from core.project import Project
    from tests.test_image_import_operations import make_image
    from core.thumbnail import generate_image_thumbnail

    ctx, store, _ = lifespan_ctx
    media = make_image(tmp_path / "image.png")
    project = Project.new()
    path = tmp_path / "images.sceneripper"
    project.save(path)
    project.close_writer()
    provider = Mock(wraps=generate_image_thumbnail)
    monkeypatch.setattr("core.thumbnail.generate_image_thumbnail", provider)
    inputs = ["image.png", "../outside.png", "/etc/outside.png"]
    out = json.loads(await start_import_images(str(path), inputs, idempotency_key="one-batch", ctx=ctx))
    assert out["success"], out
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)
    result = json.loads(await get_job_result(out["task_id"], ctx=ctx))["result"]["result"]
    assert result["imported_count"] == 1 and len(result["errors"]) == 2
    assert any("traversal" in error for error in result["errors"])
    repeated = json.loads(await start_import_images(str(path), inputs, idempotency_key="one-batch", ctx=ctx))
    assert repeated["task_id"] == out["task_id"]
    provider.assert_called_once()
    saved = Project.load(path)
    assert saved.frames[0].id == result["frame_ids"][0]
    assert saved.frames[0].file_path != media
    assert (saved.frames[0].width, saved.frames[0].height) == (400, 300)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    saved.close_writer()
    invalid = json.loads(await start_import_images(str(path), [], ctx=ctx))
    assert invalid["success"] is False
