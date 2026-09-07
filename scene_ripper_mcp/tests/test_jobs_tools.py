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
from unittest.mock import AsyncMock

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

    def fake_describe(project, clip_ids=None, **_kwargs):
        clip = project.clips_by_id["clip-1"]
        clip.description = "A person standing in a doorway."
        clip.description_model = "fake-vlm"
        clip.description_frames = 1
        return {
            "success": True,
            "result": {
                "succeeded": [{"clip_id": clip.id, "model": "fake-vlm"}],
                "failed": [],
                "skipped": [],
                "total_clips": 1,
            },
        }

    monkeypatch.setattr("core.spine.analyze.describe", fake_describe)

    out = json.loads(await start_describe(project_path=str(project_path), ctx=ctx))

    assert out["success"] is True
    _wait_for_status(store, out["task_id"], STATUS_COMPLETED)

    _sources, clips, *_ = load_project(project_path)
    assert clips[0].description == "A person standing in a doorway."
    assert clips[0].description_model == "fake-vlm"


@pytest.mark.asyncio
async def test_start_analyze_clips_uses_canonical_operation_map(
    lifespan_ctx, tmp_path, monkeypatch
):
    from core.project import load_project
    from core.spine import analyze as analyze_module

    ctx, store, _runtime = lifespan_ctx
    project_path = _make_project_file(tmp_path)

    def fake_custom_query(project, clip_ids=None, query=None, **_kwargs):
        clip = project.clips_by_id["clip-1"]
        clip.custom_queries = [
            {
                "query": query,
                "match": True,
                "confidence": 0.91,
                "model": "fake-vlm",
            }
        ]
        return {
            "success": True,
            "result": {
                "succeeded": [{"clip_id": clip.id, "query": query, "match": True}],
                "failed": [],
                "skipped": [],
                "total_clips": 1,
            },
        }

    monkeypatch.setitem(
        analyze_module.ANALYZE_CLIP_OPERATION_MAP,
        "custom_query",
        fake_custom_query,
    )

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
async def test_download_submission_freezes_url_list(tmp_path, monkeypatch):
    from scene_ripper_mcp.tools import jobs
    captured = {}
    monkeypatch.setattr(jobs, '_start_job', lambda ctx, **kwargs: captured.update(kwargs) or 'queued')
    urls = ['https://youtube.com/original']
    assert await jobs.start_download_videos(urls, str(tmp_path)) == 'queued'
    urls[0] = 'https://youtube.com/replaced'
    seen = []
    monkeypatch.setattr('core.spine.downloads.download_videos', lambda items, *a, **k: seen.extend(items))
    captured['run'](None, threading.Event())
    assert seen == ['https://youtube.com/original']
    assert captured['args']['urls'] == seen
