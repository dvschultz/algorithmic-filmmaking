"""Registry-backed generation, recipe inspection, and reconstruction over MCP."""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.project import Project
from models.clip import Clip, Source
from models.sequence import Sequence
from scene_ripper_mcp.project_sessions import SessionRuntime
from scene_ripper_mcp.tests.test_project_sessions import opened
from scene_ripper_mcp.tools import sequence as legacy
from scene_ripper_mcp.tools import sessions as tools


@pytest_asyncio.fixture
async def context():
    runtime = SessionRuntime()
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"project_sessions": runtime})
    )
    yield ctx
    await runtime.shutdown()


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "recipes.sceneripper"
    sources = [
        Source(id=f"source{i}", file_path=tmp_path / f"offline{i}.mp4", fps=30, duration_seconds=30)
        for i in range(2)
    ]
    clips = [
        Clip(
            id=f"clip{i}", source_id=f"source{i % 2}",
            start_frame=100 + i * 100, end_frame=140 + i * 100,
            dominant_colors=[(250 - i * 50, i * 50, 10)],
        )
        for i in range(4)
    ]
    project = Project(sources=sources, clips=clips, sequences=[Sequence(name="Active")])
    assert project.save(path)
    return path


@pytest.mark.asyncio
async def test_algorithm_listing_is_headless_and_matches_registry():
    from core.remix.registry import registry

    listing = json.loads(await legacy.list_sequence_algorithms())
    assert listing["success"]
    assert [entry["key"] for entry in listing["algorithms"]] == registry.keys()
    shuffle = next(entry for entry in listing["algorithms"] if entry["key"] == "shuffle")
    assert shuffle["seeded"] and {p["name"] for p in shuffle["parameters"]} >= {"hflip", "vflip", "reverse"}


@pytest.mark.asyncio
async def test_generate_inspect_reconstruct_and_undo_through_retained_session(path, context):
    sid = await opened(path, context)
    result = json.loads(await legacy.generate_sequence(
        str(path), "shuffle", parameters={"hflip": True}, seed=0, name="Hatchet", ctx=context,
    ))
    assert result["success"], result
    assert result["seed"] == 0 and result["parameters"]["hflip"] is True
    saved = Project.load(path)
    generated = next(s for s in saved.sequences if s.id == result["sequence_id"])
    recipe = generated.readable_recipe
    assert recipe is not None and recipe.id == result["recipe_id"]
    assert [e.source_clip_id for e in generated.get_all_clips()] == result["clip_ids"]
    assert [e.hflip for e in generated.get_all_clips()] == [e.hflip for e in recipe.realized]

    again = json.loads(await legacy.generate_sequence(
        str(path), "shuffle", parameters={"hflip": True}, seed=0, ctx=context,
    ))
    assert again["success"] and again["clip_ids"] == result["clip_ids"]  # deterministic
    assert again["sequence_id"] != result["sequence_id"]  # a new variation, not an overwrite

    inspected = json.loads(await legacy.get_sequence_recipe(str(path), result["sequence_id"], ctx=context))
    assert inspected["success"] and inspected["reconstructable"]
    assert inspected["recipe"]["seed"] == 0 and inspected["uses_provider"] is False

    rebuilt = json.loads(await legacy.reconstruct_sequence(str(path), result["sequence_id"], ctx=context))
    assert rebuilt["success"] and rebuilt["clip_ids"] == result["clip_ids"]
    assert rebuilt["sequence_id"] not in (result["sequence_id"], again["sequence_id"])
    # The empty starting sequence was reused by the first generation.
    assert len(Project.load(path).sequences) == 3

    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert len(Project.load(path).sequences) == 2
    assert json.loads(await tools.redo_project_session(sid, context))["success"]
    reloaded = Project.load(path)
    assert len(reloaded.sequences) == 3
    assert reloaded.sequences[-1].readable_recipe.parent_id == result["recipe_id"]


@pytest.mark.asyncio
async def test_invalid_generation_requests_do_not_write(path, context):
    await opened(path, context)
    expected = path.read_bytes()
    for kwargs in (
        {"algorithm": "storyteller"},
        {"algorithm": "color", "seed": 1},
        {"algorithm": "color", "parameters": {"direction": "nope"}},
        {"algorithm": "shuffle", "clip_ids": ["clip0", "missing"]},
    ):
        result = json.loads(await legacy.generate_sequence(str(path), ctx=context, **kwargs))
        assert not result["success"], kwargs
        assert path.read_bytes() == expected
    color = json.loads(await legacy.generate_sequence(
        str(path), "color", clip_ids=["clip3", "clip1"], parameters={"direction": "rainbow"}, ctx=context,
    ))
    assert color["success"] and sorted(color["clip_ids"]) == ["clip1", "clip3"]


@pytest.mark.asyncio
async def test_reconstruction_reports_changed_inputs_without_editing(path, context):
    await opened(path, context)
    result = json.loads(await legacy.generate_sequence(str(path), "color", ctx=context))
    assert result["success"]
    # Re-cut a clip outside the session, as an external editor would.
    project = Project.load(path)
    project.clips_by_id["clip1"].end_frame = 250
    assert project.save(path)
    inspected = json.loads(await legacy.get_sequence_recipe(str(path), result["sequence_id"]))
    assert not inspected["reconstructable"] and any("clip1" in p for p in inspected["problems"])
    expected = path.read_bytes()
    rebuilt = json.loads(await legacy.reconstruct_sequence(str(path), result["sequence_id"]))
    assert not rebuilt["success"] and "clip1" in rebuilt["error"]
    assert path.read_bytes() == expected


@pytest.mark.asyncio
async def test_start_generate_sequence_job_records_recipe_before_publishing(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from scene_ripper_mcp.jobs.runtime import JobRuntime
    from scene_ripper_mcp.jobs.store import JobStore
    from scene_ripper_mcp.tests.test_jobs_tools import _wait_for_status
    from scene_ripper_mcp.tools.jobs import get_job_result, start_generate_sequence
    from scene_ripper_mcp.tools.sequence import get_sequence_recipe

    calls = []

    def fake_narrative(clips_with_descriptions, target_duration_minutes, narrative_structure, theme=None, model=None):
        from core.remix.storyteller import NarrativeLine

        calls.append(1)
        return [NarrativeLine(clip.id, desc, "beat", i + 1) for i, (clip, desc) in enumerate(reversed(clips_with_descriptions))]

    monkeypatch.setattr("core.remix.storyteller.generate_narrative", fake_narrative)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0")
    sources = [Source(id="s0", file_path=video, fps=25, duration_seconds=30)]
    clips = [
        Clip(id=f"clip{i}", source_id="s0", start_frame=i * 25, end_frame=(i + 1) * 25, description=f"scene {i}")
        for i in range(3)
    ]
    project = Project(sources=sources, clips=clips, sequences=[Sequence(name="Active")])
    path = tmp_path / "job.sceneripper"
    assert project.save(path)

    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store, max_workers=2)
    ctx = AsyncMock()
    ctx.request_context = SimpleNamespace(lifespan_context={"job_store": store, "job_runtime": runtime})
    try:
        started = json.loads(await start_generate_sequence(
            str(path), "storyteller", parameters={"structure": "three_act"}, name="Story", ctx=ctx,
        ))
        assert started["success"], started
        assert _wait_for_status(store, started["task_id"], "completed") == "completed"
        output = json.loads(await get_job_result(started["task_id"], ctx=ctx))
        result = output["result"]
        result = result.get("result", result)
        assert result["clip_ids"] == ["clip2", "clip1", "clip0"] and result["replayed"] is False
        assert calls == [1]
        loaded = Project.load(path)
        assert loaded.sequence.name == "Story" and loaded.sequence.readable_recipe.algorithm == "storyteller"
        inspected = json.loads(await get_sequence_recipe(str(path)))
        assert inspected["recipe"]["provider_outputs"]["narrative"]

        assert not json.loads(await start_generate_sequence(str(path), "nope", ctx=ctx))["success"]
        assert not json.loads(await start_generate_sequence(str(path), "shuffle", seed=-1, ctx=ctx))["success"]
        missing_asset = json.loads(await start_generate_sequence(
            str(path), "staccato", parameters={"music_path": str(tmp_path / "none.wav")}, ctx=ctx,
        ))
        assert not missing_asset["success"]
    finally:
        runtime.shutdown(wait=True)
