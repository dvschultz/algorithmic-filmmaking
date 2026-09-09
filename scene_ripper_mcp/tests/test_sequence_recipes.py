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
