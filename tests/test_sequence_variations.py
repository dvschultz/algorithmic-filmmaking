"""U12: variation commands never overwrite earlier edits and fail actionably."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from core.project import Project
from core.remix.registry import registry
from core.spine.sequences import (
    activate_sequence, duplicate_sequence, generate_sequence, get_sequence_recipe,
    list_sequences, reconstruct_sequence, regenerate_sequence,
)
from models.clip import Clip, Source


def _project(tmp_path: Path, count: int = 5) -> Project:
    project = Project()
    path = tmp_path / "v.mp4"
    path.write_bytes(b"0")
    project.add_source(Source(id="s", file_path=path, fps=25.0, duration_seconds=30))
    project.add_clips([
        Clip(id=f"c{i}", source_id="s", start_frame=i * 25, end_frame=(i + 1) * 25, dominant_colors=[(i * 50, 20, 20)])
        for i in range(count)
    ])
    project.mark_clean()
    return project


def test_variation_leaves_previous_sequence_untouched_and_undoes_as_one_edit(tmp_path):
    project = _project(tmp_path)
    first = generate_sequence(project, "shuffle", seed=3, parameters={"hflip": True})
    assert first["success"]
    original = project.sequence
    before = original.to_dict()
    project.mark_clean()
    variation = regenerate_sequence(project, original.id, parameters={"vflip": True})
    assert variation["success"], variation
    assert project.sequence is not original
    assert original.to_dict() == before
    assert project.sequence.readable_recipe.parameters["vflip"] is True
    assert project.sequence.readable_recipe.parameters["hflip"] is True  # inherited
    assert len(project.sequences) == 2
    assert project.session.undo_text == "Generate sequence"
    project.session.undo()
    assert len(project.sequences) == 1 and project.sequence is original and not project.is_dirty
    project.session.redo()
    assert len(project.sequences) == 2


def test_keep_seed_reproduces_and_new_seed_is_recorded(tmp_path):
    project = _project(tmp_path, 8)
    first = generate_sequence(project, "shuffle", seed=11)
    same = regenerate_sequence(project, first["sequence_id"], keep_seed=True)
    assert same["seed"] == 11 and same["clip_ids"] == first["clip_ids"]
    fresh = regenerate_sequence(project, first["sequence_id"])
    assert fresh["seed"] != 11
    explicit = regenerate_sequence(project, first["sequence_id"], seed=0)
    assert explicit["seed"] == 0
    assert not regenerate_sequence(project, first["sequence_id"], seed=-2)["success"]


def test_duplicate_copies_timeline_and_recipe_without_recomputation(tmp_path, monkeypatch):
    project = _project(tmp_path)
    assert generate_sequence(project, "color", parameters={"direction": "complementary"})["success"]
    original = project.sequence

    def explode(*args, **kwargs):
        raise AssertionError("duplicate must not run the algorithm")

    monkeypatch.setattr(type(registry.require("color")), "generate", explode)
    copy = duplicate_sequence(project, original.id, name="Copy A")
    assert copy["success"] and copy["name"] == "Copy A" and copy["clip_ids"] == [e.source_clip_id for e in original.get_all_clips()]
    duplicated = project.sequence
    assert duplicated is not original and duplicated.id != original.id
    assert {e.id for e in duplicated.get_all_clips()}.isdisjoint({e.id for e in original.get_all_clips()})
    assert duplicated.readable_recipe.parent_id == original.readable_recipe.id
    assert duplicated.readable_recipe.parameters == original.readable_recipe.parameters
    project.session.undo()
    assert project.sequence is original and len(project.sequences) == 1


def test_missing_inputs_and_version_drift_are_actionable_without_overwriting(tmp_path, monkeypatch):
    project = _project(tmp_path)
    assert generate_sequence(project, "shuffle", seed=1)["success"]
    generated = project.sequence
    snapshot = [s.to_dict() for s in project.sequences]
    project.mark_clean()

    # Version drift: the stored recipe names an older algorithm version.
    monkeypatch.setattr(type(registry.require("shuffle")), "version", 2)
    drift = regenerate_sequence(project, generated.id)
    assert not drift["success"] and "version 1" in drift["error"] and drift["available_version"] == 2
    assert reconstruct_sequence(project, generated.id)["success"]  # replay still works
    project.session.undo()
    monkeypatch.undo()

    # Missing algorithm entirely.
    monkeypatch.setattr(registry, "_definitions", {k: v for k, v in registry._definitions.items() if k != "shuffle"})
    gone = regenerate_sequence(project, generated.id)
    assert not gone["success"] and "not available" in gone["error"]
    monkeypatch.undo()

    # Missing input clip.
    project.remove_clips(["c2"])
    snapshot = [s.to_dict() for s in project.sequences]
    inspected = get_sequence_recipe(project, generated.id)
    assert not inspected["reconstructable"] and any("c2" in p for p in inspected["problems"])
    result = regenerate_sequence(project, generated.id)
    assert not result["success"] and "c2" in result["error"]
    assert [s.to_dict() for s in project.sequences] == snapshot


def test_list_and_activate_expose_recipes(tmp_path):
    project = _project(tmp_path)
    project.add_to_sequence(["c0"])
    manual = project.sequence
    assert generate_sequence(project, "sequential")["success"]
    listing = list_sequences(project)["sequences"]
    assert listing[0]["recipe_id"] is None and listing[1]["algorithm"] == "sequential" and listing[1]["recipe_id"]
    result = activate_sequence(project, manual.id)
    assert result["success"] and project.sequence is manual
    assert not activate_sequence(project, "missing")["success"]


@pytest.mark.asyncio
async def test_mcp_variation_tools_round_trip(tmp_path):
    from types import SimpleNamespace

    from scene_ripper_mcp.project_sessions import SessionRuntime
    from scene_ripper_mcp.tests.test_project_sessions import opened
    from scene_ripper_mcp.tools import sequence as tools
    from scene_ripper_mcp.tools import sessions

    project = _project(tmp_path)
    path = tmp_path / "p.sceneripper"
    assert project.save(path)
    runtime = SessionRuntime()
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"project_sessions": runtime}))
    try:
        sid = await opened(path, ctx)
        first = json.loads(await tools.generate_sequence(str(path), "shuffle", seed=2, ctx=ctx))
        assert first["success"]
        variation = json.loads(await tools.regenerate_sequence(str(path), first["sequence_id"], parameters={"reverse": True}, ctx=ctx))
        assert variation["success"] and variation["parameters"]["reverse"] is True
        copy = json.loads(await tools.duplicate_sequence(str(path), first["sequence_id"], name="Dup", ctx=ctx))
        assert copy["success"] and copy["name"] == "Dup"
        listing = json.loads(await tools.list_sequences(str(path), ctx=ctx))
        assert len(listing["sequences"]) == 3
        activated = json.loads(await tools.activate_sequence(str(path), first["sequence_id"], ctx=ctx))
        assert activated["success"]
        assert Project.load(path).sequence.id == first["sequence_id"]
        assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
        assert len(Project.load(path).sequences) == 2
    finally:
        await runtime.shutdown()


def test_cli_sequence_commands_generate_inspect_and_vary(tmp_path):
    project = _project(tmp_path)
    path = tmp_path / "cli.sceneripper"
    assert project.save(path)
    root = Path(__file__).resolve().parents[1]

    def run(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "cli.main", "--json", "sequence", *args],
            cwd=root, capture_output=True, text=True, env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        return completed.returncode, completed.stdout

    code, out = run("algorithms")
    assert code == 0 and {a["key"] for a in json.loads(out)["algorithms"]} >= {"shuffle", "color", "storyteller"}
    code, out = run("generate", str(path), "shuffle", "--seed", "4", "--name", "CLI")
    assert code == 0, out
    generated = json.loads(out)
    assert generated["seed"] == 4 and generated["name"] == "CLI"
    code, out = run("recipe", str(path))
    assert code == 0 and json.loads(out)["recipe"]["seed"] == 4
    code, out = run("regenerate", str(path), "--parameters", '{"hflip": true}')
    assert code == 0 and json.loads(out)["parameters"]["hflip"] is True
    code, out = run("duplicate", str(path), "--name", "Dup")
    assert code == 0 and json.loads(out)["name"] == "Dup"
    code, out = run("reconstruct", str(path), "--sequence-id", generated["sequence_id"])
    assert code == 0 and json.loads(out)["clip_ids"] == generated["clip_ids"]
    code, out = run("list", str(path))
    assert code == 0 and len(json.loads(out)["sequences"]) == 4
    code, out = run("activate", str(path), generated["sequence_id"])
    assert code == 0 and Project.load(path).sequence.id == generated["sequence_id"]
    code, out = run("generate", str(path), "color", "--parameters", '{"direction": "nope"}')
    assert code != 0 and "must be one of" in out
