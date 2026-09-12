import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.jobs.runtime import JobRuntime
from core.jobs.store import JobStore, TERMINAL_STATUSES
from core.project import Project
from core.remix.registry import normalize_parameters, registry
from core.spine.sequences import generate_sequence
from models.clip import Clip, Source
from scene_ripper_mcp.tools import jobs


def _save_project(tmp_path: Path, *, directory: Path | None = None) -> Path:
    root = directory or tmp_path
    root.mkdir(exist_ok=True)
    media = root / "source.mp4"
    media.write_bytes(b"fixture")
    project = Project(
        sources=[Source(id="source", file_path=media, fps=25.0, duration_seconds=10)],
        clips=[Clip(id="clip", source_id="source", start_frame=0, end_frame=50)],
    )
    path = root / "project.sceneripper"
    assert project.save(path)
    return path


async def _settled(store: JobStore, runtime: JobRuntime, task_id: str):
    for _ in range(500):
        row = store.get(task_id)
        if row.status in TERMINAL_STATUSES and not runtime.is_handle_live(task_id):
            return row
        await asyncio.sleep(0.01)
    raise AssertionError("Job did not settle")


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden_location", ["stored", "override"])
async def test_regeneration_rejects_forbidden_assets(
    tmp_path, monkeypatch, forbidden_location,
):
    import core.spine.security as security

    safe = tmp_path / "allowed"
    path = _save_project(tmp_path, directory=safe)
    allowed = safe / "drawing.png"
    allowed.write_bytes(b"allowed")
    forbidden = tmp_path / "forbidden.png"
    forbidden.write_bytes(b"forbidden")

    project = Project.load(path)
    assert generate_sequence(project, "sequential")["success"]
    definition = registry.require("signature_style")
    drawing = forbidden if forbidden_location == "stored" else allowed
    assert project.sequence.recipe is not None
    project.sequence.recipe = replace(
        project.sequence.recipe,
        algorithm=definition.key,
        algorithm_version=definition.version,
        parameters=normalize_parameters(definition, {"drawing_path": str(drawing)}),
    )
    assert project.save(path)

    monkeypatch.setattr(security, "SAFE_ROOTS", [safe])
    monkeypatch.setattr(jobs, "_lifespan", lambda ctx: {"job_store": object()})
    submitted = []

    def capture(*args, **kwargs):
        submitted.append(kwargs)
        return json.dumps({"success": True, "task_id": "unexpected"})

    monkeypatch.setattr(jobs, "start_job", capture)
    overrides = {"drawing_path": str(forbidden)} if forbidden_location == "override" else None
    response = json.loads(
        await jobs.start_regenerate_sequence(
            str(path), parameters=overrides, ctx=SimpleNamespace(),
        )
    )
    assert not response["success"]
    assert not submitted


@pytest.mark.asyncio
async def test_same_key_retry_preserves_omitted_seed_and_computed_recipe(tmp_path, monkeypatch):
    import core.jobs.commits as commits
    import core.remix.registry as registry_module

    path = _save_project(tmp_path)
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store, max_workers=1)
    monkeypatch.setattr(
        jobs,
        "_lifespan",
        lambda ctx: {"job_store": store, "job_runtime": runtime},
    )
    actual_run = registry_module.run_algorithm
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("seed"))
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(registry_module, "run_algorithm", counted)
    save = commits.save_with_mtime_check
    monkeypatch.setattr(
        commits,
        "save_with_mtime_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    try:
        first = json.loads(
            await jobs.start_generate_sequence(
                str(path), "shuffle", idempotency_key="same-request", ctx=SimpleNamespace(),
            )
        )
        assert (await _settled(store, runtime, first["task_id"])).status == "failed"
        monkeypatch.setattr(commits, "save_with_mtime_check", save)
        retry = json.loads(
            await jobs.start_generate_sequence(
                str(path), "shuffle", idempotency_key="same-request", ctx=SimpleNamespace(),
            )
        )
        row = await _settled(store, runtime, retry["task_id"])
        assert row.status == "completed"
        assert row.result is not None and row.result["seed"] == calls[0]
        assert len(calls) == 1
    finally:
        runtime.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_unkeyed_successful_calls_create_distinct_variations(tmp_path, monkeypatch):
    path = _save_project(tmp_path)
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store, max_workers=1)
    monkeypatch.setattr(
        jobs,
        "_lifespan",
        lambda ctx: {"job_store": store, "job_runtime": runtime},
    )
    try:
        results = []
        for _ in range(2):
            started = json.loads(
                await jobs.start_generate_sequence(
                    str(path), "shuffle", ctx=SimpleNamespace(),
                )
            )
            row = await _settled(store, runtime, started["task_id"])
            assert row.status == "completed"
            assert row.result is not None
            results.append(row.result)
        assert results[0]["sequence_id"] != results[1]["sequence_id"]
        assert results[0]["recipe_id"] != results[1]["recipe_id"]
        assert len(Project.load(path).sequences) == 2
    finally:
        runtime.shutdown()
        store.close()
