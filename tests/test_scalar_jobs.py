"""Durable scalar analysis recovers work without publishing stale results."""

from threading import Event

import pytest

from core.jobs.scalars import run_scalar_job, scalar_job_spec
from core.jobs.store import JobStore
from core.project import Project
from tests import test_scalar_records

scalar_setup = test_scalar_records.setup


@pytest.fixture
def saved(scalar_setup, tmp_path):
    project, kind, provider = scalar_setup
    project.save(tmp_path / "project.sceneripper")
    store = JobStore(tmp_path / "jobs.db")
    yield project.path, kind, provider, store
    store.close()


def run(saved, **kwargs):
    path, kind, _, store = saved
    return run_scalar_job(
        store,
        path,
        None,
        lambda *_: None,
        kwargs.pop("cancel", Event()),
        kind=kind,
        **kwargs,
    )["result"]


def test_durable_scalar_save_and_reuse(saved):
    path, kind, provider, _ = saved
    assert len(run(saved)["succeeded"]) == 1
    assert kind in Project.load(path).clips[0].analysis_records
    assert len(run(saved)["skipped"]) == 1
    assert provider.call_count == 1


def test_durable_scalar_recovers_failed_save(saved, monkeypatch):
    from core.jobs import commits

    _, _, provider, _ = saved
    save = commits.save_with_mtime_check
    monkeypatch.setattr(
        commits,
        "save_with_mtime_check",
        lambda *args: (_ for _ in ()).throw(OSError("save failed")),
    )
    with pytest.raises(OSError, match="save failed"):
        run(saved)
    monkeypatch.setattr(commits, "save_with_mtime_check", save)
    assert len(run(saved)["succeeded"]) == 1
    assert provider.call_count == 1


def test_forced_scalar_retry_after_checkpoint_failure(saved, monkeypatch):
    _, _, provider, _ = saved
    run(saved)
    checkpoint = JobStore.checkpoint_results
    monkeypatch.setattr(
        JobStore,
        "checkpoint_results",
        lambda *args: (_ for _ in ()).throw(OSError("checkpoint failed")),
    )
    with pytest.raises(OSError, match="checkpoint failed"):
        run(saved, force=True)
    monkeypatch.setattr(JobStore, "checkpoint_results", checkpoint)
    assert len(run(saved, force=True)["skipped"]) == 1
    assert provider.call_count == 2
    run(saved, force=True)
    assert provider.call_count == 3


def test_durable_scalar_failure_preserves_previous_value(saved):
    path, kind, provider, _ = saved
    run(saved)
    before = getattr(Project.load(path).clips[0], test_scalar_records.FIELDS[kind])
    provider.side_effect = RuntimeError("decode failed")
    assert len(run(saved, force=True)["failed"]) == 1
    clip = Project.load(path).clips[0]
    assert getattr(clip, test_scalar_records.FIELDS[kind]) == before
    assert clip.analysis_records[kind].state == "failed"


def test_scalar_cancel_after_journal_recovers_without_publication(saved, monkeypatch):
    path, kind, provider, store = saved
    cancel = Event()
    record = store.record_result

    def record_then_cancel(*args, **kwargs):
        row = record(*args, **kwargs)
        cancel.set()
        return row

    monkeypatch.setattr(store, "record_result", record_then_cancel)
    assert len(run(saved, cancel=cancel)["unprocessed"]) == 1
    assert not Project.load(path).clips[0].analysis_records
    monkeypatch.setattr(store, "record_result", record)
    assert len(run(saved)["succeeded"]) == 1
    assert provider.call_count == 1


def test_queued_scalar_rejects_changed_media(saved):
    from core.jobs.commits import StaleJobResult

    path, kind, provider, _ = saved
    project = Project.load(path)
    spec = scalar_job_spec(project, None, kind, arguments={})
    project.sources[0].file_path.write_bytes(b"changed media")
    with pytest.raises(StaleJobResult, match="changed while queued"):
        run(saved, operation=spec)
    provider.assert_not_called()


def test_scalar_provenance_outlives_history(saved, monkeypatch):
    _, _, provider, store = saved
    run(saved)
    monkeypatch.setattr(store, "get_result", lambda *_: None)
    assert len(run(saved)["skipped"]) == 1
    assert provider.call_count == 1


def test_cli_scalars_use_durable_records(saved, monkeypatch):
    from types import SimpleNamespace
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    path, kind, provider, _ = saved
    register_commands()
    monkeypatch.setattr(
        "cli.utils.config.CLIConfig.load",
        lambda: SimpleNamespace(cache_dir=path.parent),
    )
    args = ["--json", "analyze", "scalars", str(path), "--operation", kind]
    for _ in range(2):
        response = CliRunner().invoke(cli, args)
        assert response.exit_code == 0, response.output
    assert provider.call_count == 1
    assert kind in Project.load(path).clips[0].analysis_records


@pytest.mark.asyncio
async def test_mcp_scalar_job_is_registered_and_persists_records(saved):
    import json
    from types import SimpleNamespace
    from core.jobs import JobRuntime
    from scene_ripper_mcp.server import mcp
    from scene_ripper_mcp.tools.jobs import start_analyze_scalars, get_job_result

    path, kind, provider, store = saved
    runtime = JobRuntime(store, max_workers=1)
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={"job_store": store, "job_runtime": runtime}
        )
    )
    try:
        names = {tool.name for tool in await mcp.list_tools()}
        assert "start_analyze_scalars" in names
        response = json.loads(await start_analyze_scalars(str(path), kind, ctx=ctx))
        assert response["success"], response
        runtime.shutdown()
        result = json.loads(await get_job_result(response["task_id"], ctx=ctx))
        assert result["success"], result
        assert store.get(response["task_id"]).kind == "analyze_scalars"
        assert provider.call_count == 1
        assert kind in Project.load(path).clips[0].analysis_records
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
async def test_mcp_invalid_scalar_options_do_not_enqueue(saved):
    import json
    from scene_ripper_mcp.tools.jobs import start_analyze_scalars

    path, _, provider, _ = saved
    for kind, samples in (("unknown", 5), ("brightness", 0), ("volume", True)):
        response = json.loads(
            await start_analyze_scalars(str(path), kind, num_samples=samples)
        )
        assert not response["success"]
        assert response["error"]["code"] == "validation_error"
    provider.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_scalar_status_uses_verified_completion(saved):
    import json
    from scene_ripper_mcp.tools.analyze import get_analysis_status

    path, kind, provider, _ = saved
    provider.return_value = None if kind == "volume" else 0.0
    run(saved)
    status = json.loads(await get_analysis_status(str(path)))
    assert status["analysis"][kind]["analyzed"] == 1
    assert status["analysis"][kind]["pending"] == 0
    project = Project.load(path)
    project.clips[0].analysis_records.clear()
    project.save()
    status = json.loads(await get_analysis_status(str(path)))
    assert status["analysis"][kind]["analyzed"] == 0
    assert status["analysis"][kind]["pending"] == 1


def test_scalar_job_reuses_valid_empty_results(saved):
    _, kind, provider, _ = saved
    provider.return_value = None if kind == "volume" else 0.0
    assert len(run(saved)["succeeded"]) == 1
    assert len(run(saved)["skipped"]) == 1
    assert provider.call_count == 1


def test_scalar_cancel_preserves_completed_prefix(saved, monkeypatch):
    from models.clip import Clip

    path, kind, provider, store = saved
    project = Project.load(path)
    project.add_clips(
        [Clip(source_id=project.sources[0].id, start_frame=60, end_frame=90)]
    )
    project.save()
    cancel = Event()
    original = store.record_result
    calls = []

    def record(*args, **kwargs):
        row = original(*args, **kwargs)
        calls.append(row)
        if len(calls) == 2:
            cancel.set()
        return row

    monkeypatch.setattr(store, "record_result", record)
    result = run(saved, cancel=cancel)
    assert len(result["succeeded"]) == len(result["unprocessed"]) == 1
    reopened = Project.load(path)
    assert kind in reopened.clips[0].analysis_records
    assert kind not in reopened.clips[1].analysis_records
    monkeypatch.setattr(store, "record_result", original)
    result = run(saved)
    assert len(result["succeeded"]) == len(result["skipped"]) == 1
    assert provider.call_count == 2


def test_recovered_scalar_identity_must_match_verified_content(saved, monkeypatch):
    from hashlib import sha256
    import json
    from core.jobs import commits
    from core.jobs.commits import StaleJobResult, canonical_json

    _, _, _, store = saved
    save = commits.save_with_mtime_check
    monkeypatch.setattr(
        commits,
        "save_with_mtime_check",
        lambda *args: (_ for _ in ()).throw(OSError("save failed")),
    )
    with pytest.raises(OSError):
        run(saved)
    monkeypatch.setattr(commits, "save_with_mtime_check", save)
    get_result = store.get_result

    def corrupt(result_id):
        row = get_result(result_id)
        if row is None:
            return None
        row = dict(row)
        payload = json.loads(row["payload_json"])
        record = json.loads(payload["record_json"])
        record["identity"]["sources"]["video"] = "0" * 64
        payload["record_json"] = json.dumps(record)
        row["payload_json"] = canonical_json(payload)
        row["payload_digest"] = sha256(row["payload_json"].encode()).hexdigest()
        return row

    monkeypatch.setattr(store, "get_result", corrupt)
    with pytest.raises(StaleJobResult, match="identity does not match"):
        run(saved)
