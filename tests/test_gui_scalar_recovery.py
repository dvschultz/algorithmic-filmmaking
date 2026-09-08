"""Scalar journals recover verified work independently of project publication."""

from threading import Event

import pytest

from core.analysis_records import AnalysisSnapshot
from core.jobs.gui_scalars import GuiScalarCache
from core.operations.scalars import scalar_task, ScalarApplication
from core.project import Project
from tests import test_scalar_records

scalar_setup = test_scalar_records.setup


@pytest.fixture
def saved(scalar_setup, tmp_path, monkeypatch):
    from core.settings import Settings

    project, operation, provider = scalar_setup
    monkeypatch.setattr("core.settings.load_settings", lambda: Settings(cache_dir=tmp_path))
    project.save(tmp_path / "project.sceneripper")
    return project, operation, provider


def run(saved, *, apply=False, cancel=None, deliver=None):
    project, operation, _ = saved
    tasks = tuple(scalar_task(clip, project.sources_by_id[clip.source_id], operation) for clip in project.clips)
    stamps = {
        path: stamp
        for task in tasks
        for _, path, stamp in AnalysisSnapshot.from_json(task.snapshot_json).inputs.files
    }
    cache = GuiScalarCache(
        project.path, project.metadata.id,
        {clip.id: clip.source_id for clip in project.clips},
        project.metadata.job_results, operation=operation, media_stamps=stamps,
    )
    applications = {task.clip_id: ScalarApplication(project, task) for task in tasks}

    def publish(outcome):
        if apply and outcome.record_json:
            assert applications[outcome.clip_id].apply(project, outcome)
            receipt = cache.results.get(outcome.clip_id)
            if receipt:
                assert receipt.matches(outcome)
                project.record_job_result(receipt.result_id, receipt.digest)
        if deliver:
            deliver(outcome)

    outcomes = cache.run(tasks, cancel or Event(), publish, lambda *_: None)
    return cache, outcomes


def test_recovery_without_project_publication(saved):
    project, operation, provider = saved
    first, outcomes = run(saved)
    assert first.results
    reopened = Project.load(project.path)
    _, recovered = run((reopened, operation, provider), apply=True)
    assert recovered == outcomes
    assert provider.call_count == 1
    assert reopened.clips[0].analysis_records[operation].provenance == "verified"
    assert not Project.load(project.path).clips[0].analysis_records


@pytest.mark.parametrize("change", ["none", "value", "record", "range", "fps"])
def test_explicit_save_checkpoints_only_matching_scalars(saved, change):
    from dataclasses import replace
    from core.jobs.store import JobStore

    project, operation, _ = saved
    cache, _ = run(saved, apply=True)
    clip = project.clips[0]
    if change == "value":
        setattr(clip, test_scalar_records.FIELDS[operation], 0.1)
    elif change == "record":
        clip.analysis_records[operation] = replace(clip.analysis_records[operation], state="failed")
    elif change == "range":
        clip.end_frame += 1
    elif change == "fps":
        project.sources[0].fps = 24
    result_id = next(iter(cache.results.values())).result_id
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not store.get_result(result_id)["committed"]
        project.save()
        assert bool(store.get_result(result_id)["committed"]) == (change == "none")
    finally:
        store.close()


def test_scalar_save_checkpoints_valid_empty_results(saved):
    from core.jobs.store import JobStore

    project, operation, provider = saved
    provider.return_value = None if operation == "volume" else 0.0
    cache, _ = run(saved, apply=True)
    project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert all(store.get_result(receipt.result_id)["committed"] for receipt in cache.results.values())
    finally:
        store.close()


def test_scalar_save_checkpoints_occurrence_delivery_ids(saved):
    from copy import deepcopy
    from dataclasses import replace
    from core.jobs.sequence_scalars import SequenceScalarJob
    from core.jobs.store import JobStore

    project, operation, _ = saved
    clip, source = project.clips[0], project.sources[0]
    job = SequenceScalarJob([(clip, source)], operation=operation, project=project)
    task = scalar_task(clip, source, operation)
    application = ScalarApplication(project, task)
    job.populate(deepcopy([(clip, source)]), Event())
    original = job.outcomes[0]
    assert original.clip_id == "0" and original.clip_id != clip.id
    assert application.apply(project, replace(original, clip_id=clip.id))
    receipt = job.cache.results["0"]
    assert receipt.matches(original)
    project.record_job_result(receipt.result_id, receipt.digest)
    project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert store.get_result(receipt.result_id)["committed"]
    finally:
        store.close()


@pytest.mark.parametrize("change", ["media", "range", "value", "runtime"])
def test_changed_inputs_do_not_recover_old_scalar(saved, change, tmp_path, monkeypatch):
    project, operation, provider = saved
    run(saved)
    clip = project.clips[0]
    if change == "media":
        project.sources[0].file_path.write_bytes(b"new media")
    elif change == "range":
        clip.end_frame += 1
    elif change == "value":
        setattr(clip, test_scalar_records.FIELDS[operation], 0.1)
    elif operation == "volume":
        (tmp_path / "ffmpeg").write_bytes(b"new binary")
    else:
        monkeypatch.setattr("core.operations.scalars.model_runtime", lambda *args: {"name": "new"})
    run(saved)
    assert provider.call_count == 2


def test_cancel_after_journaling_recovers(saved):
    project, operation, provider = saved
    cancel = Event()
    run(saved, cancel=cancel, deliver=lambda _: cancel.set())
    assert cancel.is_set()
    reopened = Project.load(project.path)
    run((reopened, operation, provider), apply=True)
    assert provider.call_count == 1


def test_failure_is_not_a_recoverable_success(saved):
    _, _, provider = saved
    provider.side_effect = RuntimeError("decode failed")
    cache, outcomes = run(saved)
    assert outcomes[0].status == "failed" and outcomes[0].record_json
    assert not cache.results
    assert cache.transient_outcomes
    provider.side_effect = None
    _, outcomes = run(saved)
    assert outcomes[0].status == "succeeded"
    assert provider.call_count == 2


def test_saved_scalar_reuses_even_when_job_history_is_missing(saved, monkeypatch):
    from core.jobs.store import JobStore

    project, _, provider = saved
    run(saved, apply=True)
    project.save()
    monkeypatch.setattr(JobStore, "get_result", lambda *args: None)
    cache, outcomes = run(saved)
    assert outcomes[0].status == "skipped"
    assert cache.transient_outcomes
    assert provider.call_count == 1


def test_sequence_worker_recovers_scalar_prerequisites(saved):
    from ui.workers.sequence_worker import SequenceWorker

    project, operation, provider = saved
    clip, source = project.clips[0], project.sources[0]
    results, errors = [], []
    for _ in range(2):
        worker = SequenceWorker(operation, [(clip, source)], project=project)
        worker.sequence_ready.connect(results.append)
        worker.error.connect(errors.append)
        worker.run()
        worker.prerequisite_job.validate_project(project)
    assert not errors and len(results) == 2
    assert provider.call_count == 1
    assert operation in results[0][0][0].analysis_records
    assert not clip.analysis_records


def test_sequence_scalar_repeated_occurrences_recover(saved):
    from copy import deepcopy
    from core.jobs.sequence_scalars import SequenceScalarJob

    project, operation, provider = saved
    pair = (project.clips[0], project.sources[0])
    for _ in range(2):
        job = SequenceScalarJob([pair, pair], operation=operation, project=project)
        copies = deepcopy([pair, pair])
        job.populate(copies, Event())
        assert len(job.outcomes) == 2
        assert all(operation in clip.analysis_records for clip, _ in copies)
        job.validate_project(project)
    assert provider.call_count == 2


def test_sequence_worker_cancel_during_scalar_compute_emits_nothing(saved):
    from ui.workers.sequence_worker import SequenceWorker

    project, operation, provider = saved
    worker = SequenceWorker(operation, [(project.clips[0], project.sources[0])], project=project)
    results, errors = [], []
    worker.sequence_ready.connect(results.append)
    worker.error.connect(errors.append)
    provider.side_effect = lambda *args, **kwargs: (worker.cancel(), 0.5)[1]
    worker.run()
    assert not results and not errors
    assert not project.clips[0].analysis_records


def test_scalar_recovery_authenticates_record_identity(saved, monkeypatch):
    from hashlib import sha256
    import json
    from core.jobs.store import JobStore
    from core.jobs.commits import StaleJobResult, canonical_json

    run(saved)
    get_result = JobStore.get_result

    def changed_record(store, result_id):
        row = get_result(store, result_id)
        if row is None:
            return None
        row = dict(row)
        payload = json.loads(row["payload_json"])
        record = json.loads(payload["record_json"])
        record["identity"]["parameters"] = {"unexpected": True}
        payload["record_json"] = json.dumps(record)
        row["payload_json"] = canonical_json(payload)
        row["payload_digest"] = sha256(row["payload_json"].encode()).hexdigest()
        return row

    monkeypatch.setattr(JobStore, "get_result", changed_record)
    with pytest.raises(StaleJobResult, match="does not match"):
        run(saved)


@pytest.mark.parametrize("change", ["media", "range", "value", "path", "session", "replacement"])
def test_sequence_scalar_delivery_rejects_changed_owner(saved, change, tmp_path):
    from copy import deepcopy
    from core.jobs.sequence_scalars import SequenceScalarJob
    from core.jobs.commits import StaleJobResult

    project, operation, _ = saved
    clip, source = project.clips[0], project.sources[0]
    job = SequenceScalarJob([(clip, source)], operation=operation, project=project)
    job.populate(deepcopy([(clip, source)]), Event())
    if change == "media":
        source.file_path.write_bytes(b"changed")
    elif change == "range":
        clip.end_frame += 1
    elif change == "value":
        setattr(clip, test_scalar_records.FIELDS[operation], 0.1)
    elif change == "path":
        project.path = tmp_path / "other.sceneripper"
    elif change == "session":
        project.session.session_id = "changed"
    else:
        project.clips[0] = deepcopy(clip)
        project._clips_by_id = None
    with pytest.raises(StaleJobResult):
        job.validate_project(project)
