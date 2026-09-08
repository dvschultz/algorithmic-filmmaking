"""Durable faces verify records independently of receipt availability."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.faces import run_face_job
from core.project import Project
from tests.test_face_recovery import setup as face_setup  # noqa: F401


@pytest.fixture
def setup(request):
    return request.getfixturevalue("face_setup")


def run(setup, **kwargs):
    path, store, _ = setup
    return run_face_job(store, path, None, lambda *_: None, Event(), **kwargs)["result"]


def test_success_and_reuse_keep_verified_records(setup):
    path, _, compute = setup
    assert len(run(setup)["succeeded"]) == 2
    assert len(run(setup)["skipped"]) == 2
    assert compute.call_count == 2
    assert all(
        c.analysis_records["face_embeddings"].provenance == "verified"
        for c in Project.load(path).clips
    )


def test_missing_receipt_rows_do_not_repeat_valid_analysis(setup, monkeypatch):
    path, store, compute = setup
    run(setup)
    assert Project.load(path).metadata.job_results
    monkeypatch.setattr(store, "get_result", lambda *_: None)
    assert len(run(setup)["skipped"]) == 2
    assert compute.call_count == 2


@pytest.mark.parametrize("change", ["legacy", "media", "weights", "fps"])
def test_changed_provenance_or_inputs_recompute(setup, change):
    path, _, compute = setup
    run(setup)
    project = Project.load(path)
    if change == "legacy":
        for clip in project.clips:
            clip.analysis_records.clear()
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"new media")
    elif change == "weights":
        pack = path.parent / "insightface" / "models" / "buffalo_l"
        (pack / "recognition.onnx").write_bytes(b"new weights")
    else:
        project.sources[0].fps = 24
    project.save()
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 4


def test_failed_force_retains_faces_and_owned_failures(setup):
    path, _, compute = setup
    run(setup)
    before = [clip.face_embeddings for clip in Project.load(path).clips]
    compute.side_effect = RuntimeError("provider failed")
    assert len(run(setup, force=True)["failed"]) == 2
    project = Project.load(path)
    assert [clip.face_embeddings for clip in project.clips] == before
    assert all(
        clip.analysis_records["face_embeddings"].state == "failed"
        for clip in project.clips
    )


@pytest.mark.parametrize("force", [False, True])
def test_first_model_download_recovers_failed_save(setup, monkeypatch, force):
    path, _, compute = setup
    pack = path.parent / "insightface" / "models" / "buffalo_l"
    contents = {item: item.read_bytes() for item in pack.glob("*.onnx")}
    for item in contents:
        item.unlink()

    def load():
        for item, content in contents.items():
            item.write_bytes(content)

    monkeypatch.setattr("core.analysis.faces._load_insightface", load)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup, force=force)
    monkeypatch.setattr(
        "core.analysis.faces._load_insightface",
        Mock(side_effect=AssertionError("must recover")),
    )
    compute.side_effect = AssertionError("must not repeat inference")
    assert len(run(setup, force=force)["succeeded"]) == 2
    assert compute.call_count == 2


def test_cancel_after_recording_retains_receipt_without_publication(setup, monkeypatch):
    path, store, compute = setup
    original = store.record_result
    cancel = Event()

    def record(*args):
        row = original(*args)
        cancel.set()
        return row

    with monkeypatch.context() as patched:
        patched.setattr(store, "record_result", record)
        result = run_face_job(store, path, None, lambda *_: None, cancel)["result"]
    assert len(result["unprocessed"]) == 2
    assert not Project.load(path).metadata.job_results
    assert all(c.face_embeddings is None for c in Project.load(path).clips)
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 2


def test_existing_model_pack_changed_during_load_is_rejected(setup, monkeypatch):
    from core.jobs.commits import StaleJobResult

    path, _, _ = setup
    weight = path.parent / "insightface" / "models" / "buffalo_l" / "recognition.onnx"
    monkeypatch.setattr(
        "core.analysis.faces._load_insightface",
        lambda: weight.write_bytes(b"changed before model opened"),
    )
    with pytest.raises(StaleJobResult):
        run(setup)
    assert not Project.load(path).metadata.job_results
    assert all(clip.face_embeddings is None for clip in Project.load(path).clips)
