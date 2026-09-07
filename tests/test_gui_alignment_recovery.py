"""Saved GUI computations survive restart without saving unrelated edits."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.gui_alignment import GuiAlignmentCache
from core.jobs.media import media_stamp
from core.jobs.commits import StaleJobResult
from core.operations.alignment import AlignmentApplication, snapshot_alignment_tasks
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 2)
    for clip in project.clips:
        clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )

    def extract(*args, **kwargs):
        path = tmp_path / "audio.wav"
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    compute = Mock(return_value=[WordTimestamp(0, 1, "hello", 0.9)])
    monkeypatch.setattr("core.analysis.alignment.align_words", compute)
    return project, compute


def run(project, *, prepare=lambda: True, force=False, cancel=None, apply=False):
    cache = GuiAlignmentCache(
        project.path,
        project.metadata.id,
        {c.id: c.source_id for c in project.clips},
        project.metadata.job_results,
        force=force,
        media_stamps={
            source.file_path: media_stamp(source.file_path)
            for source in project.sources
        },
    )
    tasks = snapshot_alignment_tasks(
        project.clips, project.sources_by_id, skip_existing=not force
    )
    tasks = tuple(t for t in tasks if t.skip_reason is None)
    application = AlignmentApplication(project, tasks)

    def deliver(outcome):
        if apply:
            assert application.apply(project, outcome)
            receipt = cache.results[outcome.clip_id]
            project.record_job_result(receipt.result_id, receipt.digest)

    return cache.run(tasks, cancel or Event(), prepare, deliver, lambda *_: None)


def test_restart_before_delivery_reuses_recorded_words_without_preflight(setup):
    project, compute = setup
    assert len(run(project)) == 2
    prepare = Mock(side_effect=AssertionError("must use cache"))
    reopened = Project.load(project.path)
    run(reopened, prepare=prepare, apply=True)
    assert compute.call_count == 2
    assert len(reopened.metadata.job_results) == 2
    assert Project.load(project.path).clips[0].transcript[0].words is None
    assert reopened.save()
    assert len(Project.load(project.path).metadata.job_results) == 2


def test_failed_save_recovers_from_disk_snapshot(setup, monkeypatch):
    project, compute = setup
    run(project, apply=True)
    with monkeypatch.context() as patch:
        patch.setattr("core.project.save_project", lambda **_: False)
        assert not project.save()
    reopened = Project.load(project.path)
    run(reopened, apply=True)
    assert compute.call_count == 2
    assert reopened.save()


def test_successful_identical_force_advances_generation(setup):
    project, compute = setup
    for _ in range(3):
        run(project, force=True, apply=True)
        assert project.save()
    assert compute.call_count == 6
    assert len(project.metadata.job_results) == 6


def test_changed_media_during_compute_is_not_published(setup):
    project, compute = setup

    def change(*args, **kwargs):
        project.sources[0].file_path.write_bytes(b"changed")
        return []

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="media changed"):
        run(project, apply=True)
    assert not project.metadata.job_results
    assert project.clips[0].transcript[0].words is None


def test_cancellation_keeps_recorded_prefix_for_restart(setup):
    project, compute = setup
    event = Event()
    count = 0

    def compute_one(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            event.set()
        return []

    compute.side_effect = compute_one
    outcomes = run(project, cancel=event)
    assert [o.status for o in outcomes] == ["succeeded", "unprocessed"]
    compute.side_effect = None
    run(Project.load(project.path), apply=True)
    assert compute.call_count == 3


def test_empty_word_data_is_recoverable(setup):
    project, compute = setup
    compute.return_value = []
    run(project)
    run(Project.load(project.path), apply=True)
    assert compute.call_count == 2


def test_queued_publication_rejects_media_edit_with_restored_mtime(setup):
    import os

    project, _ = setup
    tasks = snapshot_alignment_tasks(project.clips, project.sources_by_id)
    application = AlignmentApplication(project, tasks)
    outcome = run(project)[0]
    path = project.sources[0].file_path
    before = path.stat()
    path.write_bytes(b"x" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not application.apply(project, outcome)
    assert project.clips[0].transcript[0].words is None


def test_queued_media_edit_prevents_dependency_preparation(setup):
    project, compute = setup
    cache = GuiAlignmentCache(
        project.path,
        project.metadata.id,
        {c.id: c.source_id for c in project.clips},
        {},
        force=False,
        media_stamps={s.file_path: media_stamp(s.file_path) for s in project.sources},
    )
    tasks = snapshot_alignment_tasks(project.clips, project.sources_by_id)
    project.sources[0].file_path.write_bytes(b"changed")
    prepare = Mock()
    with pytest.raises(StaleJobResult, match="while queued"):
        cache.run(tasks, Event(), prepare, lambda *_: None, lambda *_: None)
    prepare.assert_not_called()
    compute.assert_not_called()
