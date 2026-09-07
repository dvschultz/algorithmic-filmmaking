from core.operations.transcription import TranscriptionOutcome, snapshot_tasks
from tests.test_spine_analyze import _build_project


def test_application_rejects_edits_and_duplicate_delivery(tmp_path):
    from core.operations.transcription import TranscriptionApplication

    project = _build_project(tmp_path, 2)
    tasks = snapshot_tasks(project.clips, project.sources_by_id)
    application = TranscriptionApplication(project, tasks)
    project.clips[0].start_frame += 1
    assert not application.apply(project, TranscriptionOutcome("c-0", "succeeded"))
    outcome = TranscriptionOutcome("c-1", "succeeded")
    assert application.apply(project, outcome)
    assert project.clips[1].transcript == []
    assert not application.apply(project, outcome)


def test_application_rejects_replaced_media_and_project(tmp_path):
    from core.operations.transcription import TranscriptionApplication

    project = _build_project(tmp_path, 1)
    application = TranscriptionApplication(
        project, snapshot_tasks(project.clips, project.sources_by_id)
    )
    project.sources[0].file_path.write_bytes(b"replaced media")
    assert not application.apply(project, TranscriptionOutcome("c-0", "succeeded"))
    application = TranscriptionApplication(
        project, snapshot_tasks(project.clips, project.sources_by_id)
    )
    project.clear()
    assert not application.apply(project, TranscriptionOutcome("c-0", "succeeded"))


def test_application_checks_owner_and_writability_before_mutation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch
    import pytest
    from core.operations.transcription import TranscriptionApplication

    project = _build_project(tmp_path, 1)
    application = TranscriptionApplication(
        project, snapshot_tasks(project.clips, project.sources_by_id)
    )
    outcome = TranscriptionOutcome("c-0", "succeeded")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(application.apply, project, outcome).result()
    assert project.clips[0].transcript is None
    with patch.object(
        project, "_assert_writable", side_effect=RuntimeError("read-only")
    ):
        with pytest.raises(RuntimeError, match="read-only"):
            application.apply(project, outcome)
    assert project.clips[0].transcript is None
    assert application.apply(project, outcome)


def test_batch_notifies_once_and_preserves_transcript_edits(tmp_path):
    from unittest.mock import patch
    from core.operations.transcription import TranscriptionApplication

    project = _build_project(tmp_path, 3)
    application = TranscriptionApplication(
        project, snapshot_tasks(project.clips, project.sources_by_id)
    )
    edited = [{"text": "manual edit"}]
    project.clips[0].transcript = edited
    outcomes = tuple(
        TranscriptionOutcome(clip.id, "succeeded") for clip in project.clips
    )
    with patch.object(project, "update_clips", wraps=project.update_clips) as update:
        assert application.apply_batch(project, outcomes) == (False, True, True)
        update.assert_called_once_with(project.clips[1:])
    assert project.clips[0].transcript is edited
