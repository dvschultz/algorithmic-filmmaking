"""Custom-query publication requires unchanged project and target inputs."""

from dataclasses import replace
from threading import Thread

import pytest

from core.operations.custom_query import (
    CustomQueryApplication,
    CustomQueryOutcome,
    CustomQueryTask,
)
from core.spine.analyze import custom_query
from tests.test_description_operations import project_with_thumbnails


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "edit",
        "range",
        "source",
        "media",
        "source_media",
        "replacement",
        "session",
        "query",
        "frame",
    ],
)
def test_application_rejects_stale_inputs(tmp_path, change):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.custom_queries = [{"query": "old", "match": True}]
    task = CustomQueryTask(
        clip.id,
        clip.thumbnail_path,
        "person",
        target_type="frame" if change == "frame" else "clip",
    )
    application = CustomQueryApplication(project, (task,))
    if change == "edit":
        clip.custom_queries[0]["match"] = False
    elif change == "range":
        clip.end_frame += 1
    elif change == "source":
        project.sources[0].fps += 1
    elif change == "media":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "source_media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "replacement":
        project._clips[0] = replace(clip)
        project._invalidate_caches()
    elif change == "session":
        project.clear()
    project.mark_clean()
    outcome = CustomQueryOutcome(
        clip.id,
        "other" if change == "query" else "person",
        "succeeded",
        True,
        0.98765,
        "model",
    )
    assert application.apply(project, outcome) is (change == "none")
    assert application.apply(project, outcome) is False
    assert len(clip.custom_queries) == (2 if change == "none" else 1)
    assert project.is_dirty is (change == "none")
    if change == "none":
        assert clip.custom_queries[-1]["confidence"] == 0.9877


def test_application_requires_owner_thread(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    application = CustomQueryApplication(
        project, (CustomQueryTask(clip.id, clip.thumbnail_path, "person"),)
    )
    errors = []

    def apply():
        try:
            application.apply(
                project,
                CustomQueryOutcome(clip.id, "person", "succeeded", True, 0.9, "model"),
            )
        except RuntimeError as exc:
            errors.append(exc)

    thread = Thread(target=apply)
    thread.start()
    thread.join(5)
    assert errors
    assert clip.custom_queries is None


def test_headless_rejects_edit_during_inference(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)

    def provider(**kwargs):
        project.clips[0].custom_queries = [{"query": "manual", "match": False}]
        return True, 0.9, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    monkeypatch.setattr("core.analysis.description._load_local_model", lambda *_: None)
    result = custom_query(project, query="person", tier="local")
    assert result["result"]["succeeded"] == []
    assert result["result"]["failed"] == [{"clip_id": "c-0", "code": "stale_result"}]
    assert project.clips[0].custom_queries == [{"query": "manual", "match": False}]
