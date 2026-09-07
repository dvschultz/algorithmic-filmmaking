"""Owner-thread description publication rejects replaced and edited inputs."""

from dataclasses import replace
from threading import Thread

import pytest

from core.analysis_target import AnalysisTarget
from core.operations.description import DescriptionApplication, DescriptionOutcome
from core.spine.analyze import describe
from models.frame import Frame
from tests.test_description_operations import project_with_thumbnails
from ui.workers.description_worker import DescriptionWorker


@pytest.mark.parametrize("kind", ["clip", "frame"])
@pytest.mark.parametrize(
    "change", ["none", "description", "model", "media", "path", "replace", "session"]
)
def test_application_requires_original_inputs(tmp_path, kind, change):
    project = project_with_thumbnails(tmp_path, 1)
    if kind == "frame":
        target = Frame(id="frame-1", file_path=project.clips[0].thumbnail_path)
        project.add_frames([target])
        worker = DescriptionWorker(
            [], tier="cloud", analysis_targets=[AnalysisTarget.from_frame(target)]
        )
    else:
        target = project.clips[0]
        worker = DescriptionWorker(
            project.clips, sources=project.sources_by_id, tier="cloud"
        )
    project.mark_clean()
    application = DescriptionApplication(project, worker.tasks)
    if change == "description":
        target.description = "User edit"
    elif change == "model":
        target.description_model = "User model"
    elif change == "media":
        worker.tasks[0].thumbnail_path.write_bytes(b"changed")
    elif change == "path":
        setattr(
            target,
            "file_path" if kind == "frame" else "thumbnail_path",
            tmp_path / "other",
        )
    elif change == "replace":
        if kind == "frame":
            project._frames[0] = replace(target)
        else:
            project._clips[0] = replace(target)
        project._invalidate_caches()
    elif change == "session":
        project.clear()
    outcome = DescriptionOutcome(target.id, "succeeded", "Generated", "model")
    assert application.apply(project, outcome) is (change == "none")
    assert application.apply(project, outcome) is False
    if change == "none":
        assert target.description == "Generated"
        assert target.description_model == "model"
        assert project.is_dirty
    elif change == "description":
        assert target.description == "User edit"
    else:
        assert target.description is None


@pytest.mark.parametrize("change", ["range", "fps", "source_path", "source_media"])
def test_clip_source_changes_reject_results(tmp_path, change):
    project = project_with_thumbnails(tmp_path, 1)
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, tier="cloud"
    )
    application = DescriptionApplication(project, worker.tasks)
    if change == "range":
        project.clips[0].start_frame += 1
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "source_path":
        project.sources[0].file_path = tmp_path / "other.mp4"
    else:
        project.sources[0].file_path.write_bytes(b"changed")
    assert not application.apply(
        project, DescriptionOutcome("c-0", "succeeded", "Late", "model")
    )


def test_spine_rejects_edit_during_compute(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *a, **kw: ("Generated", "model"),
    )

    def progress(*args):
        project.clips[0].description = "User edit"

    result = describe(project, tier="cloud", progress_callback=progress)["result"]
    assert result["failed"] == [{"clip_id": "c-0", "code": "stale_result"}]
    assert project.clips[0].description == "User edit"


def test_publication_requires_owner_thread(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, tier="cloud"
    )
    application = DescriptionApplication(project, worker.tasks)
    errors = []

    def publish():
        try:
            application.apply(
                project, DescriptionOutcome("c-0", "succeeded", "Wrong thread", "model")
            )
        except RuntimeError as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()
    thread.join(5)
    assert errors
    assert project.clips[0].description is None


def test_thumbnail_only_result_can_apply_with_unchanged_missing_source(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    project.sources[0].file_path.unlink()
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, tier="local"
    )
    application = DescriptionApplication(project, worker.tasks)
    assert application.apply(
        project,
        DescriptionOutcome("c-0", "succeeded", "Thumbnail description", "local"),
    )
