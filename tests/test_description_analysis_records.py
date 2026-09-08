"""Description reuse is tied to content, prompt, and actual provider execution."""

from dataclasses import replace
import json

import pytest

from core.operations.description import (
    DescriptionApplication,
    DescriptionOptions,
    description_task,
    run_description,
)
from models.analysis_record import AnalysisRecord
from tests.test_description_operations import project_with_thumbnails


OPTIONS = DescriptionOptions("cloud", model="gemini-test", input_mode="frame")


def compute(project, options=OPTIONS):
    task = description_task(project.clips[0], project.sources[0])
    return task, run_description((task,), options)[0]


def provider(monkeypatch, *, execution=None):
    calls = []

    def describe(*args, **kwargs):
        calls.append(kwargs)
        if execution is not None:
            kwargs["on_execution"](execution)
        return "A person walks", "gemini-test"

    monkeypatch.setattr("core.analysis.description.describe_frame", describe)
    return calls


def apply(project, task, outcome):
    assert DescriptionApplication(project, (task,)).apply(project, outcome)


def test_verified_result_reuses_without_inference(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = provider(monkeypatch)
    task, first = compute(project)
    apply(project, task, first)
    task, second = compute(project)
    assert second.status == "skipped"
    assert second.code == "valid_analysis"
    assert len(calls) == 1
    apply(project, task, second)
    assert project.clips[0].description == "A person walks"


@pytest.mark.parametrize(
    "change", ["prompt", "model", "range", "fps", "image", "video", "projection"]
)
def test_semantic_changes_invalidate_description(tmp_path, monkeypatch, change):
    project = project_with_thumbnails(tmp_path, 1)
    calls = provider(monkeypatch)
    task, outcome = compute(project)
    apply(project, task, outcome)
    options = OPTIONS
    if change == "prompt":
        options = replace(options, prompt="Describe the action only")
    elif change == "model":
        options = replace(options, model="gemini-other")
    elif change == "range":
        project.clips[0].end_frame += 1
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "image":
        project.clips[0].thumbnail_path.write_bytes(b"different image")
    elif change == "video":
        project.sources[0].file_path.write_bytes(b"different source")
    else:
        project.clips[0].description = "User replacement"
    assert compute(project, options)[1].status == "succeeded"
    assert len(calls) == 2


def test_legacy_text_does_not_qualify_for_verified_reuse(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].description = "Old text"
    calls = provider(monkeypatch)
    assert compute(project)[1].status == "succeeded"
    assert len(calls) == 1


def test_video_fallback_does_not_satisfy_video_request(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = provider(
        monkeypatch,
        execution={"backend": "cloud", "model": "gemini-test", "input_mode": "frame"},
    )
    options = replace(OPTIONS, input_mode="video")
    task, first = compute(project, options)
    record = AnalysisRecord.from_dict(json.loads(first.record_json))
    assert record.identity.to_dict()["model"]["execution"]["input_mode"] == "frame"
    apply(project, task, first)
    assert compute(project, options)[1].status == "succeeded"
    assert len(calls) == 2


def test_failure_records_keep_display_text_but_force_retry(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider(monkeypatch)
    task, first = compute(project)
    apply(project, task, first)

    def fail(*args, **kwargs):
        raise RuntimeError("Invalid provider response")

    monkeypatch.setattr("core.analysis.description.describe_frame", fail)
    options = replace(OPTIONS, prompt="New prompt")
    task, failed = compute(project, options)
    assert failed.status == "failed"
    apply(project, task, failed)
    assert project.clips[0].analysis_records["describe"].state == "failed"
    assert project.clips[0].description == "A person walks"
    calls = provider(monkeypatch)
    assert compute(project, options)[1].status == "succeeded"
    assert len(calls) == 1


def test_changed_media_during_inference_cannot_publish(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)

    def mutate(*args, **kwargs):
        project.clips[0].thumbnail_path.write_bytes(b"changed")
        return "Stale description", "gemini-test"

    monkeypatch.setattr("core.analysis.description.describe_frame", mutate)
    _, outcome = compute(project)
    assert outcome.status == "failed"
    assert outcome.record_json is None


def test_parallelism_does_not_invalidate_semantic_reuse(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = provider(monkeypatch)
    task, first = compute(project)
    apply(project, task, first)
    assert compute(project, replace(OPTIONS, parallelism=4))[1].status == "skipped"
    assert len(calls) == 1


def test_application_rejects_different_prompt(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider(monkeypatch)
    task, outcome = compute(project)
    assert not DescriptionApplication(
        project, (task,), replace(OPTIONS, prompt="Another prompt")
    ).apply(project, outcome)
    assert project.clips[0].description is None


def test_application_rejects_changed_prior_record(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider(monkeypatch)
    task, outcome = compute(project)
    project.clips[0].analysis_records["describe"] = AnalysisRecord.legacy(
        {"description": "Other"}
    )
    assert not DescriptionApplication(project, (task,)).apply(project, outcome)


def test_blank_response_is_a_failed_record(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *a, **kw: ("  ", "gemini-test"),
    )
    task, outcome = compute(project)
    assert outcome.status == "failed"
    apply(project, task, outcome)
    assert project.clips[0].analysis_records["describe"].state == "failed"


def test_identical_relocated_media_reuses_and_refreshes_bindings(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = provider(monkeypatch)
    task, first = compute(project)
    apply(project, task, first)
    old = project.clips[0].analysis_records["describe"]
    source = project.sources[0]
    moved = tmp_path / "moved.mp4"
    moved.write_bytes(source.file_path.read_bytes())
    source.file_path = moved
    task, reused = compute(project)
    assert reused.status == "skipped"
    apply(project, task, reused)
    current = project.clips[0].analysis_records["describe"]
    assert current.identity == old.identity
    assert current.input_json != old.input_json
    assert len(calls) == 1


def test_unified_target_preserves_description_projection(tmp_path, monkeypatch):
    from core.analysis_target import AnalysisTarget

    project = project_with_thumbnails(tmp_path, 1)
    provider(monkeypatch)
    task, first = compute(project)
    apply(project, task, first)
    target = AnalysisTarget.from_clip(project.clips[0], project.sources[0])
    task = description_task(target)
    assert run_description((task,), OPTIONS)[0].status == "skipped"


def test_headless_description_reuses_verified_records_and_retries_legacy(
    tmp_path, monkeypatch
):
    from core.spine.analyze import describe

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].description = "Legacy text"
    calls = provider(monkeypatch)
    first = describe(project, tier="cloud")["result"]
    second = describe(project, tier="cloud")["result"]
    assert len(first["succeeded"]) == 1
    assert second["skipped"] == [{"clip_id": "c-0", "reason": "valid_analysis"}]
    assert len(calls) == 1


def test_frame_target_reuses_verified_description(tmp_path, monkeypatch):
    from core.analysis_target import AnalysisTarget
    from models.frame import Frame

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(id="frame", file_path=project.clips[0].thumbnail_path)
    project.add_frames([frame])
    calls = provider(monkeypatch)
    task = description_task(AnalysisTarget.from_frame(frame))
    first = run_description((task,), OPTIONS)[0]
    apply(project, task, first)
    task = description_task(AnalysisTarget.from_frame(frame))
    reused = run_description((task,), OPTIONS)[0]
    assert reused.status == "skipped"
    apply(project, task, reused)
    assert frame.description == "A person walks"
    assert len(calls) == 1


def test_runtime_change_during_inference_is_not_success(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    runtime = {
        "execution": {
            "backend": "cloud",
            "model": "gemini-test",
            "input_mode": "frame",
        },
        "packages": {"litellm": "before"},
    }
    monkeypatch.setattr(
        "core.operations.description.description_runtime",
        lambda *args: {**runtime, "packages": dict(runtime["packages"])},
    )

    def mutate(*args, **kwargs):
        runtime["packages"]["litellm"] = "after"
        return "Stale output", "gemini-test"

    monkeypatch.setattr("core.analysis.description.describe_frame", mutate)
    _, outcome = compute(project)
    assert outcome.status == "failed"
    assert "runtime changed" in outcome.message
