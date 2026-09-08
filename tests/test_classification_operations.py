"""Classification parity, immutable results, and cancellable model admission."""

from threading import Event, Thread, get_ident
from unittest.mock import Mock

import pytest

from core.operations.classification import (
    ClassificationOptions,
    ClassificationTask,
    run_classification,
)
from core.spine.analyze import classify_content
from tests.test_description_operations import project_with_thumbnails
from ui.workers.classification_worker import ClassificationWorker


def test_verified_empty_classification_is_reused(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    assert classify_content(project)["result"]["succeeded"]
    record = project.clips[0].analysis_records["classify"]
    assert record.state == "succeeded" and record.value == {"object_labels": []}
    assert classify_content(project)["result"]["skipped"]
    assert provider.call_count == 1


def test_relocated_identical_image_reuses_without_inventing_confidence(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[("person", 0.8)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    classify_content(project)
    original = project.clips[0].analysis_records["classify"]
    moved = tmp_path / "moved.jpg"
    moved.write_bytes(project.clips[0].thumbnail_path.read_bytes())
    project.clips[0].thumbnail_path = moved
    assert classify_content(project)["result"]["skipped"]
    current = project.clips[0].analysis_records["classify"]
    assert current.identity == original.identity
    assert current.input_json != original.input_json
    assert provider.call_count == 1


def test_runtime_change_invalidates_classification(tmp_path, monkeypatch):
    from core.analysis_model_identity import classification_runtime

    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    classify_content(project)
    runtime = {**classification_runtime(), "weights": "next-version"}
    monkeypatch.setattr(
        "core.operations.classification.classification_runtime", lambda: runtime
    )
    assert classify_content(project)["result"]["succeeded"]
    assert provider.call_count == 2


@pytest.mark.parametrize("change", ["range", "source", "image", "options", "labels"])
def test_classification_changes_require_recomputation(tmp_path, monkeypatch, change):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[("person", 0.9)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    classify_content(project)
    kwargs = {}
    if change == "range":
        project.clips[0].end_frame -= 1
    elif change == "source":
        project.sources[0].file_path.write_bytes(b"new source")
    elif change == "image":
        project.clips[0].thumbnail_path.write_bytes(b"new image")
    elif change == "options":
        kwargs["top_k"] = 2
    else:
        project.clips[0].object_labels = ["edited"]
    assert classify_content(project, **kwargs)["result"]["succeeded"]
    assert provider.call_count == 2


def test_failed_classification_records_attempt_without_erasing_labels(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[("person", 0.9)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    classify_content(project)
    provider.side_effect = RuntimeError("model failed")
    assert classify_content(project, skip_existing=False)["result"]["failed"]
    assert project.clips[0].object_labels == ["person"]
    assert project.clips[0].analysis_records["classify"].state == "failed"
    provider.side_effect = None
    assert classify_content(project)["result"]["succeeded"]
    assert provider.call_count == 3


def test_gui_spine_share_inputs_and_detached_results(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    raw = [["person", 0.9]]
    provider = Mock(return_value=raw)
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    worker = ClassificationWorker(project.clips, top_k=3, threshold=0.4)
    emitted = []
    worker.labels_ready.connect(lambda _, labels: emitted.append(labels))
    worker.run()
    result = classify_content(project, top_k=3, threshold=0.4)
    assert provider.call_args_list[0] == provider.call_args_list[1]
    assert result["result"]["succeeded"] == [{"clip_id": "c-0", "label_count": 1}]
    raw[0][0] = "mutated"
    emitted[0].clear()
    assert worker.result[0].labels == (("person", 0.9),)
    assert project.clips[0].object_labels == ["person"]


@pytest.mark.parametrize(
    "raw", [None, [("person", float("nan"))], [("person", 2)], [(None, 0.8)]]
)
def test_invalid_results_fail_per_item_on_both_surfaces(tmp_path, monkeypatch, raw):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.classification.classify_frame", lambda *a, **kw: raw
    )
    worker = ClassificationWorker(project.clips)
    worker.run()
    result = classify_content(project)
    assert worker.result[0].status == "failed"
    assert result["result"]["failed"][0]["code"] == "classification_failed"
    assert project.clips[0].object_labels is None


def test_cancelled_waiter_does_not_enter_singleton_inference(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    task = ClassificationTask("c", image)
    entered, release, cancel = Event(), Event(), Event()
    second_entered = Event()
    calls, first, second = [], [], []

    def provider(*a, **kw):
        calls.append(get_ident())
        if len(calls) > 1:
            second_entered.set()
        entered.set()
        assert release.wait(5)
        return [("person", 0.9)]

    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    thread = Thread(
        target=lambda: first.extend(
            run_classification((task,), ClassificationOptions())
        )
    )
    waiter = Thread(
        target=lambda: second.extend(
            run_classification((task,), ClassificationOptions(), cancel_event=cancel)
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        waiter.start()
        assert not second_entered.wait(0.1)
        cancel.set()
        waiter.join(2)
        assert not waiter.is_alive()
        assert second[0].status == "unprocessed"
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(5)
        if waiter.ident is not None:
            waiter.join(5)
    assert first[0].status == "succeeded"


def test_cancel_during_inference_suppresses_results_and_remaining_calls(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 3)
    worker = ClassificationWorker(project.clips, parallelism=4)
    calls, emitted, completed = [], [], []

    def provider(*a, **kw):
        calls.append(True)
        worker.cancel()
        return [("person", 0.9)]

    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    worker.labels_ready.connect(lambda *args: emitted.append(args))
    worker.classification_completed.connect(lambda: completed.append(True))
    worker.run()
    assert calls == [True]
    assert emitted == []
    assert completed == [True]
    assert all(outcome.status == "unprocessed" for outcome in worker.result)


def test_empty_labels_are_successful_and_failures_do_not_abort_batch(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 2)
    monkeypatch.setattr(
        "core.analysis.classification.classify_frame",
        Mock(side_effect=[RuntimeError("model unavailable"), []]),
    )
    worker = ClassificationWorker(project.clips)
    completed = []
    worker.classification_completed.connect(lambda: completed.append(True))
    worker.run()
    assert [o.status for o in worker.result] == ["failed", "succeeded"]
    assert worker.result[1].labels == ()
    assert completed == [True]


def test_pre_cancelled_worker_completes_without_loading_model(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    worker = ClassificationWorker(project.clips)
    preload = Mock()
    monkeypatch.setattr("core.analysis.classification._load_model", preload)
    completed = []
    worker.classification_completed.connect(lambda: completed.append(True))
    worker.cancel()
    worker.run()
    assert completed == [True]
    preload.assert_not_called()


def test_frame_task_retains_target_type(tmp_path):
    from core.analysis_target import AnalysisTarget
    from models.frame import Frame

    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    frame = Frame(id="frame", file_path=image)
    worker = ClassificationWorker(
        [], analysis_targets=[AnalysisTarget.from_frame(frame)]
    )
    assert worker.tasks[0].target_type == "frame"


@pytest.mark.parametrize("confidence,expected_count", [(0.9, 1), (float("nan"), 0)])
def test_cli_uses_shared_classification_and_preserves_thumbnail_size(
    tmp_path, monkeypatch, confidence, expected_count
):
    import json
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.project import Project
    from core.settings import Settings

    project = project_with_thumbnails(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    settings = Settings(cache_dir=tmp_path)
    monkeypatch.setattr("cli.commands.analyze.CLIConfig.load", lambda: settings)
    generator = Mock()
    generator.generate_clip_thumbnail.return_value = project.clips[0].thumbnail_path
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", lambda **_: generator)
    provider = Mock(return_value=[("person", confidence)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", provider)
    register_commands()
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "analyze",
            "classify",
            str(path),
            "--top-k",
            "3",
            "--threshold",
            "0.4",
        ],
    )
    assert result.exit_code == 0, result.output
    # Older Click runners merge stderr progress into captured stdout.
    payload = result.output[result.output.index("{") :]
    assert json.loads(payload)["analyzed_clips"] == expected_count
    assert generator.generate_clip_thumbnail.call_args.kwargs["width"] == 320
    assert provider.call_args.kwargs == {"top_k": 3, "threshold": 0.4}
    assert Project.load(path).clips[0].object_labels == (
        ["person"] if expected_count else None
    )
