"""GUI OCR journals computation before owner publication and explicit saves."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.ocr import OcrApplication
from core.project import Project
from models.clip import ExtractedText
from tests.test_description_operations import project_with_thumbnails
from ui.workers.text_extraction_worker import TextExtractionWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(cache_dir=tmp_path, description_model_cloud="test"),
    )
    provider = Mock(
        side_effect=lambda **kw: [
            ExtractedText(kw["clip"].start_frame, "SIGN", 0.87654321, "vlm")
        ]
    )
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", provider)
    return project, provider


def worker_for(project, **kwargs):
    return TextExtractionWorker(
        project.clips, project.sources_by_id, project=project, **kwargs
    )


def run(project, *, apply=False, cancel=None):
    worker = worker_for(project)
    application = OcrApplication(project, worker.tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = worker.cache.results[(outcome.target_type, outcome.clip_id)]
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(worker.tasks, cancel or Event(), deliver, lambda *_: None)


@pytest.mark.parametrize("empty", [False, True])
def test_restart_reuses_then_explicit_save_checkpoints(setup, empty):
    project, provider = setup
    if empty:
        provider.side_effect = None
        provider.return_value = []
    first = run(project)
    reopened = Project.load(project.path)
    assert run(reopened, apply=True) == first
    assert provider.call_count == 2
    assert all(c.extracted_texts is None for c in Project.load(project.path).clips)
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
    finally:
        store.close()
    saved = Project.load(project.path).clips[0].extracted_texts
    assert saved == [] if empty else saved[0].confidence == 0.87654321


def test_worker_uses_durable_runtime_without_mutating_project(setup):
    project, provider = setup
    worker = worker_for(project)
    worker.run()
    assert worker.job_status == "completed"
    assert all(o.status == "succeeded" for o in worker.result)
    assert all(c.extracted_texts is None for c in project.clips)
    again = worker_for(Project.load(project.path))
    again.run()
    assert again.result == worker.result
    assert provider.call_count == 2


def test_cancelled_cache_replay_does_not_publish(setup):
    project, provider = setup
    run(project)
    cancel = Event()
    cancel.set()
    assert all(
        o.status == "unprocessed" for o in run(project, apply=True, cancel=cancel)
    )
    assert not project.metadata.job_results
    assert provider.call_count == 2


@pytest.mark.parametrize("change", ["edit", "save_as"])
def test_changed_output_or_path_is_not_checkpointed(setup, change):
    project, _ = setup
    run(project, apply=True)
    path = project.path
    if change == "edit":
        for clip in project.clips:
            clip.extracted_texts = []
    project.save(path.parent / "copy.json" if change == "save_as" else path)
    store = JobStore(path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()


def test_pipeline_skips_saved_empty_observations(setup):
    import time
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.settings import Settings
    from ui.workers.clip_analysis import ClipAnalysisController

    from core.spine.analyze import extract_text

    project, provider = setup
    provider.side_effect = None
    provider.return_value = []
    extract_text(project)
    app = QApplication.instance() or QApplication([])
    window = QObject()
    window.project = project
    window.settings = Settings(text_extraction_method="hybrid", text_extraction_vlm_model="test", description_model_cloud="test")
    controller = ClipAnalysisController(window, project.clips, ["extract_text"])
    controller.start()
    deadline = time.monotonic() + 10
    while not controller.finished and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    assert controller.finished and not controller.workers
    assert set(controller.plan.results["extract_text"].values()) == {"skipped"}
    assert provider.call_count == 2


def test_clip_frame_id_collision_keeps_recovery_and_checkpoints_separate(
    setup, monkeypatch
):
    from core.analysis_target import AnalysisTarget
    from models.frame import Frame

    project, clip_provider = setup
    clip = project.clips[0]
    frame = Frame(id=clip.id, file_path=clip.thumbnail_path)
    project.add_frames([frame])
    project.save()
    frame_provider = Mock(return_value=("", 0.9, "vlm"))
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_frame", frame_provider)

    def worker(current):
        return TextExtractionWorker(
            [],
            {},
            project=current,
            analysis_targets=[
                AnalysisTarget.from_clip(current.clips[0], current.sources[0]),
                AnalysisTarget.from_frame(current.frames[0]),
            ],
        )

    first = worker(project)
    first.run()
    reopened = Project.load(project.path)
    second = worker(reopened)
    second.run()
    assert second.result == first.result
    clip_provider.assert_called_once()
    frame_provider.assert_called_once()
    application = OcrApplication(reopened, second.tasks)
    for outcome in second.result:
        assert application.apply(reopened, outcome)
        receipt = second.cache.results[(outcome.target_type, outcome.clip_id)]
        reopened.record_job_result(receipt.result_id, receipt.digest)
    assert len(reopened.metadata.job_results) == 2
    assert reopened.save()
    saved = Project.load(project.path)
    assert saved.clips[0].extracted_texts[0].text == "SIGN"
    assert saved.frames[0].extracted_texts == []
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert all(
            store.get_result(rid)["committed"] for rid in saved.metadata.job_results
        )
    finally:
        store.close()


@pytest.mark.parametrize("change", ["media", "runtime", "model"])
def test_changed_inputs_do_not_reuse_unpublished_results(setup, monkeypatch, change):
    project, provider = setup
    run(project)
    kwargs = {}
    if change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "runtime":
        monkeypatch.setattr("core.jobs.gui_ocr._runtime", lambda: {"changed": True})
    else:
        kwargs["vlm_model"] = "different"
    worker = worker_for(project, **kwargs)
    worker.run()
    assert worker.job_status == "completed"
    assert provider.call_count == 4


def test_unsaved_worker_uses_session_only_runtime(setup):
    project, _ = setup
    project.path = None
    worker = worker_for(project)
    assert worker.cache is None
    assert worker.operation.persistence == "session_only"
    worker.run()
    assert worker.job_status == "completed"
    assert not project.metadata.job_results


def test_model_failure_stops_later_inference(setup):
    from core.errors import ModelDownloadError

    project, provider = setup
    provider.side_effect = ModelDownloadError("unavailable")
    outcomes = run(project, apply=True)
    assert [o.status for o in outcomes] == ["failed", "unprocessed"]
    provider.assert_called_once()
    assert not project.metadata.job_results


def test_pipeline_launcher_enables_recovery(setup):
    from core.settings import Settings
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    project, _ = setup
    worker, application = create_clip_analysis_worker(
        project,
        Settings(),
        "extract_text",
        project.clips,
    )
    assert worker.cache.path == project.path.resolve()
    assert application.project is project


def test_failed_inference_is_not_cached_as_empty(setup):
    project, provider = setup
    provider.side_effect = [ValueError("decode failed"), []]
    outcomes = run(project, apply=True)
    assert [o.status for o in outcomes] == ["failed", "succeeded"]
    assert project.clips[0].extracted_texts is None
    assert project.clips[1].extracted_texts == []
    assert len(project.metadata.job_results) == 1


def test_crash_after_completed_item_reuses_prefix(setup):
    project, provider = setup
    provider.side_effect = [[], SystemExit("simulated worker crash")]
    with pytest.raises(SystemExit):
        run(project)
    provider.side_effect = None
    provider.return_value = []
    assert all(o.status == "succeeded" for o in run(Project.load(project.path)))
    assert provider.call_count == 3


def test_verified_record_survives_missing_receipt(setup):
    project, provider = setup
    run(project, apply=True)
    project.save()
    (project.path.parent / "jobs.db").unlink()
    outcomes = run(Project.load(project.path))
    assert all(o.status == "skipped" and o.code == "valid_analysis" for o in outcomes)
    assert provider.call_count == 2
