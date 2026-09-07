"""Shared cinematography options, scheduling and detached result contracts."""

from threading import Event, Thread, get_ident
from unittest.mock import Mock

import pytest

from core.operations.cinematography import (
    CinematographyOptions,
    CinematographyTask,
    run_cinematography,
)
from core.spine.analyze import cinematography
from models.cinematography import CinematographyAnalysis
from tests.test_description_operations import project_with_thumbnails
from ui.workers.cinematography_worker import CinematographyWorker

OPTIONS = CinematographyOptions("cloud", "frame", "cloud-model", "local-model", 2)


def tasks(project):
    source = project.sources[0]
    return tuple(
        CinematographyTask(
            c.id,
            c.thumbnail_path,
            source.file_path,
            c.start_frame,
            c.end_frame,
            source.fps,
        )
        for c in project.clips
    )


def test_gui_spine_share_provider_arguments(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=CinematographyAnalysis(shot_size="CU"))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    worker = CinematographyWorker(
        project.clips, project.sources_by_id, mode="video", model="selected"
    )
    worker.run()
    result = cinematography(project, mode="video", model="selected")
    assert provider.call_args_list[0] == provider.call_args_list[1]
    assert result["result"]["succeeded"] == [{"clip_id": "c-0", "shot_size": "CU"}]
    assert worker.result[0].analysis.shot_size == "CU"


def test_invalid_provider_result_is_failure_on_both_surfaces(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography", lambda **_: None
    )
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    worker.run()
    result = cinematography(project)
    assert worker.result[0].status == "failed"
    assert result["result"]["failed"][0]["code"] == "cinematography_failed"


@pytest.mark.parametrize("tier,limit", [("cloud", 2), ("local", 1)])
def test_cancel_bounds_admission_and_settles_results(
    tmp_path, monkeypatch, tier, limit
):
    from dataclasses import replace

    project = project_with_thumbnails(tmp_path, 5)
    cancel, entered, release = Event(), Event(), Event()
    calls, outcomes, delivered = [], [], []

    def provider(**kwargs):
        calls.append(get_ident())
        if len(calls) == limit:
            entered.set()
        assert release.wait(5)
        return CinematographyAnalysis()

    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    thread = Thread(
        target=lambda: outcomes.extend(
            run_cinematography(
                tasks(project),
                replace(OPTIONS, tier=tier),
                cancel_event=cancel,
                on_outcome=delivered.append,
            )
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        cancel.set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(calls) == limit
    assert not delivered
    assert all(o.status == "unprocessed" for o in outcomes)


def test_cancelled_worker_completes_once(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    completed = []
    worker.analysis_completed.connect(completed.append)
    worker.cancel()
    worker.run()
    assert completed == [{}]


def test_cancelled_worker_completion_does_not_advance_pipeline(tmp_path):
    from PySide6.QtCore import QObject
    from ui.workers.analysis_pipeline_delivery import bind_pipeline_completion

    project = project_with_thumbnails(tmp_path, 1)
    window = QObject()
    window.project = project
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    window.cinematography_worker = worker
    advanced = []
    bind_pipeline_completion(
        window,
        worker,
        "cinematography_worker",
        worker.analysis_completed,
        lambda: advanced.append(True),
    )
    worker.cancel()
    worker.run()
    assert advanced == []


def test_options_remain_pinned_after_settings_change(tmp_path, monkeypatch):
    from core.settings import Settings

    settings = Settings(
        cinematography_tier="cloud",
        cinematography_model="original",
        cinematography_input_mode="video",
        cinematography_local_model="local-original",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = project_with_thumbnails(tmp_path, 1)
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    settings.cinematography_tier = "local"
    settings.cinematography_model = "changed"
    settings.cinematography_input_mode = "frame"
    settings.cinematography_local_model = "local-changed"
    provider = Mock(return_value=CinematographyAnalysis())
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    worker.run()
    kwargs = provider.call_args.kwargs
    assert (kwargs["tier"], kwargs["mode"], kwargs["model"], kwargs["local_model"]) == (
        "cloud",
        "video",
        "original",
        "local-original",
    )


def test_local_provider_uses_cinematography_model(monkeypatch, tmp_path):
    from core.analysis.cinematography import analyze_cinematography_local
    from core.settings import Settings

    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: True)
    monkeypatch.setattr(
        "core.analysis.cinematography.load_settings",
        lambda: Settings(
            cinematography_local_model="cinema-model",
            description_model_local="other-model",
        ),
    )
    provider = Mock(return_value='{"shot_size": "CU"}')
    monkeypatch.setattr("core.analysis.description.describe_frame_local", provider)
    result = analyze_cinematography_local(tmp_path / "image.jpg")
    assert provider.call_args.kwargs["model_name"] == "cinema-model"
    assert result.analysis_model == "cinema-model"


def test_outcome_and_gui_signals_have_independent_payloads(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    original = CinematographyAnalysis(shot_size="CU")
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography", lambda **_: original
    )
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    final = []
    worker.clip_completed.connect(
        lambda cid, analysis: setattr(analysis, "shot_size", "ELS")
    )
    worker.analysis_completed.connect(final.append)
    worker.run()
    original.shot_size = "MS"
    assert worker.result[0].analysis.shot_size == "CU"
    assert final[0]["c-0"].shot_size == "CU"


def test_frame_target_retains_identity_and_forces_frame_mode(tmp_path, monkeypatch):
    from core.analysis_target import AnalysisTarget

    project = project_with_thumbnails(tmp_path, 1)
    target = AnalysisTarget(
        target_type="frame", id="frame-1", image_path=project.clips[0].thumbnail_path
    )
    worker = CinematographyWorker([], {}, analysis_targets=[target], mode="video")
    provider = Mock(return_value=CinematographyAnalysis())
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    worker.run()
    assert worker.tasks[0].target_type == "frame"
    assert worker.result[0].clip_id == "frame-1"
    assert provider.call_args.kwargs["mode"] == "frame"


def test_retry_wait_is_cancellable(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)

    class CancelOnWait(Event):
        def wait(self, timeout=None):
            assert timeout == 2
            self.set()
            return True

    provider = Mock(side_effect=RuntimeError("429 rate limit"))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    result = run_cinematography(tasks(project), OPTIONS, cancel_event=CancelOnWait())
    assert result[0].status == "unprocessed"
    provider.assert_called_once()


def test_local_inference_runs_on_caller_thread(tmp_path, monkeypatch):
    from dataclasses import replace

    project = project_with_thumbnails(tmp_path, 2)
    owner = get_ident()
    threads = []

    def provider(**kwargs):
        threads.append(get_ident())
        return CinematographyAnalysis()

    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    run_cinematography(tasks(project), replace(OPTIONS, tier="local", parallelism=5))
    assert threads == [owner, owner]


@pytest.mark.parametrize(
    "error,fallback", [("ffmpeg extraction failed", True), ("401 unauthorized", False)]
)
def test_video_fallback_preserves_model_and_error_policy(
    tmp_path, monkeypatch, error, fallback
):
    from core.analysis.cinematography import analyze_cinematography

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        "core.analysis.cinematography.load_settings",
        Mock(side_effect=AssertionError("options already pinned")),
    )
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_video",
        Mock(side_effect=RuntimeError(error)),
    )
    frame = Mock(return_value=CinematographyAnalysis())
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_frame", frame
    )

    def run():
        return analyze_cinematography(
            tmp_path / "image.jpg",
            video,
            0,
            30,
            30.0,
            mode="video",
            model="selected",
            tier="cloud",
            local_model="local",
        )

    if fallback:
        run()
        assert frame.call_args.kwargs["model"] == "selected"
    else:
        with pytest.raises(RuntimeError, match="unauthorized"):
            run()
        frame.assert_not_called()
