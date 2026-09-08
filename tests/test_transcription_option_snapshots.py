"""Queue and receipt identities freeze the cloud model used by inference."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.operations.transcription import (
    TranscriptionOptions,
    run_transcription,
    snapshot_tasks,
)
from core.jobs.transcription import transcription_job_spec, run_transcription_job
from core.jobs.audio_transcription import (
    audio_transcription_job_spec,
    run_audio_transcription_job,
)
from core.jobs.store import JobStore
from models.audio_source import AudioSource
from tests.test_spine_analyze import _build_project


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        cache_dir=tmp_path, transcription_cloud_model="original-model"
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = _build_project(tmp_path, 2)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    project.add_audio_source(AudioSource(id="audio", file_path=audio))
    project.save(tmp_path / "project.json")
    from core.spine.project_io import load_with_mtime

    project, _ = load_with_mtime(project.path)
    return project, settings


@pytest.mark.parametrize("audio", [False, True])
def test_saved_job_uses_queued_cloud_model_after_settings_change(
    setup, tmp_path, monkeypatch, audio
):
    project, settings = setup
    options = TranscriptionOptions(backend="groq")
    if audio:
        spec = audio_transcription_job_spec(project, "audio", options)
        frozen = TranscriptionOptions(**spec.arguments["options"])
    else:
        spec = transcription_job_spec(project, None, options, arguments={})
        frozen = TranscriptionOptions(**json.loads(spec.inputs_json)["options"])
    assert frozen.cloud_model == "original-model"
    settings.transcription_cloud_model = "later-model"
    provider = Mock(return_value=[])
    monkeypatch.setattr(
        "core.transcription.transcribe_video"
        if audio
        else "core.transcription.transcribe_clip",
        provider,
    )
    store = JobStore(tmp_path / "jobs.db")
    try:
        if audio:
            result = run_audio_transcription_job(
                store,
                project.path,
                "audio",
                frozen,
                lambda *_: None,
                Event(),
                operation=spec,
            )
        else:
            result = run_transcription_job(
                store,
                project.path,
                None,
                frozen,
                lambda *_: None,
                Event(),
                operation=spec,
            )
        assert result["success"]
        assert provider.call_count == (1 if audio else 2)
        assert all(
            call.kwargs["cloud_model"] == "original-model"
            for call in provider.call_args_list
        )
    finally:
        store.close()


def test_shared_batch_freezes_model_once_for_all_clips(setup, monkeypatch):
    project, settings = setup
    models = []

    def provider(**kwargs):
        models.append(kwargs.get("cloud_model"))
        settings.transcription_cloud_model = "changed-between-clips"
        return []

    monkeypatch.setattr("core.transcription.transcribe_clip", provider)
    outcomes = run_transcription(
        snapshot_tasks(project.clips, project.sources_by_id, skip_existing=False),
        TranscriptionOptions(backend="groq", parallelism=1),
    )
    assert all(outcome.status == "succeeded" for outcome in outcomes)
    assert models == ["original-model", "original-model"]


@pytest.mark.parametrize("audio", [False, True])
def test_gui_worker_freezes_cloud_model_at_construction(setup, audio):
    project, settings = setup
    if audio:
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        worker = AudioTranscribeWorker(
            project.audio_sources[0], project=project, backend="groq"
        )
        options = worker.options
    else:
        from ui.workers.transcription_worker import TranscriptionWorker

        worker = TranscriptionWorker(
            project.clips, project.sources[0], project=project, backend="groq"
        )
        options = worker._options
    settings.transcription_cloud_model = "later-model"
    assert options.cloud_model == "original-model"
    assert worker.cache.options.cloud_model == "original-model"


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("same_request", [False, True])
def test_failed_save_recovery_matches_the_frozen_cloud_model(
    setup, tmp_path, monkeypatch, audio, same_request
):
    from core.operations.transcription import resolve_transcription_options

    project, settings = setup
    frozen = resolve_transcription_options(TranscriptionOptions(backend="groq"))
    provider = Mock(return_value=[])
    monkeypatch.setattr(
        "core.transcription.transcribe_video"
        if audio
        else "core.transcription.transcribe_clip",
        provider,
    )
    store = JobStore(tmp_path / "jobs.db")

    def run(options):
        if audio:
            return run_audio_transcription_job(
                store, project.path, "audio", options, lambda *_: None, Event()
            )
        return run_transcription_job(
            store,
            project.path,
            [project.clips[0].id],
            options,
            lambda *_: None,
            Event(),
        )

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                "core.jobs.commits.save_with_mtime_check",
                Mock(side_effect=OSError("save failed")),
            )
            with pytest.raises(OSError, match="save failed"):
                run(frozen)
        settings.transcription_cloud_model = "later-model"
        assert run(frozen if same_request else TranscriptionOptions(backend="groq"))[
            "success"
        ]
        assert [call.kwargs["cloud_model"] for call in provider.call_args_list] == (
            ["original-model"] if same_request else ["original-model", "later-model"]
        )
    finally:
        store.close()
