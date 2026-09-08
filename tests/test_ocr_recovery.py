"""Saved OCR jobs reuse verified results across save/checkpoint failures."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.ocr import ocr_job_spec, run_ocr_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.ocr import OcrOptions
from core.project import Project
from models.clip import ExtractedText
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    provider = Mock(
        side_effect=lambda **kw: [
            ExtractedText(kw["clip"].start_frame, "SIGN", 0.87654321, "vlm")
        ]
    )
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", provider)
    yield path, store, provider
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_ocr_job(
        store,
        path,
        None,
        lambda *_: None,
        Event(),
        options=OcrOptions(vlm_model="test"),
        **kwargs,
    )["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_after_reopen(setup, force):
    path, store, provider = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert Project.load(path).clips[0].extracted_texts is None
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        provider.side_effect = AssertionError("recomputed")
        assert len(run((path, reopened, provider), force=force)["succeeded"]) == 2
    finally:
        reopened.close()
    assert Project.load(path).clips[0].extracted_texts[0].confidence == 0.87654321


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_empty_observations(setup, force):
    path, store, provider = setup
    provider.side_effect = None
    provider.return_value = []
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert Project.load(path).clips[0].extracted_texts == []
    assert len(run(setup, force=force)["skipped"]) == 2
    assert provider.call_count == 2
    assert all(
        store.get_result(rid)["committed"]
        for rid in Project.load(path).metadata.job_results
    )


def test_force_refresh_reuses_failed_generation(setup):
    path, store, provider = setup
    run(setup)
    provider.side_effect = None
    provider.return_value = []
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    provider.side_effect = AssertionError("recomputed")
    assert len(run(setup, force=True)["succeeded"]) == 2
    assert all(c.extracted_texts == [] for c in Project.load(path).clips)


def test_manual_text_edit_requires_verified_recomputation(setup):
    path, _, provider = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].extracted_texts[0].text = "EDIT"
    project.save()
    run(setup)
    assert provider.call_count == 3
    assert Project.load(path).clips[0].extracted_texts[0].text == "SIGN"


def test_verified_ocr_survives_job_cache_removal(setup):
    path, _, provider = setup
    run(setup)
    fresh = JobStore(path.parent / "fresh-jobs.db")
    try:
        result = run((path, fresh, provider))
    finally:
        fresh.close()
    assert len(result["skipped"]) == 2
    assert provider.call_count == 2


def test_saved_failure_invalidates_previous_success(setup):
    path, _, provider = setup
    run(setup)
    provider.side_effect = RuntimeError("provider failed")
    assert len(run(setup, force=True)["failed"]) == 2
    project = Project.load(path)
    assert all(
        c.analysis_records["extract_text"].state == "failed" for c in project.clips
    )
    assert all(c.extracted_texts[0].text == "SIGN" for c in project.clips)
    provider.side_effect = None
    provider.return_value = []
    assert len(run(setup)["succeeded"]) == 2


def test_source_change_invalidates_uncommitted_cache(setup):
    path, _, provider = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup)
    Project.load(path).sources[0].file_path.write_bytes(b"new source")
    run(setup)
    assert provider.call_count == 4


def test_cancel_keeps_completed_prefix(setup):
    path, store, provider = setup
    cancel = Event()

    def infer(**kw):
        if kw["clip"].id == "c-1":
            cancel.set()
        return []

    provider.side_effect = infer
    result = run_ocr_job(
        store, path, None, lambda *_: None, cancel, options=OcrOptions(vlm_model="test")
    )["result"]
    assert len(result["succeeded"]) == 1
    assert result["unprocessed"][0]["clip_id"] == "c-1"
    saved = Project.load(path)
    assert saved.clips[0].extracted_texts == []
    assert saved.clips[1].extracted_texts is None


def test_queued_model_is_resolved_once(setup, monkeypatch):
    path, store, provider = setup
    from types import SimpleNamespace

    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(description_model_cloud="original"),
    )
    spec = ocr_job_spec(
        Project.load(path), None, OcrOptions(), arguments={"clip_ids": None}
    )
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(description_model_cloud="changed"),
    )
    run_ocr_job(store, path, None, lambda *_: None, Event(), operation=spec)
    assert all(
        call.kwargs["vlm_model"] == "original" for call in provider.call_args_list
    )


def test_queued_media_change_rejected(setup):
    path, store, provider = setup
    project = Project.load(path)
    spec = ocr_job_spec(
        project, None, OcrOptions(vlm_model="test"), arguments={"clip_ids": None}
    )
    project.sources[0].file_path.write_bytes(b"changed")
    with pytest.raises(StaleJobResult):
        run_ocr_job(store, path, None, lambda *_: None, Event(), operation=spec)
    provider.assert_not_called()


def test_queued_runtime_change_rejected(setup, monkeypatch):
    path, store, provider = setup
    spec = ocr_job_spec(
        Project.load(path), None, OcrOptions(vlm_model="test"), arguments={}
    )
    monkeypatch.setattr("core.jobs.ocr._runtime", lambda: {"algorithm": "changed"})
    with pytest.raises(StaleJobResult):
        run_ocr_job(store, path, None, lambda *_: None, Event(), operation=spec)
    provider.assert_not_called()


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_receipt_fails_without_inference(setup, column):
    import sqlite3

    path, _, provider = setup
    run(setup)
    rid = next(iter(Project.load(path).metadata.job_results))
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute(
            f"UPDATE job_results SET {column} = ? WHERE result_id = ?", ("{}", rid)
        )
    provider.reset_mock()
    with pytest.raises(StaleJobResult):
        run(setup)
    provider.assert_not_called()


def test_provider_failure_is_not_journaled_as_empty(setup):
    path, _, provider = setup
    provider.side_effect = [RuntimeError("provider failed"), []]
    result = run(setup)
    assert len(result["failed"]) == 1 and len(result["succeeded"]) == 1
    saved = Project.load(path)
    assert saved.clips[0].extracted_texts is None
    assert saved.clips[1].extracted_texts == []
    assert len(saved.metadata.job_results) == 1


def test_model_failure_stops_inference(setup):
    from core.errors import ModelDownloadError

    _, _, provider = setup
    provider.side_effect = ModelDownloadError("download failed")
    result = run(setup)
    provider.assert_called_once()
    assert len(result["unprocessed"]) == 1


def test_cli_recovers_failed_save(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, provider = setup
    register_commands()
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    args = [
        "--json",
        "analyze",
        "extract-text",
        str(path),
        "--method",
        "vlm",
        "--model",
        "test",
        "--clip-id",
        "c-0",
    ]
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        assert CliRunner().invoke(cli, args).exit_code != 0
    response = CliRunner().invoke(cli, args)
    assert response.exit_code == 0, response.output
    provider.assert_called_once()
    assert Project.load(path).clips[0].extracted_texts[0].text == "SIGN"
    assert Project.load(path).clips[1].extracted_texts is None
