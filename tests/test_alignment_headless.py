"""Shared alignment entry points preserve model and command contracts."""

import json
import inspect
from threading import Event
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from cli.main import cli, register_commands
from core.project import Project
from core.spine.analyze import align_words
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


@pytest.fixture
def runner():
    kwargs = (
        {"mix_stderr": False}
        if "mix_stderr" in inspect.signature(CliRunner).parameters
        else {}
    )
    return CliRunner(**kwargs)


@pytest.fixture
def project(tmp_path, monkeypatch):
    register_commands()
    from types import SimpleNamespace

    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    project = _build_project(tmp_path, 2)
    for clip in project.clips:
        clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda *_: (True, [])
    )

    def extract(*args, **kwargs):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake")
        return wav

    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    monkeypatch.setattr(
        "core.analysis.alignment.align_words",
        lambda *a, **k: [WordTimestamp(0, 1, "hello", 0.9)],
    )
    return project


def test_spine_selection_skip_and_force(project):
    result = align_words(project, ["c-0", "c-0"])["result"]
    assert result["succeeded"] == [{"clip_id": "c-0", "word_count": 1}]
    assert project.clips[0].transcript[0].words[0].text == "hello"
    assert project.clips[1].transcript[0].words is None
    assert len(align_words(project, ["c-0"])["result"]["skipped"]) == 1
    assert (
        len(align_words(project, ["c-0"], skip_existing=False)["result"]["succeeded"])
        == 1
    )
    with pytest.raises(ValueError, match="Unknown alignment"):
        align_words(project, ["missing"])


def test_missing_dependency_never_installs_and_cli_exits_four(
    project, tmp_path, monkeypatch, runner
):
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda *_: (False, ["ctc_forced_aligner"]),
    )
    install = Mock()
    monkeypatch.setattr("core.feature_registry.install_for_feature", install)
    path = tmp_path / "project.json"
    assert project.save(path)
    before = path.read_bytes()
    register_commands()
    result = runner.invoke(cli, ["--json", "analyze", "align", str(path)])
    assert result.exit_code == 4, result.output
    assert (
        json.loads(result.stdout)["result"]["failed"][0]["code"] == "dependency_missing"
    )
    assert path.read_bytes() == before
    install.assert_not_called()


def test_cli_saves_only_selected_alignment(project, tmp_path, runner):
    path = tmp_path / "project.json"
    assert project.save(path)
    register_commands()
    result = runner.invoke(
        cli, ["--json", "analyze", "align", str(path), "--clip", "c-1"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"]["succeeded"] == [
        {"clip_id": "c-1", "word_count": 1}
    ]
    saved = Project.load(path)
    assert saved.clips[0].transcript[0].words is None
    assert saved.clips[1].transcript[0].words[0].probability == 0.9


def test_cancel_during_capability_check_is_unprocessed(project, monkeypatch):
    cancel = Event()

    def check(*args):
        cancel.set()
        return False, ["alignment"]

    monkeypatch.setattr("core.feature_registry.check_feature_ready", check)
    result = align_words(project, cancel_event=cancel)["result"]
    assert not result["failed"]
    assert [item["code"] for item in result["unprocessed"]] == [
        "cancelled",
        "cancelled",
    ]
