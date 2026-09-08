"""Audio completion uses current provenance without owner-thread inference."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.analysis_availability import audio_transcription_is_complete
from core.operations.audio_transcription import (
    AudioTranscriptionTask,
    AudioTranscriptionApplication,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.settings import Settings
from core.spine.audio_sources import list_audio_sources
from models.audio_source import AudioSource


@pytest.fixture
def analyzed(tmp_path, monkeypatch):
    settings = Settings(
        transcription_backend="faster-whisper",
        transcription_model="small.en",
        transcription_language="en",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription.transcribe_video", Mock(return_value=[]))
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    project = Project.new()
    audio = AudioSource(id="audio", file_path=path, duration_seconds=2)
    project.add_audio_source(audio)
    task = AudioTranscriptionTask.from_audio(audio, verified=True)
    options = TranscriptionOptions(backend="faster-whisper")
    result = run_audio_transcription(task, options)
    assert AudioTranscriptionApplication(project, task, options).apply(project, result)
    return project, settings


def test_verified_silence_is_complete_without_probe_or_inference(analyzed, monkeypatch):
    project, _ = analyzed
    probe = Mock(side_effect=AssertionError("completion must not probe"))
    provider = Mock(side_effect=AssertionError("completion must not infer"))
    monkeypatch.setattr("core.transcription._has_audio_stream", probe)
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    assert audio_transcription_is_complete(project.audio_sources[0])
    assert list_audio_sources(project)["audio_sources"][0]["transcribed"]
    probe.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_mcp_audio_status_uses_verified_completion(analyzed, tmp_path, legacy):
    import asyncio
    import json
    from scene_ripper_mcp.tools.project import list_audio_sources as mcp_list
    from core.transcription_models import TranscriptSegment

    project, _ = analyzed
    if legacy:
        project.audio_sources[0].analysis_records.clear()
        project.audio_sources[0].transcript = [TranscriptSegment(start_time=0, end_time=1, text="legacy")]
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    result = json.loads(asyncio.run(mcp_list(str(path))))
    assert result["success"], result
    assert result["count"] == 1
    assert result["audio_sources"][0]["transcribed"] is (not legacy)
    assert result["audio_sources"][0]["transcript_segment_count"] == int(legacy)


@pytest.mark.parametrize(
    "change",
    ["legacy", "failure", "model", "language", "media", "path", "metadata", "value"],
)
def test_changed_audio_is_pending(analyzed, tmp_path, change):
    project, settings = analyzed
    audio = project.audio_sources[0]
    if change == "legacy":
        audio.analysis_records.clear()
    elif change == "failure":
        audio.analysis_records["transcribe"] = replace(
            audio.analysis_records["transcribe"], state="failed"
        )
    elif change == "model":
        settings.transcription_model = "medium.en"
    elif change == "language":
        settings.transcription_language = "es"
    elif change == "media":
        audio.file_path.write_bytes(b"changed")
    elif change == "path":
        audio.file_path = tmp_path / "different.wav"
    elif change == "metadata":
        audio.duration_seconds = 10
    else:
        audio.transcript = None
    assert not audio_transcription_is_complete(audio)
    assert not list_audio_sources(project)["audio_sources"][0]["transcribed"]


def test_verified_audio_card_and_settings_change(analyzed, tmp_path):
    import os
    import subprocess
    import sys

    project, _ = analyzed
    path = tmp_path / "project.json"
    project.save(path)
    project.close_writer()
    code = r"""
import sys
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QPushButton
from core.project import Project
from core.settings import Settings
from ui.widgets.audio_library_list import AudioLibraryList
app = QApplication([])
project = Project.load(Path(sys.argv[1]))
settings = Settings(transcription_backend='faster-whisper', transcription_model='small.en', transcription_language='en')
row = AudioLibraryList()
with patch('core.settings.load_settings', return_value=settings):
    row.set_sources(project.audio_sources)
    button = row._table.cellWidget(0, row._COL_TRANSCRIBE)
    assert button.text() == 'Transcribed' and button.isEnabled()
    emitted = []
    row.transcribe_requested.connect(emitted.append)
    button.click()
    assert emitted == ['audio']
    settings.transcription_model = 'medium.en'
    row.set_sources(project.audio_sources)
    button = row._table.cellWidget(0, row._COL_TRANSCRIBE)
    assert button.text() == 'Transcribe' and button.isEnabled()
project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_confirmed_no_audio_completion_does_not_reprobe(analyzed, monkeypatch):
    project, _ = analyzed
    audio = project.audio_sources[0]
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    task = AudioTranscriptionTask.from_audio(audio, verified=True)
    options = TranscriptionOptions(backend="faster-whisper")
    result = run_audio_transcription(task, options)
    assert AudioTranscriptionApplication(project, task, options).apply(project, result)
    monkeypatch.setattr(
        "core.transcription._has_audio_stream",
        Mock(side_effect=AssertionError("must not probe")),
    )
    assert audio_transcription_is_complete(audio)


def test_agent_dispatches_legacy_and_changed_settings(analyzed):
    from types import SimpleNamespace
    from core.chat_tools import transcribe_audio_source

    project, settings = analyzed
    window = SimpleNamespace(
        project=project, settings=settings, _active_audio_transcribes=set()
    )
    assert transcribe_audio_source(window, "audio")["result"]["status"] == "skipped"
    settings.transcription_model = "medium.en"
    assert (
        transcribe_audio_source(window, "audio")["_wait_for_worker"]
        == "audio_transcription"
    )
    project.audio_sources[0].analysis_records.clear()
    assert (
        transcribe_audio_source(window, "audio")["_wait_for_worker"]
        == "audio_transcription"
    )
