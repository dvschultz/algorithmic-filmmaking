"""Tests for audio source agent tools (U7)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.chat_tools import (
    get_audio_source,
    import_audio_source,
    list_audio_sources,
)
from core.project import Project
from core.transcription import TranscriptSegment
from models.audio_source import AudioSource


def _audio(tmp_path, name="song.wav", **kwargs):
    file_path = tmp_path / name
    file_path.write_bytes(b"")
    kwargs.setdefault("duration_seconds", 60.0)
    kwargs.setdefault("sample_rate", 44100)
    kwargs.setdefault("channels", 2)
    return AudioSource(file_path=file_path, **kwargs)


class TestListAudioSources:
    def test_empty_project_returns_empty_list(self):
        project = Project.new()
        result = list_audio_sources(project)
        assert result == {"success": True, "audio_sources": [], "count": 0}

    def test_lists_metadata_for_each_source(self, tmp_path):
        project = Project.new()
        a1 = _audio(tmp_path, "a.mp3", id="a1", duration_seconds=120.0)
        a2 = _audio(tmp_path, "b.wav", id="a2", duration_seconds=45.0)
        project.add_audio_source(a1)
        project.add_audio_source(a2)

        result = list_audio_sources(project)

        assert result["success"] is True
        assert result["count"] == 2
        ids = [a["id"] for a in result["audio_sources"]]
        assert ids == ["a1", "a2"]
        first = result["audio_sources"][0]
        assert first["filename"] == "a.mp3"
        assert first["duration"] == 120.0
        assert first["duration_str"] == "2:00"
        assert first["sample_rate"] == 44100
        assert first["channels"] == 2
        assert first["transcribed"] is False
        assert first["transcript_segment_count"] == 0

    def test_transcribed_flag_set_when_transcript_present(self, tmp_path):
        project = Project.new()
        a = _audio(tmp_path, "a.mp3", id="a1")
        a.transcript = [
            TranscriptSegment(start_time=0.0, end_time=1.0, text="hi", confidence=0.9),
            TranscriptSegment(start_time=1.0, end_time=2.0, text="bye", confidence=0.85),
        ]
        project.add_audio_source(a)

        result = list_audio_sources(project)
        record = result["audio_sources"][0]

        assert record["transcribed"] is True
        assert record["transcript_segment_count"] == 2


class TestGetAudioSource:
    def test_unknown_id_returns_helpful_error(self):
        project = Project.new()
        result = get_audio_source(project, audio_source_id="missing")
        assert result["success"] is False
        assert "list_audio_sources" in result["error"]

    def test_returns_full_record_without_transcript(self, tmp_path):
        project = Project.new()
        a = _audio(tmp_path, "a.mp3", id="a1")
        project.add_audio_source(a)

        result = get_audio_source(project, audio_source_id="a1")

        assert result["success"] is True
        rec = result["audio_source"]
        assert rec["id"] == "a1"
        assert rec["filename"] == "a.mp3"
        assert rec["transcript"] is None

    def test_includes_transcript_segments_when_present(self, tmp_path):
        project = Project.new()
        a = _audio(tmp_path, "a.mp3", id="a1")
        a.transcript = [
            TranscriptSegment(start_time=0.0, end_time=2.5, text="hello", confidence=0.95),
        ]
        project.add_audio_source(a)

        result = get_audio_source(project, audio_source_id="a1")

        assert result["success"] is True
        segments = result["audio_source"]["transcript"]
        assert isinstance(segments, list)
        assert len(segments) == 1
        assert segments[0]["start_time"] == 0.0
        assert segments[0]["end_time"] == 2.5
        assert segments[0]["text"] == "hello"
        assert segments[0]["confidence"] == 0.95


class TestImportAudioSource:
    def test_missing_file_returns_error(self, tmp_path):
        project = Project.new()
        result = import_audio_source(project, file_path=str(tmp_path / "ghost.mp3"))
        assert result["success"] is False
        assert "File not found" in result["error"]

    def test_unsupported_extension_returns_error(self, tmp_path):
        project = Project.new()
        bad = tmp_path / "video.mp4"
        bad.write_bytes(b"")
        result = import_audio_source(project, file_path=str(bad))
        assert result["success"] is False
        assert "Unsupported audio format" in result["error"]

    def test_happy_path_adds_to_project(self, tmp_path):
        project = Project.new()
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")

        # Patch FFmpegProcessor to avoid actually running ffprobe
        with patch("core.ffmpeg.FFmpegProcessor") as mock_proc_cls:
            instance = mock_proc_cls.return_value
            instance.ffprobe_available = True
            instance.get_audio_info.return_value = {
                "duration": 90.0,
                "sample_rate": 48000,
                "channels": 1,
                "codec": "wav",
            }

            result = import_audio_source(project, file_path=str(audio_file))

        assert result["success"] is True
        assert result["filename"] == "song.wav"
        assert result["duration"] == 90.0

        # Verify it lives in the project
        new_id = result["audio_source_id"]
        new_audio = project.get_audio_source(new_id)
        assert new_audio is not None
        assert new_audio.duration_seconds == 90.0
        assert new_audio.sample_rate == 48000
        assert new_audio.channels == 1

    def test_zero_duration_rejected(self, tmp_path):
        project = Project.new()
        audio_file = tmp_path / "empty.wav"
        audio_file.write_bytes(b"")

        with patch("core.ffmpeg.FFmpegProcessor") as mock_proc_cls:
            instance = mock_proc_cls.return_value
            instance.ffprobe_available = True
            instance.get_audio_info.return_value = {
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 0,
            }
            result = import_audio_source(project, file_path=str(audio_file))

        assert result["success"] is False
        assert "zero duration" in result["error"]
        # Project should not have any audio sources
        assert project.audio_sources == []

    def test_no_audio_stream_rejected(self, tmp_path):
        project = Project.new()
        audio_file = tmp_path / "binary.wav"
        audio_file.write_bytes(b"")

        with patch("core.ffmpeg.FFmpegProcessor") as mock_proc_cls:
            instance = mock_proc_cls.return_value
            instance.ffprobe_available = True
            instance.get_audio_info.side_effect = ValueError("No audio stream found")
            result = import_audio_source(project, file_path=str(audio_file))

        assert result["success"] is False
        assert "Not an audio file" in result["error"]


class TestMCPAudioSourceTools:
    """End-to-end tests for the MCP read-only audio source tools."""

    def test_mcp_list_audio_sources(self, tmp_path):
        from scene_ripper_mcp.tools.project import list_audio_sources as mcp_list

        # Build a project file with audio sources
        project_path = tmp_path / "p.sceneripper"
        project = Project(path=project_path)
        a1 = _audio(tmp_path, "song.mp3", id="a1", duration_seconds=180.0)
        project.add_audio_source(a1)
        assert project.save(project_path)

        # Patch validate_project_path so the MCP guard accepts our test file
        from scene_ripper_mcp.tools import project as mcp_project_tools

        with patch.object(
            mcp_project_tools, "validate_project_path",
            return_value=(True, None, project_path),
        ):
            result_json = pytest.importorskip("asyncio").run(
                mcp_list(project_path=str(project_path))
            )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["audio_sources"][0]["id"] == "a1"
        assert result["audio_sources"][0]["duration"] == 180.0

    def test_mcp_get_audio_source_unknown_id(self, tmp_path):
        from scene_ripper_mcp.tools.project import get_audio_source as mcp_get
        from scene_ripper_mcp.tools import project as mcp_project_tools

        project_path = tmp_path / "p.sceneripper"
        project = Project(path=project_path)
        assert project.save(project_path)

        with patch.object(
            mcp_project_tools, "validate_project_path",
            return_value=(True, None, project_path),
        ):
            result_json = pytest.importorskip("asyncio").run(
                mcp_get(project_path=str(project_path), audio_source_id="ghost")
            )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "list_audio_sources" in result["error"]
