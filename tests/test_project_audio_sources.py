"""Tests for AudioSource integration with Project state."""

import json
from pathlib import Path

from core.project import Project, save_project
from models.audio_source import AudioSource


def _make_audio(path: Path, **kwargs) -> AudioSource:
    return AudioSource(
        id=kwargs.pop("id", "aud-1"),
        file_path=path,
        duration_seconds=kwargs.pop("duration_seconds", 60.0),
        sample_rate=kwargs.pop("sample_rate", 44100),
        channels=kwargs.pop("channels", 2),
        **kwargs,
    )


class TestProjectAudioSourceMutators:
    def test_add_audio_source_appends_and_invalidates_cache(self, tmp_path):
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")

        project = Project()
        events: list[tuple[str, object]] = []
        project.add_observer(lambda event, data: events.append((event, data)))

        audio = _make_audio(audio_file)
        project.add_audio_source(audio)

        assert project.audio_sources == [audio]
        assert project.audio_sources_by_id == {"aud-1": audio}
        # Mutator notifies observers
        assert any(e[0] == "audio_sources_changed" for e in events)

    def test_add_then_remove_audio_source(self, tmp_path):
        a1 = _make_audio(tmp_path / "a.wav", id="a1")
        a2 = _make_audio(tmp_path / "b.wav", id="a2")
        project = Project()
        project.add_audio_source(a1)
        project.add_audio_source(a2)

        removed = project.remove_audio_source("a1")
        assert removed is a1
        assert project.audio_sources == [a2]
        assert "a1" not in project.audio_sources_by_id

    def test_remove_unknown_audio_source_returns_none(self):
        project = Project()
        assert project.remove_audio_source("does-not-exist") is None

    def test_get_audio_source(self):
        a1 = _make_audio(Path("/tmp/song.wav"), id="a1")
        project = Project()
        project.add_audio_source(a1)
        assert project.get_audio_source("a1") is a1
        assert project.get_audio_source("missing") is None

    def test_audio_sources_does_not_affect_sources_observer(self):
        """Adding audio sources fires audio_sources_changed, not source_added."""
        project = Project()
        events: list[str] = []
        project.add_observer(lambda event, data: events.append(event))

        project.add_audio_source(_make_audio(Path("/tmp/x.wav")))

        assert "audio_sources_changed" in events
        assert "source_added" not in events

    def test_clear_resets_audio_sources(self):
        project = Project()
        project.add_audio_source(_make_audio(Path("/tmp/x.wav")))
        assert len(project.audio_sources) == 1

        project.clear()
        assert project.audio_sources == []


class TestProjectAudioSourcePersistence:
    def test_save_load_round_trip(self, tmp_path):
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"")
        project_path = tmp_path / "test.sceneripper"

        project = Project(path=project_path)
        project.add_audio_source(_make_audio(audio_file, id="aud-1", duration_seconds=120.5))
        assert project.save(project_path)

        # Reload
        loaded = Project.load(project_path)
        assert len(loaded.audio_sources) == 1
        loaded_audio = loaded.audio_sources[0]
        assert loaded_audio.id == "aud-1"
        assert loaded_audio.duration_seconds == 120.5
        assert loaded_audio.file_path.resolve() == audio_file.resolve()

    def test_legacy_project_without_audio_sources_loads_cleanly(self, tmp_path):
        """Loading a project that predates audio_sources defaults to empty."""
        project_path = tmp_path / "legacy.sceneripper"
        legacy_data = {
            "id": "proj-1",
            "project_name": "Legacy",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-01-01T00:00:00",
            "sources": [],
            "clips": [],
            "sequence": None,
        }
        project_path.write_text(json.dumps(legacy_data))

        project = Project.load(project_path)
        assert project.audio_sources == []

    def test_malformed_audio_source_entry_is_skipped(self, tmp_path):
        """A malformed audio_sources entry is logged and skipped, not fatal."""
        project_path = tmp_path / "p.sceneripper"
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")
        data = {
            "id": "proj-1",
            "project_name": "Test",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-01-01T00:00:00",
            "sources": [],
            "clips": [],
            "sequence": None,
            "audio_sources": [
                "not-a-dict",  # garbage
                {  # valid one mixed in
                    "id": "aud-good",
                    "file_path": str(audio_file),
                    "duration_seconds": 30.0,
                    "sample_rate": 44100,
                    "channels": 2,
                },
            ],
        }
        project_path.write_text(json.dumps(data))

        project = Project.load(project_path)
        # The good one survives; the garbage is skipped.
        assert len(project.audio_sources) == 1
        assert project.audio_sources[0].id == "aud-good"

    def test_save_project_function_writes_audio_sources_field(self, tmp_path):
        """save_project() writes the audio_sources field when supplied."""
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")
        project_path = tmp_path / "p.sceneripper"

        save_project(
            filepath=project_path,
            sources=[],
            clips=[],
            sequence=None,
            audio_sources=[_make_audio(audio_file, id="aud-1")],
        )

        with open(project_path) as f:
            data = json.load(f)
        assert "audio_sources" in data
        assert len(data["audio_sources"]) == 1
        assert data["audio_sources"][0]["id"] == "aud-1"

    def test_save_project_omits_audio_sources_field_when_empty(self, tmp_path):
        """save_project() does not emit an audio_sources key when none are present."""
        project_path = tmp_path / "p.sceneripper"
        save_project(
            filepath=project_path,
            sources=[],
            clips=[],
            sequence=None,
        )
        with open(project_path) as f:
            data = json.load(f)
        assert "audio_sources" not in data
