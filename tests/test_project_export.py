"""Unit and integration tests for project bundle export."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.project import Project, ProjectMetadata, load_project
from core.project_export import (
    ExportResult,
    _build_filename_map,
    _strip_absolute_paths,
    export_project_bundle,
)
from models.clip import Source, Clip
from models.frame import Frame
from models.sequence import Sequence, SequenceClip


def _make_source(tmp_path: Path, name: str = "video.mp4", size: int = 1024) -> Source:
    """Create a Source with a real file on disk."""
    video_file = tmp_path / name
    video_file.write_bytes(b"\x00" * size)
    return Source(
        id=f"src-{name}",
        file_path=video_file,
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
        analyzed=True,
    )


def _make_frame(tmp_path: Path, name: str = "frame_0001.png", source_id: str = None) -> Frame:
    """Create a Frame with a real file on disk."""
    frame_file = tmp_path / name
    frame_file.write_bytes(b"\x89PNG" + b"\x00" * 100)
    return Frame(
        id=f"frm-{name}",
        file_path=frame_file,
        source_id=source_id,
        frame_number=1,
        width=1920,
        height=1080,
    )


def _make_project(sources=None, clips=None, frames=None, sequence=None, name="Test Project") -> Project:
    """Create a Project with given data."""
    return Project(
        metadata=ProjectMetadata(name=name),
        sources=sources or [],
        clips=clips or [],
        frames=frames or [],
        sequence=sequence,
    )


class TestBuildFilenameMap:
    """Tests for filename collision resolution."""

    def test_no_collisions(self, tmp_path):
        paths = [
            tmp_path / "a.mp4",
            tmp_path / "b.mp4",
            tmp_path / "c.mp4",
        ]
        result = _build_filename_map(paths, "sources")
        assert result == {
            paths[0]: "sources/a.mp4",
            paths[1]: "sources/b.mp4",
            paths[2]: "sources/c.mp4",
        }

    def test_collision_adds_suffix(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        paths = [dir_a / "video.mp4", dir_b / "video.mp4"]
        result = _build_filename_map(paths, "sources")
        assert result[paths[0]] == "sources/video.mp4"
        assert result[paths[1]] == "sources/video_2.mp4"

    def test_triple_collision(self, tmp_path):
        dirs = [tmp_path / str(i) for i in range(3)]
        for d in dirs:
            d.mkdir()
        paths = [d / "clip.mp4" for d in dirs]
        result = _build_filename_map(paths, "sources")
        assert result[paths[0]] == "sources/clip.mp4"
        assert result[paths[1]] == "sources/clip_2.mp4"
        assert result[paths[2]] == "sources/clip_3.mp4"

    def test_case_insensitive_collision(self, tmp_path):
        """Files differing only in case should be treated as collisions."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        paths = [dir_a / "Video.MP4", dir_b / "video.mp4"]
        result = _build_filename_map(paths, "sources")
        assert result[paths[0]] == "sources/Video.MP4"
        assert result[paths[1]] == "sources/video_2.mp4"

    def test_frame_subdirectory(self, tmp_path):
        paths = [tmp_path / "frame_001.png"]
        result = _build_filename_map(paths, "frames")
        assert result[paths[0]] == "frames/frame_001.png"

    def test_empty_list(self):
        result = _build_filename_map([], "sources")
        assert result == {}


class TestStripAbsolutePaths:
    """Tests for _absolute_path removal from exported JSON."""

    def test_strips_source_absolute_paths(self, tmp_path):
        json_file = tmp_path / "test.json"
        data = {
            "sources": [
                {"file_path": "sources/video.mp4", "_absolute_path": "/original/video.mp4"},
            ],
            "clips": [],
        }
        json_file.write_text(json.dumps(data))
        _strip_absolute_paths(json_file)
        result = json.loads(json_file.read_text())
        assert "_absolute_path" not in result["sources"][0]
        assert result["sources"][0]["file_path"] == "sources/video.mp4"

    def test_strips_frame_absolute_paths(self, tmp_path):
        json_file = tmp_path / "test.json"
        data = {
            "sources": [],
            "frames": [
                {"file_path": "frames/f.png", "_absolute_path": "/original/f.png"},
            ],
        }
        json_file.write_text(json.dumps(data))
        _strip_absolute_paths(json_file)
        result = json.loads(json_file.read_text())
        assert "_absolute_path" not in result["frames"][0]

    def test_no_absolute_paths_is_noop(self, tmp_path):
        json_file = tmp_path / "test.json"
        data = {"sources": [{"file_path": "sources/video.mp4"}]}
        json_file.write_text(json.dumps(data))
        _strip_absolute_paths(json_file)
        result = json.loads(json_file.read_text())
        assert result == data


class TestExportProjectBundle:
    """Tests for the main export function."""

    def test_creates_folder_structure(self, tmp_path):
        """Bundle has sources/, frames/, and .sceneripper file."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4")

        project = _make_project(sources=[source], name="MyProject")
        dest = tmp_path / "MyProject-export"

        result = export_project_bundle(project, dest)

        assert dest.is_dir()
        assert (dest / "sources").is_dir()
        assert (dest / "frames").is_dir()
        assert (dest / "MyProject.sceneripper").is_file()
        assert result.sources_copied == 1

    def test_paths_in_json_are_relative(self, tmp_path):
        """Exported JSON stores paths relative to bundle directory."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4")
        frame = _make_frame(originals, "f1.png", source_id=source.id)

        project = _make_project(sources=[source], frames=[frame], name="Test")
        dest = tmp_path / "Test-export"

        export_project_bundle(project, dest)

        project_json = json.loads((dest / "Test.sceneripper").read_text())
        source_path = project_json["sources"][0]["file_path"]
        frame_path = project_json["frames"][0]["file_path"]

        assert source_path == "sources/video.mp4"
        assert frame_path == "frames/f1.png"

    def test_absolute_paths_stripped(self, tmp_path):
        """Exported JSON must not contain _absolute_path fields."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4")
        frame = _make_frame(originals, "f1.png", source_id=source.id)

        project = _make_project(sources=[source], frames=[frame], name="Test")
        dest = tmp_path / "Test-export"

        export_project_bundle(project, dest)

        raw = (dest / "Test.sceneripper").read_text()
        assert "_absolute_path" not in raw

    def test_filename_collision_resolution(self, tmp_path):
        """Sources with same name from different dirs get unique bundle names."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        source_a = _make_source(dir_a, "interview.mp4")
        source_b = _make_source(dir_b, "interview.mp4")

        project = _make_project(sources=[source_a, source_b], name="Collision")
        dest = tmp_path / "Collision-export"

        result = export_project_bundle(project, dest)

        assert (dest / "sources" / "interview.mp4").exists()
        assert (dest / "sources" / "interview_2.mp4").exists()
        assert result.sources_copied == 2

    def test_lightweight_export_skips_videos(self, tmp_path):
        """Lightweight export writes source paths in JSON but doesn't copy files."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4", size=2048)

        project = _make_project(sources=[source], name="Light")
        dest = tmp_path / "Light-export"

        result = export_project_bundle(project, dest, include_videos=False)

        # Video not copied
        assert not (dest / "sources" / "video.mp4").exists()
        assert result.sources_copied == 0

        # But JSON still references sources/video.mp4
        project_json = json.loads((dest / "Light.sceneripper").read_text())
        assert project_json["sources"][0]["file_path"] == "sources/video.mp4"

    def test_lightweight_export_still_copies_frames(self, tmp_path):
        """Lightweight export copies frame files even when videos are excluded."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4")
        frame = _make_frame(originals, "f1.png", source_id=source.id)

        project = _make_project(sources=[source], frames=[frame], name="Light")
        dest = tmp_path / "Light-export"

        result = export_project_bundle(project, dest, include_videos=False)

        assert (dest / "frames" / "f1.png").exists()
        assert result.frames_copied == 1

    def test_missing_source_skipped_with_warning(self, tmp_path, caplog):
        """Missing source videos are skipped and logged."""
        source = Source(
            id="src-missing",
            file_path=tmp_path / "nonexistent.mp4",
            duration_seconds=30.0,
            fps=30.0,
        )
        project = _make_project(sources=[source], name="MissingSrc")
        dest = tmp_path / "MissingSrc-export"

        result = export_project_bundle(project, dest)

        assert result.sources_copied == 0
        assert len(result.sources_skipped) == 1
        assert "nonexistent.mp4" in result.sources_skipped[0]

    def test_missing_frame_skipped_with_warning(self, tmp_path, caplog):
        """Missing frame files are skipped and logged."""
        frame = Frame(
            id="frm-missing",
            file_path=tmp_path / "nonexistent.png",
            frame_number=1,
        )
        project = _make_project(frames=[frame], name="MissingFrm")
        dest = tmp_path / "MissingFrm-export"

        result = export_project_bundle(project, dest)

        assert result.frames_copied == 0
        assert len(result.frames_skipped) == 1
        assert "nonexistent.png" in result.frames_skipped[0]

    def test_dest_already_exists_raises(self, tmp_path):
        """Export raises FileExistsError if destination already exists."""
        dest = tmp_path / "existing"
        dest.mkdir()
        project = _make_project(name="Fail")

        with pytest.raises(FileExistsError):
            export_project_bundle(project, dest)

    def test_cancellation_cleans_up(self, tmp_path):
        """Cancellation removes the partial bundle directory."""
        originals = tmp_path / "originals"
        originals.mkdir()
        # Create multiple frames so cancellation can trigger between copies
        frames = [_make_frame(originals, f"f{i}.png") for i in range(5)]

        cancel_after = [2]  # Cancel after 2 files
        call_count = [0]

        def cancel_check():
            call_count[0] += 1
            return call_count[0] > cancel_after[0]

        project = _make_project(frames=frames, name="Cancel")
        dest = tmp_path / "Cancel-export"

        export_project_bundle(project, dest, cancel_check=cancel_check)

        assert not dest.exists()

    def test_empty_project_exports(self, tmp_path):
        """An empty project creates a bundle with just the project file."""
        project = _make_project(name="Empty")
        dest = tmp_path / "Empty-export"

        result = export_project_bundle(project, dest)

        assert (dest / "Empty.sceneripper").exists()
        assert result.sources_copied == 0
        assert result.frames_copied == 0

    def test_progress_callback_called(self, tmp_path):
        """Progress callback receives (current, total, filename) calls."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "v.mp4")
        frame = _make_frame(originals, "f.png", source_id=source.id)

        project = _make_project(sources=[source], frames=[frame], name="Prog")
        dest = tmp_path / "Prog-export"

        calls = []
        export_project_bundle(
            project, dest, progress_callback=lambda c, t, f: calls.append((c, t, f))
        )

        # Should have calls for frame, source, and final "Done"
        assert len(calls) >= 2
        # Last real file callback should be total == total
        assert calls[-1][0] == calls[-1][1]

    def test_project_path_unchanged(self, tmp_path):
        """Export does not change the original project's path."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "v.mp4")

        original_path = tmp_path / "original.sceneripper"
        project = _make_project(sources=[source], name="Test")
        project.path = original_path

        dest = tmp_path / "Test-export"
        export_project_bundle(project, dest)

        assert project.path == original_path

    def test_clips_preserved_in_export(self, tmp_path):
        """Clips are written to the exported project JSON."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "v.mp4")
        clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=90)

        project = _make_project(sources=[source], clips=[clip], name="Clips")
        dest = tmp_path / "Clips-export"

        export_project_bundle(project, dest)

        data = json.loads((dest / "Clips.sceneripper").read_text())
        assert len(data["clips"]) == 1
        assert data["clips"][0]["id"] == "clip-1"


class TestExportThenLoadRoundTrip:
    """Integration tests: export then load and verify data integrity."""

    def test_full_round_trip(self, tmp_path):
        """Export a full project, then load it and verify all data preserved."""
        originals = tmp_path / "originals"
        originals.mkdir()

        source = _make_source(originals, "video.mp4")
        frame = _make_frame(originals, "f1.png", source_id=source.id)
        clip = Clip(
            id="clip-1",
            source_id=source.id,
            start_frame=0,
            end_frame=90,
            shot_type="wide",
            description="A wide shot",
        )
        sequence = Sequence(name="My Sequence", fps=30.0)
        seq_clip = SequenceClip(
            source_clip_id=clip.id,
            source_id=source.id,
            track_index=0,
            start_frame=0,
            in_point=0,
            out_point=90,
        )
        sequence.tracks[0].add_clip(seq_clip)

        project = _make_project(
            sources=[source],
            clips=[clip],
            frames=[frame],
            sequence=sequence,
            name="RoundTrip",
        )
        dest = tmp_path / "RoundTrip-export"

        export_project_bundle(project, dest)

        # Load the exported project
        project_file = dest / "RoundTrip.sceneripper"
        loaded_sources, loaded_clips, loaded_seq, metadata, ui_state, loaded_frames = load_project(
            project_file
        )

        # Verify sources
        assert len(loaded_sources) == 1
        assert loaded_sources[0].id == source.id
        assert loaded_sources[0].file_path == dest / "sources" / "video.mp4"
        assert loaded_sources[0].file_path.exists()

        # Verify clips
        assert len(loaded_clips) == 1
        assert loaded_clips[0].id == "clip-1"
        assert loaded_clips[0].shot_type == "wide"
        assert loaded_clips[0].description == "A wide shot"

        # Verify sequence
        assert loaded_seq is not None
        assert loaded_seq.name == "My Sequence"
        assert len(loaded_seq.tracks[0].clips) == 1

        # Verify frames
        assert len(loaded_frames) == 1
        assert loaded_frames[0].id == frame.id
        # Frame path resolves to bundle
        assert loaded_frames[0].file_path == dest / "frames" / "f1.png"

        # Verify metadata
        assert metadata.name == "RoundTrip"

    def test_lightweight_round_trip_triggers_missing_callback(self, tmp_path):
        """Loading a lightweight bundle triggers missing_source_callback."""
        originals = tmp_path / "originals"
        originals.mkdir()
        source = _make_source(originals, "video.mp4")

        project = _make_project(sources=[source], name="LightRT")
        dest = tmp_path / "LightRT-export"

        export_project_bundle(project, dest, include_videos=False)

        # Load the lightweight bundle — source video won't exist in bundle
        project_file = dest / "LightRT.sceneripper"
        missing_callback = Mock(return_value=None)

        loaded_sources, _, _, _, _, _ = load_project(
            project_file,
            missing_source_callback=missing_callback,
        )

        # Callback should have been called for the missing source
        assert missing_callback.called
        # Source was skipped (callback returned None)
        assert len(loaded_sources) == 0
