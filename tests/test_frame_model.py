"""Unit tests for Frame model and Project frame management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from models.frame import Frame
from models.sequence import SequenceClip
from core.project import Project, ProjectMetadata, save_project, load_project, SCHEMA_VERSION


class TestFrameCreation:
    """Tests for Frame dataclass creation."""

    def test_create_frame_defaults(self):
        """Test creating a Frame with default values."""
        frame = Frame()
        assert frame.id is not None
        assert len(frame.id) > 0
        assert frame.file_path == Path()
        assert frame.source_id is None
        assert frame.clip_id is None
        assert frame.frame_number is None
        assert frame.thumbnail_path is None
        assert frame.width is None
        assert frame.height is None
        assert frame.analyzed is False
        assert frame.shot_type is None
        assert frame.dominant_colors is None
        assert frame.description is None
        assert frame.detected_objects is None
        assert frame.extracted_texts is None
        assert frame.cinematography is None
        assert frame.tags == []
        assert frame.notes == ""

    def test_create_frame_with_values(self):
        """Test creating a Frame with explicit values."""
        frame = Frame(
            id="frame-1",
            file_path=Path("/project/frames/frame_000001.png"),
            source_id="src-1",
            clip_id="clip-1",
            frame_number=42,
            width=1920,
            height=1080,
            analyzed=True,
            shot_type="wide",
            description="A wide landscape shot",
            tags=["landscape", "outdoor"],
            notes="Key frame",
        )
        assert frame.id == "frame-1"
        assert frame.file_path == Path("/project/frames/frame_000001.png")
        assert frame.source_id == "src-1"
        assert frame.clip_id == "clip-1"
        assert frame.frame_number == 42
        assert frame.width == 1920
        assert frame.height == 1080
        assert frame.analyzed is True
        assert frame.shot_type == "wide"
        assert frame.description == "A wide landscape shot"
        assert frame.tags == ["landscape", "outdoor"]
        assert frame.notes == "Key frame"


class TestFrameDisplayName:
    """Tests for Frame.display_name()."""

    def test_display_name_with_frame_number(self):
        """Frame number takes priority."""
        frame = Frame(
            file_path=Path("/test/image.png"),
            frame_number=42,
        )
        assert frame.display_name() == "Frame 42"

    def test_display_name_with_frame_zero(self):
        """Frame number 0 should still display."""
        frame = Frame(
            file_path=Path("/test/image.png"),
            frame_number=0,
        )
        assert frame.display_name() == "Frame 0"

    def test_display_name_without_frame_number(self):
        """Falls back to filename."""
        frame = Frame(file_path=Path("/test/my_photo.jpg"))
        assert frame.display_name() == "my_photo.jpg"


class TestFrameSerialization:
    """Tests for Frame to_dict/from_dict."""

    def test_to_dict_minimal(self):
        """Test serialization with minimal fields."""
        frame = Frame(id="frame-1", file_path=Path("/test/frame.png"))
        data = frame.to_dict()
        assert data["id"] == "frame-1"
        assert data["file_path"] == "/test/frame.png"
        assert data["analyzed"] is False
        # Optional fields should not be present
        assert "source_id" not in data
        assert "clip_id" not in data
        assert "frame_number" not in data
        assert "shot_type" not in data
        assert "dominant_colors" not in data

    def test_to_dict_full(self):
        """Test serialization with all fields."""
        frame = Frame(
            id="frame-1",
            file_path=Path("/test/frame.png"),
            source_id="src-1",
            clip_id="clip-1",
            frame_number=42,
            width=1920,
            height=1080,
            analyzed=True,
            shot_type="close-up",
            dominant_colors=[(255, 0, 0), (0, 128, 255)],
            description="A close-up shot",
            description_model="gpt-5.2",
            tags=["face", "portrait"],
            notes="Important frame",
        )
        data = frame.to_dict()
        assert data["source_id"] == "src-1"
        assert data["clip_id"] == "clip-1"
        assert data["frame_number"] == 42
        assert data["width"] == 1920
        assert data["height"] == 1080
        assert data["shot_type"] == "close-up"
        assert data["dominant_colors"] == [
            {"r": 255, "g": 0, "b": 0},
            {"r": 0, "g": 128, "b": 255},
        ]
        assert data["description"] == "A close-up shot"
        assert data["description_model"] == "gpt-5.2"
        assert data["tags"] == ["face", "portrait"]
        assert data["notes"] == "Important frame"

    def test_to_dict_with_base_path(self):
        """Test relative path serialization."""
        frame = Frame(
            id="frame-1",
            file_path=Path("/project/frames/frame_001.png"),
        )
        data = frame.to_dict(base_path=Path("/project"))
        assert data["file_path"] == "frames/frame_001.png"
        assert data["_absolute_path"] == "/project/frames/frame_001.png"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {"id": "frame-1", "file_path": "/test/frame.png", "analyzed": False}
        frame = Frame.from_dict(data)
        assert frame.id == "frame-1"
        assert frame.file_path == Path("/test/frame.png")
        assert frame.analyzed is False
        assert frame.source_id is None

    def test_from_dict_full(self):
        """Test deserialization with all fields."""
        data = {
            "id": "frame-1",
            "file_path": "/test/frame.png",
            "source_id": "src-1",
            "clip_id": "clip-1",
            "frame_number": 42,
            "width": 1920,
            "height": 1080,
            "analyzed": True,
            "shot_type": "wide",
            "dominant_colors": [{"r": 255, "g": 0, "b": 0}],
            "description": "A wide shot",
            "tags": ["outdoor"],
            "notes": "test",
        }
        frame = Frame.from_dict(data)
        assert frame.source_id == "src-1"
        assert frame.clip_id == "clip-1"
        assert frame.frame_number == 42
        assert frame.dominant_colors == [(255, 0, 0)]
        assert frame.tags == ["outdoor"]

    def test_round_trip(self):
        """Test to_dict -> from_dict round trip."""
        original = Frame(
            id="frame-1",
            file_path=Path("/test/frame.png"),
            source_id="src-1",
            frame_number=10,
            width=1920,
            height=1080,
            analyzed=True,
            shot_type="medium",
            dominant_colors=[(100, 200, 50)],
            description="Test frame",
            tags=["tag1"],
            notes="note",
        )
        data = original.to_dict()
        restored = Frame.from_dict(data)
        assert restored.id == original.id
        assert restored.file_path == original.file_path
        assert restored.source_id == original.source_id
        assert restored.frame_number == original.frame_number
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.analyzed == original.analyzed
        assert restored.shot_type == original.shot_type
        assert restored.dominant_colors == original.dominant_colors
        assert restored.description == original.description
        assert restored.tags == original.tags
        assert restored.notes == original.notes

    def test_from_dict_with_base_path(self, tmp_path):
        """Test relative path resolution."""
        # Create the target file
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        frame_file = frame_dir / "frame_001.png"
        frame_file.write_text("fake png")

        data = {"id": "frame-1", "file_path": "frames/frame_001.png", "analyzed": False}
        frame = Frame.from_dict(data, base_path=tmp_path)
        assert frame.file_path.exists()
        assert frame.file_path == frame_dir / "frame_001.png"

    def test_from_dict_path_traversal_rejected(self, tmp_path):
        """Test that path traversal is rejected."""
        data = {"id": "frame-1", "file_path": "../../../etc/passwd", "analyzed": False}
        with pytest.raises(ValueError, match="Path traversal detected"):
            Frame.from_dict(data, base_path=tmp_path)

    def test_from_dict_absolute_fallback(self, tmp_path):
        """Test fallback to absolute path when relative doesn't exist."""
        # Create file at absolute path
        abs_file = tmp_path / "actual_location" / "frame.png"
        abs_file.parent.mkdir()
        abs_file.write_text("fake png")

        data = {
            "id": "frame-1",
            "file_path": "old_location/frame.png",
            "_absolute_path": str(abs_file),
            "analyzed": False,
        }
        frame = Frame.from_dict(data, base_path=tmp_path)
        assert frame.file_path == abs_file


class TestSequenceClipFrameSupport:
    """Tests for SequenceClip frame-related fields."""

    def test_is_frame_entry_true(self):
        """Frame entries have frame_id set."""
        sc = SequenceClip(frame_id="frame-1", hold_frames=5)
        assert sc.is_frame_entry is True

    def test_is_frame_entry_false(self):
        """Clip entries have no frame_id."""
        sc = SequenceClip(source_clip_id="clip-1")
        assert sc.is_frame_entry is False

    def test_frame_entry_duration(self):
        """Frame entry duration equals hold_frames."""
        sc = SequenceClip(frame_id="frame-1", hold_frames=10)
        assert sc.duration_frames == 10

    def test_clip_entry_duration_unchanged(self):
        """Clip entry duration is still out_point - in_point."""
        sc = SequenceClip(
            source_clip_id="clip-1",
            in_point=0,
            out_point=30,
        )
        assert sc.duration_frames == 30

    def test_frame_entry_serialization(self):
        """Frame entry round-trips through to_dict/from_dict."""
        sc = SequenceClip(
            id="seq-1",
            frame_id="frame-1",
            source_id="src-1",
            hold_frames=5,
            start_frame=100,
        )
        data = sc.to_dict()
        assert data["frame_id"] == "frame-1"
        assert data["hold_frames"] == 5

        restored = SequenceClip.from_dict(data)
        assert restored.frame_id == "frame-1"
        assert restored.hold_frames == 5
        assert restored.is_frame_entry is True

    def test_clip_entry_serialization_omits_frame_fields(self):
        """Clip entries don't serialize frame_id or hold_frames=1."""
        sc = SequenceClip(
            source_clip_id="clip-1",
            in_point=0,
            out_point=30,
        )
        data = sc.to_dict()
        assert "frame_id" not in data
        assert "hold_frames" not in data

    def test_backward_compatible_deserialization(self):
        """Old sequence clips without frame fields load correctly."""
        data = {
            "id": "seq-1",
            "source_clip_id": "clip-1",
            "source_id": "src-1",
            "track_index": 0,
            "start_frame": 0,
            "in_point": 0,
            "out_point": 30,
        }
        sc = SequenceClip.from_dict(data)
        assert sc.frame_id is None
        assert sc.hold_frames == 1
        assert sc.is_frame_entry is False
        assert sc.duration_frames == 30


class TestProjectFrameManagement:
    """Tests for Project frame operations."""

    def _make_project_with_source(self):
        """Helper: create project with one source."""
        from models.clip import Source
        project = Project.new()
        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
        )
        project.add_source(source)
        return project

    def test_new_project_has_no_frames(self):
        """New projects start with empty frames."""
        project = Project.new()
        assert project.frames == []

    def test_add_frames(self):
        """Test adding frames to project."""
        project = self._make_project_with_source()
        frames = [
            Frame(id="f1", file_path=Path("/test/f1.png"), source_id="src-1"),
            Frame(id="f2", file_path=Path("/test/f2.png"), source_id="src-1"),
        ]
        project.add_frames(frames)
        assert len(project.frames) == 2
        assert project.is_dirty

    def test_add_frames_notifies_observer(self):
        """Test that adding frames notifies observers."""
        project = self._make_project_with_source()
        observer = Mock()
        project.add_observer(observer)

        frames = [Frame(id="f1", file_path=Path("/test/f1.png"))]
        project.add_frames(frames)
        observer.assert_called_once_with("frames_added", frames)

    def test_remove_frames(self):
        """Test removing frames by ID."""
        project = Project.new()
        frames = [
            Frame(id="f1", file_path=Path("/test/f1.png")),
            Frame(id="f2", file_path=Path("/test/f2.png")),
            Frame(id="f3", file_path=Path("/test/f3.png")),
        ]
        project.add_frames(frames)

        removed = project.remove_frames(["f1", "f3"])
        assert len(removed) == 2
        assert len(project.frames) == 1
        assert project.frames[0].id == "f2"

    def test_remove_frames_notifies_observer(self):
        """Test that removing frames notifies observers."""
        project = Project.new()
        project.add_frames([Frame(id="f1", file_path=Path("/test/f1.png"))])

        observer = Mock()
        project.add_observer(observer)
        project.remove_frames(["f1"])
        observer.assert_called_once_with("frames_removed", [project._frames[0]] if project._frames else observer.call_args[0][1])

    def test_remove_nonexistent_frames(self):
        """Removing non-existent IDs is a no-op."""
        project = Project.new()
        removed = project.remove_frames(["nonexistent"])
        assert removed == []

    def test_update_frame(self):
        """Test updating frame fields."""
        project = Project.new()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png")),
        ])

        result = project.update_frame("f1", shot_type="wide", analyzed=True)
        assert result is not None
        assert result.shot_type == "wide"
        assert result.analyzed is True

    def test_update_frame_not_found(self):
        """Updating non-existent frame returns None."""
        project = Project.new()
        result = project.update_frame("nonexistent", shot_type="wide")
        assert result is None

    def test_frames_by_id_cache(self):
        """Test frames_by_id cached property."""
        project = Project.new()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png")),
            Frame(id="f2", file_path=Path("/test/f2.png")),
        ])
        by_id = project.frames_by_id
        assert "f1" in by_id
        assert "f2" in by_id
        assert by_id["f1"].id == "f1"

    def test_frames_by_source_cache(self):
        """Test frames_by_source cached property."""
        project = Project.new()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png"), source_id="src-1"),
            Frame(id="f2", file_path=Path("/test/f2.png"), source_id="src-1"),
            Frame(id="f3", file_path=Path("/test/f3.png"), source_id="src-2"),
        ])
        by_source = project.frames_by_source
        assert len(by_source["src-1"]) == 2
        assert len(by_source["src-2"]) == 1

    def test_frames_by_clip_cache(self):
        """Test frames_by_clip cached property."""
        project = Project.new()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png"), clip_id="clip-1"),
            Frame(id="f2", file_path=Path("/test/f2.png"), clip_id="clip-1"),
        ])
        by_clip = project.frames_by_clip
        assert len(by_clip["clip-1"]) == 2

    def test_remove_source_removes_frames(self):
        """Removing a source also removes its frames."""
        project = self._make_project_with_source()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png"), source_id="src-1"),
            Frame(id="f2", file_path=Path("/test/f2.png"), source_id="src-1"),
            Frame(id="f3", file_path=Path("/test/f3.png")),  # No source
        ])
        project.remove_source("src-1")
        assert len(project.frames) == 1
        assert project.frames[0].id == "f3"

    def test_clear_clears_frames(self):
        """Clearing project clears frames."""
        project = Project.new()
        project.add_frames([Frame(id="f1", file_path=Path("/test/f1.png"))])
        project.clear()
        assert project.frames == []

    def test_add_frames_to_sequence(self):
        """Test adding frames to sequence."""
        project = self._make_project_with_source()
        project.add_frames([
            Frame(id="f1", file_path=Path("/test/f1.png"), source_id="src-1"),
            Frame(id="f2", file_path=Path("/test/f2.png"), source_id="src-1"),
        ])
        project.add_frames_to_sequence(["f1", "f2"], hold_frames=3)

        assert project.sequence is not None
        seq_clips = project.sequence.get_all_clips()
        assert len(seq_clips) == 2
        assert seq_clips[0].frame_id == "f1"
        assert seq_clips[0].hold_frames == 3
        assert seq_clips[0].is_frame_entry is True
        assert seq_clips[1].start_frame == 3  # After first frame's hold


class TestProjectFramePersistence:
    """Tests for saving and loading projects with frames."""

    def test_save_load_with_frames(self, tmp_path):
        """Test save/load round trip with frames."""
        from models.clip import Source

        project_file = tmp_path / "test_project.json"

        # Create frame files
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(3):
            (frame_dir / f"frame_{i:06d}.png").write_text("fake png")

        # Create project with frames
        source = Source(id="src-1", file_path=Path("/test/video.mp4"))
        frames = [
            Frame(
                id=f"f{i}",
                file_path=frame_dir / f"frame_{i:06d}.png",
                source_id="src-1",
                frame_number=i,
                width=1920,
                height=1080,
            )
            for i in range(3)
        ]

        save_project(
            filepath=project_file,
            sources=[],
            clips=[],
            sequence=None,
            frames=frames,
        )

        # Verify file was written
        assert project_file.exists()
        with open(project_file) as f:
            data = json.load(f)
        assert "frames" in data
        assert len(data["frames"]) == 3

        # Load and verify
        _, _, _, _, _, loaded_frames = load_project(
            filepath=project_file,
            missing_source_callback=lambda p, sid: None,
        )
        assert len(loaded_frames) == 3
        assert loaded_frames[0].id == "f0"
        assert loaded_frames[0].frame_number == 0
        assert loaded_frames[0].width == 1920

    def test_load_old_project_without_frames(self, tmp_path):
        """Test loading a v1.0 project without frames field."""
        project_file = tmp_path / "old_project.json"
        old_data = {
            "id": "proj-1",
            "project_name": "Old Project",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-01-01T00:00:00",
            "sources": [],
            "clips": [],
            "sequence": None,
        }
        with open(project_file, "w") as f:
            json.dump(old_data, f)

        _, _, _, metadata, _, frames = load_project(
            filepath=project_file,
            missing_source_callback=lambda p, sid: None,
        )
        assert frames == []
        assert metadata.name == "Old Project"

    def test_project_class_save_load_with_frames(self, tmp_path):
        """Test Project.save() and Project.load() with frames."""
        project_file = tmp_path / "project.json"

        # Create frame files
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        (frame_dir / "frame_000001.png").write_text("fake")

        project = Project.new(name="Frame Test")
        project.add_frames([
            Frame(
                id="f1",
                file_path=frame_dir / "frame_000001.png",
                frame_number=1,
                analyzed=True,
                shot_type="wide",
            ),
        ])
        project.save(path=project_file)

        loaded = Project.load(
            path=project_file,
            missing_source_callback=lambda p, sid: None,
        )
        assert len(loaded.frames) == 1
        assert loaded.frames[0].id == "f1"
        assert loaded.frames[0].shot_type == "wide"
        assert loaded.frames[0].analyzed is True

    def test_schema_version_bumped(self):
        """Schema version should be 1.1."""
        assert SCHEMA_VERSION == "1.1"

    def test_repr_includes_frames(self):
        """Project repr includes frame count."""
        project = Project.new()
        project.add_frames([Frame(id="f1", file_path=Path("/test/f1.png"))])
        r = repr(project)
        assert "frames=1" in r
