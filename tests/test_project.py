"""Unit tests for the unified Project class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.project import Project, ProjectMetadata
from models.clip import Source, Clip


class TestProjectCreation:
    """Tests for Project creation."""

    def test_new_project(self):
        """Test creating a new empty project."""
        project = Project.new(name="Test Project")
        assert project.metadata.name == "Test Project"
        assert project.path is None
        assert project.sources == []
        assert project.clips == []
        assert project.sequence is None
        assert not project.is_dirty

    def test_new_project_default_name(self):
        """Test creating project with default name."""
        project = Project.new()
        assert project.metadata.name == "Untitled Project"

    def test_project_has_uuid(self):
        """Test that new projects have a UUID."""
        project = Project.new()
        assert project.metadata.id is not None
        assert len(project.metadata.id) > 0


class TestProjectObserverPattern:
    """Tests for the observer pattern implementation."""

    def test_add_observer(self):
        """Test adding an observer."""
        project = Project.new()
        observer = Mock()
        project.add_observer(observer)
        assert observer in project._observers

    def test_remove_observer(self):
        """Test removing an observer."""
        project = Project.new()
        observer = Mock()
        project.add_observer(observer)
        project.remove_observer(observer)
        assert observer not in project._observers

    def test_observer_called_on_source_added(self):
        """Test observer is called when source is added."""
        project = Project.new()
        observer = Mock()
        project.add_observer(observer)

        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)

        observer.assert_called_once_with("source_added", source)

    def test_observer_called_on_clips_added(self):
        """Test observer is called when clips are added."""
        project = Project.new()
        observer = Mock()
        project.add_observer(observer)

        clips = [
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30),
            Clip(id="clip-2", source_id="src-1", start_frame=30, end_frame=60),
        ]
        project.add_clips(clips)

        observer.assert_called_once_with("clips_added", clips)

    def test_observer_called_on_sequence_changed(self):
        """Test observer is called when sequence changes."""
        project = Project.new()
        observer = Mock()

        # Add a clip first
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])

        # Now add observer and add to sequence
        project.add_observer(observer)
        project.add_to_sequence(["clip-1"])

        observer.assert_called_with("sequence_changed", ["clip-1"])

    def test_observer_error_handling(self):
        """Test that observer errors don't break notifications."""
        project = Project.new()

        # First observer throws exception
        bad_observer = Mock(side_effect=Exception("Test error"))
        good_observer = Mock()

        project.add_observer(bad_observer)
        project.add_observer(good_observer)

        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)

        # Good observer should still be called despite bad observer error
        good_observer.assert_called_once()


class TestProjectCachedProperties:
    """Tests for cached property lookups."""

    @pytest.fixture
    def project_with_data(self):
        """Create a project with sources and clips."""
        project = Project.new()

        source1 = Source(
            id="src-1",
            file_path=Path("/test/video1.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        source2 = Source(
            id="src-2",
            file_path=Path("/test/video2.mp4"),
            duration_seconds=120.0,
            fps=24.0,
            width=1280,
            height=720,
        )
        project.add_source(source1)
        project.add_source(source2)

        clips = [
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30),
            Clip(id="clip-2", source_id="src-1", start_frame=30, end_frame=60),
            Clip(id="clip-3", source_id="src-2", start_frame=0, end_frame=24),
        ]
        project.add_clips(clips)

        return project

    def test_sources_by_id(self, project_with_data):
        """Test sources_by_id lookup."""
        lookup = project_with_data.sources_by_id
        assert "src-1" in lookup
        assert "src-2" in lookup
        assert lookup["src-1"].duration_seconds == 60.0
        assert lookup["src-2"].fps == 24.0

    def test_clips_by_id(self, project_with_data):
        """Test clips_by_id lookup."""
        lookup = project_with_data.clips_by_id
        assert "clip-1" in lookup
        assert "clip-2" in lookup
        assert "clip-3" in lookup
        assert lookup["clip-1"].source_id == "src-1"
        assert lookup["clip-3"].source_id == "src-2"

    def test_clips_by_source(self, project_with_data):
        """Test clips_by_source grouping."""
        lookup = project_with_data.clips_by_source
        assert "src-1" in lookup
        assert "src-2" in lookup
        assert len(lookup["src-1"]) == 2
        assert len(lookup["src-2"]) == 1

    def test_cache_invalidation_on_source_add(self):
        """Test caches are invalidated when source is added."""
        project = Project.new()

        # Access cache to populate it
        _ = project.sources_by_id

        # Add source
        source = Source(
            id="src-new",
            file_path=Path("/test/new.mp4"),
            duration_seconds=30.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)

        # Cache should be invalidated and rebuilt
        assert "src-new" in project.sources_by_id

    def test_cache_invalidation_on_clips_add(self):
        """Test caches are invalidated when clips are added."""
        project = Project.new()

        # Access cache to populate it
        _ = project.clips_by_id

        # Add clip
        clip = Clip(id="clip-new", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])

        # Cache should be invalidated and rebuilt
        assert "clip-new" in project.clips_by_id


class TestProjectDirtyState:
    """Tests for dirty state tracking."""

    def test_new_project_not_dirty(self):
        """Test new project is not dirty."""
        project = Project.new()
        assert not project.is_dirty

    def test_add_source_marks_dirty(self):
        """Test adding source marks project dirty."""
        project = Project.new()
        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)
        assert project.is_dirty

    def test_add_clips_marks_dirty(self):
        """Test adding clips marks project dirty."""
        project = Project.new()
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])
        assert project.is_dirty

    def test_add_to_sequence_marks_dirty(self):
        """Test adding to sequence marks project dirty."""
        project = Project.new()
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])
        project.mark_clean()  # Reset dirty

        project.add_to_sequence(["clip-1"])
        assert project.is_dirty

    def test_mark_clean(self):
        """Test mark_clean clears dirty state."""
        project = Project.new()
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])
        assert project.is_dirty

        project.mark_clean()
        assert not project.is_dirty


class TestProjectOperations:
    """Tests for project operations."""

    def test_add_source(self):
        """Test adding a source."""
        project = Project.new()
        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)

        assert len(project.sources) == 1
        assert project.sources[0].id == "src-1"

    def test_add_clips(self):
        """Test adding clips."""
        project = Project.new()
        clips = [
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30),
            Clip(id="clip-2", source_id="src-1", start_frame=30, end_frame=60),
        ]
        project.add_clips(clips)

        assert len(project.clips) == 2

    def test_add_to_sequence_creates_sequence(self):
        """Test add_to_sequence creates sequence if none exists."""
        project = Project.new()
        assert project.sequence is None

        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])
        project.add_to_sequence(["clip-1"])

        assert project.sequence is not None

    def test_add_to_sequence_with_invalid_clip(self):
        """Test add_to_sequence ignores invalid clip IDs."""
        project = Project.new()
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])

        # Try to add non-existent clip
        project.add_to_sequence(["clip-1", "nonexistent"])

        # Should only add the valid clip
        assert project.sequence is not None

    def test_clear_project(self):
        """Test clearing project state."""
        project = Project.new()

        # Add data
        source = Source(
            id="src-1",
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])
        project.add_to_sequence(["clip-1"])

        # Clear
        project.clear()

        assert project.sources == []
        assert project.clips == []
        assert project.sequence is None


class TestProjectPersistence:
    """Tests for project save/load."""

    def test_save_project(self):
        """Test saving a project to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test_project.json"

            project = Project.new(name="Test Save")
            source = Source(
                id="src-1",
                file_path=Path("/test/video.mp4"),
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
            project.add_clips([clip])

            assert project.save(project_path) is True
            assert project_path.exists()
            assert project.path == project_path
            assert not project.is_dirty  # Should be clean after save

    def test_save_notifies_observers(self):
        """Test save notifies observers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test.json"
            project = Project.new()

            observer = Mock()
            project.add_observer(observer)
            project.save(project_path)

            # Should have been called with project_saved event
            calls = [call for call in observer.call_args_list if call[0][0] == "project_saved"]
            assert len(calls) == 1

    def test_load_project(self):
        """Test loading a project from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test_project.json"

            # Create and save project
            original = Project.new(name="Test Load")
            source = Source(
                id="src-1",
                file_path=Path(tmpdir) / "video.mp4",
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            # Create dummy video file
            (Path(tmpdir) / "video.mp4").touch()

            original.add_source(source)
            clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
            original.add_clips([clip])
            original.save(project_path)

            # Load project
            loaded = Project.load(project_path)

            assert loaded.metadata.name == "Test Load"
            assert len(loaded.sources) == 1
            assert len(loaded.clips) == 1
            assert loaded.path == project_path
            assert not loaded.is_dirty

    def test_save_without_path_raises(self):
        """Test save raises error when no path specified."""
        project = Project.new()
        with pytest.raises(ValueError, match="No path specified"):
            project.save()

    def test_save_updates_path(self):
        """Test save with new path updates project path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "project1.json"
            path2 = Path(tmpdir) / "project2.json"

            project = Project.new()
            project.save(path1)
            assert project.path == path1

            project.save(path2)
            assert project.path == path2


class TestProjectLoadError:
    """Tests for project loading error handling."""

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        from core.project import ProjectLoadError

        with pytest.raises((FileNotFoundError, ProjectLoadError)):
            Project.load(Path("/nonexistent/project.json"))

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        from core.project import ProjectLoadError

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "invalid.json"
            project_path.write_text("not valid json {{{")

            with pytest.raises((json.JSONDecodeError, ProjectLoadError)):
                Project.load(project_path)
