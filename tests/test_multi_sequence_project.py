"""Tests for multi-sequence project support (R1-R9)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.project import Project, save_project, load_project, _validate_project_structure
from models.clip import Source, Clip
from models.sequence import Sequence, SequenceClip


# --- Fixtures ---


@pytest.fixture
def project():
    """A new empty project."""
    return Project.new()


@pytest.fixture
def project_with_clips():
    """A project with a source and 3 clips ready for sequencing."""
    p = Project.new()
    source = Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=90.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    p.add_source(source)
    clips = [
        Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30),
        Clip(id="clip-2", source_id="src-1", start_frame=30, end_frame=60),
        Clip(id="clip-3", source_id="src-1", start_frame=60, end_frame=90),
    ]
    p.add_clips(clips)
    return p


# --- Unit 1: Project model — sequences list + compatibility property ---


class TestR2Invariant:
    """New projects always have exactly 1 empty sequence."""

    def test_new_project_has_one_sequence(self, project):
        assert len(project.sequences) == 1
        assert project.active_sequence_index == 0

    def test_new_project_sequence_is_empty(self, project):
        assert len(project.sequence.get_all_clips()) == 0

    def test_clear_resets_to_one_empty_sequence(self, project_with_clips):
        project_with_clips.add_to_sequence(["clip-1"])
        project_with_clips.clear()
        assert len(project_with_clips.sequences) == 1
        assert project_with_clips.active_sequence_index == 0
        assert len(project_with_clips.sequence.get_all_clips()) == 0


class TestCompatibilityProperty:
    """project.sequence returns active sequence, setter replaces active entry."""

    def test_property_returns_active_sequence(self, project):
        assert project.sequence is project.sequences[0]

    def test_property_returns_active_when_index_nonzero(self, project):
        seq2 = Sequence(name="Second")
        project.add_sequence(seq2)
        project.set_active_sequence(1)
        assert project.sequence is seq2
        assert project.sequence is project.sequences[1]

    def test_setter_replaces_active_entry(self, project):
        new_seq = Sequence(name="Replacement")
        project.sequence = new_seq
        assert project.sequences[0] is new_seq
        assert len(project.sequences) == 1

    def test_setter_replaces_only_active_entry(self, project):
        seq2 = Sequence(name="Second")
        project.add_sequence(seq2)
        project.set_active_sequence(1)

        replacement = Sequence(name="Replacement")
        project.sequence = replacement
        # Index 0 untouched, index 1 replaced
        assert project.sequences[0].name != "Replacement"
        assert project.sequences[1] is replacement

    def test_setter_none_substitutes_empty_sequence(self, project):
        project.sequence = None
        assert project.sequence is not None
        assert len(project.sequences) == 1
        assert len(project.sequence.get_all_clips()) == 0

    def test_existing_methods_work_through_property(self, project_with_clips):
        """add_to_sequence operates on active sequence via the property shim."""
        project_with_clips.add_to_sequence(["clip-1", "clip-2"])
        assert len(project_with_clips.sequence.get_all_clips()) == 2
        # The clips are on the sequence in the list
        assert len(project_with_clips.sequences[0].get_all_clips()) == 2


class TestAddSequence:
    """project.add_sequence() appends to list and notifies observers."""

    def test_add_sequence_appends(self, project):
        seq2 = Sequence(name="Chromatics")
        project.add_sequence(seq2)
        assert len(project.sequences) == 2
        assert project.sequences[1] is seq2

    def test_add_sequence_marks_dirty(self, project):
        project.mark_clean()
        project.add_sequence(Sequence(name="New"))
        assert project.is_dirty

    def test_add_sequence_notifies_observers(self, project):
        observer = Mock()
        project.add_observer(observer)
        seq2 = Sequence(name="New")
        project.add_sequence(seq2)
        observer.assert_called_once_with("sequences_changed", project.sequences)


class TestRemoveSequence:
    """project.remove_sequence() with R2 invariant and R8 fallback."""

    def test_remove_middle_sequence(self, project):
        project.add_sequence(Sequence(name="A"))
        project.add_sequence(Sequence(name="B"))
        assert len(project.sequences) == 3
        project.remove_sequence(1)
        assert len(project.sequences) == 2
        assert project.sequences[1].name == "B"

    def test_remove_last_sequence_creates_empty(self, project):
        """R2: removing the only sequence auto-creates a fresh empty one."""
        project.remove_sequence(0)
        assert len(project.sequences) == 1
        assert project.active_sequence_index == 0
        assert len(project.sequence.get_all_clips()) == 0

    def test_remove_active_switches_to_zero(self, project):
        """R8: deleting active sequence switches to index 0."""
        project.add_sequence(Sequence(name="A"))
        project.add_sequence(Sequence(name="B"))
        project.set_active_sequence(2)
        project.remove_sequence(2)
        assert project.active_sequence_index == 0

    def test_remove_before_active_decrements_index(self, project):
        project.add_sequence(Sequence(name="A"))
        project.add_sequence(Sequence(name="B"))
        project.set_active_sequence(2)
        # Active is at index 2 ("B"), remove index 0 (initial empty)
        project.remove_sequence(0)
        # Active should now point to index 1 (still "B")
        assert project.active_sequence_index == 1
        assert project.sequence.name == "B"

    def test_remove_after_active_keeps_index(self, project):
        project.add_sequence(Sequence(name="A"))
        project.add_sequence(Sequence(name="B"))
        project.set_active_sequence(0)
        project.remove_sequence(2)
        assert project.active_sequence_index == 0

    def test_remove_marks_dirty(self, project):
        project.mark_clean()
        project.add_sequence(Sequence(name="A"))
        project.mark_clean()
        project.remove_sequence(1)
        assert project.is_dirty

    def test_remove_notifies_observers(self, project):
        project.add_sequence(Sequence(name="A"))
        observer = Mock()
        project.add_observer(observer)
        project.remove_sequence(1)
        # Should fire both sequences_changed and active_sequence_changed
        calls = [call[0] for call in observer.call_args_list]
        assert ("sequences_changed", project.sequences) in calls
        assert ("active_sequence_changed", project.active_sequence_index) in calls

    def test_remove_out_of_range_is_noop(self, project):
        project.mark_clean()
        project.remove_sequence(5)
        assert not project.is_dirty
        assert len(project.sequences) == 1

    def test_remove_negative_index_is_noop(self, project):
        project.remove_sequence(-1)
        assert len(project.sequences) == 1


class TestSetActiveSequence:
    """project.set_active_sequence() switches the active index."""

    def test_set_active_sequence(self, project):
        project.add_sequence(Sequence(name="Second"))
        project.set_active_sequence(1)
        assert project.active_sequence_index == 1
        assert project.sequence.name == "Second"

    def test_set_active_notifies_observers(self, project):
        project.add_sequence(Sequence(name="Second"))
        observer = Mock()
        project.add_observer(observer)
        project.set_active_sequence(1)
        observer.assert_called_once_with("active_sequence_changed", 1)

    def test_set_active_out_of_range_raises(self, project):
        with pytest.raises(IndexError):
            project.set_active_sequence(5)

    def test_set_active_negative_raises(self, project):
        with pytest.raises(IndexError):
            project.set_active_sequence(-1)


class TestInitSequenceCompat:
    """Constructor handles legacy sequence= param and new sequences= param."""

    def test_init_with_legacy_sequence_param(self):
        seq = Sequence(name="Legacy")
        p = Project(sequence=seq)
        assert len(p.sequences) == 1
        assert p.sequence is seq

    def test_init_with_sequences_param(self):
        seqs = [Sequence(name="A"), Sequence(name="B")]
        p = Project(sequences=seqs, active_sequence_index=1)
        assert len(p.sequences) == 2
        assert p.sequence.name == "B"

    def test_init_sequences_takes_precedence(self):
        seq_legacy = Sequence(name="Legacy")
        seqs = [Sequence(name="A"), Sequence(name="B")]
        p = Project(sequence=seq_legacy, sequences=seqs)
        assert len(p.sequences) == 2
        assert p.sequences[0].name == "A"

    def test_init_no_sequence_creates_default(self):
        p = Project()
        assert len(p.sequences) == 1
        assert p.sequence is not None

    def test_init_clamps_active_index(self):
        seqs = [Sequence(name="Only")]
        p = Project(sequences=seqs, active_sequence_index=5)
        assert p.active_sequence_index == 0


# --- Unit 2: Save/load pipeline — schema 1.4 + backward compatibility ---


@pytest.fixture
def saved_project_with_sequences(tmp_path):
    """Save a project with 3 sequences and return (path, project)."""
    project_path = tmp_path / "test_project.json"

    # Create a real source file so load_project doesn't raise MissingSourceError
    source_path = tmp_path / "video.mp4"
    source_path.write_bytes(b"\x00" * 100)

    project = Project.new(name="Multi-Seq Test")
    source = Source(
        id="src-1",
        file_path=source_path,
        duration_seconds=90.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)
    clips = [
        Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30),
        Clip(id="clip-2", source_id="src-1", start_frame=30, end_frame=60),
        Clip(id="clip-3", source_id="src-1", start_frame=60, end_frame=90),
    ]
    project.add_clips(clips)

    # Sequence 0 (initial): add clip-1
    project.add_to_sequence(["clip-1"])

    # Sequence 1: "Chromatics" with clip-2
    seq2 = Sequence(name="Chromatics", algorithm="color")
    seq2_clip = SequenceClip(
        source_clip_id="clip-2", source_id="src-1",
        start_frame=0, in_point=30, out_point=60,
    )
    seq2.tracks[0].add_clip(seq2_clip)
    project.add_sequence(seq2)

    # Sequence 2: "Storyteller" with clip-3
    seq3 = Sequence(name="Storyteller", algorithm="storyteller")
    seq3_clip = SequenceClip(
        source_clip_id="clip-3", source_id="src-1",
        start_frame=0, in_point=60, out_point=90,
    )
    seq3.tracks[0].add_clip(seq3_clip)
    project.add_sequence(seq3)

    # Set active to index 1
    project.set_active_sequence(1)

    project.save(path=project_path)
    return project_path, project


class TestSaveLoadRoundTrip:
    """Multi-sequence projects survive save → load round-trip."""

    def test_round_trip_preserves_all_sequences(self, saved_project_with_sequences):
        project_path, original = saved_project_with_sequences
        loaded = Project.load(project_path)

        assert len(loaded.sequences) == 3
        assert loaded.active_sequence_index == 1

    def test_round_trip_preserves_sequence_names(self, saved_project_with_sequences):
        project_path, _ = saved_project_with_sequences
        loaded = Project.load(project_path)

        names = [s.name for s in loaded.sequences]
        assert names[1] == "Chromatics"
        assert names[2] == "Storyteller"

    def test_round_trip_preserves_algorithm_labels(self, saved_project_with_sequences):
        project_path, _ = saved_project_with_sequences
        loaded = Project.load(project_path)

        assert loaded.sequences[1].algorithm == "color"
        assert loaded.sequences[2].algorithm == "storyteller"

    def test_round_trip_preserves_clip_data(self, saved_project_with_sequences):
        project_path, _ = saved_project_with_sequences
        loaded = Project.load(project_path)

        # Active sequence (index 1) has 1 clip
        assert len(loaded.sequence.get_all_clips()) == 1
        assert loaded.sequence.get_all_clips()[0].source_clip_id == "clip-2"

        # Index 0 has 1 clip
        assert len(loaded.sequences[0].get_all_clips()) == 1
        assert loaded.sequences[0].get_all_clips()[0].source_clip_id == "clip-1"

    def test_file_contains_both_keys(self, saved_project_with_sequences):
        """File writes both 'sequence' (legacy) and 'sequences' (new)."""
        project_path, _ = saved_project_with_sequences
        with open(project_path, "r") as f:
            data = json.load(f)
        assert "sequence" in data
        assert "sequences" in data
        assert "active_sequence_index" in data
        assert isinstance(data["sequences"], list)
        assert len(data["sequences"]) == 3

    def test_file_version_is_1_4(self, saved_project_with_sequences):
        project_path, _ = saved_project_with_sequences
        with open(project_path, "r") as f:
            data = json.load(f)
        assert data["version"] == "1.4"


class TestBackwardCompatLoad:
    """Loading old v1.3 project files (single 'sequence' key)."""

    def test_load_v13_single_sequence(self, tmp_path):
        """Old format with 'sequence' dict → wraps into 1-element list."""
        project_path = tmp_path / "old_project.json"
        source_path = tmp_path / "video.mp4"
        source_path.write_bytes(b"\x00" * 100)

        source = Source(
            id="src-1",
            file_path=source_path,
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        # Create a v1.3 file using save_project (no extra_data)
        seq = Sequence(name="Old Sequence")
        seq_clip = SequenceClip(
            source_clip_id="clip-1", source_id="src-1",
            start_frame=0, in_point=0, out_point=30,
        )
        seq.tracks[0].add_clip(seq_clip)

        save_project(
            filepath=project_path,
            sources=[source],
            clips=[Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)],
            sequence=seq,
        )
        # Manually downgrade to v1.3 format (no "sequences" key)
        with open(project_path, "r") as f:
            data = json.load(f)
        data.pop("sequences", None)
        data.pop("active_sequence_index", None)
        data["version"] = "1.3"
        with open(project_path, "w") as f:
            json.dump(data, f)

        loaded = Project.load(project_path)
        assert len(loaded.sequences) == 1
        assert loaded.active_sequence_index == 0
        assert loaded.sequence.name == "Old Sequence"

    def test_load_v13_null_sequence(self, tmp_path):
        """Old format with 'sequence': null → creates 1 empty sequence (R2)."""
        project_path = tmp_path / "null_seq.json"
        data = {
            "version": "1.3",
            "name": "Null Seq",
            "id": "test-id",
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-01-01T00:00:00",
            "sources": [],
            "clips": [],
            "sequence": None,
        }
        with open(project_path, "w") as f:
            json.dump(data, f)

        loaded = Project.load(project_path)
        # R2: at least one sequence
        assert len(loaded.sequences) == 1
        assert len(loaded.sequence.get_all_clips()) == 0


class TestValidateProjectStructure:
    """_validate_project_structure accepts both old and new formats."""

    def test_accepts_old_format(self):
        data = {"version": "1.3", "sequence": {"id": "s1", "tracks": []}}
        assert _validate_project_structure(data) == []

    def test_accepts_new_format(self):
        data = {
            "version": "1.4",
            "sequence": {"id": "s1", "tracks": []},
            "sequences": [{"id": "s1", "tracks": []}, {"id": "s2", "tracks": []}],
            "active_sequence_index": 0,
        }
        assert _validate_project_structure(data) == []

    def test_rejects_sequences_not_list(self):
        data = {"version": "1.4", "sequences": "bad"}
        errors = _validate_project_structure(data)
        assert any("sequences" in e and "list" in e for e in errors)

    def test_rejects_sequences_entry_not_dict(self):
        data = {"version": "1.4", "sequences": ["not_a_dict"]}
        errors = _validate_project_structure(data)
        assert any("sequences[0]" in e for e in errors)


class TestMCPReadModifyWrite:
    """MCP/CLI callers using save_project() directly preserve non-active sequences."""

    def test_mcp_save_preserves_other_sequences(self, saved_project_with_sequences):
        """When save_project() is called without extra_data on a v1.4 file,
        it reads existing sequences and only replaces the active one."""
        project_path, original = saved_project_with_sequences

        # Simulate MCP: load with load_project(), modify the active sequence,
        # then save back with save_project() (no extra_data)
        sources, clips, sequence, metadata, ui_state, frames = load_project(project_path)

        # MCP modifies the active sequence (clears it)
        sequence.tracks[0].clips.clear()

        save_project(
            filepath=project_path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            metadata=metadata,
        )

        # Reload and verify all 3 sequences are still there
        with open(project_path, "r") as f:
            data = json.load(f)

        assert len(data["sequences"]) == 3
        # Active sequence (index 1) was cleared
        active_clips = data["sequences"][1].get("tracks", [{}])[0].get("clips", [])
        assert len(active_clips) == 0
        # Non-active sequences preserved
        assert len(data["sequences"][0]["tracks"][0]["clips"]) == 1
        assert len(data["sequences"][2]["tracks"][0]["clips"]) == 1

    def test_mcp_save_on_old_format_no_sequences_key(self, tmp_path):
        """save_project() without extra_data on a file with no 'sequences' key
        doesn't add one — preserves old format."""
        project_path = tmp_path / "old.json"
        data = {
            "version": "1.3",
            "name": "Old",
            "id": "test",
            "created_at": "2025-01-01",
            "modified_at": "2025-01-01",
            "sources": [],
            "clips": [],
            "sequence": None,
        }
        with open(project_path, "w") as f:
            json.dump(data, f)

        save_project(
            filepath=project_path,
            sources=[],
            clips=[],
            sequence=None,
        )

        with open(project_path, "r") as f:
            result = json.load(f)
        assert "sequences" not in result

    def test_load_project_signature_unchanged(self):
        """load_project() still returns a 6-tuple with single Optional[Sequence]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {
                "version": "1.4",
                "name": "Test",
                "id": "test",
                "created_at": "2025-01-01",
                "modified_at": "2025-01-01",
                "sources": [],
                "clips": [],
                "sequence": None,
                "sequences": [
                    {"id": "s1", "name": "A", "fps": 30, "tracks": [{"id": "t1", "name": "V1", "clips": []}]},
                    {"id": "s2", "name": "B", "fps": 30, "tracks": [{"id": "t2", "name": "V1", "clips": []}]},
                ],
                "active_sequence_index": 1,
            }
            with open(path, "w") as f:
                json.dump(data, f)

            result = load_project(path)
            assert len(result) == 6
            sources, clips, sequence, metadata, ui_state, frames = result
            # Returns the active sequence (index 1 = "B")
            assert sequence.name == "B"
