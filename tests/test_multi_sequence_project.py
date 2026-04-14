"""Tests for multi-sequence project support (R1-R9)."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from core.project import Project
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
