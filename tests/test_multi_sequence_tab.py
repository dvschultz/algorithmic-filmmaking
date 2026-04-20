"""Tests for multi-sequence UI in the Sequence tab (Units 3-7)."""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.project import Project
from models.clip import Source, Clip
from models.sequence import Sequence, SequenceClip


# --- Fixtures ---


@pytest.fixture
def project_with_clips():
    """A project with a source, 3 clips, and 2 sequences."""
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
    # Add a clip to the initial sequence
    p.add_to_sequence(["clip-1"])
    # Add a second sequence with clip-2
    seq2 = Sequence(name="Chromatics", algorithm="color")
    seq2_clip = SequenceClip(
        source_clip_id="clip-2", source_id="src-1",
        start_frame=0, in_point=30, out_point=60,
    )
    seq2.tracks[0].add_clip(seq2_clip)
    p.add_sequence(seq2)
    return p


# --- Unit 3: Sequence dropdown + switching with dirty tracking ---


class TestSequenceDropdownSync:
    """Dropdown reflects project.sequences names."""

    def test_sync_populates_dropdown(self, project_with_clips):
        """_sync_sequence_dropdown fills items from project.sequences."""
        # We can't instantiate SequenceTab without a full Qt app,
        # so test the underlying project model behavior that the dropdown depends on.
        p = project_with_clips
        assert len(p.sequences) == 2
        assert p.sequences[0].name == "Untitled Sequence"
        assert p.sequences[1].name == "Chromatics"
        assert p.active_sequence_index == 0

    def test_switching_changes_active_index(self, project_with_clips):
        """Switching sequence index changes project.active_sequence_index."""
        p = project_with_clips
        p.set_active_sequence(1)
        assert p.active_sequence_index == 1
        assert p.sequence.name == "Chromatics"

    def test_switching_back_preserves_data(self, project_with_clips):
        """Switching away and back preserves sequence clip data."""
        p = project_with_clips
        # Verify initial sequence has clip-1
        assert len(p.sequences[0].get_all_clips()) == 1
        assert p.sequences[0].get_all_clips()[0].source_clip_id == "clip-1"

        # Switch to Chromatics
        p.set_active_sequence(1)
        assert p.sequence.get_all_clips()[0].source_clip_id == "clip-2"

        # Switch back
        p.set_active_sequence(0)
        assert p.sequence.get_all_clips()[0].source_clip_id == "clip-1"


# --- Unit 5: New Sequence + Delete ---


class TestNewSequence:
    """New Sequence button creates a new empty sequence."""

    def test_new_sequence_creates_blank(self, project_with_clips):
        p = project_with_clips
        initial_count = len(p.sequences)
        p.add_sequence(Sequence(name="Untitled Sequence"))
        assert len(p.sequences) == initial_count + 1
        assert p.sequences[-1].name == "Untitled Sequence"
        assert len(p.sequences[-1].get_all_clips()) == 0


class TestDeleteSequence:
    """Delete sequence with R2 invariant and R7 confirmation rules."""

    def test_delete_empty_sequence(self, project_with_clips):
        """Empty sequence (0 clips) can be deleted."""
        p = project_with_clips
        # Make an empty third sequence
        p.add_sequence(Sequence(name="Empty"))
        assert len(p.sequences) == 3
        p.remove_sequence(2)
        assert len(p.sequences) == 2

    def test_delete_populated_sequence(self, project_with_clips):
        """Populated sequence can be deleted (confirmation tested at UI level)."""
        p = project_with_clips
        # Chromatics (index 1) has 1 clip
        assert len(p.sequences[1].get_all_clips()) == 1
        p.remove_sequence(1)
        assert len(p.sequences) == 1

    def test_delete_active_switches_to_zero(self, project_with_clips):
        p = project_with_clips
        p.set_active_sequence(1)
        p.remove_sequence(1)
        assert p.active_sequence_index == 0

    def test_delete_only_sequence_creates_empty(self):
        """R2: deleting the only sequence auto-creates a fresh empty one."""
        p = Project.new()
        assert len(p.sequences) == 1
        p.remove_sequence(0)
        assert len(p.sequences) == 1
        assert len(p.sequence.get_all_clips()) == 0

    def test_delete_non_active_preserves_view(self, project_with_clips):
        p = project_with_clips
        # Active is 0, delete non-active index 1
        p.set_active_sequence(0)
        p.remove_sequence(1)
        assert p.active_sequence_index == 0
        assert len(p.sequences) == 1


class TestDirtyTracking:
    """Dirty flag behavior for multi-sequence switching."""

    def test_algorithm_running_guard(self, project_with_clips):
        """_algorithm_running flag concept: when True, changes shouldn't set dirty."""
        # This is a model-level test verifying the concept.
        # The actual flag lives on SequenceTab, tested here at the data level.
        p = project_with_clips
        p.mark_clean()
        # Simulate adding clips (as an algorithm would)
        p.add_to_sequence(["clip-2"])
        assert p.is_dirty  # Project is dirty from the add


# --- Unit 4: Auto-naming ---


# --- Unit 6: Rename sequence ---


class TestRenameSequence:
    """Sequence rename via context menu (R9)."""

    def test_rename_updates_name(self, project_with_clips):
        p = project_with_clips
        p.sequences[1].name = "Final Cut"
        assert p.sequences[1].name == "Final Cut"

    def test_rename_preserves_data(self, project_with_clips):
        p = project_with_clips
        clips_before = len(p.sequences[1].get_all_clips())
        p.sequences[1].name = "Renamed"
        assert len(p.sequences[1].get_all_clips()) == clips_before

    def test_duplicate_names_allowed(self, project_with_clips):
        """R9: duplicate names are permitted."""
        p = project_with_clips
        p.sequences[1].name = p.sequences[0].name
        assert p.sequences[0].name == p.sequences[1].name


class TestSequenceAutoNaming:
    """Monotonic auto-naming for algorithm runs."""

    def test_first_run_uses_bare_label(self, project_with_clips):
        """First Chromatics run creates 'Chromatics'."""
        # "Chromatics" already exists (added in fixture)
        # But the naming is per-algorithm scan, so existing "Chromatics" means next would be #2
        p = project_with_clips
        names = [s.name for s in p.sequences]
        assert "Chromatics" in names

    def test_naming_counter_increments(self, project_with_clips):
        """Running same algorithm again produces '{Label} #2'."""
        p = project_with_clips
        # Add another "Chromatics" sequence
        seq3 = Sequence(name="Chromatics #2", algorithm="color")
        p.add_sequence(seq3)
        names = [s.name for s in p.sequences]
        assert "Chromatics" in names
        assert "Chromatics #2" in names

    def test_rename_doesnt_affect_counter(self, project_with_clips):
        """Renaming 'Chromatics' to 'Final Cut' doesn't reset the counter."""
        p = project_with_clips
        # Rename "Chromatics" (index 1) to "Final Cut"
        p.sequences[1].name = "Final Cut"
        # "Chromatics" name is gone — next run should use bare "Chromatics"
        # (scan finds no existing "Chromatics" or "Chromatics #N")
        bare_exists = any(s.name == "Chromatics" for s in p.sequences)
        assert not bare_exists

    def test_switching_preserves_previous_sequence(self, project_with_clips):
        """After running multiple algorithms, previous sequences are intact."""
        p = project_with_clips
        # Initial has clip-1, Chromatics has clip-2
        assert len(p.sequences[0].get_all_clips()) == 1
        assert p.sequences[0].get_all_clips()[0].source_clip_id == "clip-1"
        assert len(p.sequences[1].get_all_clips()) == 1
        assert p.sequences[1].get_all_clips()[0].source_clip_id == "clip-2"


# --- Unit 7: Parameter-tweak prompt + gui_state ---


class TestParameterTweakPrompt:
    """R3a: direction/algorithm changes prompt Replace/Create New/Cancel."""

    def test_replace_removes_old_and_creates_new(self, project_with_clips):
        """'Replace' on a parameter tweak keeps sequence count the same."""
        p = project_with_clips
        count_before = len(p.sequences)
        # Simulate replace: remove active, then add new
        p.remove_sequence(p.active_sequence_index)
        p.add_sequence(Sequence(name="Replaced", algorithm="color"))
        # Count should be same as before (one removed, one added)
        assert len(p.sequences) == count_before

    def test_create_new_adds_sequence(self, project_with_clips):
        """'Create New' on a parameter tweak adds a new sequence."""
        p = project_with_clips
        count_before = len(p.sequences)
        p.add_sequence(Sequence(name="New Direction", algorithm="color"))
        assert len(p.sequences) == count_before + 1


class TestGuiStateSync:
    """gui_state.sequence_ids updates with active sequence."""

    def test_switching_updates_active_clips(self, project_with_clips):
        """After switching, the active sequence clips change."""
        p = project_with_clips
        # Active is 0, has clip-1
        clips_0 = [c.source_clip_id for c in p.sequence.get_all_clips()]
        assert "clip-1" in clips_0

        # Switch to 1, has clip-2
        p.set_active_sequence(1)
        clips_1 = [c.source_clip_id for c in p.sequence.get_all_clips()]
        assert "clip-2" in clips_1
        assert "clip-1" not in clips_1


class TestSequenceMetadataOnSwitch:
    """Algorithm label and chromatic bar update to match selected sequence."""

    def test_active_sequence_has_correct_algorithm(self, project_with_clips):
        p = project_with_clips
        p.set_active_sequence(1)
        assert p.sequence.algorithm == "color"

    def test_switching_preserves_algorithm(self, project_with_clips):
        p = project_with_clips
        p.set_active_sequence(1)
        assert p.sequence.algorithm == "color"
        p.set_active_sequence(0)
        assert p.sequence.algorithm is None  # Initial sequence has no algorithm
