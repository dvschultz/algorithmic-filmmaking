"""Tests for clip disabled feature: model field, project toggle, undo command, and downstream filtering."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.clip import Clip, Source
from core.project import Project, save_project, load_project


# --- Clip model field ---


class TestClipDisabledField:
    def test_default_is_false(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100)
        assert clip.disabled is False

    def test_serialization_omitted_when_false(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100)
        data = clip.to_dict()
        assert "disabled" not in data

    def test_serialization_included_when_true(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100, disabled=True)
        data = clip.to_dict()
        assert data["disabled"] is True

    def test_deserialization_defaults_false(self):
        """Old project files without 'disabled' key should default to False."""
        data = {"id": "c1", "source_id": "s1", "start_frame": 0, "end_frame": 100}
        clip = Clip.from_dict(data)
        assert clip.disabled is False

    def test_deserialization_reads_true(self):
        data = {"id": "c1", "source_id": "s1", "start_frame": 0, "end_frame": 100, "disabled": True}
        clip = Clip.from_dict(data)
        assert clip.disabled is True

    def test_round_trip(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100, disabled=True)
        data = clip.to_dict()
        restored = Clip.from_dict(data)
        assert restored.disabled is True


# --- Project toggle ---


def _make_project_with_clips(n=3):
    """Helper: create a project with n clips."""
    source = Source(id="s1", file_path=Path("/tmp/test.mp4"), fps=30.0, duration_seconds=10.0)
    clips = [
        Clip(id=f"c{i}", source_id="s1", start_frame=i * 100, end_frame=(i + 1) * 100)
        for i in range(n)
    ]
    return Project(sources=[source], clips=clips)


class TestProjectToggleDisabled:
    def test_toggle_on(self):
        project = _make_project_with_clips()
        toggled = project.toggle_clips_disabled(["c0"])
        assert len(toggled) == 1
        assert toggled[0].disabled is True

    def test_toggle_off(self):
        project = _make_project_with_clips()
        project.toggle_clips_disabled(["c0"])  # on
        toggled = project.toggle_clips_disabled(["c0"])  # off
        assert toggled[0].disabled is False

    def test_toggle_multiple(self):
        project = _make_project_with_clips()
        toggled = project.toggle_clips_disabled(["c0", "c1"])
        assert len(toggled) == 2
        assert all(c.disabled for c in toggled)

    def test_invalid_ids_ignored(self):
        project = _make_project_with_clips()
        toggled = project.toggle_clips_disabled(["nonexistent"])
        assert toggled == []

    def test_sets_dirty(self):
        project = _make_project_with_clips()
        project.mark_clean()
        project.toggle_clips_disabled(["c0"])
        assert project.is_dirty

    def test_no_dirty_when_nothing_toggled(self):
        project = _make_project_with_clips()
        project.mark_clean()
        project.toggle_clips_disabled(["nonexistent"])
        assert not project.is_dirty

    def test_notifies_observers(self):
        project = _make_project_with_clips()
        events = []
        project.add_observer(lambda event, data: events.append((event, data)))
        project.toggle_clips_disabled(["c0"])
        assert len(events) == 1
        assert events[0][0] == "clips_updated"

    def test_enabled_clips_property(self):
        project = _make_project_with_clips()
        project.toggle_clips_disabled(["c0", "c2"])
        enabled = project.enabled_clips
        assert len(enabled) == 1
        assert enabled[0].id == "c1"


# --- Undo command ---


class TestToggleClipDisabledCommand:
    def test_redo_toggles(self):
        from ui.commands.toggle_clip_disabled import ToggleClipDisabledCommand

        project = _make_project_with_clips()
        cmd = ToggleClipDisabledCommand(project, ["c0"])
        cmd.redo()
        assert project.clips_by_id["c0"].disabled is True

    def test_undo_reverts(self):
        from ui.commands.toggle_clip_disabled import ToggleClipDisabledCommand

        project = _make_project_with_clips()
        cmd = ToggleClipDisabledCommand(project, ["c0"])
        cmd.redo()
        assert project.clips_by_id["c0"].disabled is True
        cmd.undo()
        assert project.clips_by_id["c0"].disabled is False

    def test_menu_text(self):
        from ui.commands.toggle_clip_disabled import ToggleClipDisabledCommand

        project = _make_project_with_clips()
        cmd_single = ToggleClipDisabledCommand(project, ["c0"])
        assert "1 clip" in cmd_single.text()
        cmd_multi = ToggleClipDisabledCommand(project, ["c0", "c1"])
        assert "2 clips" in cmd_multi.text()


# --- Disabled clips excluded from sequence ---


class TestDisabledClipsExcludedFromSequence:
    def test_add_to_sequence_skips_disabled(self):
        project = _make_project_with_clips()
        project.clips_by_id["c0"].disabled = True
        project.add_to_sequence(["c0", "c1"])
        seq_clips = project.sequence.tracks[0].clips
        assert len(seq_clips) == 1
        assert seq_clips[0].source_clip_id == "c1"

    def test_all_disabled_adds_nothing(self):
        project = _make_project_with_clips()
        for c in project.clips:
            c.disabled = True
        project.add_to_sequence(["c0", "c1", "c2"])
        seq_clips = project.sequence.tracks[0].clips
        assert len(seq_clips) == 0


# --- Save/load round-trip ---


class TestDisabledClipSaveLoad:
    def test_save_load_preserves_disabled(self, tmp_path):
        # Create project with a disabled clip
        source = Source(
            id="s1",
            file_path=tmp_path / "test.mp4",
            fps=30.0,
            duration_seconds=10.0,
        )
        # Create the fake video file so load doesn't skip it
        (tmp_path / "test.mp4").touch()

        clips = [
            Clip(id="c0", source_id="s1", start_frame=0, end_frame=100, disabled=True),
            Clip(id="c1", source_id="s1", start_frame=100, end_frame=200),
        ]

        project_path = tmp_path / "test.srproj"
        save_project(project_path, [source], clips, None)

        # Load and verify
        loaded_sources, loaded_clips, _, _, _, _ = load_project(project_path)
        by_id = {c.id: c for c in loaded_clips}
        assert by_id["c0"].disabled is True
        assert by_id["c1"].disabled is False
