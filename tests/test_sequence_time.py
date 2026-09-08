"""All sequence creation routes preserve source ranges and exact durations."""

from fractions import Fraction
from pathlib import Path

from core.project import Project
from core.sequence_time import video_entry
from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip


def test_one_frame_cuts_survive_lower_timeline_rate_without_duration_drift():
    project = Project.new()
    project.sequence = Sequence(fps=24)
    source = Source(file_path=Path("short.mp4"), fps=30)
    clip = Clip(source_id=source.id, start_frame=240, end_frame=241)
    project.add_source(source)
    project.add_clips([clip])
    for _ in range(30):
        project.add_to_sequence([clip.id])
    entries = project.sequence.get_all_clips()
    assert len(entries) == 30
    assert any(entry.duration_frames == 0 for entry in entries)
    assert project.sequence.duration_time == 1
    assert project.sequence.duration_frames == 24
    assert all(left.timeline_range.end == right.timeline_range.start
               for left, right in zip(entries, entries[1:]))
    before = project.sequence.to_dict()
    project.update_sequence_metadata(project.sequence, fps=30)
    assert all(entry.duration_frames == 1 for entry in entries)
    project.session.undo()
    assert project.sequence.to_dict() == before


def test_subframe_still_can_be_edited_after_lowering_timeline_rate():
    from models.frame import Frame
    project = Project.new()
    project.sequence = Sequence(fps=60)
    frame = Frame(file_path=Path("still.png"))
    project.add_frames([frame])
    project.add_frames_to_sequence([frame.id], hold_frames=1)
    project.update_sequence_metadata(project.sequence, fps=24)
    entry = project.sequence.get_all_clips()[0]
    assert entry.hold_frames == 0
    project.update_sequence_clip(entry.id, hflip=True)
    assert entry.hflip
    assert entry.source_range.duration == Fraction(1, 60)
    project.session.undo()
    assert not entry.hflip


def test_project_append_mixed_rates_uses_timeline_seconds():
    project = Project.new()
    project.sequence = Sequence(fps=30)
    ids = []
    for rate in (24, 25, 30):
        source = Source(file_path=Path(f"{rate}.mp4"), fps=rate)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=240 + rate)
        project.add_source(source)
        project.add_clips([clip])
        ids.append(clip.id)
    project.add_to_sequence(ids)
    entries = project.sequence.get_all_clips()
    assert [entry.start_frame for entry in entries] == [0, 30, 60]
    assert project.sequence.duration_frames == 90
    assert all(entry.in_point == 240 for entry in entries)
    assert [entry.source_range.start for entry in entries] == [Fraction(10), Fraction(48, 5), Fraction(8)]


def test_relative_selection_is_converted_only_once_and_round_trips():
    source = Source(file_path=Path("video.mp4"), fps=24)
    clip = Clip(source_id=source.id, start_frame=240, end_frame=300)
    entry = video_entry(clip, source, timeline_fps=30, start=Fraction(1, 24), relative_range=(12, 36))
    restored = SequenceClip.from_dict(entry.to_dict())
    assert restored.in_point == 252
    assert restored.out_point == 276
    assert restored.timeline_range.start == Fraction(1, 24)
    assert restored.source_range.duration == 1


def test_rate_change_preserves_media_time_and_undo_restores_exact_state():
    project = Project.new()
    project.sequence = Sequence(fps=30)
    source = Source(file_path=Path("video.mp4"), fps=24)
    clip = Clip(source_id=source.id, start_frame=240, end_frame=241)
    project.add_source(source)
    project.add_clips([clip])
    for _ in range(30):
        project.add_to_sequence([clip.id])
    before = project.sequence.to_dict()
    assert project.sequence.duration_time == Fraction(5, 4)
    project.update_sequence_metadata(project.sequence, fps=60)
    assert project.sequence.duration_time == Fraction(5, 4)
    assert project.sequence.duration_frames == 75
    project.session.undo()
    assert project.sequence.to_dict() == before
    project.session.redo()
    assert project.sequence.duration_frames == 75


def test_reorder_mixed_rates_preserves_shared_boundaries_and_undo():
    project = Project.new()
    project.sequence = Sequence(fps=30)
    for rate in (24, 25, 30):
        source = Source(file_path=Path(f"{rate}.mp4"), fps=rate)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=241)
        project.add_source(source)
        project.add_clips([clip])
        project.add_to_sequence([clip.id])
    before = project.sequence.to_dict()
    ids = [entry.id for entry in project.sequence.get_all_clips()]
    project.reorder_sequence(list(reversed(ids)))
    entries = project.sequence.get_all_clips()
    assert entries[0].timeline_range.end == entries[1].timeline_range.start
    assert entries[1].timeline_range.end == entries[2].timeline_range.start
    project.session.undo()
    assert project.sequence.to_dict() == before


def test_vfr_trim_uses_presentation_boundaries_and_undo():
    source = Source(
        file_path=Path("vfr.mp4"), fps=30, variable_frame_rate=True,
        frame_timestamps=("0", "1/50", "7/100", "1/10"),
    )
    clip = Clip(source_id=source.id, start_frame=0, end_frame=3)
    project = Project(sources=[source], clips=[clip])
    project.add_to_sequence([clip.id])
    entry = project.sequence.get_all_clips()[0]
    before = entry.to_dict()
    assert entry.source_presentation == ("0", "1/10")
    project.update_sequence_clip(entry.id, in_point=1)
    assert entry.source_range.start == Fraction(1, 50)
    assert entry.source_range.duration == Fraction(2, 25)
    project.session.undo()
    assert entry.to_dict() == before


def test_still_duration_survives_frame_rate_change():
    from models.frame import Frame
    project = Project.new()
    project.sequence = Sequence(fps=20)
    frame = Frame(file_path=Path("still.png"))
    project.add_frames([frame])
    project.add_frames_to_sequence([frame.id], hold_frames=1)
    before = project.sequence.to_dict()
    project.update_sequence_metadata(project.sequence, fps=30)
    entry = project.sequence.get_all_clips()[0]
    assert entry.hold_frames == 2
    assert entry.source_range.duration == Fraction(1, 20)
    assert project.sequence.duration_time == Fraction(1, 20)
    project.session.undo()
    assert project.sequence.to_dict() == before


def test_changed_vfr_map_changes_new_entries_without_reinterpreting_old_ones():
    source = Source(
        file_path=Path("vfr.mp4"), fps=30, variable_frame_rate=True,
        frame_timestamps=("0", "1/50", "7/100", "1/10"),
    )
    clip = Clip(source_id=source.id, start_frame=0, end_frame=3)
    first = video_entry(clip, source, timeline_fps=30, start=Fraction(0))
    source.frame_timestamps = ("0", "1/25", "7/50", "1/5")
    second = video_entry(clip, source, timeline_fps=30, start=Fraction(0))
    assert first.source_range.duration == Fraction(1, 10)
    assert second.source_range.duration == Fraction(1, 5)
    assert len(second.to_dict()["source_presentation"]) == 2
