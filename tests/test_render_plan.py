"""Render adapters share references, intervals and representability checks."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from core.project import Project
from core.render_plan import RenderPlanError, compile_render_plan
from models.clip import Clip, Source
from models.frame import Frame
from models.sequence import Sequence, Track


def media_project(tmp_path, rates=(24, 25, 30, 30000 / 1001)):
    project = Project.new()
    project.sequence = Sequence(fps=30)
    for index, rate in enumerate(rates):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"media")
        source = Source(file_path=path, fps=rate)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=264)
        project.add_source(source)
        project.add_clips([clip])
        project.add_to_sequence([clip.id])
    return project


def compile_project(project, **kwargs):
    return compile_render_plan(
        project.sequence, project.sources_by_id,
        {clip.id: (clip, project.sources_by_id[clip.source_id]) for clip in project.clips},
        frames=project.frames_by_id, **kwargs,
    )


def test_mixed_rate_plan_preserves_adjacent_boundaries_and_source_offsets(tmp_path):
    project = media_project(tmp_path)
    plan = compile_project(project)
    assert len(plan.segments) == 4
    assert all(segment.media.start_frame == 240 for segment in plan.segments)
    for left, right in zip(plan.segments, plan.segments[1:]):
        assert left.timeline.end == right.timeline.start
        assert left.end_frame == right.start_frame
    assert plan.segments[-1].timeline.end == project.sequence.duration_time


def test_plan_is_detached_and_immutable(tmp_path):
    project = media_project(tmp_path, (24,))
    plan = compile_project(project)
    project.sequence.tracks[0].clips[0].in_point += 1
    project.sources[0].file_path = tmp_path / "different.mp4"
    assert plan.segments[0].media.start_frame == 240
    assert plan.segments[0].path.name == "0.mp4"
    with pytest.raises(FrozenInstanceError):
        plan.segments[0].reverse = True


def test_gui_mapping_can_resolve_offline_media_but_render_preflight_rejects_it(tmp_path):
    project = media_project(tmp_path, (24,))
    project.sources[0].file_path.unlink()
    plan = compile_project(project, check_media=False)
    assert plan.source_frame_at(0) == 240
    assert plan.media_stamps == ()
    with pytest.raises(RenderPlanError, match="missing"):
        compile_project(project)


def test_gap_uses_output_rate_and_edl_retains_record_position(tmp_path):
    project = media_project(tmp_path, (24,))
    entry = project.sequence.get_all_clips()[0]
    project.update_sequence_clip(entry.id, start_frame=60)
    plan = compile_project(project, output_fps=25)
    assert [segment.kind for segment in plan.segments] == ["gap", "video"]
    assert plan.segments[0].frame_count == 50
    assert plan.segments[1].start_frame == 50
    assert plan.duration == 3
    with pytest.raises(RenderPlanError, match="rates to match"):
        plan.validate_edl()


def test_edl_retains_gap_and_absolute_source_timecodes(tmp_path):
    from core.edl_export import EDLExportConfig, export_edl

    project = media_project(tmp_path, (30,))
    entry = project.sequence.get_all_clips()[0]
    project.update_sequence_clip(entry.id, start_frame=60)
    path = tmp_path / "edit.edl"
    assert export_edl(project.sequence, project.sources_by_id, EDLExportConfig(path))
    assert "00:00:08:00 00:00:08:24 00:00:02:00 00:00:02:24" in path.read_text()


def test_edl_failure_preserves_previous_file_and_explains_transform(tmp_path):
    from core.edl_export import EDLExportConfig, export_edl

    project = media_project(tmp_path, (30,))
    project.update_sequence_clip(project.sequence.get_all_clips()[0].id, reverse=True)
    path = tmp_path / "edit.edl"
    path.write_text("previous edit")
    config = EDLExportConfig(path)
    assert not export_edl(project.sequence, project.sources_by_id, config)
    assert "transforms" in config.error_message
    assert path.read_text() == "previous edit"


@pytest.mark.parametrize("other_track", [False, True])
def test_overlaps_fail_preflight_even_across_tracks(tmp_path, other_track):
    project = media_project(tmp_path, (24, 24))
    entry = project.sequence.get_all_clips()[1]
    if other_track:
        project.sequence.tracks.append(Track())
    project.update_sequence_clip(entry.id, start_frame=15, track_index=int(other_track))
    with pytest.raises(RenderPlanError, match="Overlapping"):
        compile_project(project)


@pytest.mark.parametrize("missing", ["source", "clip", "file", "music"])
def test_missing_references_fail_preflight(tmp_path, missing):
    project = media_project(tmp_path, (24,))
    if missing == "source":
        sources = {}
        with pytest.raises(RenderPlanError, match="Source missing"):
            compile_render_plan(project.sequence, sources)
        return
    if missing == "clip":
        with pytest.raises(RenderPlanError, match="Library clip reference"):
            compile_render_plan(project.sequence, project.sources_by_id, {})
        return
    if missing == "file":
        project.sources[0].file_path.unlink()
    else:
        project.sequence.music_path = str(tmp_path / "absent.wav")
    with pytest.raises(RenderPlanError, match="media is missing"):
        compile_project(project)


@pytest.mark.parametrize("effect", ["reverse", "hflip", "vflip"])
def test_edl_refuses_unrepresentable_transforms(tmp_path, effect):
    project = media_project(tmp_path, (24,))
    project.update_sequence_clip(project.sequence.get_all_clips()[0].id, **{effect: True})
    with pytest.raises(RenderPlanError, match="EDL cannot represent transforms"):
        compile_project(project).validate_edl()


def test_still_duration_is_independent_of_output_rate(tmp_path):
    path = tmp_path / "still.png"
    path.write_bytes(b"image")
    project = Project.new()
    project.sequence = Sequence(fps=24)
    frame = Frame(file_path=path)
    project.add_frames([frame])
    project.add_frames_to_sequence([frame.id], hold_frames=12)
    plan = compile_project(project, output_fps=30)
    assert plan.segments[0].frame_count == 15
    assert plan.duration == Fraction(1, 2)
    with pytest.raises(RenderPlanError, match="still hold"):
        plan.validate_edl()


def test_cli_edl_exports_the_loaded_project(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from types import SimpleNamespace

    project = media_project(tmp_path, (30,))
    path = tmp_path / "project.json"
    project.save(path)
    output = tmp_path / "cli.edl"
    monkeypatch.setattr("cli.commands.export.CLIConfig.load", lambda: SimpleNamespace(export_dir=tmp_path))
    register_commands()
    result = CliRunner().invoke(cli, ["export", "edl", str(path), "-o", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "00:00:08:00 00:00:08:24" in output.read_text()
