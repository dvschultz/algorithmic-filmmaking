"""Observable color contracts across existing application surfaces.

Only the media computation is stubbed; selection, persistence and result
formatting run through the real adapters.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from core.project import Project
from tests.test_spine_analyze import _build_project


@pytest.mark.parametrize("surface", ["spine", "cli", "mcp", "worker"])
def test_color_surfaces_use_source_ranges_and_preserve_metadata(tmp_path, surface):
    project = _build_project(tmp_path, n_clips=2)
    project.clips[1].description = "Keep this description"
    path = tmp_path / "project.json"
    assert project.save(path)
    palette = [(10, 20, 30)]

    with patch(
        "core.analysis.color.extract_dominant_colors", return_value=palette
    ) as extract:
        if surface == "spine":
            from core.spine.analyze import analyze_colors

            assert analyze_colors(project)["success"]
            assert project.save()
        elif surface == "cli":
            from cli.main import cli, register_commands

            register_commands()
            result = CliRunner().invoke(cli, ["--json", "analyze", "colors", str(path)])
            assert result.exit_code == 0, result.output
            # Older Click versions merge stderr progress into captured stdout.
            assert (
                json.loads(result.output[result.output.index("{") :])["analyzed_clips"]
                == 2
            )
        elif surface == "mcp":
            from scene_ripper_mcp.tools.analyze import _analyze_colors_sync

            assert json.loads(_analyze_colors_sync(path, 5))["analyzed_clips"] == 2
        else:
            from ui.workers.color_worker import ColorAnalysisWorker

            worker = ColorAnalysisWorker(
                project.clips, sources_by_id=project.sources_by_id
            )
            results = {}
            worker.color_ready.connect(
                lambda clip_id, colors: results.update({clip_id: colors})
            )
            worker.run()
            assert {
                key: [tuple(rgb) for rgb in colors] for key, colors in results.items()
            } == {clip.id: palette for clip in project.clips}
            # The GUI adapter currently owns result application.
            for clip in project.clips:
                clip.dominant_colors = results[clip.id]
            project.update_clips(project.clips)
            assert project.save()

    assert sorted(
        (c.kwargs["start_frame"], c.kwargs["end_frame"]) for c in extract.call_args_list
    ) == [
        (0, 60),
        (60, 120),
    ]
    loaded = Project.load(path)
    assert [c.dominant_colors for c in loaded.clips] == [palette, palette]
    assert loaded.clips[1].description == "Keep this description"


@pytest.mark.parametrize(
    "surface, expected_calls", [("spine", 1), ("cli", 1), ("mcp", 2), ("worker", 1)]
)
def test_color_default_recompute_policy(tmp_path, surface, expected_calls):
    project = _build_project(tmp_path, n_clips=2, populate_colors=1)
    path = tmp_path / "project.json"
    assert project.save(path)
    with patch(
        "core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]
    ) as extract:
        if surface == "spine":
            from core.spine.analyze import analyze_colors

            analyze_colors(project)
        elif surface == "cli":
            from cli.main import cli, register_commands

            register_commands()
            result = CliRunner().invoke(cli, ["analyze", "colors", str(path)])
            assert result.exit_code == 0, result.output
        elif surface == "mcp":
            from scene_ripper_mcp.tools.analyze import _analyze_colors_sync

            assert json.loads(_analyze_colors_sync(path, 5))["success"]
        else:
            from ui.workers.color_worker import ColorAnalysisWorker

            ColorAnalysisWorker(
                project.clips, sources_by_id=project.sources_by_id
            ).run()
    assert extract.call_count == expected_calls


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.3", "1.4"])
def test_supported_project_versions_round_trip(tmp_path, version):
    fixture = Path(__file__).parent / "fixtures" / "projects" / f"v{version}.json"
    path = tmp_path / "project.json"
    path.write_text(fixture.read_text())
    for name in ("video.mp4", "still.png", "music.wav"):
        (tmp_path / name).touch()
    project = Project.load(path)
    assert project.clips[0].start_frame == 48
    assert project.sources[0].fps == 23.976
    assert project.sequence.fps == 30
    entry = project.sequence.tracks[0].clips[0]
    assert (entry.in_point, entry.out_point, entry.hflip, entry.reverse) == (
        48,
        96,
        True,
        True,
    )
    assert project.save()
    restored = Project.load(path)
    assert restored.clips[0].start_frame == 48
    if version == "1.4":
        assert len(restored.sequences) == 2
        assert restored.frames[0].id == "frame-a"
        assert restored.audio_sources[0].id == "audio-a"
        assert restored.sequences[1].tracks[0].clips[0].hold_frames == 24


def test_current_offline_media_loading_requires_explicit_resolution(tmp_path):
    from core.project import MissingSourceError

    fixture = Path(__file__).parent / "fixtures" / "projects" / "v1.0.json"
    path = tmp_path / "project.json"
    path.write_text(fixture.read_text())
    with pytest.raises(MissingSourceError):
        Project.load(path)
    # Characterizes a known destructive policy, not the intended U5 behavior.
    project = Project.load(path, missing_source_callback=lambda *_: None)
    assert project.sources == []
    assert project.clips == []
    assert project.sequence.tracks[0].clips == []


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_color_updates_preserve_all_sequences_frames_and_audio(tmp_path, surface):
    fixture = Path(__file__).parent / "fixtures" / "projects" / "v1.4.json"
    path = tmp_path / "project.json"
    path.write_text(fixture.read_text())
    for name in ("video.mp4", "still.png", "music.wav"):
        (tmp_path / name).touch()
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        if surface == "cli":
            from cli.main import cli, register_commands

            register_commands()
            result = CliRunner().invoke(cli, ["analyze", "colors", str(path)])
            assert result.exit_code == 0, result.output
        else:
            from scene_ripper_mcp.tools.analyze import _analyze_colors_sync

            assert json.loads(_analyze_colors_sync(path, 5))["success"]
    restored = Project.load(path)
    assert len(restored.sequences) == 2
    assert restored.frames[0].id == "frame-a"
    assert restored.audio_sources[0].id == "audio-a"


@pytest.mark.parametrize("surface", ["cli", "mcp", "worker"])
def test_color_partial_failures_are_visible_without_discarding_success(
    tmp_path, surface
):
    project = _build_project(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)

    def extract(**kwargs):
        if kwargs["start_frame"] == 0:
            raise RuntimeError("unreadable frame")
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=extract):
        if surface == "cli":
            from cli.main import cli, register_commands

            register_commands()
            output = CliRunner().invoke(cli, ["--json", "analyze", "colors", str(path)])
            assert output.exit_code == 0, output.output
            result = json.loads(output.output[output.output.index("{") :])
            assert result["analyzed_clips"] == 1
            assert result["errors"] == 1
            assert "unreadable frame" in result["error_details"][0]
        elif surface == "mcp":
            from scene_ripper_mcp.tools.analyze import _analyze_colors_sync

            result = json.loads(_analyze_colors_sync(path, 5))
            assert result["analyzed_clips"] == 1
            assert result["error_details"][0]["code"] == "extraction_failed"
        else:
            from ui.workers.color_worker import ColorAnalysisWorker

            worker = ColorAnalysisWorker(
                project.clips, sources_by_id=project.sources_by_id, project=project
            )
            errors = []
            worker.error.connect(errors.append)
            worker.result_ready.connect(
                lambda application, result: application.apply(result)
            )
            worker.run()
            assert errors == ["unreadable frame"]
            assert project.save()
    restored = Project.load(path)
    assert restored.clips[0].dominant_colors is None
    assert restored.clips[1].dominant_colors == [(1, 2, 3)]
