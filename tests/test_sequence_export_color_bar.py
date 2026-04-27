"""Tests for chromatic color-bar export behavior."""

from pathlib import Path
from types import SimpleNamespace

from core.sequence_export import ExportConfig, SequenceExporter
from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip


def test_export_segment_includes_drawbox_filter_when_color_bar_enabled(monkeypatch, tmp_path):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.sequence_export.subprocess.run", fake_run)

    config = ExportConfig(
        output_path=tmp_path / "out.mp4",
        width=1280,
        height=720,
        show_chromatic_color_bar=True,
    )
    success = exporter._export_segment(
        source_path=Path("input.mp4"),
        output_path=tmp_path / "segment.mp4",
        start_frame=0,
        end_frame=30,
        source_fps=30.0,
        config=config,
        bar_color=(255, 0, 0),
    )

    assert success is True
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert "fps=30.0" in vf
    assert "drawbox=" in vf
    assert "0xff0000@1.0" in vf
    assert "\\," in vf


def test_export_resolves_clip_color_and_falls_back_to_black(monkeypatch, tmp_path):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    bar_colors = []
    source_fps_values = []

    def fake_export_segment(
        *,
        source_path,
        output_path,
        start_frame,
        end_frame,
        source_fps,
        config,
        bar_color=None,
        seq_clip=None,
    ):
        bar_colors.append(bar_color)
        source_fps_values.append(source_fps)
        return True

    def fake_concat_segments(*, segment_paths, output_path, config):
        return True

    monkeypatch.setattr(exporter, "_export_segment", fake_export_segment)
    monkeypatch.setattr(exporter, "_concat_segments", fake_concat_segments)

    source = Source(id="src-1", file_path=Path("src.mp4"), fps=24.0)
    clip_a = Clip(
        id="clip-a",
        source_id=source.id,
        start_frame=0,
        end_frame=30,
        dominant_colors=[(10, 20, 30)],
    )
    clip_b = Clip(
        id="clip-b",
        source_id=source.id,
        start_frame=30,
        end_frame=60,
        dominant_colors=None,
    )
    sequence = Sequence(fps=30.0)
    sequence.tracks[0].clips = [
        SequenceClip(
            source_clip_id=clip_a.id,
            source_id=source.id,
            start_frame=0,
            in_point=clip_a.start_frame,
            out_point=clip_a.end_frame,
        ),
        SequenceClip(
            source_clip_id=clip_b.id,
            source_id=source.id,
            start_frame=30,
            in_point=clip_b.start_frame,
            out_point=clip_b.end_frame,
        ),
    ]

    config = ExportConfig(
        output_path=tmp_path / "sequence.mp4",
        fps=30.0,
        show_chromatic_color_bar=True,
    )

    success = exporter.export(
        sequence=sequence,
        sources={source.id: source},
        clips={clip_a.id: (clip_a, source), clip_b.id: (clip_b, source)},
        config=config,
    )

    assert success is True
    assert bar_colors == [(10, 20, 30), (0, 0, 0)]
    assert source_fps_values == [24.0, 24.0]


def test_export_inserts_black_silent_segments_for_timeline_gaps(monkeypatch, tmp_path):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    calls = []

    def fake_export_gap_segment(output_path, duration_seconds, config):
        calls.append(("gap", duration_seconds))
        return True

    def fake_export_segment(
        *,
        source_path,
        output_path,
        start_frame,
        end_frame,
        source_fps,
        config,
        bar_color=None,
        seq_clip=None,
    ):
        calls.append(("clip", start_frame, end_frame))
        return True

    def fake_concat_segments(*, segment_paths, output_path, config):
        return True

    monkeypatch.setattr(exporter, "_export_gap_segment", fake_export_gap_segment)
    monkeypatch.setattr(exporter, "_export_segment", fake_export_segment)
    monkeypatch.setattr(exporter, "_concat_segments", fake_concat_segments)

    source = Source(id="src-1", file_path=Path("src.mp4"), fps=30.0, width=1280, height=720)
    clip = Clip(id="clip-a", source_id=source.id, start_frame=0, end_frame=30)
    sequence = Sequence(fps=30.0)
    sequence.tracks[0].clips = [
        SequenceClip(
            source_clip_id=clip.id,
            source_id=source.id,
            start_frame=30,
            in_point=0,
            out_point=30,
        )
    ]

    success = exporter.export(
        sequence=sequence,
        sources={source.id: source},
        clips={clip.id: (clip, source)},
        config=ExportConfig(output_path=tmp_path / "out.mp4", fps=30.0, width=1280, height=720),
    )

    assert success is True
    assert calls == [("gap", 1.0), ("clip", 0, 30)]


def test_export_segment_uses_source_fps_for_trim_and_sequence_fps_for_output(
    monkeypatch, tmp_path,
):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.sequence_export.subprocess.run", fake_run)

    config = ExportConfig(
        output_path=tmp_path / "out.mp4",
        fps=30.0,
    )

    success = exporter._export_segment(
        source_path=Path("input.mp4"),
        output_path=tmp_path / "segment.mp4",
        start_frame=48,
        end_frame=96,
        source_fps=24.0,
        config=config,
    )

    assert success is True
    cmd = captured["cmd"]
    # Double -ss: coarse seek (max(0, 2.0-5.0)=0.0) before -i, precise (2.0) after -i
    ss_indices = [i for i, v in enumerate(cmd) if v == "-ss"]
    assert len(ss_indices) == 2
    assert cmd[ss_indices[0] + 1] == "0"  # coarse seek
    assert cmd[ss_indices[1] + 1] == "2.0"  # precise seek
    # Duration subtracts one frame (1/24s) to prevent audio bleed at cut boundaries
    expected_duration = (96 - 48) / 24.0 - 1.0 / 24.0
    assert cmd[cmd.index("-t") + 1] == str(expected_duration)
    assert cmd[cmd.index("-vf") + 1] == "fps=30.0"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    # No -af when there's no reverse: output -ss + -t already cuts audio
    # sample-accurately. Adding atrim=0:N here would silently strip audio
    # because the precise -ss after -i shifts audio PTS past the atrim window.
    assert "-af" not in cmd


def test_export_segment_omits_atrim_so_audio_survives_double_ss(monkeypatch, tmp_path):
    """Regression: double -ss + atrim=0:duration silently produced silent
    segments because output -ss shifts audio PTS past the atrim window.

    The audio filter chain must NOT contain atrim/asetpts. If reverse is
    not requested, -af should be absent entirely.
    """
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.sequence_export.subprocess.run", fake_run)

    config = ExportConfig(output_path=tmp_path / "out.mp4", fps=30.0)
    exporter._export_segment(
        source_path=Path("input.mp4"),
        output_path=tmp_path / "segment.mp4",
        start_frame=240,  # 10s in at 24fps -> exercises the precise -ss
        end_frame=264,
        source_fps=24.0,
        config=config,
    )

    cmd = captured["cmd"]
    cmd_str = " ".join(cmd)
    assert "atrim" not in cmd_str
    assert "asetpts" not in cmd_str
    assert "-af" not in cmd


def test_export_segment_uses_areverse_only_when_reverse_requested(monkeypatch, tmp_path):
    """When seq_clip.reverse is set, -af should be present with areverse only."""
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.sequence_export.subprocess.run", fake_run)

    seq_clip = SequenceClip(
        source_clip_id="c1",
        source_id="s1",
        in_point=0,
        out_point=24,
        reverse=True,
    )
    config = ExportConfig(output_path=tmp_path / "out.mp4", fps=30.0)
    exporter._export_segment(
        source_path=Path("input.mp4"),
        output_path=tmp_path / "segment.mp4",
        start_frame=0,
        end_frame=24,
        source_fps=24.0,
        config=config,
        seq_clip=seq_clip,
    )

    cmd = captured["cmd"]
    assert "-af" in cmd
    af = cmd[cmd.index("-af") + 1]
    assert af == "areverse"
