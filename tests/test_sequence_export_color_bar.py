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
        fps=30.0,
        config=config,
        bar_color=(255, 0, 0),
    )

    assert success is True
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert "drawbox=" in vf
    assert "0xff0000@1.0" in vf
    assert "\\," in vf


def test_export_resolves_clip_color_and_falls_back_to_black(monkeypatch, tmp_path):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")
    bar_colors = []

    def fake_export_segment(*, source_path, output_path, start_frame, end_frame, fps, config, bar_color=None, seq_clip=None):
        bar_colors.append(bar_color)
        return True

    def fake_concat_segments(*, segment_paths, output_path, config):
        return True

    monkeypatch.setattr(exporter, "_export_segment", fake_export_segment)
    monkeypatch.setattr(exporter, "_concat_segments", fake_concat_segments)

    source = Source(id="src-1", file_path=Path("src.mp4"), fps=30.0)
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
