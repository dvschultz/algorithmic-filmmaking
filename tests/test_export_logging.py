"""Focused tests for export failure logging."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from core.ffmpeg import FFmpegProcessor
from core.sequence_export import ExportConfig, SequenceExporter


def test_extract_clip_logs_ffmpeg_stderr_on_failure(monkeypatch, tmp_path, caplog):
    processor = FFmpegProcessor.__new__(FFmpegProcessor)
    processor.ffmpeg_path = "ffmpeg"
    processor.ffprobe_path = "ffprobe"
    processor.ffmpeg_available = True
    processor.ffprobe_available = True

    monkeypatch.setattr(
        "core.ffmpeg.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="encoder blew up"),
    )

    with caplog.at_level(logging.INFO):
        success = processor.extract_clip(
            input_path=Path("input.mov"),
            output_path=tmp_path / "output.mp4",
            start_seconds=1.0,
            duration_seconds=2.0,
            fps=30.0,
        )

    assert success is False
    assert "Extracting clip:" in caplog.text
    assert "FFmpeg clip extraction failed" in caplog.text
    assert "encoder blew up" in caplog.text


def test_sequence_export_logs_concat_failure(monkeypatch, tmp_path, caplog):
    exporter = SequenceExporter(ffmpeg_path="ffmpeg")

    source = SimpleNamespace(id="src-1", file_path=Path("source.mp4"), fps=30.0)
    clip = SimpleNamespace(id="clip-1", start_frame=0, end_frame=30, dominant_colors=None)
    seq_clip = SimpleNamespace(
        id="seq-1",
        is_frame_entry=False,
        source_clip_id="clip-1",
        in_point=0,
        out_point=30,
        reverse=False,
        hflip=False,
        vflip=False,
    )
    sequence = SimpleNamespace(
        fps=30.0,
        get_all_clips=lambda: [seq_clip],
    )

    monkeypatch.setattr(exporter, "_export_segment", lambda **kwargs: True)
    monkeypatch.setattr(exporter, "_concat_segments", lambda **kwargs: False)

    config = ExportConfig(output_path=tmp_path / "sequence.mp4", fps=30.0)

    with caplog.at_level(logging.INFO):
        success = exporter.export(
            sequence=sequence,
            sources={source.id: source},
            clips={clip.id: (clip, source)},
            config=config,
        )

    assert success is False
    assert "Starting sequence export" in caplog.text
    assert "Sequence export concat failed" in caplog.text
