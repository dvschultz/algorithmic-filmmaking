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

    from models.clip import Clip, Source
    from models.sequence import Sequence, SequenceClip

    source = Source(id="src-1", file_path=tmp_path / "source.mp4", fps=30.0)
    source.file_path.write_bytes(b"source")
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30)
    sequence = Sequence(fps=30.0)
    sequence.tracks[0].clips = [SequenceClip(
        source_clip_id=clip.id, source_id=source.id, in_point=0, out_point=30,
    )]
    monkeypatch.setattr(exporter, "_export_segment", lambda **kwargs: True)
    monkeypatch.setattr(exporter, "_has_audio", lambda *args: False)
    monkeypatch.setattr("core.media_timing.probe_video_timing", lambda *args: SimpleNamespace(frame_count=30, variable=False, rate=30, origin=0))
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
