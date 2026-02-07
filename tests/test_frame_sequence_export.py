"""Tests for frame support in sequence, EDL, and SRT export."""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from core.edl_export import EDLExportConfig, export_edl, frames_to_timecode
from core.sequence_export import ExportConfig, SequenceExporter
from core.srt_export import SRTExportConfig, export_srt
from models.clip import Clip, Source
from models.frame import Frame
from models.sequence import Sequence, SequenceClip, Track


# -- Helpers -----------------------------------------------------------------

def _make_source(source_id: str = "src1", fps: float = 30.0) -> Source:
    return Source(
        id=source_id,
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=fps,
    )


def _make_clip(clip_id: str, source_id: str = "src1") -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=90,
        description="Test clip description",
    )


def _make_frame(frame_id: str, file_path: Path, description: str = None) -> Frame:
    return Frame(
        id=frame_id,
        file_path=file_path,
        description=description,
        frame_number=42,
    )


def _make_frame_seq_clip(
    frame_id: str,
    start_frame: int = 0,
    hold_frames: int = 30,
) -> SequenceClip:
    return SequenceClip(
        source_clip_id="",
        source_id="",
        frame_id=frame_id,
        hold_frames=hold_frames,
        start_frame=start_frame,
    )


def _make_clip_seq_clip(
    clip_id: str,
    source_id: str = "src1",
    start_frame: int = 0,
    in_point: int = 0,
    out_point: int = 90,
) -> SequenceClip:
    return SequenceClip(
        source_clip_id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        in_point=in_point,
        out_point=out_point,
    )


def _create_test_png(path: Path, width: int = 64, height: int = 64) -> None:
    """Create a minimal valid PNG at *path* using raw bytes (no PIL needed)."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # Create uncompressed pixel data: each row = filter byte (0) + RGB * width
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\x80\x00\x00" * width  # red pixels
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


skip_no_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="FFmpeg not installed"
)


# -- EDL export tests -------------------------------------------------------

class TestEDLExportFrameEntries:
    """EDL export with frame-based SequenceClips."""

    def test_frame_only_sequence(self, tmp_path):
        """EDL with only frame entries produces correct timecodes and clip name."""
        frame_path = tmp_path / "frame.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path, description="Sunset shot")

        seq = Sequence(name="Frame EDL", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1", start_frame=0, hold_frames=60)]

        edl_path = tmp_path / "output.edl"
        config = EDLExportConfig(output_path=edl_path)

        result = export_edl(seq, {}, config, frames={"f1": frame})
        assert result is True

        content = edl_path.read_text()
        assert "TITLE:" in content
        # Frame 42 -> display_name = "Frame 42"
        assert "Frame 42" in content
        # Source timecodes should start at 00:00:00:00
        assert "00:00:00:00" in content

    def test_mixed_clip_and_frame_sequence(self, tmp_path):
        """EDL with both clip and frame entries."""
        source = _make_source()
        frame_path = tmp_path / "still.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path)

        seq = Sequence(name="Mixed EDL", fps=30.0)
        clip_entry = _make_clip_seq_clip("c1", start_frame=0)
        frame_entry = _make_frame_seq_clip("f1", start_frame=90, hold_frames=30)
        seq.tracks[0].clips = [clip_entry, frame_entry]

        edl_path = tmp_path / "mixed.edl"
        config = EDLExportConfig(output_path=edl_path)

        result = export_edl(
            seq,
            {"src1": source},
            config,
            frames={"f1": frame},
        )
        assert result is True

        content = edl_path.read_text()
        lines = [l for l in content.split("\n") if l.startswith("*")]
        # Two FROM CLIP NAME comments: one for video, one for frame
        assert len(lines) == 2
        assert "video.mp4" in lines[0]
        assert "Frame 42" in lines[1]

    def test_frame_entry_skipped_without_frames_dict(self, tmp_path):
        """Frame entries are gracefully skipped when no frames dict provided."""
        seq = Sequence(name="No Frames", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1")]

        edl_path = tmp_path / "skip.edl"
        config = EDLExportConfig(output_path=edl_path)

        # File is written (header only) but the frame entry is skipped
        result = export_edl(seq, {}, config, frames=None)
        assert result is True
        content = edl_path.read_text()
        assert "TITLE:" in content
        # No FROM CLIP NAME lines since all entries were skipped
        assert "FROM CLIP NAME" not in content

    def test_frame_entry_skipped_missing_frame_id(self, tmp_path):
        """Frame entry with unknown frame_id is skipped."""
        seq = Sequence(name="Missing Frame", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("nonexistent")]

        edl_path = tmp_path / "missing.edl"
        config = EDLExportConfig(output_path=edl_path)

        result = export_edl(seq, {}, config, frames={})
        assert result is True
        content = edl_path.read_text()
        assert "FROM CLIP NAME" not in content


# -- SRT export tests --------------------------------------------------------

class TestSRTExportFrameEntries:
    """SRT export with frame-based SequenceClips."""

    def test_frame_with_description(self, tmp_path):
        """Frame entry uses description as subtitle text."""
        frame_path = tmp_path / "frame.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path, description="Golden hour sunset")

        seq = Sequence(name="Frame SRT", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1", hold_frames=90)]

        srt_path = tmp_path / "output.srt"
        config = SRTExportConfig(output_path=srt_path)

        success, exported, skipped = export_srt(
            seq, {}, {}, config, frames={"f1": frame}
        )
        assert success is True
        assert exported == 1
        assert skipped == 0

        content = srt_path.read_text()
        assert "Golden hour sunset" in content
        assert "00:00:00,000" in content

    def test_frame_without_description_uses_display_name(self, tmp_path):
        """Frame entry without description falls back to display_name."""
        frame_path = tmp_path / "frame.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path, description=None)

        seq = Sequence(name="Frame SRT", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1", hold_frames=60)]

        srt_path = tmp_path / "output.srt"
        config = SRTExportConfig(output_path=srt_path)

        success, exported, skipped = export_srt(
            seq, {}, {}, config, frames={"f1": frame}
        )
        assert success is True
        assert exported == 1
        assert "Frame 42" in srt_path.read_text()

    def test_mixed_clip_and_frame_srt(self, tmp_path):
        """SRT export with both clip and frame entries."""
        source = _make_source()
        clip = _make_clip("c1")
        frame_path = tmp_path / "still.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path, description="A beautiful still")

        seq = Sequence(name="Mixed", fps=30.0)
        clip_entry = _make_clip_seq_clip("c1", start_frame=0)
        frame_entry = _make_frame_seq_clip("f1", start_frame=90, hold_frames=30)
        seq.tracks[0].clips = [clip_entry, frame_entry]

        srt_path = tmp_path / "mixed.srt"
        config = SRTExportConfig(output_path=srt_path)

        success, exported, skipped = export_srt(
            seq,
            {"c1": clip},
            {"src1": source},
            config,
            frames={"f1": frame},
        )
        assert success is True
        assert exported == 2
        assert skipped == 0

        content = srt_path.read_text()
        assert "Test clip description" in content
        assert "A beautiful still" in content

    def test_frame_entry_skipped_without_frames_dict(self, tmp_path):
        """Frame entries skipped when no frames dict -> counted as skipped."""
        seq = Sequence(name="No Frames", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1")]

        srt_path = tmp_path / "skip.srt"
        config = SRTExportConfig(output_path=srt_path)

        success, exported, skipped = export_srt(seq, {}, {}, config)
        # File is written (empty) but entry is skipped
        assert success is True
        assert exported == 0
        assert skipped == 1

    def test_frame_timecodes_are_correct(self, tmp_path):
        """Verify frame entry timecodes are based on hold_frames / fps."""
        frame_path = tmp_path / "f.png"
        _create_test_png(frame_path)
        frame = _make_frame("f1", frame_path, description="X")

        seq = Sequence(name="TC", fps=30.0)
        # 30 frames at 30fps = 1.0s duration, starting at frame 90 = 3.0s
        entry = _make_frame_seq_clip("f1", start_frame=90, hold_frames=30)
        seq.tracks[0].clips = [entry]

        srt_path = tmp_path / "tc.srt"
        config = SRTExportConfig(output_path=srt_path)

        success, _, _ = export_srt(seq, {}, {}, config, frames={"f1": frame})
        assert success is True

        content = srt_path.read_text()
        assert "00:00:03,000 --> 00:00:04,000" in content


# -- Video export tests (require FFmpeg) -------------------------------------

class TestSequenceExportFrameEntries:
    """SequenceExporter.export() with frame-based entries."""

    @skip_no_ffmpeg
    def test_frame_segment_generation(self, tmp_path):
        """Frame entry produces a valid video segment."""
        frame_path = tmp_path / "frame.png"
        _create_test_png(frame_path, width=320, height=240)
        frame = _make_frame("f1", frame_path)

        exporter = SequenceExporter()
        segment_path = tmp_path / "segment.mp4"

        success = exporter._export_frame_segment(
            frame_path=frame.file_path,
            output_path=segment_path,
            hold_seconds=1.0,
            fps=30.0,
            config=ExportConfig(output_path=tmp_path / "out.mp4"),
        )
        assert success is True
        assert segment_path.exists()
        assert segment_path.stat().st_size > 0

        # Verify duration with ffprobe
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(segment_path),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            duration = float(probe.stdout.strip())
            assert 0.8 <= duration <= 1.5  # Allow codec tolerance

    @skip_no_ffmpeg
    def test_frame_only_sequence_export(self, tmp_path):
        """Full export of a sequence containing only frame entries."""
        frame_path = tmp_path / "frame.png"
        _create_test_png(frame_path, width=320, height=240)
        frame = _make_frame("f1", frame_path)

        seq = Sequence(name="Frame Only", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1", hold_frames=15)]

        output = tmp_path / "frame_only.mp4"
        config = ExportConfig(output_path=output, fps=30.0)

        exporter = SequenceExporter()
        success = exporter.export(
            sequence=seq,
            sources={},
            clips={},
            config=config,
            frames={"f1": frame},
        )
        assert success is True
        assert output.exists()

    @skip_no_ffmpeg
    def test_frame_with_resolution_scaling(self, tmp_path):
        """Frame segment respects width/height config with letterboxing."""
        frame_path = tmp_path / "wide.png"
        _create_test_png(frame_path, width=640, height=240)
        frame = _make_frame("f1", frame_path)

        exporter = SequenceExporter()
        segment_path = tmp_path / "scaled.mp4"

        success = exporter._export_frame_segment(
            frame_path=frame.file_path,
            output_path=segment_path,
            hold_seconds=0.5,
            fps=30.0,
            config=ExportConfig(
                output_path=tmp_path / "out.mp4",
                width=320,
                height=240,
            ),
        )
        assert success is True
        assert segment_path.exists()

    def test_frame_entry_skipped_without_frames_dict(self, tmp_path):
        """Frame entries are skipped when frames dict is None."""
        seq = Sequence(name="No Frames", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("f1")]

        output = tmp_path / "out.mp4"
        config = ExportConfig(output_path=output, fps=30.0)

        exporter = SequenceExporter()
        success = exporter.export(
            sequence=seq,
            sources={},
            clips={},
            config=config,
            frames=None,
        )
        # No segments produced -> False
        assert success is False

    def test_frame_entry_skipped_missing_frame_id(self, tmp_path):
        """Frame entry with unknown frame_id is skipped gracefully."""
        seq = Sequence(name="Missing", fps=30.0)
        seq.tracks[0].clips = [_make_frame_seq_clip("nonexistent")]

        output = tmp_path / "out.mp4"
        config = ExportConfig(output_path=output, fps=30.0)

        exporter = SequenceExporter()
        success = exporter.export(
            sequence=seq,
            sources={},
            clips={},
            config=config,
            frames={},
        )
        assert success is False
