"""Tests for SRT export functionality."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.srt_export import (
    export_srt,
    SRTExportConfig,
    _format_colors_hex,
    _seconds_to_srt_time,
    _sanitize_srt_text,
)
from models.clip import Clip, Source, ExtractedText
from models.sequence import Sequence, SequenceClip, Track


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_colors_hex_basic(self):
        """Test hex formatting of RGB colors."""
        colors = [(255, 87, 51), (199, 0, 57)]
        result = _format_colors_hex(colors)
        assert result == "#FF5733, #C70039"

    def test_format_colors_hex_single(self):
        """Test hex formatting of single color."""
        colors = [(0, 0, 0)]
        result = _format_colors_hex(colors)
        assert result == "#000000"

    def test_format_colors_hex_empty(self):
        """Test hex formatting of empty list."""
        assert _format_colors_hex([]) is None
        assert _format_colors_hex(None) is None

    def test_seconds_to_srt_time_zero(self):
        """Test timecode for zero seconds."""
        assert _seconds_to_srt_time(0) == "00:00:00,000"

    def test_seconds_to_srt_time_basic(self):
        """Test timecode for basic values."""
        assert _seconds_to_srt_time(1.5) == "00:00:01,500"
        assert _seconds_to_srt_time(65.123) == "00:01:05,123"

    def test_seconds_to_srt_time_hours(self):
        """Test timecode with hours."""
        assert _seconds_to_srt_time(3661.5) == "01:01:01,500"

    def test_seconds_to_srt_time_negative(self):
        """Test timecode handles negative values."""
        assert _seconds_to_srt_time(-1) == "00:00:00,000"

    def test_sanitize_srt_text_basic(self):
        """Test basic text sanitization."""
        assert _sanitize_srt_text("Hello World") == "Hello World"

    def test_sanitize_srt_text_newlines(self):
        """Test newline handling."""
        assert _sanitize_srt_text("Line 1\nLine 2") == "Line 1\nLine 2"
        assert _sanitize_srt_text("Line 1\r\nLine 2") == "Line 1\nLine 2"

    def test_sanitize_srt_text_double_newlines(self):
        """Test double newline collapsing."""
        assert _sanitize_srt_text("Line 1\n\nLine 2") == "Line 1\nLine 2"

    def test_sanitize_srt_text_whitespace(self):
        """Test whitespace trimming."""
        assert _sanitize_srt_text("  Hello  ") == "Hello"


class TestExportSrt:
    """Tests for the main export_srt function."""

    def create_source(self, source_id: str = "src1", fps: float = 30.0) -> Source:
        """Create a test source."""
        return Source(
            id=source_id,
            file_path=Path("/test/video.mp4"),
            duration_seconds=60.0,
            fps=fps,
        )

    def create_clip(
        self,
        clip_id: str,
        source_id: str,
        start: int = 0,
        end: int = 30,
        description: str = None,
        shot_type: str = None,
        colors: list = None,
        ocr_texts: list = None,
    ) -> Clip:
        """Create a test clip with optional metadata."""
        clip = Clip(
            id=clip_id,
            source_id=source_id,
            start_frame=start,
            end_frame=end,
            description=description,
            shot_type=shot_type,
            dominant_colors=colors,
        )
        if ocr_texts:
            clip.extracted_texts = [
                ExtractedText(
                    frame_number=0,
                    text=text,
                    confidence=0.9,
                    source="tesseract",
                )
                for text in ocr_texts
            ]
        return clip

    def create_sequence_with_clips(
        self,
        clips: list[Clip],
        algorithm: str = None,
        fps: float = 30.0,
    ) -> Sequence:
        """Create a sequence with clips on the timeline."""
        sequence = Sequence(fps=fps, algorithm=algorithm)
        current_frame = 0
        for clip in clips:
            seq_clip = SequenceClip(
                source_clip_id=clip.id,
                source_id=clip.source_id,
                track_index=0,
                start_frame=current_frame,
                in_point=clip.start_frame,
                out_point=clip.end_frame,
            )
            sequence.tracks[0].add_clip(seq_clip)
            current_frame += clip.duration_frames
        return sequence

    def test_export_empty_sequence(self):
        """Test export with no clips returns False."""
        sequence = Sequence()
        config = SRTExportConfig(output_path=Path("/tmp/test.srt"))
        success, exported, skipped = export_srt(sequence, {}, {}, config)
        assert success is False
        assert exported == 0
        assert skipped == 0

    def test_export_storyteller_with_descriptions(self):
        """Test exporting Storyteller sequence with descriptions."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, description="A woman walks"),
                self.create_clip("c2", "src1", 30, 60, description="Close-up of hands"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="storyteller")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 2
            assert skipped == 0
            assert output_path.exists()

            content = output_path.read_text()
            assert "A woman walks" in content
            assert "Close-up of hands" in content
            assert "00:00:00,000 --> 00:00:01,000" in content

    def test_export_color_with_colors(self):
        """Test exporting Color sequence with dominant colors."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip(
                    "c1", "src1", 0, 30, colors=[(255, 87, 51), (199, 0, 57)]
                ),
                self.create_clip(
                    "c2", "src1", 30, 60, colors=[(0, 128, 255)]
                ),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="color")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 2
            assert skipped == 0

            content = output_path.read_text()
            assert "#FF5733, #C70039" in content
            assert "#0080FF" in content

    def test_export_shot_type(self):
        """Test exporting Shot Type sequence."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, shot_type="close-up"),
                self.create_clip("c2", "src1", 30, 60, shot_type="wide"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="shot_type")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 2

            content = output_path.read_text()
            assert "close-up" in content
            assert "wide" in content

    def test_export_exquisite_corpus(self):
        """Test exporting Exquisite Corpus with OCR text."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, ocr_texts=["CHAPTER ONE", "THE BEGINNING"]),
                self.create_clip("c2", "src1", 30, 60, ocr_texts=["FADE IN"]),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="exquisite_corpus")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 2

            content = output_path.read_text()
            # combined_text uses " | " separator
            assert "CHAPTER ONE | THE BEGINNING" in content
            assert "FADE IN" in content

    def test_export_skips_clips_without_metadata(self):
        """Test that clips without required metadata are skipped."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, description="Has description"),
                self.create_clip("c2", "src1", 30, 60),  # No description
                self.create_clip("c3", "src1", 60, 90, description="Also has description"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="storyteller")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 2
            assert skipped == 1

            content = output_path.read_text()
            assert "Has description" in content
            assert "Also has description" in content
            # Entry numbers should be consecutive (1, 2), not (1, 3)
            assert "\n1\n" in content
            assert "\n2\n" in content
            assert "\n3\n" not in content

    def test_export_fallback_to_description(self):
        """Test that unknown algorithm falls back to description."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, description="Fallback text"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm=None)
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert exported == 1

            content = output_path.read_text()
            assert "Fallback text" in content

    def test_export_adds_srt_extension(self):
        """Test that .srt extension is added if missing."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, description="Test"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="storyteller")
            output_path = Path(tmpdir) / "test"  # No extension
            config = SRTExportConfig(output_path=output_path)

            success, exported, skipped = export_srt(
                sequence, clips_by_id, sources_by_id, config
            )

            assert success is True
            assert (Path(tmpdir) / "test.srt").exists()

    def test_srt_format_validation(self):
        """Test that exported SRT follows standard format."""
        with TemporaryDirectory() as tmpdir:
            source = self.create_source()
            clips = [
                self.create_clip("c1", "src1", 0, 30, description="First clip"),
                self.create_clip("c2", "src1", 30, 60, description="Second clip"),
            ]
            clips_by_id = {c.id: c for c in clips}
            sources_by_id = {source.id: source}

            sequence = self.create_sequence_with_clips(clips, algorithm="storyteller")
            output_path = Path(tmpdir) / "test.srt"
            config = SRTExportConfig(output_path=output_path)

            export_srt(sequence, clips_by_id, sources_by_id, config)

            content = output_path.read_text()
            lines = content.strip().split("\n")

            # First entry
            assert lines[0] == "1"
            assert " --> " in lines[1]
            assert "First clip" in lines[2]

            # Blank line between entries
            assert lines[3] == ""

            # Second entry
            assert lines[4] == "2"


class TestSequenceModelAlgorithm:
    """Tests for Sequence.algorithm field persistence."""

    def test_algorithm_serialization(self):
        """Test algorithm field is serialized."""
        sequence = Sequence(algorithm="storyteller")
        data = sequence.to_dict()
        assert data.get("algorithm") == "storyteller"

    def test_algorithm_deserialization(self):
        """Test algorithm field is deserialized."""
        data = {
            "id": "test-id",
            "name": "Test Sequence",
            "fps": 30.0,
            "tracks": [],
            "algorithm": "color",
        }
        sequence = Sequence.from_dict(data)
        assert sequence.algorithm == "color"

    def test_algorithm_none_not_serialized(self):
        """Test None algorithm is not included in serialization."""
        sequence = Sequence(algorithm=None)
        data = sequence.to_dict()
        assert "algorithm" not in data

    def test_algorithm_missing_from_dict(self):
        """Test missing algorithm defaults to None."""
        data = {
            "id": "test-id",
            "name": "Test Sequence",
            "fps": 30.0,
            "tracks": [],
        }
        sequence = Sequence.from_dict(data)
        assert sequence.algorithm is None
