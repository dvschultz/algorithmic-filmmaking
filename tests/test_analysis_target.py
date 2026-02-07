"""Tests for AnalysisTarget and worker integration with Frame objects."""

import tempfile
from pathlib import Path

import pytest

from core.analysis_target import AnalysisTarget
from models.clip import Clip, Source
from models.frame import Frame


class TestAnalysisTargetFromClip:
    """Test AnalysisTarget.from_clip factory method."""

    def test_from_clip_basic(self):
        source = Source(
            id="src-1",
            file_path=Path("/video.mp4"),
            fps=30.0,
        )
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=100,
            end_frame=200,
            thumbnail_path=Path("/thumb.jpg"),
        )
        target = AnalysisTarget.from_clip(clip, source)

        assert target.target_type == "clip"
        assert target.id == "clip-1"
        assert target.image_path == Path("/thumb.jpg")
        assert target.video_path == Path("/video.mp4")
        assert target.start_frame == 100
        assert target.end_frame == 200
        assert target.fps == 30.0

    def test_from_clip_preserves_existing_analysis(self):
        source = Source(id="src-1", file_path=Path("/video.mp4"), fps=24.0)
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            shot_type="close-up",
            dominant_colors=[(255, 0, 0)],
            description="A red ball",
        )
        target = AnalysisTarget.from_clip(clip, source)

        assert target.shot_type == "close-up"
        assert target.dominant_colors == [(255, 0, 0)]
        assert target.description == "A red ball"


class TestAnalysisTargetFromFrame:
    """Test AnalysisTarget.from_frame factory method."""

    def test_from_frame_basic(self):
        frame = Frame(
            id="frame-1",
            file_path=Path("/image.png"),
        )
        target = AnalysisTarget.from_frame(frame)

        assert target.target_type == "frame"
        assert target.id == "frame-1"
        assert target.image_path == Path("/image.png")
        assert target.video_path is None
        assert target.start_frame is None
        assert target.end_frame is None
        assert target.fps is None

    def test_from_frame_preserves_existing_analysis(self):
        frame = Frame(
            id="frame-1",
            file_path=Path("/image.png"),
            shot_type="wide shot",
            dominant_colors=[(0, 128, 255)],
            description="A sunset",
        )
        target = AnalysisTarget.from_frame(frame)

        assert target.shot_type == "wide shot"
        assert target.dominant_colors == [(0, 128, 255)]
        assert target.description == "A sunset"

    def test_from_frame_no_object_labels(self):
        """Frame model doesn't have object_labels field."""
        frame = Frame(id="frame-1", file_path=Path("/image.png"))
        target = AnalysisTarget.from_frame(frame)
        assert target.object_labels is None


class TestWorkerTaskBuilding:
    """Test that workers correctly build tasks from AnalysisTarget objects."""

    @pytest.fixture
    def image_file(self, tmp_path):
        """Create a temporary image file for testing."""
        img_path = tmp_path / "test_image.png"
        # Create a minimal valid PNG (1x1 pixel)
        import struct
        import zlib

        def create_minimal_png():
            signature = b"\x89PNG\r\n\x1a\n"
            # IHDR chunk
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT chunk
            raw = zlib.compress(b"\x00\xff\x00\x00")
            idat_crc = zlib.crc32(b"IDAT" + raw)
            idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
            # IEND chunk
            iend_crc = zlib.crc32(b"IEND")
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return signature + ihdr + idat + iend

        img_path.write_bytes(create_minimal_png())
        return img_path

    def test_color_worker_accepts_targets(self, image_file):
        from ui.workers.color_worker import ColorAnalysisWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = ColorAnalysisWorker(
            clips=[],
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"
        assert worker._tasks[0].thumbnail_path == image_file

    def test_color_worker_skips_existing(self, image_file):
        from ui.workers.color_worker import ColorAnalysisWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
            dominant_colors=[(255, 0, 0)],
        )
        worker = ColorAnalysisWorker(
            clips=[],
            analysis_targets=[target],
            skip_existing=True,
        )
        assert len(worker._tasks) == 0

    def test_color_worker_processes_existing_when_disabled(self, image_file):
        from ui.workers.color_worker import ColorAnalysisWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
            dominant_colors=[(255, 0, 0)],
        )
        worker = ColorAnalysisWorker(
            clips=[],
            analysis_targets=[target],
            skip_existing=False,
        )
        assert len(worker._tasks) == 1

    def test_shot_type_worker_accepts_targets(self, image_file):
        from ui.workers.shot_type_worker import ShotTypeWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = ShotTypeWorker(
            clips=[],
            sources_by_id={},
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"

    def test_description_worker_accepts_targets(self, image_file):
        from ui.workers.description_worker import DescriptionWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = DescriptionWorker(
            clips=[],
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"

    def test_object_detection_worker_accepts_targets(self, image_file):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = ObjectDetectionWorker(
            clips=[],
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"

    def test_cinematography_worker_accepts_targets(self, image_file):
        from ui.workers.cinematography_worker import CinematographyWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = CinematographyWorker(
            clips=[],
            sources_by_id={},
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"

    def test_classification_worker_accepts_targets(self, image_file):
        from ui.workers.classification_worker import ClassificationWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = ClassificationWorker(
            clips=[],
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "frame-1"

    def test_text_extraction_worker_accepts_targets(self, image_file):
        from ui.workers.text_extraction_worker import TextExtractionWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=image_file,
        )
        worker = TextExtractionWorker(
            clips=[],
            sources_by_id={},
            analysis_targets=[target],
        )
        assert worker._analysis_targets is not None
        assert len(worker._analysis_targets) == 1

    def test_target_skips_missing_image(self, tmp_path):
        from ui.workers.color_worker import ColorAnalysisWorker

        target = AnalysisTarget(
            target_type="frame",
            id="frame-1",
            image_path=tmp_path / "nonexistent.png",
        )
        worker = ColorAnalysisWorker(
            clips=[],
            analysis_targets=[target],
        )
        assert len(worker._tasks) == 0

    def test_clips_still_work_without_targets(self, image_file):
        """Ensure existing clip-based workflow is unaffected."""
        from ui.workers.color_worker import ColorAnalysisWorker

        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            thumbnail_path=image_file,
        )
        worker = ColorAnalysisWorker(clips=[clip])
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "clip-1"
