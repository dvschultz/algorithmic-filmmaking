"""Tests for gaze analysis background worker.

Tests validate worker behavior without requiring mediapipe.
All core.analysis.gaze functions are mocked.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import make_test_clip
from models.clip import Source


@pytest.fixture
def source(tmp_path):
    """Create a source with a real file path."""
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"\x00" * 100)
    return Source(
        id="src-1",
        file_path=video_file,
        duration_seconds=60.0,
        fps=30.0,
    )


@pytest.fixture
def sources_by_id(source):
    return {source.id: source}


class TestGazeWorkerProcessing:
    """Test that the worker processes clips and sets gaze fields."""

    def test_processes_clips_and_sets_gaze_fields(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip = make_test_clip("c1", start_frame=0, end_frame=90)

        mock_result = {
            "gaze_yaw": 5.2,
            "gaze_pitch": -3.1,
            "gaze_category": "at_camera",
        }

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=mock_result), \
             patch("core.analysis.gaze.unload_model"):

            gaze_results = []
            completed = []

            worker = GazeAnalysisWorker(
                [clip], sources_by_id, sample_interval=1.0, skip_existing=True
            )
            worker.gaze_ready.connect(
                lambda cid, yaw, pitch, cat: gaze_results.append((cid, yaw, pitch, cat))
            )
            worker.detection_completed.connect(lambda: completed.append(True))

            worker.run()

        # Clip fields mutated in-place
        assert clip.gaze_yaw == 5.2
        assert clip.gaze_pitch == -3.1
        assert clip.gaze_category == "at_camera"

        # Signal emitted
        assert len(gaze_results) == 1
        assert gaze_results[0] == ("c1", 5.2, -3.1, "at_camera")

        # Completion emitted
        assert completed == [True]

    def test_no_face_detected_skips_emission(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip = make_test_clip("c1", start_frame=0, end_frame=90)

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=None), \
             patch("core.analysis.gaze.unload_model"):

            gaze_results = []
            completed = []

            worker = GazeAnalysisWorker([clip], sources_by_id)
            worker.gaze_ready.connect(
                lambda cid, yaw, pitch, cat: gaze_results.append(cid)
            )
            worker.detection_completed.connect(lambda: completed.append(True))

            worker.run()

        # No gaze signal emitted for clips with no face
        assert gaze_results == []
        # Clip fields remain None
        assert clip.gaze_yaw is None
        assert clip.gaze_pitch is None
        assert clip.gaze_category is None
        # Completion still emitted
        assert completed == [True]


class TestGazeWorkerProgress:
    """Test that progress signals emit correct counts."""

    def test_progress_signals_correct_counts(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clips = [
            make_test_clip("c1", start_frame=0, end_frame=90),
            make_test_clip("c2", start_frame=90, end_frame=180),
            make_test_clip("c3", start_frame=180, end_frame=270),
        ]

        mock_result = {
            "gaze_yaw": 0.0,
            "gaze_pitch": 0.0,
            "gaze_category": "at_camera",
        }

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=mock_result), \
             patch("core.analysis.gaze.unload_model"):

            progress_updates = []
            worker = GazeAnalysisWorker(clips, sources_by_id)
            worker.progress.connect(
                lambda current, total: progress_updates.append((current, total))
            )

            worker.run()

        assert progress_updates == [(1, 3), (2, 3), (3, 3)]


class TestGazeWorkerSkipExisting:
    """Test skip_existing behavior."""

    def test_skip_existing_true_skips_clips_with_gaze_category(
        self, source, sources_by_id
    ):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip_with = make_test_clip("c1", start_frame=0, end_frame=90)
        clip_with.gaze_yaw = 5.0
        clip_with.gaze_pitch = -2.0
        clip_with.gaze_category = "at_camera"

        clip_without = make_test_clip("c2", start_frame=90, end_frame=180)

        mock_result = {
            "gaze_yaw": 15.0,
            "gaze_pitch": 1.0,
            "gaze_category": "looking_right",
        }

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=mock_result) as mock_extract, \
             patch("core.analysis.gaze.unload_model"):

            worker = GazeAnalysisWorker(
                [clip_with, clip_without], sources_by_id, skip_existing=True
            )
            worker.run()

        # Only called once (for clip_without)
        assert mock_extract.call_count == 1

    def test_skip_existing_false_processes_all(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip_with = make_test_clip("c1", start_frame=0, end_frame=90)
        clip_with.gaze_category = "at_camera"

        clip_without = make_test_clip("c2", start_frame=90, end_frame=180)

        mock_result = {
            "gaze_yaw": 0.0,
            "gaze_pitch": 0.0,
            "gaze_category": "at_camera",
        }

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=mock_result) as mock_extract, \
             patch("core.analysis.gaze.unload_model"):

            worker = GazeAnalysisWorker(
                [clip_with, clip_without], sources_by_id, skip_existing=False
            )
            worker.run()

        # Both clips processed
        assert mock_extract.call_count == 2


class TestGazeWorkerAllAlreadyAnalyzed:
    """Test that all-analyzed clips emit detection_completed immediately."""

    def test_all_analyzed_emits_completed_immediately(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip1 = make_test_clip("c1")
        clip1.gaze_category = "at_camera"
        clip2 = make_test_clip("c2")
        clip2.gaze_category = "looking_left"

        completed = []
        progress_updates = []

        worker = GazeAnalysisWorker(
            [clip1, clip2], sources_by_id, skip_existing=True
        )
        worker.detection_completed.connect(lambda: completed.append(True))
        worker.progress.connect(
            lambda c, t: progress_updates.append((c, t))
        )

        worker.run()

        assert completed == [True]
        # No progress emitted since nothing was processed
        assert progress_updates == []


class TestGazeWorkerMissingSource:
    """Test that missing sources are handled gracefully."""

    def test_missing_source_logs_warning_and_continues(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip_missing_source = make_test_clip(
            "c1", source_id="nonexistent", start_frame=0, end_frame=90
        )
        clip_valid = make_test_clip("c2", start_frame=0, end_frame=90)

        mock_result = {
            "gaze_yaw": 2.0,
            "gaze_pitch": -1.0,
            "gaze_category": "at_camera",
        }

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", return_value=mock_result) as mock_extract, \
             patch("core.analysis.gaze.unload_model"):

            gaze_results = []
            completed = []

            worker = GazeAnalysisWorker(
                [clip_missing_source, clip_valid], sources_by_id
            )
            worker.gaze_ready.connect(
                lambda cid, yaw, pitch, cat: gaze_results.append(cid)
            )
            worker.detection_completed.connect(lambda: completed.append(True))

            worker.run()

        # Only the valid clip was processed
        assert mock_extract.call_count == 1
        assert gaze_results == ["c2"]
        assert completed == [True]

    def test_nonexistent_source_file_skips_clip(self):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        source_bad_path = Source(
            id="src-bad",
            file_path=Path("/nonexistent/video.mp4"),
            duration_seconds=60.0,
            fps=30.0,
        )
        sources = {source_bad_path.id: source_bad_path}

        clip = make_test_clip("c1", source_id="src-bad")

        completed = []
        worker = GazeAnalysisWorker([clip], sources)
        worker.detection_completed.connect(lambda: completed.append(True))

        worker.run()

        # Completed without processing (no valid clips)
        assert completed == [True]
        assert clip.gaze_category is None


class TestGazeWorkerExceptionHandling:
    """Test that per-clip exceptions are caught and logged."""

    def test_per_clip_exception_continues_processing(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip1 = make_test_clip("c1", start_frame=0, end_frame=90)
        clip2 = make_test_clip("c2", start_frame=90, end_frame=180)

        mock_result = {
            "gaze_yaw": 10.0,
            "gaze_pitch": 0.0,
            "gaze_category": "looking_right",
        }

        call_count = 0

        def extract_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("decode error")
            return mock_result

        with patch("core.analysis.gaze.is_model_loaded", return_value=True), \
             patch("core.analysis.gaze.load_face_mesh"), \
             patch("core.analysis.gaze.extract_gaze_from_clip", side_effect=extract_side_effect), \
             patch("core.analysis.gaze.unload_model"):

            gaze_results = []
            progress_updates = []

            worker = GazeAnalysisWorker([clip1, clip2], sources_by_id)
            worker.gaze_ready.connect(
                lambda cid, yaw, pitch, cat: gaze_results.append(cid)
            )
            worker.progress.connect(
                lambda c, t: progress_updates.append((c, t))
            )

            worker.run()

        # First clip failed, second succeeded
        assert gaze_results == ["c2"]
        # Progress emitted for both
        assert progress_updates == [(1, 2), (2, 2)]
        # First clip fields unchanged
        assert clip1.gaze_category is None
        # Second clip fields set
        assert clip2.gaze_category == "looking_right"

    def test_model_load_failure_emits_error(self, source, sources_by_id):
        from ui.workers.gaze_worker import GazeAnalysisWorker

        clip = make_test_clip("c1")

        with patch("core.analysis.gaze.is_model_loaded", return_value=False), \
             patch("core.analysis.gaze.load_face_mesh", side_effect=RuntimeError("mediapipe not installed")), \
             patch("core.analysis.gaze.unload_model"):

            errors = []
            completed = []

            worker = GazeAnalysisWorker([clip], sources_by_id)
            worker.error.connect(errors.append)
            worker.detection_completed.connect(lambda: completed.append(True))

            worker.run()

        assert len(errors) == 1
        assert "Failed to load gaze detection model" in errors[0]
        assert "mediapipe not installed" in errors[0]
        # detection_completed still emits so pipeline never stalls
        assert completed == [True]
