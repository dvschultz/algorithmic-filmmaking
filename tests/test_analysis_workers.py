"""Tests for analysis worker task building and skip_existing logic.

These tests validate that workers correctly:
- Skip clips that already have analysis results (skip_existing=True)
- Include all clips when skip_existing=False
- Skip clips without valid thumbnails
- Build correct frozen dataclass tasks
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from tests.conftest import make_test_clip
from models.clip import Source


@pytest.fixture
def source():
    return Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=30.0,
    )


@pytest.fixture
def sources_by_id(source):
    return {source.id: source}


@pytest.fixture
def thumbnail_path(tmp_path):
    """Create a real thumbnail file for testing."""
    p = tmp_path / "thumb.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0")  # Minimal JPEG header
    return p


def _make_clip_with_thumb(clip_id, thumbnail_path, source_id="src-1", **kwargs):
    """Create a clip with a real thumbnail path."""
    clip = make_test_clip(clip_id, source_id=source_id, **kwargs)
    clip.thumbnail_path = thumbnail_path
    return clip


# --- ColorAnalysisWorker ---

class TestColorWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_colors(self, source, sources_by_id, tmp_path):
        from ui.workers.color_worker import ColorAnalysisWorker

        # Create a real file so source.file_path.exists() passes
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        source.file_path = video_file

        clip_with = _make_clip_with_thumb(
            "c1", None, dominant_colors=[(255, 0, 0)]
        )
        clip_without = _make_clip_with_thumb("c2", None)

        worker = ColorAnalysisWorker(
            [clip_with, clip_without], parallelism=1, skip_existing=True,
            sources_by_id=sources_by_id,
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, source, sources_by_id, tmp_path):
        from ui.workers.color_worker import ColorAnalysisWorker

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        source.file_path = video_file

        clip_with = _make_clip_with_thumb(
            "c1", None, dominant_colors=[(255, 0, 0)]
        )
        clip_without = _make_clip_with_thumb("c2", None)

        worker = ColorAnalysisWorker(
            [clip_with, clip_without], parallelism=1, skip_existing=False,
            sources_by_id=sources_by_id,
        )
        assert len(worker._tasks) == 2

    def test_skips_clips_without_source(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip = make_test_clip("c1", source_id="missing-source")

        worker = ColorAnalysisWorker([clip], parallelism=1, sources_by_id={})
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_source(self, source, sources_by_id):
        from ui.workers.color_worker import ColorAnalysisWorker

        # source.file_path points to /test/video.mp4 which doesn't exist
        clip = make_test_clip("c1")

        worker = ColorAnalysisWorker([clip], parallelism=1, sources_by_id=sources_by_id)
        assert len(worker._tasks) == 0

    def test_empty_clips_produces_empty_tasks(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        worker = ColorAnalysisWorker([], parallelism=1)
        assert len(worker._tasks) == 0

    def test_parallelism_clamped(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        worker = ColorAnalysisWorker([], parallelism=100)
        assert worker._parallelism == 8  # Max is 8

        worker = ColorAnalysisWorker([], parallelism=0)
        assert worker._parallelism == 1  # Min is 1


# --- ShotTypeWorker ---

class TestShotTypeWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_shot_type(
        self, thumbnail_path, sources_by_id
    ):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, shot_type="wide"
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ShotTypeWorker(
            [clip_with, clip_without], sources_by_id, parallelism=1
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(
        self, thumbnail_path, sources_by_id
    ):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, shot_type="close-up"
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ShotTypeWorker(
            [clip_with, clip_without], sources_by_id, skip_existing=False
        )
        assert len(worker._tasks) == 2

    def test_task_includes_source_data(
        self, thumbnail_path, sources_by_id, source
    ):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clip = _make_clip_with_thumb(
            "c1", thumbnail_path, start_frame=100, end_frame=200
        )

        worker = ShotTypeWorker([clip], sources_by_id)
        assert len(worker._tasks) == 1
        task = worker._tasks[0]
        assert task.source_path == source.file_path
        assert task.start_frame == 100
        assert task.end_frame == 200
        assert task.fps == 30.0

    def test_skips_clips_without_thumbnail(self, sources_by_id):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        worker = ShotTypeWorker([clip], sources_by_id)
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_thumbnail(self, sources_by_id):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = Path("/nonexistent/thumb.jpg")

        worker = ShotTypeWorker([clip], sources_by_id)
        assert len(worker._tasks) == 0


class TestShotTypeWorkerErrors:
    def test_emits_aggregated_error_summary(
        self,
        monkeypatch,
        thumbnail_path,
        sources_by_id,
    ):
        from ui.workers.shot_type_worker import ShotTypeWorker

        clips = [
            _make_clip_with_thumb("clip-1", thumbnail_path),
            _make_clip_with_thumb("clip-2", thumbnail_path),
        ]
        worker = ShotTypeWorker(
            clips,
            sources_by_id,
            parallelism=1,
            skip_existing=False,
        )

        def _raise_for_all(*_args, **_kwargs):
            raise RuntimeError("torch import failed")

        monkeypatch.setattr(
            "core.analysis.shots.classify_shot_type_tiered",
            _raise_for_all,
        )
        # Patch model pre-loading so it doesn't fail on CI (no torch/GPU)
        monkeypatch.setattr(
            "core.analysis.shots.is_model_loaded",
            lambda: True,
        )

        errors = []
        completed = []
        worker.error.connect(errors.append)
        worker.analysis_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "Shot type classification failed for 2 clips" in errors[0]
        assert "clip-1" in errors[0]
        assert "clip-2" in errors[0]
        assert "torch import failed" in errors[0]


class TestColorWorkerErrors:
    def test_emits_aggregated_error_summary(self, tmp_path, source, sources_by_id):
        from ui.workers.color_worker import ColorAnalysisWorker

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        source.file_path = video_file

        clips = [make_test_clip("clip-1"), make_test_clip("clip-2")]
        worker = ColorAnalysisWorker(
            clips,
            parallelism=1,
            skip_existing=False,
            sources_by_id=sources_by_id,
        )

        with patch(
            "core.analysis.color.extract_dominant_colors",
            side_effect=RuntimeError("ffmpeg read failed"),
        ):
            errors = []
            completed = []
            worker.error.connect(errors.append)
            worker.analysis_completed.connect(lambda: completed.append(True))
            worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "Color extraction failed for 2 clips" in errors[0]
        assert "clip-1" in errors[0]
        assert "clip-2" in errors[0]
        assert "ffmpeg read failed" in errors[0]


class TestClassificationWorkerErrors:
    def test_emits_aggregated_error_summary(self, thumbnail_path):
        from ui.workers.classification_worker import ClassificationWorker

        clips = [
            _make_clip_with_thumb("clip-1", thumbnail_path),
            _make_clip_with_thumb("clip-2", thumbnail_path),
        ]
        worker = ClassificationWorker(
            clips,
            parallelism=1,
            skip_existing=False,
        )

        with patch(
            "core.analysis.classification.classify_frame",
            side_effect=RuntimeError("torch import failed"),
        ), patch(
            "core.analysis.classification._load_model",
            return_value=None,
        ):
            errors = []
            completed = []
            worker.error.connect(errors.append)
            worker.classification_completed.connect(lambda: completed.append(True))
            worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "Content classification failed for 2 clips" in errors[0]
        assert "clip-1" in errors[0]
        assert "clip-2" in errors[0]
        assert "torch import failed" in errors[0]


class TestObjectDetectionWorkerErrors:
    def test_emits_aggregated_error_summary(self, thumbnail_path):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clips = [
            _make_clip_with_thumb("clip-1", thumbnail_path),
            _make_clip_with_thumb("clip-2", thumbnail_path),
        ]
        worker = ObjectDetectionWorker(
            clips,
            parallelism=1,
            skip_existing=False,
        )

        with patch(
            "core.analysis.detection.ensure_default_detection_model_loaded",
            return_value=None,
        ), patch(
            "core.analysis.detection.detect_objects",
            side_effect=RuntimeError("yolo weights missing"),
        ):
            errors = []
            completed = []
            worker.error.connect(errors.append)
            worker.detection_completed.connect(lambda: completed.append(True))
            worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "Object detection failed for 2 clips" in errors[0]
        assert "clip-1" in errors[0]
        assert "clip-2" in errors[0]
        assert "yolo weights missing" in errors[0]

    def test_model_load_failure_emits_single_batch_error(self, thumbnail_path, monkeypatch):
        from core.errors import ModelDownloadError
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clips = [
            _make_clip_with_thumb("clip-1", thumbnail_path),
            _make_clip_with_thumb("clip-2", thumbnail_path),
        ]
        worker = ObjectDetectionWorker(
            clips,
            parallelism=1,
            skip_existing=False,
        )

        def _raise_model_load_failure():
            raise ModelDownloadError("Failed to load YOLO26n model: network down")

        def _process_should_not_run(_task):
            raise AssertionError("per-clip detection should not run after model preload fails")

        monkeypatch.setattr(
            "core.analysis.detection.ensure_default_detection_model_loaded",
            _raise_model_load_failure,
        )
        monkeypatch.setattr(worker, "_process_task", _process_should_not_run)

        errors = []
        completed = []
        worker.error.connect(errors.append)
        worker.detection_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert errors == ["Failed to load YOLO26n model: network down"]


# --- TranscriptionWorker ---

class TestTranscriptionWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_transcript(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        clip_with = make_test_clip(
            "c1", transcript_text="Hello world"
        )
        clip_without = make_test_clip("c2")

        worker = TranscriptionWorker(
            [clip_with, clip_without], source, skip_existing=True
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        clip_with = make_test_clip("c1", transcript_text="Hello")
        clip_without = make_test_clip("c2")

        worker = TranscriptionWorker(
            [clip_with, clip_without], source, skip_existing=False
        )
        assert len(worker._tasks) == 2

    def test_task_has_correct_timing(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        clip = make_test_clip("c1", start_frame=300, end_frame=600)

        worker = TranscriptionWorker([clip], source)
        assert len(worker._tasks) == 1
        task = worker._tasks[0]
        assert task.start_time == 300 / 30.0  # 10.0 seconds
        assert task.end_time == 600 / 30.0  # 20.0 seconds

    def test_parallelism_clamped(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        worker = TranscriptionWorker([], source, parallelism=100, backend="faster-whisper")
        assert worker._parallelism == 4  # Max is 4

        worker = TranscriptionWorker([], source, parallelism=0, backend="faster-whisper")
        assert worker._parallelism == 1  # Min is 1

    def test_mlx_backend_forces_serial_parallelism(self, source, monkeypatch):
        from ui.workers.transcription_worker import TranscriptionWorker

        monkeypatch.setattr(
            "ui.workers.transcription_worker.TranscriptionWorker._resolve_backend",
            staticmethod(lambda _backend: "mlx-whisper"),
        )

        worker = TranscriptionWorker([], source, parallelism=4, backend="mlx-whisper")
        assert worker._parallelism == 1

    def test_auto_backend_forces_serial_parallelism_when_mlx_selected(self, source, monkeypatch):
        from ui.workers.transcription_worker import TranscriptionWorker

        monkeypatch.setattr(
            "ui.workers.transcription_worker.TranscriptionWorker._resolve_backend",
            staticmethod(lambda _backend: "mlx-whisper"),
        )

        worker = TranscriptionWorker([], source, parallelism=4, backend="auto")
        assert worker._parallelism == 1


# --- ClassificationWorker ---

class TestClassificationWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_labels(self, thumbnail_path):
        from ui.workers.classification_worker import ClassificationWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, object_labels=["dog", "cat"]
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ClassificationWorker([clip_with, clip_without])
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, thumbnail_path):
        from ui.workers.classification_worker import ClassificationWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, object_labels=["dog"]
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ClassificationWorker(
            [clip_with, clip_without], skip_existing=False
        )
        assert len(worker._tasks) == 2

    def test_skips_clips_without_thumbnail(self):
        from ui.workers.classification_worker import ClassificationWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        worker = ClassificationWorker([clip])
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_thumbnail(self):
        from ui.workers.classification_worker import ClassificationWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = Path("/nonexistent/thumb.jpg")

        worker = ClassificationWorker([clip])
        assert len(worker._tasks) == 0


# --- ObjectDetectionWorker ---

class TestObjectDetectionWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_detections(self, thumbnail_path):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clip_with = _make_clip_with_thumb(
            "c1",
            thumbnail_path,
            detected_objects=[{"label": "person", "confidence": 0.9}],
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ObjectDetectionWorker([clip_with, clip_without])
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, thumbnail_path):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clip_with = _make_clip_with_thumb(
            "c1",
            thumbnail_path,
            detected_objects=[{"label": "car"}],
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ObjectDetectionWorker(
            [clip_with, clip_without], skip_existing=False
        )
        assert len(worker._tasks) == 2

    def test_skips_clips_without_thumbnail(self):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        worker = ObjectDetectionWorker([clip])
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_thumbnail(self):
        from ui.workers.object_detection_worker import ObjectDetectionWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = Path("/nonexistent/thumb.jpg")

        worker = ObjectDetectionWorker([clip])
        assert len(worker._tasks) == 0


# --- DescriptionWorker ---

class TestDescriptionWorkerTaskBuilding:
    def test_skip_existing_skips_clips_with_description(
        self, thumbnail_path, sources_by_id
    ):
        from ui.workers.description_worker import DescriptionWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, description="A person walking"
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = DescriptionWorker(
            [clip_with, clip_without], sources=sources_by_id
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(
        self, thumbnail_path, sources_by_id
    ):
        from ui.workers.description_worker import DescriptionWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, description="A dog"
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = DescriptionWorker(
            [clip_with, clip_without],
            sources=sources_by_id,
            skip_existing=False,
        )
        assert len(worker._tasks) == 2

    def test_task_includes_source_data(
        self, thumbnail_path, sources_by_id, source
    ):
        from ui.workers.description_worker import DescriptionWorker

        clip = _make_clip_with_thumb("c1", thumbnail_path)

        worker = DescriptionWorker([clip], sources=sources_by_id)
        assert len(worker._tasks) == 1
        task = worker._tasks[0]
        assert task.source_path == source.file_path
        assert task.fps == 30.0

    def test_parallelism_clamped(self):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker([], tier="cloud", parallelism=100)
        assert worker._parallelism == 5  # Max is 5

        worker = DescriptionWorker([], tier="cloud", parallelism=0)
        assert worker._parallelism == 1  # Min is 1

    def test_local_tier_forces_serial_parallelism(self):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker([], tier="local", parallelism=5)
        assert worker._parallelism == 1

    @patch("ui.workers.description_worker.load_settings")
    def test_default_local_setting_forces_serial_parallelism(self, mock_load_settings):
        from core.settings import Settings
        from ui.workers.description_worker import DescriptionWorker

        mock_load_settings.return_value = Settings(description_model_tier="local")

        worker = DescriptionWorker([], parallelism=5)
        assert worker._parallelism == 1

    def test_skips_clips_without_thumbnail(self, sources_by_id):
        from ui.workers.description_worker import DescriptionWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        worker = DescriptionWorker([clip], sources=sources_by_id)
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_thumbnail(self, sources_by_id):
        from ui.workers.description_worker import DescriptionWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = Path("/nonexistent/thumb.jpg")

        worker = DescriptionWorker([clip], sources=sources_by_id)
        assert len(worker._tasks) == 0


# --- Settings round-trip ---

class TestAnalysisParallelismSettings:
    def test_defaults(self):
        from core.settings import Settings

        s = Settings()
        assert s.color_analysis_parallelism == 4
        assert s.description_parallelism == 3
        assert s.transcription_parallelism == 2
        assert s.local_model_parallelism == 1

    def test_json_round_trip(self, tmp_path):
        from core.settings import Settings, _settings_to_json, _load_from_json

        s = Settings()
        s.color_analysis_parallelism = 6
        s.description_parallelism = 4
        s.transcription_parallelism = 3
        s.local_model_parallelism = 2

        data = _settings_to_json(s)
        assert data["analysis"]["color_analysis_parallelism"] == 6
        assert data["analysis"]["description_parallelism"] == 4
        assert data["analysis"]["transcription_parallelism"] == 3
        assert data["analysis"]["local_model_parallelism"] == 2

        # Round-trip through JSON load
        import json

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(data))

        loaded = _load_from_json(config_path, Settings())
        assert loaded.color_analysis_parallelism == 6
        assert loaded.description_parallelism == 4
        assert loaded.transcription_parallelism == 3
        assert loaded.local_model_parallelism == 2
