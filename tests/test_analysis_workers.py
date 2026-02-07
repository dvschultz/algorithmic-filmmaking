"""Tests for analysis worker task building and skip_existing logic.

These tests validate that workers correctly:
- Skip clips that already have analysis results (skip_existing=True)
- Include all clips when skip_existing=False
- Skip clips without valid thumbnails
- Build correct frozen dataclass tasks
"""

from pathlib import Path
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
    def test_skip_existing_skips_clips_with_colors(self, thumbnail_path):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, dominant_colors=[(255, 0, 0)]
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ColorAnalysisWorker(
            [clip_with, clip_without], parallelism=1, skip_existing=True
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, thumbnail_path):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip_with = _make_clip_with_thumb(
            "c1", thumbnail_path, dominant_colors=[(255, 0, 0)]
        )
        clip_without = _make_clip_with_thumb("c2", thumbnail_path)

        worker = ColorAnalysisWorker(
            [clip_with, clip_without], parallelism=1, skip_existing=False
        )
        assert len(worker._tasks) == 2

    def test_skips_clips_without_thumbnail(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        worker = ColorAnalysisWorker([clip], parallelism=1)
        assert len(worker._tasks) == 0

    def test_skips_clips_with_nonexistent_thumbnail(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = Path("/nonexistent/thumb.jpg")

        worker = ColorAnalysisWorker([clip], parallelism=1)
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

        worker = TranscriptionWorker([], source, parallelism=100)
        assert worker._parallelism == 4  # Max is 4


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

        worker = DescriptionWorker([], parallelism=100)
        assert worker._parallelism == 5  # Max is 5

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
