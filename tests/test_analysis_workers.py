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
        with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
            worker.run()
        assert [(o.target_id, o.status) for o in worker.result.outcomes] == [
            ("c1", "skipped"), ("c2", "succeeded"),
        ]

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
        with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
            worker.run()
        assert [o.status for o in worker.result.outcomes] == ["succeeded", "succeeded"]

    def test_reports_clips_without_source(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        clip = make_test_clip("c1", source_id="missing-source")

        worker = ColorAnalysisWorker([clip], parallelism=1, sources_by_id={})
        worker.run()
        assert worker.result.outcomes[0].code == "source_file_missing"

    def test_reports_clips_with_nonexistent_source(self, source, sources_by_id):
        from ui.workers.color_worker import ColorAnalysisWorker

        # source.file_path points to /test/video.mp4 which doesn't exist
        clip = make_test_clip("c1")

        worker = ColorAnalysisWorker([clip], parallelism=1, sources_by_id=sources_by_id)
        worker.run()
        assert worker.result.outcomes[0].code == "source_file_missing"

    def test_empty_clips_produces_empty_tasks(self):
        from ui.workers.color_worker import ColorAnalysisWorker

        worker = ColorAnalysisWorker([], parallelism=1)
        worker.run()
        assert worker.result.outcomes == ()

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

        def _raise_model_load_failure(*args, **kwargs):
            raise ModelDownloadError("Failed to load YOLO26n model: network down")

        monkeypatch.setattr(
            "core.analysis.detection.detect_objects",
            _raise_model_load_failure,
        )

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
            [clip_with, clip_without],
            source,
            skip_existing=True,
            backend="faster-whisper",
        )
        assert len(worker._tasks) == 1
        assert worker._tasks[0].clip_id == "c2"

    def test_skip_existing_false_includes_all(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        clip_with = make_test_clip("c1", transcript_text="Hello")
        clip_without = make_test_clip("c2")

        worker = TranscriptionWorker(
            [clip_with, clip_without],
            source,
            skip_existing=False,
            backend="faster-whisper",
        )
        assert len(worker._tasks) == 2

    def test_task_has_correct_timing(self, source):
        from ui.workers.transcription_worker import TranscriptionWorker

        clip = make_test_clip("c1", start_frame=300, end_frame=600)

        worker = TranscriptionWorker([clip], source, backend="faster-whisper")
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


class TestTranscriptionWorkerErrors:
    def test_missing_ffmpeg_emits_single_batch_error(self, source, monkeypatch):
        from ui.workers.transcription_worker import TranscriptionWorker

        clips = [make_test_clip("clip-1"), make_test_clip("clip-2")]
        worker = TranscriptionWorker(
            clips,
            source,
            backend="faster-whisper",
            skip_existing=False,
        )

        monkeypatch.setattr("core.binary_resolver.find_binary", lambda _name: None)
        monkeypatch.setattr(
            "core.transcription.get_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model should not load when ffmpeg is missing")
            ),
        )

        errors = []
        completed = []
        worker.error.connect(errors.append)
        worker.transcription_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "FFmpeg is required for transcription" in errors[0]

    def test_emits_aggregated_error_summary(self, source, monkeypatch):
        from ui.workers.transcription_worker import TranscriptionWorker

        clips = [make_test_clip("clip-1"), make_test_clip("clip-2")]
        worker = TranscriptionWorker(
            clips,
            source,
            parallelism=1,
            backend="faster-whisper",
            skip_existing=False,
        )

        monkeypatch.setattr("core.binary_resolver.find_binary", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("core.transcription.get_model", lambda *_args, **_kwargs: object())
        from core.operations.transcription import TranscriptionOutcome
        monkeypatch.setattr(
            "core.operations.transcription.compute_task",
            lambda task, options: TranscriptionOutcome(task.clip_id, "failed", message="audio extraction failed"),
        )

        errors = []
        completed = []
        worker.error.connect(errors.append)
        worker.transcription_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert len(errors) == 1
        assert "Transcription failed for 2 clips" in errors[0]
        assert "clip-1" in errors[0]
        assert "clip-2" in errors[0]
        assert "audio extraction failed" in errors[0]

    def test_low_disk_space_aborts_before_model_load(self, source, monkeypatch, tmp_path):
        from ui.workers.transcription_worker import TranscriptionWorker

        clips = [make_test_clip("clip-1")]
        worker = TranscriptionWorker(
            clips,
            source,
            backend="faster-whisper",
            skip_existing=False,
            model_cache_dir=tmp_path,
            min_free_disk_gb=3.0,
        )

        monkeypatch.setattr("core.binary_resolver.find_binary", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            "core.transcription_storage.validate_transcription_disk_space",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("low disk")),
        )
        monkeypatch.setattr(
            "core.transcription.get_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model should not load after disk preflight failure")
            ),
        )

        errors = []
        completed = []
        worker.error.connect(errors.append)
        worker.transcription_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert errors == ["low disk"]

    def test_status_signal_reports_model_load_and_elapsed_time(self, source, monkeypatch, tmp_path):
        from ui.workers.transcription_worker import TranscriptionWorker

        clips = [make_test_clip("clip-1")]
        worker = TranscriptionWorker(
            clips,
            source,
            backend="faster-whisper",
            skip_existing=False,
            model_cache_dir=tmp_path,
            min_free_disk_gb=0.5,
        )

        monkeypatch.setattr("core.binary_resolver.find_binary", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("core.transcription_storage.validate_transcription_disk_space", lambda *_args, **_kwargs: None)
        monkeypatch.setattr("core.transcription.get_model", lambda *_args, **_kwargs: object())
        from core.operations.transcription import TranscriptionOutcome
        monkeypatch.setattr("core.operations.transcription.compute_task", lambda task, options: TranscriptionOutcome(task.clip_id, "succeeded"))

        statuses = []
        completed = []
        worker.status.connect(statuses.append)
        worker.transcription_completed.connect(lambda: completed.append(True))

        worker.run()

        assert completed == [True]
        assert any("checking disk space" in message for message in statuses)
        assert any("loading faster-whisper model" in message for message in statuses)
        assert any("completed in" in message for message in statuses)


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


class TestDescriptionWorkerLifecycle:
    @pytest.mark.parametrize("tier", ["local", "cpu", "gpu"])
    def test_preload_failure_reports_each_target_and_completes(
        self, monkeypatch, thumbnail_path, tier
    ):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker(
            [_make_clip_with_thumb(cid, thumbnail_path) for cid in ("clip-1", "clip-2")],
            tier=tier,
        )
        monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)

        def fail(*_):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr("core.analysis.description._load_local_model", fail)
        errors, completed = [], []
        worker.error.connect(lambda cid, message: errors.append((cid, message)))
        worker.description_completed.connect(lambda: completed.append(True))
        worker.run()
        assert errors == [
            (cid, "Failed to load local VLM: model unavailable")
            for cid in ("clip-1", "clip-2")
        ]
        assert worker.error_count == 2
        assert worker.last_error == errors[0][1]
        assert completed == [True]

    @pytest.mark.parametrize("cancel_during_load", [False, True])
    def test_cancel_completes_without_inference(
        self, monkeypatch, thumbnail_path, cancel_during_load
    ):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker(
            [_make_clip_with_thumb("clip-1", thumbnail_path)], tier="local"
        )
        loaded, completed = [], []

        def load(*_):
            loaded.append(True)
            worker.cancel()

        monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)
        monkeypatch.setattr("core.analysis.description._load_local_model", load)
        monkeypatch.setattr("core.analysis.description.describe_frame", lambda *a, **kw: pytest.fail("inference"))
        worker.description_completed.connect(lambda: completed.append(True))
        if not cancel_during_load:
            worker.cancel()
        worker.run()
        assert loaded == ([True] if cancel_during_load else [])
        assert completed == [True]

    def test_cloud_override_does_not_preload_local_settings(
        self, monkeypatch, thumbnail_path
    ):
        from types import SimpleNamespace
        from ui.workers.description_worker import DescriptionWorker

        monkeypatch.setattr(
            "ui.workers.description_worker.load_settings",
            lambda: SimpleNamespace(description_model_tier="local"),
        )
        worker = DescriptionWorker(
            [_make_clip_with_thumb("clip-1", thumbnail_path)], tier="cloud"
        )
        monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)
        monkeypatch.setattr(
            "core.analysis.description._load_local_model",
            lambda *_: pytest.fail("cloud must not load local model"),
        )
        monkeypatch.setattr(
            "core.analysis.description.describe_frame", lambda *a, **kw: ("A frame", "cloud")
        )
        worker.run()
        assert worker.success_count == 1

    def test_empty_batch_completes_once(self):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker([], tier="local")
        completed = []
        worker.description_completed.connect(lambda: completed.append(True))
        worker.run()
        assert completed == [True]

    def test_cancel_during_failed_preload_does_not_report_errors(
        self, monkeypatch, thumbnail_path
    ):
        from ui.workers.description_worker import DescriptionWorker

        worker = DescriptionWorker(
            [_make_clip_with_thumb("clip-1", thumbnail_path)], tier="local"
        )

        def load(*_):
            worker.cancel()
            raise RuntimeError("interrupted")

        monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)
        monkeypatch.setattr("core.analysis.description._load_local_model", load)
        errors, completed = [], []
        worker.error.connect(lambda *args: errors.append(args))
        worker.description_completed.connect(lambda: completed.append(True))
        worker.run()
        assert errors == []
        assert worker.error_count == 0
        assert completed == [True]


class TestDescriptionWorkerRetries:
    def test_retries_transient_provider_500(
        self,
        monkeypatch,
        thumbnail_path,
        sources_by_id,
    ):
        from ui.workers.description_worker import DescriptionWorker

        clip = _make_clip_with_thumb("clip-1", thumbnail_path)
        worker = DescriptionWorker(
            [clip],
            sources=sources_by_id,
            tier="cloud",
            skip_existing=False,
        )

        attempts = []
        sleeps = []

        def _describe_frame(*_args, **_kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError(
                    "Video description failed (gemini-3.1-flash-lite-preview): "
                    "litellm.InternalServerError: 500 Internal error encountered."
                )
            return "Recovered description", "gemini-3.1-flash-lite-preview (video)"

        monkeypatch.setattr("core.analysis.description.describe_frame", _describe_frame)
        monkeypatch.setattr(
            worker._cancel_event, "wait",
            lambda delay: sleeps.append(delay),
        )

        clip_id, description, model, error = worker._process_task(worker._tasks[0])

        assert clip_id == "clip-1"
        assert description == "Recovered description"
        assert model == "gemini-3.1-flash-lite-preview (video)"
        assert error is None
        assert len(attempts) == 2
        assert sleeps == [2]

    def test_does_not_retry_auth_errors(
        self,
        monkeypatch,
        thumbnail_path,
        sources_by_id,
    ):
        from ui.workers.description_worker import DescriptionWorker

        clip = _make_clip_with_thumb("clip-1", thumbnail_path)
        worker = DescriptionWorker(
            [clip],
            sources=sources_by_id,
            tier="cloud",
            skip_existing=False,
        )

        attempts = []

        def _describe_frame(*_args, **_kwargs):
            attempts.append(True)
            raise RuntimeError("Video description failed: authentication failed")

        monkeypatch.setattr("core.analysis.description.describe_frame", _describe_frame)
        monkeypatch.setattr(
            worker._cancel_event, "wait",
            lambda _delay: (_ for _ in ()).throw(
                AssertionError("auth errors should not sleep for retry")
            ),
        )

        clip_id, description, model, error = worker._process_task(worker._tasks[0])

        assert clip_id == "clip-1"
        assert description is None
        assert model is None
        assert "authentication failed" in error
        assert len(attempts) == 1


# --- CustomQueryWorker ---

class TestCustomQueryWorkerTaskBuilding:
    def test_parallelism_clamped(self):
        from ui.workers.custom_query_worker import CustomQueryWorker

        worker = CustomQueryWorker([], "person", {}, tier="cloud", parallelism=100)
        assert worker._parallelism == 5  # Max is 5

        worker = CustomQueryWorker([], "person", {}, tier="cloud", parallelism=0)
        assert worker._parallelism == 1  # Min is 1

    def test_local_tier_forces_serial_parallelism(self):
        from ui.workers.custom_query_worker import CustomQueryWorker

        worker = CustomQueryWorker([], "person", {}, tier="local", parallelism=5)
        assert worker._parallelism == 1

    @patch("ui.workers.custom_query_worker.load_settings")
    def test_default_local_setting_forces_serial_parallelism(self, mock_load_settings):
        from core.settings import Settings
        from ui.workers.custom_query_worker import CustomQueryWorker

        mock_load_settings.return_value = Settings(description_model_tier="local")

        worker = CustomQueryWorker([], "person", {}, parallelism=5)
        assert worker._parallelism == 1

    def test_local_preload_and_inference_share_runtime_thread(
        self, monkeypatch, thumbnail_path
    ):
        import threading

        from ui.workers.custom_query_worker import CustomQueryWorker

        clip = _make_clip_with_thumb("clip-1", thumbnail_path)
        worker = CustomQueryWorker(
            [clip],
            "person",
            {},
            tier="local",
            parallelism=5,
            skip_existing=False,
        )
        caller_thread_id = threading.get_ident()
        evaluation_thread_ids = []
        preload_thread_ids = []
        results = []

        monkeypatch.setattr(
            "core.analysis.description.is_model_loaded",
            lambda *_: False,
        )
        monkeypatch.setattr(
            "core.analysis.description._load_local_model",
            lambda *_: preload_thread_ids.append(threading.get_ident()),
        )

        def _evaluate_custom_query(*_args, **_kwargs):
            evaluation_thread_ids.append(threading.get_ident())
            return True, 0.9, "local-test-model"

        monkeypatch.setattr(
            "core.analysis.custom_query.evaluate_custom_query",
            _evaluate_custom_query,
        )
        worker.query_result_ready.connect(
            lambda clip_id, query, match, confidence, model: results.append(
                (clip_id, query, match, confidence, model)
            )
        )

        worker.run()

        assert len(evaluation_thread_ids) == 1
        assert evaluation_thread_ids == preload_thread_ids
        assert evaluation_thread_ids[0] != caller_thread_id
        assert results == [
            ("clip-1", "person", True, 0.9, "local-test-model")
        ]


# --- Settings round-trip ---

class TestAnalysisParallelismSettings:
    def test_defaults(self):
        from core.settings import Settings

        s = Settings()
        assert s.color_analysis_parallelism == 4
        assert s.description_parallelism == 3
        assert s.transcription_parallelism == 2
        assert s.transcription_min_free_disk_gb == 3.0
        assert s.transcription_segmentation_mode == "backend"
        assert s.transcription_segment_max_seconds == 12.0
        assert s.local_model_parallelism == 1

    def test_json_round_trip(self, tmp_path):
        from core.settings import Settings, _settings_to_json, _load_from_json

        s = Settings()
        s.color_analysis_parallelism = 6
        s.description_parallelism = 4
        s.transcription_parallelism = 3
        s.transcription_min_free_disk_gb = 6.5
        s.transcription_segmentation_mode = "fixed"
        s.transcription_segment_max_seconds = 8.0
        s.local_model_parallelism = 2

        data = _settings_to_json(s)
        assert data["analysis"]["color_analysis_parallelism"] == 6
        assert data["analysis"]["description_parallelism"] == 4
        assert data["analysis"]["transcription_parallelism"] == 3
        assert data["transcription"]["min_free_disk_gb"] == 6.5
        assert data["transcription"]["segmentation_mode"] == "fixed"
        assert data["transcription"]["segment_max_seconds"] == 8.0
        assert data["analysis"]["local_model_parallelism"] == 2

        # Round-trip through JSON load
        import json

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(data))

        loaded = _load_from_json(config_path, Settings())
        assert loaded.color_analysis_parallelism == 6
        assert loaded.description_parallelism == 4
        assert loaded.transcription_parallelism == 3
        assert loaded.transcription_min_free_disk_gb == 6.5
        assert loaded.transcription_segmentation_mode == "fixed"
        assert loaded.transcription_segment_max_seconds == 8.0
        assert loaded.local_model_parallelism == 2
