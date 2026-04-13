"""Tests for EmbeddingAnalysisWorker.

Following the convention from tests/test_gaze_worker.py: call worker.run()
directly (synchronously) rather than start() to avoid QThread scheduling
complications in the test environment.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_test_clip


@pytest.fixture
def source(tmp_path):
    """A Source with a real file path (used only for the worker's ignored sources_by_id param)."""
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"\x00" * 100)
    from models.clip import Source

    return Source(
        id="src-1",
        file_path=video_file,
        duration_seconds=60.0,
        fps=30.0,
    )


@pytest.fixture
def sources_by_id(source):
    return {source.id: source}


@pytest.fixture
def clip_with_thumb(tmp_path):
    """A clip with a real (but empty) thumbnail file on disk."""
    thumb = tmp_path / "thumb-c1.jpg"
    thumb.write_bytes(b"\x00" * 10)
    clip = make_test_clip("c1")
    clip.thumbnail_path = thumb
    return clip


class TestEmbeddingWorkerFiltering:
    def test_skips_clips_already_having_embeddings(self, qapp_fixture, sources_by_id, tmp_path):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"\x00")
        clip1 = make_test_clip("c1")
        clip1.thumbnail_path = thumb
        clip1.embedding = [0.5] * 768  # already analyzed

        clip2 = make_test_clip("c2")
        clip2.thumbnail_path = thumb

        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            return_value=[[0.1] * 768],
        ) as mock_extract, patch("core.analysis.embeddings.unload_model"):
            worker = EmbeddingAnalysisWorker(
                [clip1, clip2], sources_by_id=sources_by_id, skip_existing=True
            )
            worker.run()

        # Only clip2 was processed
        mock_extract.assert_called_once()
        passed_paths = mock_extract.call_args[0][0]
        assert len(passed_paths) == 1
        # clip2 got a new embedding; clip1 kept its original
        assert clip1.embedding == [0.5] * 768
        assert clip2.embedding == [0.1] * 768

    def test_skips_clips_without_thumbnails(self, qapp_fixture, sources_by_id):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        clip = make_test_clip("c1")
        clip.thumbnail_path = None

        completions = []
        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch"
        ) as mock_extract, patch("core.analysis.embeddings.unload_model"):
            worker = EmbeddingAnalysisWorker([clip], sources_by_id=sources_by_id)
            worker.analysis_completed.connect(lambda: completions.append(True))
            worker.run()

        # No clips to process → extract never called, completion still emits
        mock_extract.assert_not_called()
        assert completions == [True]

    def test_empty_clip_list_emits_completion_and_returns(self, qapp_fixture):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        completions = []
        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch"
        ) as mock_extract, patch("core.analysis.embeddings.unload_model") as mock_unload:
            worker = EmbeddingAnalysisWorker([])
            worker.analysis_completed.connect(lambda: completions.append(True))
            worker.run()

        # Early return — neither extract nor unload called
        mock_extract.assert_not_called()
        mock_unload.assert_not_called()
        assert completions == [True]


class TestEmbeddingWorkerHappyPath:
    def test_processes_clips_and_sets_embedding_fields(
        self, qapp_fixture, clip_with_thumb, sources_by_id
    ):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        mock_vector = [0.1] * 768
        ready_ids: list[str] = []
        progress_events: list[tuple[int, int]] = []
        completions: list[bool] = []

        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            return_value=[mock_vector],
        ), patch("core.analysis.embeddings.unload_model") as mock_unload:
            worker = EmbeddingAnalysisWorker(
                [clip_with_thumb], sources_by_id=sources_by_id
            )
            worker.embedding_ready.connect(lambda cid: ready_ids.append(cid))
            worker.progress.connect(lambda c, t: progress_events.append((c, t)))
            worker.analysis_completed.connect(lambda: completions.append(True))
            worker.run()

        # Clip fields mutated in place
        assert clip_with_thumb.embedding == mock_vector
        assert clip_with_thumb.embedding_model == "dinov2-vit-b-14"

        # Signals emitted
        assert ready_ids == ["c1"]
        assert progress_events == [(1, 1)]
        assert completions == [True]

        # Always-unload pattern — unload is always called in finally
        mock_unload.assert_called_once()

    def test_chunking_emits_progress_between_chunks(
        self, qapp_fixture, sources_by_id, tmp_path
    ):
        """With chunk_size=2 and 4 clips, progress fires twice (after each chunk)."""
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"\x00")
        clips = []
        for i in range(4):
            c = make_test_clip(f"c{i}")
            c.thumbnail_path = thumb
            clips.append(c)

        progress_events = []
        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            side_effect=lambda paths: [[0.1] * 768 for _ in paths],
        ), patch("core.analysis.embeddings.unload_model"):
            worker = EmbeddingAnalysisWorker(
                clips, sources_by_id=sources_by_id, chunk_size=2
            )
            worker.progress.connect(lambda c, t: progress_events.append((c, t)))
            worker.run()

        # 4 clips, chunk_size=2 → 2 chunks, 2 progress events (2/4 and 4/4)
        assert progress_events == [(2, 4), (4, 4)]
        # All clips got embeddings
        for clip in clips:
            assert clip.embedding == [0.1] * 768


class TestEmbeddingWorkerCancellation:
    def test_cancellation_between_chunks_stops_processing(
        self, qapp_fixture, sources_by_id, tmp_path
    ):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"\x00")
        clips = []
        for i in range(4):
            c = make_test_clip(f"c{i}")
            c.thumbnail_path = thumb
            clips.append(c)

        call_count = {"n": 0}

        def cancel_after_first_chunk(paths):
            call_count["n"] += 1
            if call_count["n"] == 1:
                worker.cancel()
            return [[0.1] * 768 for _ in paths]

        completions: list[bool] = []
        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            side_effect=cancel_after_first_chunk,
        ), patch("core.analysis.embeddings.unload_model"):
            worker = EmbeddingAnalysisWorker(
                clips, sources_by_id=sources_by_id, chunk_size=2
            )
            worker.analysis_completed.connect(lambda: completions.append(True))
            worker.run()

        # First chunk's two clips got embeddings
        assert clips[0].embedding == [0.1] * 768
        assert clips[1].embedding == [0.1] * 768
        # Second chunk never ran
        assert clips[2].embedding is None
        assert clips[3].embedding is None
        # Completion signal still fires so the pipeline advances
        assert completions == [True]


class TestEmbeddingWorkerErrorPath:
    def test_chunk_exception_emits_error_breaks_loop_preserves_prior_chunks(
        self, qapp_fixture, sources_by_id, tmp_path
    ):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"\x00")
        clips = []
        for i in range(4):
            c = make_test_clip(f"c{i}")
            c.thumbnail_path = thumb
            clips.append(c)

        call_count = {"n": 0}

        def fail_on_second_chunk(paths):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Simulated OOM")
            return [[0.1] * 768 for _ in paths]

        errors: list[str] = []
        completions: list[bool] = []
        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            side_effect=fail_on_second_chunk,
        ), patch("core.analysis.embeddings.unload_model") as mock_unload:
            worker = EmbeddingAnalysisWorker(
                clips, sources_by_id=sources_by_id, chunk_size=2
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.analysis_completed.connect(lambda: completions.append(True))
            worker.run()

        # First chunk succeeded
        assert clips[0].embedding == [0.1] * 768
        assert clips[1].embedding == [0.1] * 768
        # Second chunk failed — no embeddings written
        assert clips[2].embedding is None
        assert clips[3].embedding is None
        # Error signaled exactly once
        assert len(errors) == 1
        assert "Simulated OOM" in errors[0]
        # Completion still emits so the pipeline advances
        assert completions == [True]
        # Model is still unloaded in finally
        mock_unload.assert_called_once()


class TestEmbeddingWorkerSourcesByIdIsIgnored:
    """sources_by_id is accepted for launch-site API parity but never read."""

    def test_worker_runs_with_none_sources_by_id(
        self, qapp_fixture, clip_with_thumb
    ):
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker

        with patch(
            "core.analysis.embeddings.extract_clip_embeddings_batch",
            return_value=[[0.1] * 768],
        ), patch("core.analysis.embeddings.unload_model"):
            worker = EmbeddingAnalysisWorker([clip_with_thumb], sources_by_id=None)
            worker.run()

        assert clip_with_thumb.embedding == [0.1] * 768


# Qt application fixture (needed since the worker is a QThread subclass)
@pytest.fixture(scope="module")
def qapp_fixture():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
