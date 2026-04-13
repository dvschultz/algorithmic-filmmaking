"""Tests for embedding analysis dispatch and pipeline completion handler.

These tests target MainWindow's _launch_analysis_worker branch, the
_launch_embeddings_worker factory, and the guard-flag-protected
_on_pipeline_embeddings_finished handler. They don't exercise a real
QThread — the worker is instantiated but its .start() is mocked so we
can inspect the signal wiring.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestDispatchBranch:
    def test_dispatch_routes_embeddings_to_launcher(self, qapp):
        """_launch_analysis_worker('embeddings', clips) calls _launch_embeddings_worker."""
        from ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)  # bypass full __init__
        mw.progress_bar = MagicMock()
        mw.status_bar = MagicMock()

        launcher_calls = []
        mw._launch_embeddings_worker = lambda clips: launcher_calls.append(clips)

        mw._launch_analysis_worker("embeddings", ["clip-a", "clip-b"])

        assert launcher_calls == [["clip-a", "clip-b"]]


class TestLauncherSignalWiring:
    def test_launcher_creates_worker_and_connects_signals(self, qapp):
        """_launch_embeddings_worker instantiates EmbeddingAnalysisWorker and wires up signals."""
        from ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        mw._embeddings_finished_handled = False

        # Patch the class in main_window's namespace so instantiation
        # returns our mock instead of a real QThread.
        fake_worker = MagicMock()
        with patch("ui.main_window.EmbeddingAnalysisWorker", return_value=fake_worker):
            mw._launch_embeddings_worker(["clip-a"])

        # Worker constructed with the clips
        assert mw._embeddings_worker is fake_worker
        # Required signal connections
        fake_worker.progress.connect.assert_called()
        fake_worker.embedding_ready.connect.assert_called()
        fake_worker.analysis_completed.connect.assert_called()
        fake_worker.error.connect.assert_called()
        # Lifecycle cleanup connected
        assert fake_worker.finished.connect.call_count >= 2
        # Worker started
        fake_worker.start.assert_called_once()
        # Guard flag reset
        assert mw._embeddings_finished_handled is False


class TestPipelineCompletionHandler:
    def test_guard_flag_prevents_double_invocation(self, qapp):
        """Qt finished/analysis_completed signal can fire twice; the guard flag prevents
        the pipeline-advance handler from firing twice."""
        from ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        mw._embeddings_finished_handled = False

        calls = []
        mw._on_analysis_phase_worker_finished = lambda op_key: calls.append(op_key)

        # First call advances the pipeline
        mw._on_pipeline_embeddings_finished()
        assert calls == ["embeddings"]
        assert mw._embeddings_finished_handled is True

        # Second call (duplicate signal) is suppressed
        mw._on_pipeline_embeddings_finished()
        assert calls == ["embeddings"]  # unchanged

    def test_error_handler_logs_and_shows_status_message(self, qapp):
        from ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        status_bar = MagicMock()
        mw.statusBar = lambda: status_bar

        mw._on_embeddings_error("boom")

        status_bar.showMessage.assert_called_once()
        msg = status_bar.showMessage.call_args[0][0]
        assert "boom" in msg
        assert "Embedding" in msg
