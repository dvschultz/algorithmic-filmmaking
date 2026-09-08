"""Clip worker factories preserve operation ownership and provider settings."""

from unittest.mock import MagicMock

import pytest

from core.analysis_operations import OPERATIONS_BY_KEY
from core.project import Project
from core.settings import Settings
from models.clip import Clip, Source
from ui.workers.clip_analysis_work import create_clip_analysis_worker


@pytest.fixture(scope="module", autouse=True)
def qapp():
    import os
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("operation", list(OPERATIONS_BY_KEY))
def test_factory_builds_shared_application_without_starting(tmp_path, operation):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    source.file_path.write_bytes(b"source")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    clip.thumbnail_path = tmp_path / "thumb.jpg"
    clip.thumbnail_path.write_bytes(b"thumbnail")
    project.add_source(source)
    project.add_clips([clip])
    worker, application = create_clip_analysis_worker(
        project,
        Settings(),
        operation,
        [clip],
        query="a flower",
    )
    assert not worker.isRunning()
    assert application.project is project
    assert "finished" not in type(worker).__dict__
    ids = (
        [target.target_id for target in worker.request.targets]
        if operation == "colors"
        else [task.clip_id for task in worker.tasks]
    )
    assert ids == [clip.id]


def test_error_handler_logs_and_shows_status_message():
    from types import SimpleNamespace
    from ui.main_window import MainWindow

    status_bar = MagicMock()
    window = SimpleNamespace(statusBar=lambda: status_bar)
    MainWindow._on_embeddings_error(window, "boom")
    status_bar.showMessage.assert_called_once()
    message = status_bar.showMessage.call_args.args[0]
    assert "boom" in message and "Embedding" in message


@pytest.mark.parametrize(
    "operation", ["shots", "describe", "custom_query", "cinematography", "extract_text"]
)
def test_provider_options_use_the_run_settings(tmp_path, monkeypatch, operation):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    source.file_path.write_bytes(b"source")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    clip.thumbnail_path = tmp_path / "thumb.jpg"
    clip.thumbnail_path.write_bytes(b"thumbnail")
    project.add_source(source)
    project.add_clips([clip])
    settings = Settings()
    settings.text_extraction_method = "hybrid"
    settings.text_extraction_vlm_model = None
    monkeypatch.setattr(
        "core.settings.load_settings",
        MagicMock(side_effect=AssertionError("settings reloaded")),
    )
    worker, _ = create_clip_analysis_worker(
        project, settings, operation, [clip], query="a flower"
    )
