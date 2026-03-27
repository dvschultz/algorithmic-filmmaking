"""Focused tests for single-clip export helpers on MainWindow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import logging

from models.clip import Source
from tests.conftest import make_test_clip
from ui.main_window import MainWindow


@pytest.fixture
def source():
    return Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )


class _DummyProgressBar:
    def __init__(self):
        self.visible = False
        self.ranges = []

    def setVisible(self, visible: bool):
        self.visible = visible

    def setRange(self, minimum: int, maximum: int):
        self.ranges.append((minimum, maximum))


class _DummyStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message: str, timeout: int = 0):
        self.messages.append((message, timeout))


def _make_window(tmp_path):
    window = SimpleNamespace(
        settings=SimpleNamespace(export_dir=tmp_path),
        progress_bar=_DummyProgressBar(),
        status_bar=_DummyStatusBar(),
        sources_by_id={},
        _sanitize_filename=MainWindow._sanitize_filename,
    )
    window._default_clip_export_filename = (
        lambda clip, source, ordinal=None:
        MainWindow._default_clip_export_filename(window, clip, source, ordinal)
    )
    return window


def test_single_clip_export_appends_mp4_and_uses_clicked_source(tmp_path, source, monkeypatch):
    clip = make_test_clip("c1")
    clip.source_id = source.id
    window = _make_window(tmp_path)

    requested_paths = []

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "chosen_clip"), "Video Files (*.mp4)"),
    )
    monkeypatch.setattr(
        "ui.main_window.QDesktopServices.openUrl",
        lambda url: requested_paths.append(Path(url.toLocalFile())),
    )
    monkeypatch.setattr(
        "ui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: pytest.fail("export should not fail"),
    )

    export_calls = []

    def fake_export(req_clip, req_source, output_path):
        export_calls.append((req_clip, req_source, output_path))
        return True

    window._export_clip_to_path = fake_export

    MainWindow._on_clip_export_requested(window, clip, source)

    assert export_calls == [(clip, source, tmp_path / "chosen_clip.mp4")]
    assert requested_paths == [tmp_path]
    assert window.status_bar.messages[-1] == ("Exported clip to chosen_clip.mp4", 5000)
    assert window.progress_bar.ranges == [(0, 0), (0, 100)]


def test_single_clip_export_cancel_does_nothing(tmp_path, source, monkeypatch):
    clip = make_test_clip("c1")
    window = _make_window(tmp_path)

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )

    export_calls = []
    window._export_clip_to_path = lambda *args: export_calls.append(args) or True

    MainWindow._on_clip_export_requested(window, clip, source)

    assert export_calls == []
    assert window.status_bar.messages == []


def test_default_clip_export_filename_uses_clip_name_when_present(tmp_path, source):
    clip = make_test_clip("c1")
    clip.name = 'Hero "Moment"'
    window = _make_window(tmp_path)

    filename = MainWindow._default_clip_export_filename(window, clip, source)

    assert filename == "video_Hero _Moment_.mp4"


def test_export_clip_to_path_logs_failure(tmp_path, source, monkeypatch, caplog):
    clip = make_test_clip("c1")
    clip.source_id = source.id
    output_path = tmp_path / "clip.mp4"

    class _FakeProcessor:
        def extract_clip(self, **_kwargs):
            return False

    monkeypatch.setattr("ui.main_window.FFmpegProcessor", lambda: _FakeProcessor())

    window = _make_window(tmp_path)

    with caplog.at_level(logging.INFO):
        success = MainWindow._export_clip_to_path(window, clip, source, output_path)

    assert success is False
    assert "Manual clip export requested" in caplog.text
    assert "Manual clip export failed" in caplog.text
