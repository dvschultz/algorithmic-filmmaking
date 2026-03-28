"""Regression tests for URL import resolution selection."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from core.downloader import (
    DEFAULT_DOWNLOAD_RESOLUTION,
    build_download_format_selector,
    normalize_download_resolution,
)
from ui.main_window import MainWindow
from ui.tabs.collect_tab import CollectTab


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_build_download_format_selector_defaults_to_best_available():
    assert build_download_format_selector(None) == (
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    )


def test_build_download_format_selector_caps_requested_height():
    selector = build_download_format_selector("720p")

    assert "height<=720" in selector
    assert "height<=1080" not in selector


def test_normalize_download_resolution_rejects_unknown_tier():
    with pytest.raises(ValueError, match="Unsupported download resolution"):
        normalize_download_resolution("360p")


def test_url_import_dialog_defaults_to_1080p(qapp):
    from ui.dialogs.url_import_dialog import URLImportDialog

    dialog = URLImportDialog(initial_url="")

    assert dialog.selected_resolution() == DEFAULT_DOWNLOAD_RESOLUTION
    assert dialog._import_btn.isEnabled() is False

    dialog.url_edit.setText("https://youtube.com/watch?v=abc")
    qapp.processEvents()

    assert dialog._import_btn.isEnabled() is True


def test_collect_tab_emits_url_and_resolution_from_shared_dialog(qapp, monkeypatch):
    collected = []

    monkeypatch.setattr(
        "ui.tabs.collect_tab.URLImportDialog.get_import_request",
        lambda *_args, **_kwargs: ("https://youtube.com/watch?v=abc", "480p"),
    )

    tab = CollectTab()
    tab.download_requested.connect(lambda url, resolution: collected.append((url, resolution)))

    tab._on_url_click()
    qapp.processEvents()

    assert collected == [("https://youtube.com/watch?v=abc", "480p")]


def test_import_url_click_forwards_selected_resolution(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "ui.main_window.URLImportDialog.get_import_request",
        lambda *_args, **_kwargs: ("https://vimeo.com/123", "4k"),
    )

    harness = SimpleNamespace()
    harness._download_video = lambda url, resolution=None: captured.update(
        {"url": url, "resolution": resolution}
    )

    MainWindow._on_import_url_click(harness)

    assert captured == {"url": "https://vimeo.com/123", "resolution": "4k"}


def test_download_video_passes_resolution_to_single_worker(monkeypatch):
    captured = {}

    class _Signal:
        def connect(self, _callback):
            return None

    class _FakeWorker:
        def __init__(self, url, resolution=None):
            captured["url"] = url
            captured["resolution"] = resolution
            self.progress = _Signal()
            self.download_completed = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            return None

    monkeypatch.setattr("ui.main_window.DownloadWorker", _FakeWorker)

    harness = SimpleNamespace(
        collect_tab=SimpleNamespace(set_downloading=lambda *_args: None),
        progress_bar=SimpleNamespace(setVisible=lambda *_args: None, setRange=lambda *_args: None),
        _gui_state=SimpleNamespace(set_processing=lambda *_args: None),
        _ensure_video_download_available=lambda: True,
        _on_download_progress=lambda *_args: None,
        _on_download_finished=lambda *_args: None,
        _on_download_error=lambda *_args: None,
        download_worker=None,
    )

    MainWindow._download_video(harness, "https://youtube.com/watch?v=abc", resolution="720p")

    assert captured == {
        "url": "https://youtube.com/watch?v=abc",
        "resolution": "720p",
        "started": True,
    }
