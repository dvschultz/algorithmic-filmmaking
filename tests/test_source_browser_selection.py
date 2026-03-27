"""Selection behavior tests for SourceBrowser."""

from pathlib import Path

import pytest
from PySide6.QtCore import QRect

from models.clip import Source


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_source(source_id: str, filename: str) -> Source:
    return Source(
        id=source_id,
        file_path=Path(f"/test/{filename}"),
        duration_seconds=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )


def test_marquee_selection_replaces_previous_sources(qapp, monkeypatch):
    from ui.source_browser import SourceBrowser

    browser = SourceBrowser()
    sources = [
        make_source("s1", "video-1.mp4"),
        make_source("s2", "video-2.mp4"),
        make_source("s3", "video-3.mp4"),
    ]
    for source in sources:
        browser.add_source(source)

    geometries = {
        "s1": QRect(0, 0, 100, 100),
        "s2": QRect(120, 0, 100, 100),
        "s3": QRect(240, 0, 100, 100),
    }

    monkeypatch.setattr("ui.source_thumbnail.SourceThumbnail.isVisible", lambda self: True)
    monkeypatch.setattr(
        "ui.source_thumbnail.SourceThumbnail.geometry",
        lambda self: geometries[self.source.id],
    )

    browser._set_selected_ids({"s3"})
    browser._apply_marquee_selection(QRect(0, 0, 230, 110), additive=False)

    assert browser.selected_source_ids == {"s1", "s2"}


def test_marquee_selection_shift_adds_sources(qapp, monkeypatch):
    from ui.source_browser import SourceBrowser

    browser = SourceBrowser()
    sources = [
        make_source("s1", "video-1.mp4"),
        make_source("s2", "video-2.mp4"),
        make_source("s3", "video-3.mp4"),
    ]
    for source in sources:
        browser.add_source(source)

    geometries = {
        "s1": QRect(0, 0, 100, 100),
        "s2": QRect(120, 0, 100, 100),
        "s3": QRect(240, 0, 100, 100),
    }

    monkeypatch.setattr("ui.source_thumbnail.SourceThumbnail.isVisible", lambda self: True)
    monkeypatch.setattr(
        "ui.source_thumbnail.SourceThumbnail.geometry",
        lambda self: geometries[self.source.id],
    )

    browser._set_selected_ids({"s1"})
    browser._marquee_base_selection = set(browser.selected_source_ids)
    browser._apply_marquee_selection(QRect(200, 0, 160, 110), additive=True)

    assert browser.selected_source_ids == {"s1", "s2", "s3"}


def test_marquee_selection_only_targets_source_cards(qapp, monkeypatch):
    from ui.source_browser import SourceBrowser

    browser = SourceBrowser()
    for source in [
        make_source("s1", "video-1.mp4"),
        make_source("s2", "video-2.mp4"),
    ]:
        browser.add_source(source)

    geometries = {
        "s1": QRect(0, 0, 100, 100),
        "s2": QRect(120, 0, 100, 100),
    }

    monkeypatch.setattr("ui.source_thumbnail.SourceThumbnail.isVisible", lambda self: True)
    monkeypatch.setattr(
        "ui.source_thumbnail.SourceThumbnail.geometry",
        lambda self: geometries[self.source.id],
    )

    browser._apply_marquee_selection(QRect(0, 0, 400, 110), additive=False)

    assert browser.selected_source_ids == {"s1", "s2"}
