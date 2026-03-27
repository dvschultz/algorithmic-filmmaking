"""Selection behavior tests for ClipBrowser."""

from pathlib import Path

import pytest
from PySide6.QtCore import QRect

from models.clip import Source
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def test_select_all_excludes_disabled_clips(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [
        make_test_clip("c1"),
        make_test_clip("c2"),
        make_test_clip("c3"),
        make_test_clip("c4"),
    ]
    clips[0].disabled = True
    clips[2].disabled = True

    for clip in clips:
        browser.add_clip(clip, source)

    # In headless test runs, widgets may never become "visible" even though
    # selection logic depends on isVisible(). Force deterministic visibility.
    monkeypatch.setattr(
        "ui.clip_browser.ClipThumbnail.isVisible",
        lambda self: True,
    )

    browser.select_all()

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {"c2", "c4"}


def test_toggle_disabled_unselects_only_toggled_clip(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    for clip in clips:
        browser.add_clip(clip, source)

    browser.set_selection(["c2", "c3"])
    browser.toggle_disabled(["c3"])

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {"c2"}
    assert browser._thumbnail_by_id["c3"].clip.disabled is True


def test_toggle_disabled_does_not_select_first_clip(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    for clip in clips:
        browser.add_clip(clip, source)

    browser.set_selection(["c3"])
    browser.toggle_disabled(["c3"])

    assert browser.get_selected_clips() == []
    assert "c1" not in browser.selected_clips


def test_marquee_selection_replaces_previous_selection(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    for clip in clips:
        browser.add_clip(clip, source)

    geometries = {
        "c1": QRect(0, 0, 100, 100),
        "c2": QRect(120, 0, 100, 100),
        "c3": QRect(240, 0, 100, 100),
    }

    monkeypatch.setattr("ui.clip_browser.ClipThumbnail.isVisible", lambda self: True)
    monkeypatch.setattr(
        "ui.clip_browser.ClipThumbnail.geometry",
        lambda self: geometries[self.clip.id],
    )

    browser.set_selection(["c3"])
    browser._apply_marquee_selection(QRect(0, 0, 230, 110), additive=False)

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {"c1", "c2"}


def test_marquee_selection_shift_adds_to_previous_selection(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    for clip in clips:
        browser.add_clip(clip, source)

    geometries = {
        "c1": QRect(0, 0, 100, 100),
        "c2": QRect(120, 0, 100, 100),
        "c3": QRect(240, 0, 100, 100),
    }

    monkeypatch.setattr("ui.clip_browser.ClipThumbnail.isVisible", lambda self: True)
    monkeypatch.setattr(
        "ui.clip_browser.ClipThumbnail.geometry",
        lambda self: geometries[self.clip.id],
    )

    browser.set_selection(["c1"])
    browser._marquee_base_selection = set(browser.selected_clips)
    browser._apply_marquee_selection(QRect(200, 0, 160, 110), additive=True)

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {"c1", "c2", "c3"}


def test_marquee_selection_skips_disabled_and_hidden_clips(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    clips[1].disabled = True
    for clip in clips:
        browser.add_clip(clip, source)

    geometries = {
        "c1": QRect(0, 0, 100, 100),
        "c2": QRect(120, 0, 100, 100),
        "c3": QRect(240, 0, 100, 100),
    }
    visibility = {"c1": True, "c2": True, "c3": False}

    monkeypatch.setattr(
        "ui.clip_browser.ClipThumbnail.isVisible",
        lambda self: visibility[self.clip.id],
    )
    monkeypatch.setattr(
        "ui.clip_browser.ClipThumbnail.geometry",
        lambda self: geometries[self.clip.id],
    )

    browser._apply_marquee_selection(QRect(0, 0, 400, 110), additive=False)

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {"c1"}


def test_export_request_forwards_clicked_clip_and_source(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clip = make_test_clip("c1")
    browser.add_clip(clip, source)

    requested = []
    browser.export_requested.connect(lambda req_clip, req_source: requested.append((req_clip, req_source)))

    thumb = browser._thumbnail_by_id[clip.id]
    thumb._emit_export_requested()

    assert requested == [(clip, source)]
