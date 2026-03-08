"""Selection behavior tests for ClipBrowser."""

from pathlib import Path

import pytest

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
