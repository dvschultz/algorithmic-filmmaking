"""Tests for Unit 5 filter predicates (person count, has_audio, volume, etc.)."""

import os

import pytest

from models.clip import Source
from pathlib import Path
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def browser_with_clips(qapp):
    from ui.clip_browser import ClipBrowser
    browser = ClipBrowser()
    src = Source(
        id="src",
        file_path=Path("/video.mp4"),
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    return browser, src


def _add(browser, src, clip_id, **kwargs):
    clip = make_test_clip(clip_id)
    for k, v in kwargs.items():
        setattr(clip, k, v)
    browser.add_clip(clip, src)
    return clip


def test_person_count_greater_than(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", person_count=0)
    _add(browser, src, "c2", person_count=1)
    _add(browser, src, "c3", person_count=3)
    qapp.processEvents()

    browser.apply_filters({"person_count": (">", 1)})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1  # only c3


def test_person_count_equal(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", person_count=1)
    _add(browser, src, "c2", person_count=2)
    _add(browser, src, "c3", person_count=1)
    qapp.processEvents()

    browser.apply_filters({"person_count": ("=", 1)})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 2


def test_person_count_less_than_treats_none_as_zero(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1")  # person_count=None → treated as 0
    _add(browser, src, "c2", person_count=2)
    qapp.processEvents()

    browser.apply_filters({"person_count": ("<", 1)})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1  # only c1


def test_has_transcript_yes(browser_with_clips, qapp):
    from core.transcription import TranscriptSegment
    browser, src = browser_with_clips
    _add(browser, src, "c1", transcript=[])
    _add(browser, src, "c2", transcript=[TranscriptSegment(text="hi", start_time=0.0, end_time=1.0)])
    qapp.processEvents()

    browser.apply_filters({"has_transcript": True})
    qapp.processEvents()

    # Only c2 has a non-empty transcript
    assert browser.get_visible_clip_count() == 1


def test_has_on_screen_text_no(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", extracted_texts=None)
    _add(browser, src, "c2", extracted_texts=[{"text": "hello"}])
    qapp.processEvents()

    browser.apply_filters({"has_on_screen_text": False})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1  # only c1


def test_volume_range(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", rms_volume=-40.0)
    _add(browser, src, "c2", rms_volume=-20.0)
    _add(browser, src, "c3", rms_volume=-10.0)
    qapp.processEvents()

    browser.apply_filters({"min_volume": -30.0, "max_volume": -15.0})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1  # only c2


def test_enabled_filter_yes(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", disabled=False)
    _add(browser, src, "c2", disabled=True)
    qapp.processEvents()

    browser.apply_filters({"enabled_filter": True})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1


def test_tag_note_search(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", tags=["important"])
    _add(browser, src, "c2", notes="not great")
    _add(browser, src, "c3")
    qapp.processEvents()

    browser.apply_filters({"tag_note_search": "important"})
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1

    browser.apply_filters({"tag_note_search": "great"})
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1


def test_has_analysis_ops_describe(browser_with_clips, qapp):
    """When 'describe' is selected, only clips with a description pass."""
    browser, src = browser_with_clips
    _add(browser, src, "c1", description="a scene description")
    _add(browser, src, "c2")  # no description
    qapp.processEvents()

    browser.apply_filters({"has_analysis_ops": ["describe"]})
    qapp.processEvents()

    assert browser.get_visible_clip_count() == 1
