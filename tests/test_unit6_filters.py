"""Tests for Unit 6 filter predicates (ImageNet, YOLO)."""

import os

import pytest

from pathlib import Path

from models.clip import Source
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
        id="src", file_path=Path("/v.mp4"), duration_seconds=60.0,
        fps=30.0, width=1920, height=1080,
    )
    return browser, src


def _add(browser, src, cid, **kwargs):
    clip = make_test_clip(cid)
    for k, v in kwargs.items():
        setattr(clip, k, v)
    browser.add_clip(clip, src)
    return clip


def test_imagenet_any(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", object_labels=["dog", "ball"])
    _add(browser, src, "c2", object_labels=["cat"])
    _add(browser, src, "c3", object_labels=[])
    qapp.processEvents()

    browser.apply_filters({
        "imagenet_labels": ["dog", "cat"],
        "imagenet_mode": "any",
    })
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 2  # c1 + c2


def test_imagenet_all(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", object_labels=["dog", "ball"])
    _add(browser, src, "c2", object_labels=["dog", "cat", "ball"])
    _add(browser, src, "c3", object_labels=["dog"])
    qapp.processEvents()

    browser.apply_filters({
        "imagenet_labels": ["dog", "ball"],
        "imagenet_mode": "all",
    })
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 2  # c1 + c2


def test_yolo_labels(browser_with_clips, qapp):
    browser, src = browser_with_clips
    _add(browser, src, "c1", detected_objects=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]}])
    _add(browser, src, "c2", detected_objects=[{"label": "car", "confidence": 0.8, "bbox": [0, 0, 100, 100]}])
    _add(browser, src, "c3", detected_objects=[])
    qapp.processEvents()

    browser.apply_filters({"yolo_labels": ["person"]})
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1


def test_yolo_total_count_gt(browser_with_clips, qapp):
    browser, src = browser_with_clips
    det = lambda label: {"label": label, "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    _add(browser, src, "c1", detected_objects=[det("person")])
    _add(browser, src, "c2", detected_objects=[det("person"), det("person"), det("car")])
    _add(browser, src, "c3", detected_objects=[])
    qapp.processEvents()

    browser.apply_filters({"yolo_total_count": (">", 2)})
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1  # c2 has 3 detections


def test_yolo_per_label_rules(browser_with_clips, qapp):
    browser, src = browser_with_clips
    det = lambda label: {"label": label, "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    _add(browser, src, "c1", detected_objects=[det("person")])
    _add(browser, src, "c2", detected_objects=[det("person"), det("person"), det("car")])
    _add(browser, src, "c3", detected_objects=[det("car"), det("car")])
    qapp.processEvents()

    # Rule: exactly 1 person
    browser.apply_filters({"yolo_per_label_rules": [["person", "=", 1]]})
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1  # only c1

    # Rule: > 0 car AND > 1 person
    browser.apply_filters({
        "yolo_per_label_rules": [["car", ">", 0], ["person", ">", 1]],
    })
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1  # only c2


def test_active_filter_chips_renders(qapp):
    from core.filter_state import FilterState
    from ui.widgets.active_filter_chips import ActiveFilterChips
    fs = FilterState()
    bar = ActiveFilterChips(fs)
    # Empty state: no chips
    assert len(bar._chips) == 0

    fs.shot_type = "Close-up"
    qapp.processEvents()
    assert len(bar._chips) == 1


def test_active_filter_chip_clear_removes_filter(qapp):
    from core.filter_state import FilterState
    from ui.widgets.active_filter_chips import ActiveFilterChips

    fs = FilterState()
    fs.shot_type = "Close-up"
    bar = ActiveFilterChips(fs)
    qapp.processEvents()
    assert len(bar._chips) == 1

    bar._chips[0].click()
    qapp.processEvents()
    assert fs.shot_type == set()


def test_reset_button_clears_everything(qapp):
    from core.filter_state import FilterState

    fs = FilterState()
    fs.apply_dict({
        "shot_type": ["Close-up"],
        "person_count": (">", 1),
        "has_audio": True,
        "imagenet_labels": ["dog"],
    })
    assert fs.has_active() is True

    fs.clear_all()
    assert fs.has_active() is False


def test_filter_state_change_triggers_grid_rebuild(browser_with_clips, qapp):
    """Regression: setting a FilterState field directly must hide clips.

    This is the bug that v0.3.12 shipped with — FilterState.changed was
    not connected to _rebuild_grid, so sidebar chip toggles updated state
    but the grid never refiltered.
    """
    browser, src = browser_with_clips
    det = lambda label: {"label": label, "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    _add(browser, src, "c1", detected_objects=[det("person")])
    _add(browser, src, "c2", detected_objects=[det("car")])
    _add(browser, src, "c3", detected_objects=[])
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 3

    # Set FilterState directly, bypassing apply_filters
    browser._filter_state.yolo_labels = {"person"}
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1


def test_has_active_filters_reflects_unit5_6_fields(browser_with_clips, qapp):
    """Regression: has_active_filters() must report Unit 5/6 filters as active."""
    browser, src = browser_with_clips
    _add(browser, src, "c1")
    qapp.processEvents()

    browser._filter_state.has_audio = True
    assert browser.has_active_filters() is True

    browser._filter_state.has_audio = None
    browser._filter_state.imagenet_labels = {"dog"}
    assert browser.has_active_filters() is True


def test_clear_all_filters_resets_unit5_6(browser_with_clips, qapp):
    """Regression: browser.clear_all_filters() must reset Unit 5/6 fields."""
    browser, src = browser_with_clips
    _add(browser, src, "c1")
    qapp.processEvents()

    browser._filter_state.apply_dict({
        "shot_type": ["Close-up"],
        "person_count": (">", 0),
        "imagenet_labels": ["dog"],
        "yolo_total_count": (">", 0),
    })
    assert browser.has_active_filters() is True

    browser.clear_all_filters()
    qapp.processEvents()
    assert browser.has_active_filters() is False
    assert browser._filter_state.person_count is None
    assert browser._filter_state.imagenet_labels == set()
    assert browser._filter_state.yolo_total_count is None
