"""UI tests for custom query display and filtering."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

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
def source():
    return Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )


def test_clip_browser_shows_custom_query_badges_and_filters(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()

    clip_match = make_test_clip("c1")
    clip_match.custom_queries = [
        {"query": "red hat", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"}
    ]
    clip_no_match = make_test_clip("c2")
    clip_no_match.custom_queries = [
        {"query": "red hat", "match": False, "confidence": 0.08, "model": "qwen3-vl-4b"}
    ]
    clip_none = make_test_clip("c3")

    browser.add_clip(clip_match, source)
    browser.add_clip(clip_no_match, source)
    browser.add_clip(clip_none, source)
    qapp.processEvents()

    match_thumb = browser._thumbnail_by_id["c1"]
    no_match_thumb = browser._thumbnail_by_id["c2"]
    none_thumb = browser._thumbnail_by_id["c3"]

    assert match_thumb.custom_query_label.text() == "Query Match"
    assert match_thumb.custom_query_label.isHidden() is False
    assert "YES: red hat" in match_thumb.custom_query_label.toolTip()

    assert no_match_thumb.custom_query_label.text() == "Query No Match"
    assert no_match_thumb.custom_query_label.isHidden() is False
    assert "NO: red hat" in no_match_thumb.custom_query_label.toolTip()

    assert none_thumb.custom_query_label.isHidden() is True

    browser.custom_query_filter_combo.setCurrentText("Match")
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1
    assert browser.get_active_filters()["custom_query"] == "Match"

    browser.custom_query_filter_combo.setCurrentText("No Match")
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1
    assert browser.get_active_filters()["custom_query"] == "No Match"

    browser.clear_all_filters()
    qapp.processEvents()
    assert browser.get_active_filters()["custom_query"] is None
    assert browser.get_visible_clip_count() == 3


def test_custom_query_ready_updates_tabs_sidebar_and_dirty_state(source):
    from ui.main_window import MainWindow

    clip = make_test_clip("c1")
    clip.custom_queries = None

    analyze_calls = []
    cut_calls = []
    sidebar_calls = []
    dirty_calls = []

    window = SimpleNamespace(
        project=SimpleNamespace(clips_by_id={clip.id: clip}),
        analyze_tab=SimpleNamespace(
            update_clip_custom_queries=lambda clip_id, queries: analyze_calls.append(
                (clip_id, queries)
            )
        ),
        cut_tab=SimpleNamespace(
            clip_browser=SimpleNamespace(
                update_clip_custom_queries=lambda clip_id, queries: cut_calls.append(
                    (clip_id, queries)
                )
            )
        ),
        clip_details_sidebar=SimpleNamespace(
            refresh_custom_queries_if_showing=lambda clip_id, queries: sidebar_calls.append(
                (clip_id, queries)
            )
        ),
        _mark_dirty=lambda: dirty_calls.append(True),
    )

    MainWindow._on_custom_query_ready(
        window,
        clip.id,
        "red hat",
        True,
        0.92,
        "qwen3-vl-4b",
    )

    assert clip.custom_queries == [
        {
            "query": "red hat",
            "match": True,
            "confidence": 0.92,
            "model": "qwen3-vl-4b",
        }
    ]
    assert analyze_calls == [(clip.id, clip.custom_queries)]
    assert cut_calls == [(clip.id, clip.custom_queries)]
    assert sidebar_calls == [(clip.id, clip.custom_queries)]
    assert dirty_calls == [True]
