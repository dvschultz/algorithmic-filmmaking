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


def test_clip_browser_filters_by_selected_custom_queries_with_and_logic(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()

    clip_both = make_test_clip("c1")
    clip_both.custom_queries = [
        {"query": "blue car", "match": False, "confidence": 0.21, "model": "qwen3-vl-4b"},
        {"query": "blue car", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"},
        {"query": "person running", "match": True, "confidence": 0.87, "model": "qwen3-vl-4b"},
    ]
    clip_only_blue = make_test_clip("c2")
    clip_only_blue.custom_queries = [
        {"query": "blue car", "match": True, "confidence": 0.83, "model": "qwen3-vl-4b"}
    ]
    clip_latest_no_match = make_test_clip("c3")
    clip_latest_no_match.custom_queries = [
        {"query": "blue car", "match": True, "confidence": 0.88, "model": "qwen3-vl-4b"},
        {"query": "blue car", "match": False, "confidence": 0.12, "model": "qwen3-vl-4b"},
    ]
    clip_none = make_test_clip("c4")

    browser.add_clip(clip_both, source)
    browser.add_clip(clip_only_blue, source)
    browser.add_clip(clip_latest_no_match, source)
    browser.add_clip(clip_none, source)
    qapp.processEvents()

    both_thumb = browser._thumbnail_by_id["c1"]
    latest_no_match_thumb = browser._thumbnail_by_id["c3"]
    none_thumb = browser._thumbnail_by_id["c4"]

    assert both_thumb.custom_query_label.text() == "Query Match"
    assert both_thumb.custom_query_label.isHidden() is False
    assert "YES: blue car" in both_thumb.custom_query_label.toolTip()
    assert "YES: person running" in both_thumb.custom_query_label.toolTip()
    assert "NO: blue car" not in both_thumb.custom_query_label.toolTip()

    assert latest_no_match_thumb.custom_query_label.text() == "Query No Match"
    assert latest_no_match_thumb.custom_query_label.isHidden() is False
    assert "NO: blue car" in latest_no_match_thumb.custom_query_label.toolTip()
    assert "YES: blue car" not in latest_no_match_thumb.custom_query_label.toolTip()

    assert none_thumb.custom_query_label.isHidden() is True

    assert sorted(browser._custom_query_filter_actions) == ["blue car", "person running"]

    browser._custom_query_filter_actions["blue car"].trigger()
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 2
    assert browser.get_active_filters()["custom_query"] == ["blue car"]

    browser._custom_query_filter_actions["person running"].trigger()
    qapp.processEvents()
    assert browser.get_visible_clip_count() == 1
    assert browser.get_active_filters()["custom_query"] == ["blue car", "person running"]

    browser.clear_all_filters()
    qapp.processEvents()
    assert browser.get_active_filters()["custom_query"] is None
    assert browser.get_visible_clip_count() == 4


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


def test_clip_details_sidebar_allows_editing_custom_queries(qapp, source):
    from ui.clip_details_sidebar import ClipDetailsSidebar

    clip = make_test_clip("c1")
    clip.custom_queries = [
        {"query": "blue car", "match": False, "confidence": 0.21, "model": "qwen3-vl-4b"},
        {"query": "blue car", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"},
        {"query": "person running", "match": True, "confidence": 0.87, "model": "qwen3-vl-4b"},
    ]

    sidebar = ClipDetailsSidebar()
    sidebar.video_player._setup_player = lambda: None
    sidebar.video_player._player_ready = True
    edited_queries = []
    sidebar.clip_edited.connect(lambda updated_clip: edited_queries.append(updated_clip.custom_queries))

    sidebar.show_clip(clip, source)
    qapp.processEvents()

    assert sidebar.custom_queries_header.isHidden() is False
    assert len(sidebar._custom_query_row_widgets) == 2

    sidebar._on_custom_query_match_changed(0, "Match")
    qapp.processEvents()
    assert clip.custom_queries == [
        {"query": "blue car", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"},
        {"query": "person running", "match": True, "confidence": 0.87, "model": "qwen3-vl-4b"},
    ]
    assert edited_queries[-1] == clip.custom_queries

    sidebar._on_custom_query_changed(0, "red car")
    qapp.processEvents()
    assert clip.custom_queries == [
        {"query": "red car", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"},
        {"query": "person running", "match": True, "confidence": 0.87, "model": "qwen3-vl-4b"},
    ]
    assert edited_queries[-1] == clip.custom_queries

    sidebar._on_custom_query_removed(1)
    qapp.processEvents()
    assert clip.custom_queries == [
        {"query": "red car", "match": True, "confidence": 0.92, "model": "qwen3-vl-4b"},
    ]
    assert len(sidebar._custom_query_row_widgets) == 1
    assert edited_queries[-1] == clip.custom_queries
