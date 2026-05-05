"""Selection behavior tests for ClipBrowser."""

from pathlib import Path

import pytest
from PySide6.QtCore import QRect
from PySide6.QtTest import QTest

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


def test_add_clips_bulk_populates_lookup_once(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip("c1"), make_test_clip("c2"), make_test_clip("c3")]
    rebuilds = []

    monkeypatch.setattr(browser, "_rebuild_grid", lambda: rebuilds.append(True))

    browser.add_clips([(clip, source) for clip in clips])

    assert [thumb.clip.id for thumb in browser.thumbnails] == ["c1", "c2", "c3"]
    assert set(browser._thumbnail_by_id) == {"c1", "c2", "c3"}
    assert all(browser.get_source_for_clip(clip.id) is source for clip in clips)
    assert rebuilds == [True]


def test_add_clips_defer_rebuild_skips_until_finalize(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    rebuilds = []
    monkeypatch.setattr(browser, "_rebuild_grid", lambda: rebuilds.append(True))

    # Three deferred batches must not trigger any rebuild.
    for batch_idx in range(3):
        clips = [make_test_clip(f"b{batch_idx}-{i}") for i in range(2)]
        browser.add_clips([(c, source) for c in clips], defer_rebuild=True)

    assert rebuilds == []
    assert len(browser.thumbnails) == 6

    # finalize_batch_load() flushes a single rebuild.
    browser.finalize_batch_load()
    assert rebuilds == [True]


def test_add_clips_can_defer_filter_sync_until_finalize(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    syncs = []
    rebuilds = []
    monkeypatch.setattr(browser, "_sync_custom_query_filter_options", lambda: syncs.append(True))
    monkeypatch.setattr(browser, "_rebuild_grid", lambda: rebuilds.append(True))

    clips = [make_test_clip(f"c{i}") for i in range(3)]
    browser.add_clips(
        [(clip, source) for clip in clips],
        defer_rebuild=True,
        defer_filter_sync=True,
    )

    assert syncs == []
    assert rebuilds == []

    browser.finalize_batch_load()
    assert syncs == [True]
    assert rebuilds == [True]


def test_virtual_clip_browser_realizes_only_visible_window(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    browser.resize(700, 500)
    clips = [make_test_clip(f"v{i}") for i in range(100)]

    browser.set_virtual_clips([(clip, source) for clip in clips])
    qapp.processEvents()

    assert browser._virtual_mode is True
    assert len(browser._virtual_entries) == 100
    assert 0 < len(browser.thumbnails) < 100
    assert browser.is_virtualized() is True
    assert browser.get_total_clip_count() == 100
    assert browser.get_realized_clip_count() == len(browser.thumbnails)
    assert all(browser.get_source_for_clip(clip.id) is source for clip in clips)


def test_virtual_clip_browser_selection_uses_all_filtered_data(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    clips = [make_test_clip(f"v{i}") for i in range(25)]
    clips[3].disabled = True
    clips[10].disabled = True

    browser.set_virtual_clips([(clip, source) for clip in clips])
    qapp.processEvents()
    browser.select_all()

    selected_ids = {clip.id for clip in browser.get_selected_clips()}
    assert selected_ids == {clip.id for clip in clips if not clip.disabled}


def test_virtual_clip_browser_update_preserves_multi_source_entries(qapp, source):
    from ui.clip_browser import ClipBrowser

    other_source = Source(
        id="src-2",
        file_path=Path("/test/new-video.mp4"),
        duration_seconds=8.0,
        fps=24.0,
        width=1280,
        height=720,
    )
    old_clips = [make_test_clip(f"old-{i}", source_id=source.id) for i in range(75)]
    new_clips = [make_test_clip(f"new-{i}", source_id=other_source.id) for i in range(25)]

    browser = ClipBrowser()
    browser.resize(700, 500)
    browser.set_virtual_clips(
        [(clip, source) for clip in old_clips]
        + [(clip, other_source) for clip in new_clips]
    )
    qapp.processEvents()

    updated_new_clip = make_test_clip(
        "new-3",
        source_id=other_source.id,
        shot_type="close-up",
    )
    browser.update_clips([updated_new_clip])
    qapp.processEvents()

    assert browser.get_total_clip_count() == 100
    assert sum(1 for clip, src in browser._virtual_entries if src.id == source.id) == 75
    assert sum(1 for clip, src in browser._virtual_entries if src.id == other_source.id) == 25
    stored = {clip.id: clip for clip, _source in browser._virtual_entries}
    assert stored["new-3"].shot_type == "close-up"


def test_virtual_refresh_layout_skips_when_columns_unchanged(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    browser.resize(700, 500)
    clips = [make_test_clip(f"v{i}") for i in range(100)]
    browser.set_virtual_clips([(clip, source) for clip in clips])
    qapp.processEvents()
    browser._last_column_count = browser._calculate_columns()

    rebuilds = []
    monkeypatch.setattr(browser, "_rebuild_grid", lambda: rebuilds.append(True))

    browser.refresh_layout()

    assert rebuilds == []


def test_virtual_scroll_reuses_cached_rows(qapp, source, monkeypatch):
    from ui.clip_browser import ClipBrowser, VIRTUAL_CARD_ROW_HEIGHT

    browser = ClipBrowser()
    browser.resize(700, 500)
    clips = [make_test_clip(f"v{i}") for i in range(100)]
    browser.set_virtual_clips([(clip, source) for clip in clips])
    qapp.processEvents()

    calls = []
    original = browser._build_virtual_rows

    def wrapped_build(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(browser, "_build_virtual_rows", wrapped_build)
    browser.scroll.verticalScrollBar().setValue(VIRTUAL_CARD_ROW_HEIGHT * 3)
    browser._on_scroll_changed(browser.scroll.verticalScrollBar().value())
    QTest.qWait(25)
    qapp.processEvents()

    assert calls == []


def test_virtual_scroll_reuses_cached_thumbnail_widgets(qapp, source):
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    browser.resize(700, 500)
    clips = [make_test_clip(f"v{i}") for i in range(100)]
    browser.set_virtual_clips([(clip, source) for clip in clips])
    qapp.processEvents()

    first_id = browser.thumbnails[0].clip.id
    first_widget = browser.thumbnails[0]

    browser._clear_realized_virtual_widgets()
    reused = browser._get_virtual_thumbnail(clips[0], source)

    assert reused is first_widget
    assert browser._virtual_widget_cache[first_id] is first_widget


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
