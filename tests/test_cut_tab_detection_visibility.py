"""Tests for Cut tab clip visibility after detection."""

from pathlib import Path

from PySide6.QtWidgets import QApplication

from models.clip import Clip, Source
from ui.tabs.cut_tab import CutTab


def test_cut_tab_add_clips_shows_detected_clips_without_thumbnails():
    app = QApplication.instance() or QApplication([])
    tab = CutTab()
    source = Source(id="source-1", file_path=Path("/tmp/video.mp4"), fps=30.0)
    clips = [
        Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30),
        Clip(id="clip-2", source_id=source.id, start_frame=30, end_frame=60),
    ]

    tab.set_source(source)
    tab.add_clips([(clip, source) for clip in clips])
    app.processEvents()

    assert tab.clip_browser.get_total_clip_count() == 2
    assert tab.clip_count_label.text() == "2 clips"
    assert [thumb.clip.id for thumb in tab.clip_browser.thumbnails] == ["clip-1", "clip-2"]


def test_cut_tab_full_sync_populates_browser_after_completion():
    app = QApplication.instance() or QApplication([])
    tab = CutTab()
    source = Source(id="source-1", file_path=Path("/tmp/video.mp4"), fps=30.0)
    clips = [
        Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30),
        Clip(id="clip-2", source_id=source.id, start_frame=30, end_frame=60),
    ]

    tab.set_source(source)
    tab.set_clips(clips)
    assert tab.clip_browser.get_total_clip_count() == 0

    tab.set_clip_source_pairs([(clip, source) for clip in clips])
    app.processEvents()

    assert tab.clip_browser.get_total_clip_count() == 2
    assert tab.clip_count_label.text() == "2 clips"
    assert [thumb.clip.id for thumb in tab.clip_browser.thumbnails] == ["clip-1", "clip-2"]
