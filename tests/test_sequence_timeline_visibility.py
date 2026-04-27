"""Regression tests for Sequence tab timeline visibility and restoration."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.project import Project
from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip

# SequenceTab contains video widgets; offscreen avoids display-dependent failures in CI/headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_sequence_tab_cards_view_explains_empty_timeline(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()

    assert "Timeline is empty." in tab.cards_hint_label.text()


def test_sequence_timeline_splitter_sections_cannot_be_collapsed(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()

    assert tab.timeline_splitter.childrenCollapsible() is False
    assert tab.timeline_splitter.isCollapsible(0) is False
    assert tab.timeline_splitter.isCollapsible(1) is False
    assert tab.timeline_splitter.isCollapsible(2) is False
    assert tab.video_player.minimumHeight() >= 180
    assert tab.timeline_preview.minimumHeight() >= 90
    assert tab.timeline.minimumHeight() >= 180


def test_timeline_move_and_delete_emit_sequence_changed(qapp):
    from ui.timeline.timeline_widget import TimelineWidget

    timeline = TimelineWidget()
    changes = []
    timeline.sequence_changed.connect(lambda: changes.append(True))

    source = Source(
        id="src-1",
        file_path=Path("/tmp/test.mp4"),
        duration_seconds=10.0,
        fps=24.0,
        width=1920,
        height=1080,
    )
    clip = Clip(
        id="clip-1",
        source_id=source.id,
        start_frame=0,
        end_frame=24,
    )
    timeline.add_clip(clip, source)
    changes.clear()

    seq_clip = timeline.sequence.get_all_clips()[0]
    timeline.scene.clip_moved.emit(seq_clip.id, seq_clip.start_frame)
    timeline.scene.remove_clip(seq_clip.id)

    assert len(changes) == 2


class _FakeTimeline:
    def __init__(self):
        self.loaded = None
        self.zoom_fit_called = False

    def load_sequence(self, sequence, sources, clips):
        self.loaded = (sequence, sources, clips)

    def _on_zoom_fit(self):
        self.zoom_fit_called = True


class _FakePreview:
    def __init__(self):
        self.clips = None
        self.sources = None
        self.cleared = False

    def set_clips(self, clips, sources):
        self.clips = clips
        self.sources = sources
        self.cleared = False

    def clear(self):
        self.clips = []
        self.sources = None
        self.cleared = True


class _FakeSequenceTab:
    STATE_CARDS = 0
    STATE_TIMELINE = 1

    def __init__(self):
        self.timeline = _FakeTimeline()
        self.timeline_preview = _FakePreview()
        self._sources = {}
        self._available_clips = []
        self._clips = []
        self.state = None
        self.synced_sequence = None

    def _set_state(self, state):
        self.state = state

    def sync_sequence_metadata(self, sequence):
        self.synced_sequence = sequence


def _make_project_with_sequence(*, clip_on_second_track: bool) -> Project:
    project = Project.new(name="Timeline Visibility")
    source = Source(
        id="src-1",
        file_path=Path("/tmp/test.mp4"),
        duration_seconds=10.0,
        fps=24.0,
        width=1920,
        height=1080,
    )
    clip = Clip(
        id="clip-1",
        source_id=source.id,
        start_frame=0,
        end_frame=24,
    )
    project.add_source(source)
    project.add_clips([clip])

    sequence = Sequence(fps=24.0)
    if clip_on_second_track:
        sequence.add_track("Video 2")
        target_track = sequence.tracks[1]
        track_index = 1
    else:
        target_track = sequence.tracks[0]
        track_index = 0

    target_track.add_clip(
        SequenceClip(
            source_clip_id=clip.id,
            source_id=source.id,
            track_index=track_index,
            start_frame=0,
            in_point=clip.start_frame,
            out_point=clip.end_frame,
        )
    )
    project.sequence = sequence
    return project


def test_refresh_timeline_from_project_shows_timeline_for_non_primary_track_clip():
    from ui.main_window import MainWindow

    project = _make_project_with_sequence(clip_on_second_track=True)
    fake_window = SimpleNamespace(
        project=project,
        sequence_tab=_FakeSequenceTab(),
        _update_sequence_chromatic_bar=lambda: None,
    )

    MainWindow._refresh_timeline_from_project(fake_window)

    assert fake_window.sequence_tab.state == fake_window.sequence_tab.STATE_TIMELINE
    assert fake_window.sequence_tab.timeline.loaded is not None
    assert fake_window.sequence_tab.timeline.zoom_fit_called is True
    assert [clip.id for clip, _ in fake_window.sequence_tab.timeline_preview.clips] == ["clip-1"]


def test_refresh_timeline_from_project_returns_to_cards_when_sequence_is_empty():
    from ui.main_window import MainWindow

    project = Project.new(name="Empty Sequence")
    source = Source(
        id="src-1",
        file_path=Path("/tmp/test.mp4"),
        duration_seconds=10.0,
        fps=24.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)
    project.sequence = Sequence(fps=24.0)

    fake_window = SimpleNamespace(
        project=project,
        sequence_tab=_FakeSequenceTab(),
        _update_sequence_chromatic_bar=lambda: None,
    )

    MainWindow._refresh_timeline_from_project(fake_window)

    assert fake_window.sequence_tab.state == fake_window.sequence_tab.STATE_CARDS
    assert fake_window.sequence_tab.timeline_preview.cleared is True
