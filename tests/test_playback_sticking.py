"""Tests for playback sticking bug fixes.

Bug 1 — Load failure deadlock prevention:
  When media_load_failed is emitted, all pending/loading state must be cleared
  and playback must be stopped (if active) to prevent the player from getting
  stuck in an unrecoverable loading state.

Bug 2 — Premature advance guard:
  When _on_video_state_changed(playing=False) fires while
  _sequence_preview_loading is True, the handler must NOT call
  _play_clip_at_frame — otherwise a stale pause event from the *previous*
  source overwrites the pending load for the *next* source, causing the
  player to skip or stick.
"""

import os

import pytest

# Must be set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Bug 1: VideoPlayer._on_load_failed clears pending state
# ---------------------------------------------------------------------------

class TestVideoPlayerLoadFailed:
    """VideoPlayer._on_load_failed must clear all pending state and emit
    media_load_failed so that MainWindow (or any listener) can react."""

    def test_on_load_failed_clears_media_loaded(self, qapp):
        from ui.video_player import VideoPlayer

        player = VideoPlayer()
        player._media_loaded = True

        player._on_load_failed()

        assert player._media_loaded is False

    def test_on_load_failed_clears_pending_seek(self, qapp):
        from ui.video_player import VideoPlayer

        player = VideoPlayer()
        player._pending_seek_seconds = 5.0

        player._on_load_failed()

        assert player._pending_seek_seconds is None

    def test_on_load_failed_clears_pending_clip_range(self, qapp):
        from ui.video_player import VideoPlayer

        player = VideoPlayer()
        player._pending_clip_range = (1.0, 5.0)

        player._on_load_failed()

        assert player._pending_clip_range is None

    def test_on_load_failed_clears_pending_play_on_load(self, qapp):
        from ui.video_player import VideoPlayer

        player = VideoPlayer()
        player._pending_play_on_load = True

        player._on_load_failed()

        assert player._pending_play_on_load is False

    def test_on_load_failed_emits_signal(self, qapp):
        from ui.video_player import VideoPlayer

        player = VideoPlayer()
        received = []
        player.media_load_failed.connect(lambda: received.append(True))

        player._on_load_failed()

        assert received == [True]


# ---------------------------------------------------------------------------
# Bug 1 (continued): MainWindow._on_sequence_video_load_failed clears state
# ---------------------------------------------------------------------------

class _FakeMainWindow:
    """Minimal stand-in for MainWindow, carrying only the state variables
    and methods touched by _on_sequence_video_load_failed and
    _on_video_state_changed.  Avoids instantiating the real MainWindow
    (which requires a full Qt widget tree, project, etc.)."""

    def __init__(self):
        self._sequence_preview_loading = False
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        self._sequence_preview_source_id = "src-1"
        self._is_playing = False
        self._current_playback_clip = None
        self._gui_state = None
        self._stop_playback_called = False

    # Minimal stub — records the call rather than doing real work.
    def _stop_playback(self):
        self._stop_playback_called = True
        self._is_playing = False

    # Bind the real methods from MainWindow at class level so they
    # operate on our fake instance.
    @staticmethod
    def _bind():
        """Import and attach the real method implementations."""
        from ui.main_window import MainWindow

        _FakeMainWindow._on_sequence_video_load_failed = (
            MainWindow._on_sequence_video_load_failed
        )
        _FakeMainWindow._on_video_state_changed = (
            MainWindow._on_video_state_changed
        )


class TestMainWindowLoadFailedHandler:
    """_on_sequence_video_load_failed must clear all loading/pending state
    and stop playback if the player was playing."""

    @pytest.fixture(autouse=True)
    def _bind_methods(self):
        _FakeMainWindow._bind()

    def _make_loading_window(self, playing=False):
        """Return a _FakeMainWindow that looks like it's mid-load."""
        w = _FakeMainWindow()
        w._sequence_preview_loading = True
        w._pending_sequence_preview_source_id = "src-2"
        w._pending_sequence_preview_clip_range = (0.0, 3.0)
        w._pending_sequence_preview_seek_seconds = 1.5
        w._pending_sequence_playback_source_id = "src-2"
        w._pending_sequence_playback_range = (0, 72)
        w._is_playing = playing
        return w

    def test_clears_loading_flag(self):
        w = self._make_loading_window()
        w._on_sequence_video_load_failed()
        assert w._sequence_preview_loading is False

    def test_clears_pending_preview_state(self):
        w = self._make_loading_window()
        w._on_sequence_video_load_failed()
        assert w._pending_sequence_preview_source_id is None
        assert w._pending_sequence_preview_clip_range is None
        assert w._pending_sequence_preview_seek_seconds is None

    def test_clears_pending_playback_state(self):
        w = self._make_loading_window()
        w._on_sequence_video_load_failed()
        assert w._pending_sequence_playback_source_id is None
        assert w._pending_sequence_playback_range is None

    def test_invalidates_source_id(self):
        w = self._make_loading_window()
        w._on_sequence_video_load_failed()
        assert w._sequence_preview_source_id is None

    def test_stops_playback_when_playing(self):
        w = self._make_loading_window(playing=True)
        w._on_sequence_video_load_failed()
        assert w._stop_playback_called is True
        assert w._is_playing is False

    def test_does_not_stop_playback_when_not_playing(self):
        w = self._make_loading_window(playing=False)
        w._on_sequence_video_load_failed()
        assert w._stop_playback_called is False


# ---------------------------------------------------------------------------
# Bug 2: Premature advance guard in _on_video_state_changed
# ---------------------------------------------------------------------------

class TestPrematureAdvanceGuard:
    """When _on_video_state_changed(playing=False) fires while
    _sequence_preview_loading is True, it must be a no-op (beyond GUI state
    update) — specifically, it must NOT call _play_clip_at_frame."""

    @pytest.fixture(autouse=True)
    def _bind_methods(self):
        _FakeMainWindow._bind()

    def test_loading_guard_prevents_advance(self):
        """playing=False during loading must not call _play_clip_at_frame."""
        w = _FakeMainWindow()
        w._is_playing = True
        w._sequence_preview_loading = True
        w._current_playback_clip = "should-not-matter"

        # _play_clip_at_frame should never be reached; attach a sentinel.
        called = []
        w._play_clip_at_frame = lambda frame: called.append(frame)

        w._on_video_state_changed(playing=False)

        assert called == [], (
            "_play_clip_at_frame should not be called while loading"
        )

    def test_not_playing_returns_early(self):
        """If _is_playing is False, the handler returns immediately."""
        w = _FakeMainWindow()
        w._is_playing = False
        w._sequence_preview_loading = False
        w._current_playback_clip = "should-not-matter"

        called = []
        w._play_clip_at_frame = lambda frame: called.append(frame)

        w._on_video_state_changed(playing=False)

        assert called == [], (
            "_play_clip_at_frame should not be called when not playing"
        )

    def test_normal_advance_when_not_loading(self):
        """Without the loading flag, playing=False should try to advance.

        We verify this by checking that _play_clip_at_frame IS called when
        _sequence_preview_loading is False and _current_playback_clip exists
        with an end_frame.
        """
        from types import SimpleNamespace

        w = _FakeMainWindow()
        w._is_playing = True
        w._sequence_preview_loading = False

        # Fake clip with an end_frame
        w._current_playback_clip = SimpleNamespace(end_frame=lambda: 48)

        # Fake sequence_tab.timeline for the set_playhead_time call
        w.sequence_tab = SimpleNamespace(
            timeline=SimpleNamespace(
                set_playhead_time=lambda t: None,
                sequence=SimpleNamespace(fps=24.0),
            )
        )

        called = []
        w._play_clip_at_frame = lambda frame: called.append(frame)

        w._on_video_state_changed(playing=False)

        assert called == [48], (
            "_play_clip_at_frame should be called with end_frame when not loading"
        )
