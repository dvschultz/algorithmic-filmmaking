"""Tests for VideoPlayer public API with mock MPV backend.

Tests the VideoPlayer contract — public methods, signals, and properties —
by mocking the underlying mpv.MPV instance. This validates the API surface
without requiring libmpv to be installed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_mpv_instance():
    """Create a fresh mock MPV instance with sensible defaults."""
    mock = MagicMock()
    mock.pause = True
    mock.speed = 1.0
    mock.mute = False
    mock.time_pos = 0.0
    mock.duration = None
    mock.eof_reached = False
    mock.core_shutdown = False
    return mock


@pytest.fixture
def player():
    """Create a VideoPlayer with a mock MPV backend."""
    mock_instance = _make_mock_mpv_instance()

    # Need QApplication for widget creation
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Patch mpv.MPV where video_player imports it so no real MPV is created
    with patch('ui.video_player.mpv.MPV', return_value=mock_instance):
        from ui.video_player import VideoPlayer
        p = VideoPlayer()
    p._mock = mock_instance
    yield p
    p.shutdown()


class TestVideoPlayerPublicAPI:
    """Test the public API contract."""

    def test_load_video_calls_mpv_play(self, player):
        test_path = Path("/tmp/test_video.mp4")
        player._shutting_down = False
        player.load_video(test_path)
        player._mock.play.assert_called_once_with(str(test_path))

    def test_load_video_clears_clip_range(self, player):
        player._shutting_down = False
        player._clip_start_ms = 1000
        player._clip_end_ms = 5000
        player.load_video(Path("/tmp/test.mp4"))
        assert player._clip_start_ms is None
        assert player._clip_end_ms is None

    def test_seek_to_calls_mpv_seek(self, player):
        player._shutting_down = False
        player.seek_to(10.5)
        player._mock.seek.assert_called_once_with(10.5, 'absolute', 'exact')

    def test_play_sets_pause_false(self, player):
        player._shutting_down = False
        player.play()
        assert player._mock.pause is False

    def test_pause_sets_pause_true(self, player):
        player._shutting_down = False
        player._mock.pause = False
        player.pause()
        assert player._mock.pause is True

    def test_stop_pauses_and_seeks_to_start(self, player):
        player._shutting_down = False
        player.stop()
        assert player._mock.pause is True
        player._mock.seek.assert_called_once_with(0, 'absolute')

    def test_stop_in_clip_mode_seeks_to_clip_start(self, player):
        player._shutting_down = False
        player._clip_start_ms = 5000
        player.stop()
        player._mock.seek.assert_called_once_with(5.0, 'absolute', 'exact')

    def test_is_playing_when_not_paused(self, player):
        player._shutting_down = False
        player._mock.pause = False
        assert player.is_playing is True

    def test_is_playing_when_paused(self, player):
        player._shutting_down = False
        player._mock.pause = True
        assert player.is_playing is False

    def test_duration_ms_converts_from_seconds(self, player):
        player._duration_s = 120.5
        assert player.duration_ms == 120500

    def test_playback_speed_getter(self, player):
        player._mock.speed = 2.0
        assert player.playback_speed == 2.0

    def test_playback_speed_setter(self, player):
        player._shutting_down = False
        player.playback_speed = 0.5
        assert player._mock.speed == 0.5


class TestClipRange:
    """Test clip range (A/B loop) behavior."""

    def test_set_clip_range_sets_ab_loop(self, player):
        player._shutting_down = False
        player.set_clip_range(10.0, 20.0)
        assert player._mock.ab_loop_a == 10.0
        assert player._mock.ab_loop_b == 20.0

    def test_set_clip_range_updates_internal_state(self, player):
        player._shutting_down = False
        player.set_clip_range(10.0, 20.0)
        assert player._clip_start_ms == 10000
        assert player._clip_end_ms == 20000

    def test_set_clip_range_seeks_to_start(self, player):
        player._shutting_down = False
        player.set_clip_range(10.0, 20.0)
        player._mock.seek.assert_called_with(10.0, 'absolute', 'exact')

    def test_clear_clip_range_clears_ab_loop(self, player):
        player._shutting_down = False
        player.set_clip_range(10.0, 20.0)
        player.clear_clip_range()
        assert player._mock.ab_loop_a == 'no'
        assert player._mock.ab_loop_b == 'no'
        assert player._clip_start_ms is None
        assert player._clip_end_ms is None

    def test_play_range_sets_range_and_plays(self, player):
        player._shutting_down = False
        player.play_range(5.0, 15.0)
        assert player._clip_start_ms == 5000
        assert player._clip_end_ms == 15000
        assert player._mock.pause is False


class TestFrameStepping:
    """Test frame step methods."""

    def test_frame_step_forward_pauses_first(self, player):
        player._shutting_down = False
        player._mock.pause = False
        player.frame_step_forward()
        assert player._mock.pause is True
        player._mock.frame_step.assert_called_once()

    def test_frame_step_backward_pauses_first(self, player):
        player._shutting_down = False
        player._mock.pause = False
        player.frame_step_backward()
        assert player._mock.pause is True
        player._mock.frame_back_step.assert_called_once()


class TestMuteProperty:
    """Test mute property."""

    def test_mute_getter(self, player):
        player._mock.mute = True
        assert player.mute is True

    def test_mute_setter(self, player):
        player._shutting_down = False
        player.mute = True
        assert player._mock.mute is True


class TestABLoop:
    """Test manual A/B loop (not clip range)."""

    def test_set_ab_loop_when_no_clip_range(self, player):
        player._shutting_down = False
        player.set_ab_loop(5.0, 25.0)
        assert player._mock.ab_loop_a == 5.0
        assert player._mock.ab_loop_b == 25.0

    def test_set_ab_loop_blocked_by_clip_range(self, player):
        player._shutting_down = False
        player._clip_start_ms = 1000
        # Set initial values
        player._mock.ab_loop_a = 99.0
        player._mock.ab_loop_b = 99.0
        player.set_ab_loop(5.0, 25.0)
        # Should NOT have changed since clip range is active
        assert player._mock.ab_loop_a == 99.0
        assert player._mock.ab_loop_b == 99.0

    def test_clear_ab_loop_when_no_clip_range(self, player):
        player._shutting_down = False
        player.set_ab_loop(5.0, 25.0)
        player.clear_ab_loop()
        assert player._mock.ab_loop_a == 'no'
        assert player._mock.ab_loop_b == 'no'

    def test_clear_ab_loop_blocked_by_clip_range(self, player):
        player._shutting_down = False
        player._clip_start_ms = 1000
        player._mock.ab_loop_a = 10.0
        player.clear_ab_loop()
        # Should NOT have cleared since clip range is active
        assert player._mock.ab_loop_a == 10.0


class TestShutdown:
    """Test shutdown behavior."""

    def test_shutdown_terminates_mpv(self, player):
        player._shutting_down = False
        player.shutdown()
        player._mock.terminate.assert_called_once()

    def test_shutdown_prevents_further_operations(self, player):
        player._shutting_down = False
        player.shutdown()
        # Reset mock to track new calls
        player._mock.reset_mock()
        # These should not call mpv methods
        player.play()
        player.pause()
        player.stop()
        player.seek_to(10.0)
        player.frame_step_forward()
        player.frame_step_backward()
        # No mpv methods should have been called
        player._mock.seek.assert_not_called()
        player._mock.frame_step.assert_not_called()
        player._mock.frame_back_step.assert_not_called()

    def test_double_shutdown_is_safe(self, player):
        player._shutting_down = False
        player.shutdown()
        player.shutdown()  # Should not raise

    def test_is_playing_returns_false_after_shutdown(self, player):
        player._shutting_down = False
        player.shutdown()
        assert player.is_playing is False


class TestSpeedControl:
    """Test speed control widget."""

    def test_set_speed_control_disabled(self, player):
        player._shutting_down = False
        player.set_speed_control_enabled(False)
        assert not player.speed_combo.isEnabled()
        assert player._mock.speed == 1.0

    def test_set_speed_control_re_enabled(self, player):
        player._shutting_down = False
        player.set_speed_control_enabled(False)
        player.set_speed_control_enabled(True)
        assert player.speed_combo.isEnabled()


class TestPositionSignals:
    """Test internal signal routing."""

    def test_position_changed_emits_milliseconds(self, player):
        received = []
        player.position_updated.connect(lambda v: received.append(v))
        player._on_position_changed(10.5)
        assert received == [10500]

    def test_duration_changed_updates_internal_state(self, player):
        player._on_duration_changed(120.0)
        assert player._duration_s == 120.0
        assert player.duration_ms == 120000

    def test_duration_changed_emits_signal(self, player):
        received = []
        player.duration_changed.connect(lambda v: received.append(v))
        player._on_duration_changed(120.0)
        assert received == [120000]

    def test_pause_changed_emits_playback_state(self, player):
        received = []
        player.playback_state_changed.connect(lambda v: received.append(v))
        player._on_pause_changed(False)  # not paused = playing
        assert received == [True]

    def test_file_loaded_emits_media_loaded(self, player):
        received = []
        player.media_loaded.connect(lambda: received.append(True))
        player._on_file_loaded()
        assert received == [True]
