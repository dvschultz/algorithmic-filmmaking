"""Legacy prerender paths cannot bypass the verified sequence preview."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.clip import Source
from models.sequence import Sequence, SequenceClip, Track
from ui.main_window import MainWindow, _resolve_playback_source


def test_plain_playback_ignores_an_existing_legacy_prerender(tmp_path):
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    legacy = tmp_path / "old.mp4"
    legacy.write_bytes(b"a render of different source media")
    entry = SequenceClip(
        in_point=90, out_point=120, source_rate="30", timeline_rate="24",
        prerendered_path=str(legacy),
    )
    assert _resolve_playback_source(entry, source, 12, 24) == (
        source.file_path, 3.0, 4.0, 3.5,
    )


@pytest.mark.parametrize("transform", ["hflip", "vflip", "reverse"])
def test_transformed_legacy_playback_requests_verified_preview(tmp_path, transform):
    legacy = tmp_path / "old.mp4"
    legacy.write_bytes(b"stale render")
    entry = SequenceClip(
        in_point=90, out_point=120, source_rate="30", timeline_rate="30",
        prerendered_path=str(legacy), **{transform: True},
    )
    sequence = Sequence(fps=30, tracks=[Track(clips=[entry])])
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    player = Mock()
    render = Mock()
    window = SimpleNamespace(
        sequence_tab=SimpleNamespace(
            timeline=SimpleNamespace(
                get_sequence=lambda: sequence,
                get_clip_at_playhead=lambda: (entry, None, source),
            ),
            video_player=player,
        ),
        _start_sequence_preview_render=render,
        _update_sequence_chromatic_bar=Mock(),
        _sequence_preview_source_id=None,
        _is_playing=True,
        _playback_timer=Mock(),
        clip_details_sidebar=SimpleNamespace(video_player=Mock()),
    )
    window._pause_playback = lambda: MainWindow._pause_playback(window)
    render.side_effect = lambda **kwargs: not window._is_playing or pytest.fail("playback must pause before rendering")
    MainWindow._play_clip_at_frame(window, 0)
    render.assert_called_once_with(play_after_frame=0)
    player.load_video.assert_not_called()
    player.pause.assert_called_once()
    window._playback_timer.stop.assert_called_once()


@pytest.mark.parametrize("transform", ["hflip", "vflip", "reverse"])
def test_direct_source_resolution_cannot_drop_transforms(tmp_path, transform):
    source = Source(file_path=tmp_path / "source.mp4", fps=30)
    entry = SequenceClip(
        in_point=0, out_point=30, source_rate="30", timeline_rate="30",
        **{transform: True},
    )
    with pytest.raises(ValueError, match="preview"):
        _resolve_playback_source(entry, source, 0, 30)
