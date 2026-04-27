"""Tests for rendered sequence preview playback routing."""

import os
from pathlib import Path
from types import SimpleNamespace

from models.sequence import Sequence, SequenceClip

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _sequence_with_clip() -> Sequence:
    sequence = Sequence(id="seq-1", fps=24.0)
    sequence.tracks[0].clips = [
        SequenceClip(
            source_clip_id="clip-1",
            source_id="src-1",
            start_frame=0,
            in_point=0,
            out_point=24,
        )
    ]
    return sequence


def test_playback_request_uses_ready_rendered_preview():
    from ui.main_window import MainWindow

    started = []
    rendered = []
    sequence = _sequence_with_clip()
    window = SimpleNamespace(
        _is_playing=False,
        sequence_tab=SimpleNamespace(
            timeline=SimpleNamespace(get_sequence=lambda: sequence)
        ),
        _has_ready_sequence_preview=lambda: True,
        _start_rendered_sequence_preview_playback=lambda frame: started.append(frame),
        _start_sequence_preview_render=lambda play_after_frame: rendered.append(play_after_frame),
    )

    MainWindow._on_playback_requested(window, 12)

    assert started == [12]
    assert rendered == []


def test_playback_request_blocks_on_preview_render_when_missing():
    from ui.main_window import MainWindow

    rendered = []
    legacy_calls = []
    sequence = _sequence_with_clip()
    window = SimpleNamespace(
        _is_playing=False,
        sequence_tab=SimpleNamespace(
            timeline=SimpleNamespace(get_sequence=lambda: sequence)
        ),
        _has_ready_sequence_preview=lambda: False,
        _start_sequence_preview_render=lambda play_after_frame: rendered.append(play_after_frame),
        _play_clip_at_frame=lambda frame: legacy_calls.append(frame),
    )

    MainWindow._on_playback_requested(window, 18)

    assert rendered == [18]
    assert legacy_calls == []


def test_stale_preview_render_completion_does_not_activate_or_play(tmp_path):
    from core.sequence_preview import SequencePreviewSettings
    from ui.main_window import MainWindow

    statuses = []
    messages = []
    played = []
    progress_visible = []
    window = SimpleNamespace(
        _sequence_preview_play_after_render_frame=24,
        _get_sequence_preview_cache_entry=lambda: (
            "current-signature",
            tmp_path / "current.mp4",
            SequencePreviewSettings(),
        ),
        progress_bar=SimpleNamespace(
            setVisible=lambda visible: progress_visible.append(visible)
        ),
        sequence_tab=SimpleNamespace(
            set_sequence_preview_status=lambda *args, **kwargs: statuses.append((args, kwargs))
        ),
        status_bar=SimpleNamespace(
            showMessage=lambda *args: messages.append(args)
        ),
        _start_rendered_sequence_preview_playback=lambda frame: played.append(frame),
        _rendered_sequence_preview_path=None,
        _rendered_sequence_preview_signature=None,
        _rendered_sequence_preview_profile="720p proxy",
    )

    MainWindow._on_sequence_preview_finished(
        window,
        Path(tmp_path / "stale.mp4"),
        "stale-signature",
        "720p proxy",
        False,
    )

    assert window._rendered_sequence_preview_path is None
    assert window._rendered_sequence_preview_signature is None
    assert window._sequence_preview_play_after_render_frame is None
    assert played == []
    assert progress_visible == [False]
    assert statuses[0][0][0] == "Stale"


def test_matching_preview_render_completion_activates_and_continues_playback(tmp_path):
    from core.sequence_preview import SequencePreviewSettings
    from ui.main_window import MainWindow

    preview_path = tmp_path / "preview.mp4"
    statuses = []
    played = []
    window = SimpleNamespace(
        _sequence_preview_play_after_render_frame=36,
        _get_sequence_preview_cache_entry=lambda: (
            "matching-signature",
            preview_path,
            SequencePreviewSettings(),
        ),
        progress_bar=SimpleNamespace(setVisible=lambda visible: None),
        sequence_tab=SimpleNamespace(
            set_sequence_preview_status=lambda *args, **kwargs: statuses.append((args, kwargs))
        ),
        status_bar=SimpleNamespace(showMessage=lambda *args: None),
        _start_rendered_sequence_preview_playback=lambda frame: played.append(frame),
        _rendered_sequence_preview_path=None,
        _rendered_sequence_preview_signature=None,
        _rendered_sequence_preview_profile="",
    )

    MainWindow._on_sequence_preview_finished(
        window,
        preview_path,
        "matching-signature",
        "720p proxy",
        False,
    )

    assert window._rendered_sequence_preview_path == preview_path
    assert window._rendered_sequence_preview_signature == "matching-signature"
    assert window._rendered_sequence_preview_profile == "720p proxy"
    assert window._sequence_preview_play_after_render_frame is None
    assert played == [36]
    assert statuses[0][0] == ("Ready", "720p proxy")
