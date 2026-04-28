"""Integration tests for Cassette Tape's apply path on SequenceTab.

Covers the U3↔U4 contract: the dialog emits 4-tuples
``(Clip, Source, in_frame, out_frame)`` and ``_apply_cassette_tape_sequence``
must place the right number of sub-clips on the timeline at the correct
frame offsets, with the algorithm key recorded on the sequence.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# SequenceTab contains video widgets — use offscreen for headless runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_source(source_id: str = "src-1", fps: float = 30.0):
    from models.clip import Source

    return Source(
        id=source_id,
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=fps,
        width=1920,
        height=1080,
    )


def _make_clip(clip_id: str, source_id: str = "src-1", *, start_frame: int = 0,
               end_frame: int = 900):
    from models.clip import Clip

    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )


class TestApplyCassetteTapeSequence:
    def _make_tab(self, qapp):
        from ui.tabs.sequence_tab import SequenceTab
        tab = SequenceTab()
        source = _make_source(fps=30.0)
        tab._sources = {source.id: source}
        tab.video_player = MagicMock()
        tab.timeline_preview = MagicMock()
        return tab, source

    def test_three_subclips_create_three_timeline_clips_at_correct_offsets(self, qapp):
        tab, source = self._make_tab(qapp)
        clip = _make_clip("c1")

        # Three sub-clips: 0-30, 60-90, 120-150 (all relative to clip.start_frame)
        payload = [
            (clip, source, 0, 30),
            (clip, source, 60, 90),
            (clip, source, 120, 150),
        ]

        tab._apply_cassette_tape_sequence(payload)

        sequence = tab.timeline.get_sequence()
        seq_clips = sequence.get_all_clips()
        assert len(seq_clips) == 3

        # in/out are clip.start_frame + offset (0 + offset here)
        assert seq_clips[0].in_point == 0
        assert seq_clips[0].out_point == 30
        assert seq_clips[1].in_point == 60
        assert seq_clips[1].out_point == 90
        assert seq_clips[2].in_point == 120
        assert seq_clips[2].out_point == 150

        # Cumulative timeline placement: each clip starts where the previous ended.
        # First sub-clip duration = 30, second starts at 30, etc.
        assert seq_clips[0].start_frame == 0
        assert seq_clips[1].start_frame == 30
        assert seq_clips[2].start_frame == 60  # 30 + 30

        assert sequence.algorithm == "cassette_tape"
        assert sequence.allow_repeats is False

    def test_apply_with_empty_payload_is_no_op(self, qapp):
        tab, _ = self._make_tab(qapp)
        # Should not raise
        tab._apply_cassette_tape_sequence([])

    def test_apply_sets_algorithm_dropdown_label(self, qapp):
        tab, source = self._make_tab(qapp)
        clip = _make_clip("c1")
        tab._apply_cassette_tape_sequence([(clip, source, 0, 15)])
        assert tab.algorithm_dropdown.currentText() == "Cassette Tape"
        assert tab._current_algorithm == "cassette_tape"

    def test_apply_with_zero_fps_falls_back_to_30(self, qapp):
        from ui.tabs.sequence_tab import SequenceTab
        tab = SequenceTab()
        # Source with fps=0 (corrupted metadata)
        bad_source = _make_source(fps=0.0)
        tab._sources = {bad_source.id: bad_source}
        tab.video_player = MagicMock()
        tab.timeline_preview = MagicMock()
        clip = _make_clip("c1")
        # Should not raise (would divide-by-zero without the guard)
        tab._apply_cassette_tape_sequence([(clip, bad_source, 0, 15)])
        # Timeline fps was set to the fallback, not 0
        assert tab.timeline.sequence.fps == 30.0
