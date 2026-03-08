"""Tests for sequence timeline/source playback position mapping."""

from ui.main_window import _timeline_frame_to_source_seconds, _source_ms_to_timeline_seconds


class _DummySequenceClip:
    def __init__(self, start_frame: int, in_point: int, out_point: int):
        self.start_frame = start_frame
        self.in_point = in_point
        self.out_point = out_point

    def end_frame(self) -> int:
        return self.start_frame + (self.out_point - self.in_point)


def test_timeline_frame_to_source_seconds_with_in_point_offset():
    seq_clip = _DummySequenceClip(
        start_frame=300,   # 10s on timeline at 30fps
        in_point=900,      # 30s in source at 30fps
        out_point=1200,    # 40s in source at 30fps
    )

    # 60 frames into the timeline clip -> 60 frames past in_point in source.
    source_seconds = _timeline_frame_to_source_seconds(seq_clip, timeline_frame=360, source_fps=30.0)
    assert source_seconds == 32.0


def test_source_ms_to_timeline_seconds_with_in_point_offset():
    seq_clip = _DummySequenceClip(
        start_frame=300,   # 12.5s on timeline at 24fps
        in_point=900,      # 30s in source at 30fps
        out_point=1200,    # 40s in source at 30fps
    )

    # 32s source position => 60 frames into the sequence clip => timeline 15.0s at 24fps.
    timeline_seconds = _source_ms_to_timeline_seconds(
        seq_clip,
        position_ms=32000,
        source_fps=30.0,
        timeline_fps=24.0,
    )
    assert timeline_seconds == 15.0
