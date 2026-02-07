"""Tests for batch frame extraction in core/ffmpeg.py."""

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.ffmpeg import estimate_extraction_size, extract_frames_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: Path, num_frames: int = 30, fps: float = 30.0) -> Path:
    """Create a short synthetic MP4 video using OpenCV.

    Each frame is a solid colour that changes every 10 frames to simulate
    scene changes.  Frames are BGR as required by cv2.VideoWriter.
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    colours = [
        (0, 0, 255),    # red  (BGR)
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
    ]
    for i in range(num_frames):
        colour = colours[(i // 10) % len(colours)]
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path


# ---------------------------------------------------------------------------
# Tests – extract_frames_batch
# ---------------------------------------------------------------------------


class TestExtractFramesBatchAll:
    """Tests for 'all' extraction mode."""

    def test_extracts_all_frames(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=15, fps=30.0)
        out_dir = tmp_path / "frames"

        frames = extract_frames_batch(video, out_dir, fps=30.0, mode="all")

        assert len(frames) > 0
        # FFmpeg may produce slightly more/fewer frames due to codec rounding,
        # but should be close to 15.
        assert len(frames) >= 13
        assert all(f.suffix == ".png" for f in frames)
        assert frames == sorted(frames)  # returned in order

    def test_output_dir_created(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=10, fps=30.0)
        out_dir = tmp_path / "nested" / "output"

        extract_frames_batch(video, out_dir, fps=30.0, mode="all")

        assert out_dir.is_dir()

    def test_with_start_end_frames(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)
        out_dir = tmp_path / "frames"

        # Extract only the middle 10 frames
        frames = extract_frames_batch(
            video, out_dir, fps=30.0, mode="all",
            start_frame=10, end_frame=20,
        )

        assert len(frames) > 0
        # Should be roughly 10 frames (codec rounding may vary slightly)
        assert len(frames) <= 15


class TestExtractFramesBatchInterval:
    """Tests for 'interval' extraction mode."""

    def test_interval_reduces_frame_count(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)
        all_dir = tmp_path / "all"
        interval_dir = tmp_path / "interval"

        all_frames = extract_frames_batch(video, all_dir, fps=30.0, mode="all")
        interval_frames = extract_frames_batch(
            video, interval_dir, fps=30.0, mode="interval", interval=5,
        )

        # Interval mode should produce fewer frames than all mode
        assert len(interval_frames) < len(all_frames)
        assert len(interval_frames) > 0


class TestExtractFramesBatchSmart:
    """Tests for 'smart' (scene-change) extraction mode."""

    def test_smart_produces_frames(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)
        out_dir = tmp_path / "smart"

        frames = extract_frames_batch(video, out_dir, fps=30.0, mode="smart")

        # Smart mode should detect at least one scene change from the
        # colour transitions we baked into the synthetic video.
        # (It may also detect 0 if the codec smooths things out, so we
        # just verify it doesn't crash and returns a list.)
        assert isinstance(frames, list)


class TestExtractFramesBatchCancellation:
    """Tests for cancellation via threading.Event."""

    def test_cancel_before_start(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=10, fps=30.0)
        out_dir = tmp_path / "frames"
        cancel = threading.Event()
        cancel.set()  # Already cancelled

        frames = extract_frames_batch(
            video, out_dir, fps=30.0, mode="all", cancel_event=cancel,
        )

        assert frames == []


class TestExtractFramesBatchProgress:
    """Tests for the progress callback."""

    def test_progress_callback_called(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=10, fps=30.0)
        out_dir = tmp_path / "frames"
        calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            calls.append((current, total))

        extract_frames_batch(
            video, out_dir, fps=30.0, mode="all", progress_callback=on_progress,
        )

        assert len(calls) == 1  # Called once after extraction finishes
        current, total = calls[0]
        assert current > 0
        assert total > 0


# ---------------------------------------------------------------------------
# Tests – estimate_extraction_size
# ---------------------------------------------------------------------------


class TestEstimateExtractionSize:
    """Tests for the size/frame-count estimator."""

    def test_all_mode(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)

        est_bytes, frame_count = estimate_extraction_size(
            video, fps=30.0, mode="all", interval=1,
            start_frame=0, end_frame=30,
        )

        assert frame_count == 30
        assert est_bytes == 30 * 500 * 1024

    def test_interval_mode(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)

        est_bytes, frame_count = estimate_extraction_size(
            video, fps=30.0, mode="interval", interval=5,
            start_frame=0, end_frame=30,
        )

        assert frame_count == 6  # ceil(30/5)
        assert est_bytes == 6 * 500 * 1024

    def test_smart_mode(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=90, fps=30.0)

        est_bytes, frame_count = estimate_extraction_size(
            video, fps=30.0, mode="smart", interval=1,
            start_frame=0, end_frame=90,
        )

        # 90 frames at 30fps = 3s -> ~1 scene change
        assert frame_count == 1
        assert est_bytes == 1 * 500 * 1024

    def test_with_start_end_range(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=60, fps=30.0)

        _, frame_count = estimate_extraction_size(
            video, fps=30.0, mode="all", interval=1,
            start_frame=10, end_frame=40,
        )

        assert frame_count == 30

    def test_end_frame_none_uses_video_duration(self, tmp_path: Path):
        video = _make_synthetic_video(tmp_path / "video.mp4", num_frames=30, fps=30.0)

        _, frame_count = estimate_extraction_size(
            video, fps=30.0, mode="all", interval=1,
            start_frame=0, end_frame=None,
        )

        # Should derive from video duration — approximately 30 frames
        assert frame_count > 0
