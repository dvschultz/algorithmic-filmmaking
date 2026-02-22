"""FFmpeg wrapper for video processing operations."""

import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from core.binary_resolver import find_binary
from core.paths import is_frozen

logger = logging.getLogger(__name__)


class FFmpegProcessor:
    """Handles FFmpeg operations for video processing."""

    def __init__(self):
        self.ffmpeg_path = find_binary("ffmpeg")
        self.ffprobe_path = find_binary("ffprobe")
        self.ffmpeg_available = self.ffmpeg_path is not None
        self.ffprobe_available = self.ffprobe_path is not None

        if not self.ffmpeg_available:
            if is_frozen():
                logger.warning("FFmpeg not found — features requiring FFmpeg will be disabled")
            else:
                raise RuntimeError(
                    "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
                )

        if not self.ffprobe_available:
            if is_frozen():
                logger.warning("FFprobe not found — features requiring FFprobe will be disabled")
            else:
                raise RuntimeError(
                    "FFprobe not found. Please install FFmpeg and ensure it's in your PATH."
                )

    def refresh_binaries(self) -> None:
        """Re-check for FFmpeg/FFprobe after a download completes."""
        self.ffmpeg_path = find_binary("ffmpeg")
        self.ffprobe_path = find_binary("ffprobe")
        self.ffmpeg_available = self.ffmpeg_path is not None
        self.ffprobe_available = self.ffprobe_path is not None

    def extract_clip(
        self,
        input_path: Path,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
        fps: float = 30.0,
        accurate: bool = True,
    ) -> bool:
        """
        Extract a clip from a video file.

        Args:
            input_path: Source video file
            output_path: Output clip file
            start_seconds: Start time in seconds
            duration_seconds: Duration in seconds
            fps: Frame rate (used for frame-accurate cutting)
            accurate: If True, re-encode for frame-accurate cuts (slower but precise)

        Returns:
            True if successful
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate end time, subtract one frame to exclude first frame of next scene
        frame_duration = 1.0 / fps
        end_seconds = start_seconds + duration_seconds - frame_duration

        if accurate:
            # Re-encode for frame-accurate cutting
            # Use -ss before -i for fast seeking, then accurate cut
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(start_seconds),
                "-i", str(input_path),
                "-to", str(duration_seconds - frame_duration),  # Relative to seek point
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",  # High quality
                "-c:a", "aac",
                "-b:a", "192k",
            ]
        else:
            # Stream copy - fast but only keyframe-accurate
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(start_seconds),
                "-i", str(input_path),
                "-to", str(duration_seconds - frame_duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
            ]

        cmd.append(str(output_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for clip extraction
        )

        return result.returncode == 0

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get video metadata using FFprobe.

        Returns dict with: duration, fps, width, height, codec
        """
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        import json
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("No video stream found")

        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "fps": fps,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
        }


def extract_frame(
    video_path: Path,
    frame_number: int,
    output_path: Path,
    fps: float,
) -> bool:
    """Extract a single frame from a video file.

    Args:
        video_path: Path to the video file
        frame_number: The frame number to extract (0-indexed)
        output_path: Path where the frame image will be saved
        fps: Video frame rate (used to calculate timestamp)

    Returns:
        True if extraction was successful, False otherwise
    """
    ffmpeg_path = find_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("FFmpeg not found")

    # Calculate timestamp from frame number
    timestamp = frame_number / fps

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",  # High quality JPEG
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    return result.returncode == 0 and output_path.exists()


# ---------------------------------------------------------------------------
# Batch frame extraction
# ---------------------------------------------------------------------------

_DEFAULT_PNG_SIZE_ESTIMATE = 500 * 1024  # ~500 KB per 1080p PNG frame


def estimate_extraction_size(
    video_path: Path,
    fps: float,
    mode: str,
    interval: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> tuple[int, int]:
    """Estimate disk usage and frame count for a batch extraction.

    Args:
        video_path: Source video file (used only for duration lookup when
            *end_frame* is ``None``).
        fps: Video frame rate.
        mode: Extraction mode – ``"all"``, ``"interval"``, or ``"smart"``.
        interval: Every Nth frame (only meaningful for ``"interval"`` mode).
        start_frame: First frame to extract (0-indexed).
        end_frame: Last frame (exclusive).  ``None`` means until end of video.

    Returns:
        ``(estimated_bytes, frame_count)`` tuple.
    """
    if end_frame is None:
        # Derive total frames from video duration
        ffprobe_path = find_binary("ffprobe")
        if ffprobe_path is None:
            raise RuntimeError("FFprobe not found")
        import json

        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        end_frame = int(duration * fps)

    total_frames_in_range = max(0, end_frame - start_frame)

    if mode == "interval":
        frame_count = max(0, (total_frames_in_range + interval - 1) // interval)
    elif mode == "smart":
        # Heuristic: assume ~1 scene change per 3 seconds of footage
        duration_secs = total_frames_in_range / fps if fps > 0 else 0
        frame_count = max(1, int(duration_secs / 3))
    else:  # "all"
        frame_count = total_frames_in_range

    estimated_bytes = frame_count * _DEFAULT_PNG_SIZE_ESTIMATE
    return estimated_bytes, frame_count


def extract_frames_batch(
    video_path: Path,
    output_dir: Path,
    fps: float,
    mode: str = "all",
    interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[Path]:
    """Extract frames from a video in batch.

    Args:
        video_path: Source video file.
        output_dir: Directory to write extracted PNG frames into.
        fps: Video frame rate.
        mode: ``"all"`` extracts every frame, ``"interval"`` every *interval*-th
            frame, ``"smart"`` extracts frames at scene changes.
        interval: Frame interval for ``"interval"`` mode.
        start_frame: First frame (0-indexed, inclusive).
        end_frame: Last frame (exclusive).  ``None`` means until end of video.
        progress_callback: ``callback(current, total)`` invoked after extraction.
        cancel_event: When set, the extraction is aborted early.

    Returns:
        Sorted list of ``Path`` objects for extracted frames.
    """
    ffmpeg_path = find_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("FFmpeg not found")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Early cancellation check
    if cancel_event is not None and cancel_event.is_set():
        return []

    # Compute estimated total for progress reporting
    _, estimated_total = estimate_extraction_size(
        video_path, fps, mode, interval, start_frame, end_frame,
    )

    # Build the FFmpeg command ------------------------------------------------
    cmd: list[str] = [ffmpeg_path, "-y"]

    # Time-range flags (applied before -i for fast seeking)
    if start_frame > 0:
        cmd.extend(["-ss", str(start_frame / fps)])
    cmd.extend(["-i", str(video_path)])
    if end_frame is not None:
        # Duration relative to the seek point
        duration = (end_frame - start_frame) / fps
        cmd.extend(["-t", str(duration)])

    # Mode-specific filters
    if mode == "interval":
        cmd.extend([
            "-vf", f"select=not(mod(n\\,{interval}))",
            "-vsync", "vfr",
        ])
    elif mode == "smart":
        cmd.extend([
            "-vf", "select='gt(scene\\,0.3)'",
            "-vsync", "vfr",
        ])
    # "all" mode – no filter needed

    output_pattern = str(output_dir / "frame_%06d.png")
    cmd.append(output_pattern)

    # Run FFmpeg --------------------------------------------------------------
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll for cancellation while FFmpeg is running
    while True:
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            process.wait(timeout=10)
            return []
        try:
            process.wait(timeout=0.5)
            break  # Process finished
        except subprocess.TimeoutExpired:
            continue

    if process.returncode != 0:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"FFmpeg frame extraction failed (rc={process.returncode}): {stderr}")

    # Collect results ---------------------------------------------------------
    extracted = sorted(output_dir.glob("frame_*.png"))

    if progress_callback is not None:
        progress_callback(len(extracted), estimated_total)

    return extracted
