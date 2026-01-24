"""FFmpeg wrapper for video processing operations."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


class FFmpegProcessor:
    """Handles FFmpeg operations for video processing."""

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable."""
        path = shutil.which("ffmpeg")
        if path is None:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
            )
        return path

    def _find_ffprobe(self) -> str:
        """Find FFprobe executable."""
        path = shutil.which("ffprobe")
        if path is None:
            raise RuntimeError(
                "FFprobe not found. Please install FFmpeg and ensure it's in your PATH."
            )
        return path

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
