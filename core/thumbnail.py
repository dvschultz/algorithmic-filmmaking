"""Thumbnail generation using FFmpeg."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


class ThumbnailGenerator:
    """Generates thumbnails from video files."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.ffmpeg_path = shutil.which("ffmpeg")
        if self.ffmpeg_path is None:
            raise RuntimeError("FFmpeg not found")

        # Default cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "scene-ripper" / "thumbnails"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_thumbnail(
        self,
        video_path: Path,
        timestamp_seconds: float,
        output_path: Optional[Path] = None,
        width: int = 160,
        height: int = 90,
    ) -> Path:
        """
        Generate a thumbnail at a specific timestamp.

        Args:
            video_path: Source video file
            timestamp_seconds: Time position for thumbnail
            output_path: Where to save thumbnail (auto-generated if None)
            width: Thumbnail width
            height: Thumbnail height

        Returns:
            Path to generated thumbnail
        """
        if output_path is None:
            # Generate cache path based on video and timestamp
            video_hash = str(hash(str(video_path) + str(timestamp_seconds)))[-8:]
            output_path = self.cache_dir / f"thumb_{video_hash}.jpg"

        if output_path.exists():
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite
            "-ss", str(timestamp_seconds),  # Seek to timestamp
            "-i", str(video_path),
            "-vframes", "1",  # Extract 1 frame
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-q:v", "2",  # High quality JPEG
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Thumbnail generation failed: {result.stderr}")

        return output_path

    def generate_clip_thumbnail(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        output_path: Optional[Path] = None,
        width: int = 160,
        height: int = 90,
    ) -> Path:
        """
        Generate a thumbnail for a clip (at 1/3 into the clip).

        Args:
            video_path: Source video file
            start_seconds: Clip start time
            end_seconds: Clip end time
            output_path: Where to save thumbnail
            width: Thumbnail width
            height: Thumbnail height

        Returns:
            Path to generated thumbnail
        """
        # Take thumbnail at 1/3 into the clip for a representative frame
        duration = end_seconds - start_seconds
        timestamp = start_seconds + (duration / 3)

        return self.generate_thumbnail(
            video_path=video_path,
            timestamp_seconds=timestamp,
            output_path=output_path,
            width=width,
            height=height,
        )

    def clear_cache(self):
        """Clear all cached thumbnails."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
