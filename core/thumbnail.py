"""Thumbnail generation using FFmpeg."""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from core.binary_resolver import find_binary
from core.paths import is_frozen

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Generates thumbnails from video files."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.ffmpeg_path = find_binary("ffmpeg")
        self.ffmpeg_available = self.ffmpeg_path is not None

        if not self.ffmpeg_available:
            if is_frozen():
                logger.warning("FFmpeg not found — thumbnail generation disabled")
            else:
                raise RuntimeError("FFmpeg not found")

        # Default cache directory
        default_cache = Path.home() / ".cache" / "scene-ripper" / "thumbnails"
        if cache_dir is None:
            cache_dir = default_cache

        # Try to create cache directory, fall back to default if inaccessible
        self.cache_dir = cache_dir
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # Configured cache dir is inaccessible (e.g., unmounted drive)
            # Fall back to default location
            self.cache_dir = default_cache
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
            "-strict", "unofficial",  # Allow non-full-range YUV (required for FFmpeg 8.x)
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout for thumbnail generation
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

    def generate_first_frame(
        self,
        video_path: Path,
        source_id: str,
        width: int = 160,
        height: int = 90,
    ) -> Optional[Path]:
        """
        Generate a thumbnail from the first frame of a video (for library grid).

        Args:
            video_path: Source video file
            source_id: Unique ID for the source (used for caching)
            width: Thumbnail width
            height: Thumbnail height

        Returns:
            Path to generated thumbnail, or None if generation failed
        """
        output_path = self.cache_dir / f"source_{source_id}.jpg"

        if output_path.exists():
            return output_path

        try:
            return self.generate_thumbnail(
                video_path=video_path,
                timestamp_seconds=0.0,  # First frame
                output_path=output_path,
                width=width,
                height=height,
            )
        except Exception:
            return None

    def clear_cache(self):
        """Clear all cached thumbnails."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


def generate_image_thumbnail(
    image_path: Path,
    output_path: Path,
    max_size: tuple[int, int] = (220, 160),
) -> Path:
    """Generate a JPEG thumbnail from an image file.

    Supports PNG, JPG, TIFF, BMP, and WebP input formats.  The thumbnail
    maintains the original aspect ratio and fits within *max_size*.

    Args:
        image_path: Source image file.
        output_path: Destination for the JPEG thumbnail.
        max_size: ``(width, height)`` bounding box.

    Returns:
        *output_path* after writing.
    """
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.LANCZOS)
        # Convert to RGB in case source has alpha (PNG) or palette mode
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(output_path, format="JPEG", quality=85)

    return output_path
