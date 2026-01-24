"""Video downloader using yt-dlp for YouTube/Vimeo support."""

import subprocess
import shutil
import re
import json
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class DownloadResult:
    """Result of a video download."""

    success: bool
    file_path: Optional[Path] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class VideoDownloader:
    """Downloads videos from YouTube, Vimeo, and other supported sites."""

    # Whitelist of allowed domains for security
    ALLOWED_DOMAINS = {
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "vimeo.com",
        "www.vimeo.com",
        "player.vimeo.com",
    }

    def __init__(self, download_dir: Optional[Path] = None):
        self.ytdlp_path = self._find_ytdlp()

        if download_dir is None:
            download_dir = Path.home() / "Movies" / "Scene Ripper Downloads"
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _find_ytdlp(self) -> str:
        """Find yt-dlp executable."""
        path = shutil.which("yt-dlp")
        if path is None:
            raise RuntimeError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )
        return path

    def is_valid_url(self, url: str) -> tuple[bool, str]:
        """
        Check if URL is from an allowed domain with safe scheme.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            # Validate scheme first - only allow HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                return False, "Only HTTP/HTTPS URLs are supported"

            host = parsed.netloc.lower()

            if not host:
                return False, "Invalid URL format"

            # Remove port if present (e.g., youtube.com:443)
            if ":" in host:
                host = host.rsplit(":", 1)[0]

            # Remove credentials if present (e.g., user:pass@youtube.com)
            if "@" in host:
                host = host.rsplit("@", 1)[-1]

            # Check against whitelist
            if host not in self.ALLOWED_DOMAINS:
                return False, f"Domain not supported: {host}. Supported: YouTube, Vimeo"

            return True, ""

        except (ValueError, AttributeError) as e:
            return False, f"URL parsing error: {e}"

    def get_video_info(self, url: str) -> dict:
        """
        Get video metadata without downloading.

        Returns dict with: title, duration, uploader, thumbnail
        """
        valid, error = self.is_valid_url(url)
        if not valid:
            raise ValueError(error)

        cmd = [
            self.ytdlp_path,
            "--no-download",
            "--print-json",
            "--no-playlist",
            "--",  # End of options
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")

        data = json.loads(result.stdout)
        return {
            "title": data.get("title", "Unknown"),
            "duration": data.get("duration", 0),
            "uploader": data.get("uploader", "Unknown"),
            "thumbnail": data.get("thumbnail"),
        }

    def download(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DownloadResult:
        """
        Download a video from URL.

        Args:
            url: Video URL (YouTube or Vimeo)
            progress_callback: Optional callback (progress 0-100, status message)

        Returns:
            DownloadResult with file path if successful
        """
        # Validate URL
        valid, error = self.is_valid_url(url)
        if not valid:
            return DownloadResult(success=False, error=error)

        if progress_callback:
            progress_callback(0, "Getting video info...")

        # Get video info first
        try:
            info = self.get_video_info(url)
            title = info["title"]
        except Exception as e:
            return DownloadResult(success=False, error=str(e))

        if progress_callback:
            progress_callback(5, f"Downloading: {title}")

        # Sanitize filename
        safe_title = self._sanitize_filename(title)
        output_template = str(self.download_dir / f"{safe_title}.%(ext)s")

        # Build download command
        cmd = [
            self.ytdlp_path,
            "--no-playlist",  # Don't download playlists
            "--no-exec",  # Don't run post-processing scripts
            "--max-filesize", "4G",  # Limit file size
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--newline",  # Progress on new lines for parsing
            "--",  # End of options
            url,
        ]

        # Run download with progress parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_file = None
        try:
            for line in process.stdout:
                line = line.strip()

                # Parse progress from yt-dlp output
                if "[download]" in line:
                    # Try to extract percentage
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match and progress_callback:
                        percent = float(match.group(1))
                        # Scale to 5-95% range (leave room for start/end)
                        scaled = 5 + (percent * 0.9)
                        progress_callback(scaled, f"Downloading: {percent:.1f}%")

                    # Check for destination file
                    if "Destination:" in line:
                        dest_match = re.search(r"Destination:\s*(.+)$", line)
                        if dest_match:
                            output_file = Path(dest_match.group(1))

                elif "[Merger]" in line:
                    if progress_callback:
                        progress_callback(95, "Merging audio and video...")

            process.wait()
        finally:
            # Ensure subprocess is cleaned up even on exception
            if process.stdout:
                process.stdout.close()
            if process.poll() is None:  # Process still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        if process.returncode != 0:
            return DownloadResult(
                success=False,
                error="Download failed. Check URL and try again."
            )

        # Find the output file if we didn't catch it
        if output_file is None or not output_file.exists():
            # Look for recently created mp4 files
            mp4_files = list(self.download_dir.glob(f"{safe_title}*.mp4"))
            if mp4_files:
                output_file = max(mp4_files, key=lambda p: p.stat().st_mtime)

        if output_file is None or not output_file.exists():
            return DownloadResult(
                success=False,
                error="Download completed but file not found"
            )

        if progress_callback:
            progress_callback(100, "Download complete!")

        return DownloadResult(
            success=True,
            file_path=output_file,
            title=title,
            duration=info.get("duration"),
        )

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        sanitized = sanitized.strip('. ')
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized or "video"
