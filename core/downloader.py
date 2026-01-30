"""Video downloader using yt-dlp for YouTube/Vimeo support."""

import logging
import os
import subprocess
import shutil
import re
import json
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _get_subprocess_env() -> dict:
    """Get environment for subprocess with paths to find Deno and other tools.

    macOS GUI apps don't inherit shell environment, so we need to explicitly
    add common tool paths (Homebrew, etc.) to ensure yt-dlp can find Deno
    for the JavaScript challenge solver.
    """
    env = os.environ.copy()
    path = env.get("PATH", "")

    # Common paths that might be missing in GUI apps
    additional_paths = [
        "/opt/homebrew/bin",  # Homebrew on Apple Silicon
        "/opt/homebrew/sbin",
        "/usr/local/bin",  # Homebrew on Intel Mac
        "/usr/local/sbin",
        str(Path.home() / ".local" / "bin"),  # User local bin
        str(Path.home() / ".deno" / "bin"),  # Deno default install location
    ]

    # Add any paths not already present
    path_parts = path.split(os.pathsep)
    for p in additional_paths:
        if p not in path_parts:
            path_parts.insert(0, p)

    env["PATH"] = os.pathsep.join(path_parts)
    return env


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

    def get_video_info(self, url: str, include_format_details: bool = False) -> dict:
        """
        Get video metadata without downloading.

        Args:
            url: Video URL
            include_format_details: If True, include width/height/filesize from best format

        Returns dict with: title, duration, uploader, thumbnail
        If include_format_details is True, also includes: width, height, aspect_ratio, filesize_approx
        """
        valid, error = self.is_valid_url(url)
        if not valid:
            raise ValueError(error)

        cmd = [
            self.ytdlp_path,
            "--no-download",
            "--print-json",
            "--no-playlist",
            # Enable remote challenge solver for YouTube n-sig challenges (required for 2026+)
            "--remote-components", "ejs:github",
            "--",  # End of options
            url,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, env=_get_subprocess_env()
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out getting video info (30 seconds)")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")

        data = json.loads(result.stdout)
        info = {
            "title": data.get("title", "Unknown"),
            "duration": data.get("duration", 0),
            "uploader": data.get("uploader", "Unknown"),
            "thumbnail": data.get("thumbnail"),
        }

        if include_format_details:
            # Try to get dimensions from the top-level (merged format) first
            width = data.get("width")
            height = data.get("height")
            filesize = data.get("filesize") or data.get("filesize_approx")

            # If not available at top level, find best video format
            if not width or not height:
                formats = data.get("formats", [])
                # Look for best video format by height
                best_video = None
                for fmt in formats:
                    if fmt.get("vcodec") != "none" and fmt.get("height"):
                        if best_video is None or fmt.get("height", 0) > best_video.get("height", 0):
                            best_video = fmt
                if best_video:
                    width = width or best_video.get("width")
                    height = height or best_video.get("height")
                    if not filesize:
                        filesize = best_video.get("filesize") or best_video.get("filesize_approx")

            info["width"] = width
            info["height"] = height
            info["filesize_approx"] = filesize

            # Calculate aspect ratio
            if width and height and height > 0:
                info["aspect_ratio"] = round(width / height, 2)
            else:
                info["aspect_ratio"] = None

        return info

    def download(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        max_download_seconds: int = 3600,
    ) -> DownloadResult:
        """
        Download a video from URL.

        Args:
            url: Video URL (YouTube or Vimeo)
            progress_callback: Optional callback (progress 0-100, status message)
            cancel_check: Optional callback that returns True to cancel download
            max_download_seconds: Maximum download time before timeout (default 1 hour)

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
            # Enable remote challenge solver for YouTube n-sig challenges (required for 2026+)
            "--remote-components", "ejs:github",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--newline",  # Progress on new lines for parsing
            "--",  # End of options
            url,
        ]

        logger.debug(f"yt-dlp command: {' '.join(cmd)}")

        # Run download with progress parsing
        # Use augmented environment to ensure Deno is findable for challenge solver
        env = _get_subprocess_env()
        logger.debug(f"Subprocess PATH includes /opt/homebrew/bin: {'/opt/homebrew/bin' in env.get('PATH', '')}")

        # Verify Deno is actually findable
        deno_check = subprocess.run(['which', 'deno'], capture_output=True, text=True, env=env)
        logger.debug(f"Deno location: {deno_check.stdout.strip()} (found: {deno_check.returncode == 0})")
        logger.debug(f"Full PATH being used: {env.get('PATH', 'NOT SET')[:300]}...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        output_file = None
        cancelled = False
        timed_out = False
        start_time = time.time()
        last_error = None  # Capture yt-dlp error messages
        recent_lines = []  # Keep recent output for debugging
        deno_solver_ran = False  # Track if challenge solver was invoked

        try:
            for line in process.stdout:
                # Check for cancellation
                if cancel_check and cancel_check():
                    cancelled = True
                    break

                # Check for timeout
                if time.time() - start_time > max_download_seconds:
                    timed_out = True
                    break

                line = line.strip()

                # Keep last 30 lines for error reporting (increased for debugging)
                recent_lines.append(line)
                if len(recent_lines) > 30:
                    recent_lines.pop(0)

                # Log challenge solver and JS runtime status for debugging
                if "[jsc:deno]" in line:
                    deno_solver_ran = True
                    logger.info(f"yt-dlp challenge solver: {line}")
                elif "JS runtime" in line or "[jsc]" in line:
                    logger.info(f"yt-dlp JS runtime: {line}")

                # Capture error messages from yt-dlp
                if "ERROR:" in line:
                    last_error = line

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

            if not cancelled and not timed_out:
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

        if cancelled:
            # Clean up partial download
            if output_file and output_file.exists():
                try:
                    output_file.unlink()
                except OSError:
                    pass
            return DownloadResult(success=False, error="Download cancelled")

        if timed_out:
            # Clean up partial download
            if output_file and output_file.exists():
                try:
                    output_file.unlink()
                except OSError:
                    pass
            return DownloadResult(success=False, error="Download timed out")

        if process.returncode != 0:
            # Use captured error message if available, otherwise show recent output
            logger.error(f"yt-dlp failed with code {process.returncode}")
            logger.error(f"Recent output: {recent_lines}")
            logger.error(f"Deno challenge solver ran: {deno_solver_ran}")

            # Check for specific known issues
            recent_text = " ".join(recent_lines)

            # JavaScript runtime missing (yt-dlp 2025+ requirement)
            if "No supported JavaScript runtime" in recent_text:
                error_msg = (
                    "yt-dlp requires a JavaScript runtime for YouTube downloads. "
                    "Install Deno with: brew install deno (macOS) or see "
                    "https://github.com/yt-dlp/yt-dlp/wiki/EJS"
                )
            # 403 with JS runtime issue
            elif "HTTP Error 403: Forbidden" in recent_text and "JavaScript runtime" in recent_text:
                error_msg = (
                    "YouTube download blocked (403). This is usually caused by missing "
                    "JavaScript runtime. Install Deno with: brew install deno"
                )
            # SABR streaming 403 errors (GitHub issue #12482)
            elif "HTTP Error 403: Forbidden" in recent_text and "SABR" in recent_text:
                if deno_solver_ran:
                    error_msg = (
                        "YouTube is blocking this download despite challenge solver running. "
                        "This may be YouTube rate limiting. Try again in a few minutes or "
                        "try a different video."
                    )
                else:
                    error_msg = (
                        "YouTube is blocking this download (SABR streaming restriction). "
                        "Try updating yt-dlp: pip install -U yt-dlp"
                    )
            # Generic 403 - could be geo-restriction, age-restriction, or other
            elif "HTTP Error 403: Forbidden" in recent_text:
                error_msg = (
                    "YouTube blocked the download (403 Forbidden). This may be due to "
                    "geo-restrictions, age-restrictions, or YouTube rate limiting. "
                    "Try again later or try a different video."
                )
            elif last_error:
                error_msg = last_error
            elif recent_lines:
                # Show last few lines that might contain error info
                error_msg = f"Download failed: {'; '.join(recent_lines[-3:])}"
            else:
                error_msg = "Download failed. Check URL and try again."
            return DownloadResult(
                success=False,
                error=error_msg
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
        # Remove control characters (0x00-0x1F)
        sanitized = ''.join(c for c in name if c.isprintable())

        # Remove filesystem-unsafe characters AND % (yt-dlp template char)
        sanitized = re.sub(r'[<>:"/\\|?*%]', '', sanitized)
        sanitized = sanitized.strip('. ')

        # Check for Windows reserved names
        RESERVED = {'CON', 'PRN', 'AUX', 'NUL',
                    'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                    'COM6', 'COM7', 'COM8', 'COM9',
                    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5',
                    'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        if sanitized.upper() in RESERVED:
            sanitized = f"video_{sanitized}"

        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]

        return sanitized or "video"
