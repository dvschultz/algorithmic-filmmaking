"""On-demand dependency manager for binary tools and Python packages.

Downloads and manages external dependencies that are not bundled in the .app:
- FFmpeg / FFprobe (static arm64 builds)
- yt-dlp (standalone binary)
- Python packages via pip install --target (using standalone Python)

All managed files are stored in ~/Library/Application Support/Scene Ripper/.
"""

import logging
import os
import platform
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from core.paths import get_managed_bin_dir, get_managed_packages_dir, get_managed_python_dir

logger = logging.getLogger(__name__)

# Download URLs for static arm64 macOS binaries
_FFMPEG_URL = "https://www.osxexperts.net/ffmpeg7arm.zip"
_FFPROBE_URL = "https://www.osxexperts.net/ffprobe7arm.zip"
_YTDLP_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"

# Minimum expected sizes to validate downloads (bytes)
_MIN_FFMPEG_SIZE = 10 * 1024 * 1024   # 10 MB
_MIN_FFPROBE_SIZE = 5 * 1024 * 1024   # 5 MB
_MIN_YTDLP_SIZE = 1 * 1024 * 1024     # 1 MB

ProgressCallback = Optional[Callable[[float, str], None]]


def _download_file(
    url: str,
    dest: Path,
    progress_callback: ProgressCallback = None,
    label: str = "",
) -> Path:
    """Download a file from a URL with progress reporting.

    Args:
        url: URL to download from.
        dest: Destination file path.
        progress_callback: Optional (progress_0_to_1, message) callback.
        label: Human-readable label for progress messages.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails or file is too small.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp file first, then move (atomic-ish)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        if progress_callback:
            progress_callback(0.0, f"Downloading {label}...")

        req = urllib.request.Request(url, headers={"User-Agent": "Scene-Ripper/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 64 * 1024  # 64 KB chunks

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback and total > 0:
                        progress_callback(
                            downloaded / total,
                            f"Downloading {label}: {downloaded // (1024 * 1024)} MB"
                            + (f" / {total // (1024 * 1024)} MB" if total else ""),
                        )

        # Move temp file to final location
        os.replace(tmp, dest)

        if progress_callback:
            progress_callback(1.0, f"{label} downloaded")

        logger.info(f"Downloaded {label} to {dest} ({dest.stat().st_size} bytes)")
        return dest

    except Exception as e:
        # Clean up partial download
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {label}: {e}") from e


def _extract_zip_binary(zip_path: Path, binary_name: str, dest_dir: Path) -> Path:
    """Extract a single binary from a ZIP archive.

    Args:
        zip_path: Path to the ZIP file.
        binary_name: Name of the binary to extract.
        dest_dir: Directory to extract to.

    Returns:
        Path to the extracted binary.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the binary in the archive (may be in a subdirectory)
        candidates = [n for n in zf.namelist() if n.endswith(binary_name)]
        if not candidates:
            raise RuntimeError(f"{binary_name} not found in archive")

        target = candidates[0]
        zf.extract(target, dest_dir)

        extracted = dest_dir / target
        final = dest_dir / binary_name

        # Move to dest_dir root if it was in a subdirectory
        if extracted != final:
            os.replace(extracted, final)

    # Make executable
    final.chmod(final.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return final


def ensure_ffmpeg(progress_callback: ProgressCallback = None) -> Path:
    """Ensure FFmpeg is available, downloading if needed.

    Returns:
        Path to the ffmpeg binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    bin_dir = get_managed_bin_dir()
    ffmpeg_path = bin_dir / "ffmpeg"

    if ffmpeg_path.is_file() and os.access(ffmpeg_path, os.X_OK):
        return ffmpeg_path

    if platform.machine() != "arm64":
        raise RuntimeError("Automatic FFmpeg download only supports Apple Silicon (arm64)")

    # Download and extract
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "ffmpeg.zip"
        _download_file(_FFMPEG_URL, zip_path, progress_callback, "FFmpeg")
        _extract_zip_binary(zip_path, "ffmpeg", bin_dir)

    # Validate
    if not ffmpeg_path.is_file() or ffmpeg_path.stat().st_size < _MIN_FFMPEG_SIZE:
        ffmpeg_path.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg download appears corrupt (too small)")

    logger.info(f"FFmpeg installed: {ffmpeg_path}")
    return ffmpeg_path


def ensure_ffprobe(progress_callback: ProgressCallback = None) -> Path:
    """Ensure FFprobe is available, downloading if needed.

    Returns:
        Path to the ffprobe binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    bin_dir = get_managed_bin_dir()
    ffprobe_path = bin_dir / "ffprobe"

    if ffprobe_path.is_file() and os.access(ffprobe_path, os.X_OK):
        return ffprobe_path

    if platform.machine() != "arm64":
        raise RuntimeError("Automatic FFprobe download only supports Apple Silicon (arm64)")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "ffprobe.zip"
        _download_file(_FFPROBE_URL, zip_path, progress_callback, "FFprobe")
        _extract_zip_binary(zip_path, "ffprobe", bin_dir)

    if not ffprobe_path.is_file() or ffprobe_path.stat().st_size < _MIN_FFPROBE_SIZE:
        ffprobe_path.unlink(missing_ok=True)
        raise RuntimeError("FFprobe download appears corrupt (too small)")

    logger.info(f"FFprobe installed: {ffprobe_path}")
    return ffprobe_path


def ensure_yt_dlp(progress_callback: ProgressCallback = None) -> Path:
    """Ensure yt-dlp is available, downloading if needed.

    Returns:
        Path to the yt-dlp binary.

    Raises:
        RuntimeError: If download fails.
    """
    bin_dir = get_managed_bin_dir()
    ytdlp_path = bin_dir / "yt-dlp"

    if ytdlp_path.is_file() and os.access(ytdlp_path, os.X_OK):
        return ytdlp_path

    _download_file(_YTDLP_URL, ytdlp_path, progress_callback, "yt-dlp")

    # Make executable
    ytdlp_path.chmod(ytdlp_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if ytdlp_path.stat().st_size < _MIN_YTDLP_SIZE:
        ytdlp_path.unlink(missing_ok=True)
        raise RuntimeError("yt-dlp download appears corrupt (too small)")

    logger.info(f"yt-dlp installed: {ytdlp_path}")
    return ytdlp_path


def update_yt_dlp(progress_callback: ProgressCallback = None) -> Path:
    """Force re-download of yt-dlp to get the latest version.

    Returns:
        Path to the updated yt-dlp binary.
    """
    bin_dir = get_managed_bin_dir()
    ytdlp_path = bin_dir / "yt-dlp"

    # Remove existing to force re-download
    ytdlp_path.unlink(missing_ok=True)
    return ensure_yt_dlp(progress_callback)


def is_binary_available(name: str) -> bool:
    """Check if a managed binary is available.

    Args:
        name: Binary name (ffmpeg, ffprobe, yt-dlp).

    Returns:
        True if the binary exists and is executable.
    """
    path = get_managed_bin_dir() / name
    return path.is_file() and os.access(path, os.X_OK)


def install_package(
    specifier: str,
    progress_callback: ProgressCallback = None,
) -> bool:
    """Install a Python package into the managed packages directory.

    Uses the standalone managed Python (not the frozen app's Python) to run
    pip install --target.

    Args:
        specifier: pip install specifier (e.g., "torch>=2.4,<2.6").
        progress_callback: Optional progress callback.

    Returns:
        True if installation succeeded.

    Raises:
        RuntimeError: If managed Python is not available.
    """
    python_dir = get_managed_python_dir()
    packages_dir = get_managed_packages_dir()
    packages_dir.mkdir(parents=True, exist_ok=True)

    # Find managed Python interpreter
    python_bin = python_dir / "bin" / "python3"
    if not python_bin.is_file():
        raise RuntimeError(
            "Managed Python not found. Please download Python first via Settings."
        )

    if progress_callback:
        progress_callback(0.0, f"Installing {specifier}...")

    cmd = [
        str(python_bin),
        "-m", "pip", "install",
        "--target", str(packages_dir),
        "--no-user",
        "--disable-pip-version-check",
        specifier,
    ]

    logger.info(f"Installing package: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"pip install failed: {result.stderr}")
        if progress_callback:
            progress_callback(0.0, f"Failed to install {specifier}")
        return False

    if progress_callback:
        progress_callback(1.0, f"Installed {specifier}")

    logger.info(f"Package installed: {specifier}")
    return True


def is_package_available(module_name: str) -> bool:
    """Check if a Python package is importable (from managed packages or bundled).

    Args:
        module_name: Top-level module name (e.g., "torch", "ultralytics").

    Returns:
        True if the module can be imported.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
