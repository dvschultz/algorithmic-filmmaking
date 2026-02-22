"""On-demand dependency manager for binary tools and Python packages.

Downloads and manages external dependencies that are not bundled in the .app:
- FFmpeg / FFprobe (static arm64 builds)
- yt-dlp (standalone binary)
- Python 3.11 standalone (for pip install --target)
- Python packages via pip install --target (using standalone Python)

All managed files are stored in ~/Library/Application Support/Scene Ripper/.
"""

import json
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

# Standalone Python 3.11 (python-build-standalone from Astral/indygreg)
# Self-contained tarball — no .pkg extraction or sudo needed
_PYTHON_VERSION = "3.11.11"
_PYTHON_BUILD_TAG = "20250317"
_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{_PYTHON_BUILD_TAG}/cpython-{_PYTHON_VERSION}+{_PYTHON_BUILD_TAG}-aarch64-apple-darwin-install_only.tar.gz"
)

# Minimum expected sizes to validate downloads (bytes)
_MIN_FFMPEG_SIZE = 10 * 1024 * 1024   # 10 MB
_MIN_FFPROBE_SIZE = 5 * 1024 * 1024   # 5 MB
_MIN_YTDLP_SIZE = 1 * 1024 * 1024     # 1 MB
_MIN_PYTHON_SIZE = 30 * 1024 * 1024   # 30 MB (extracted)

# ABI compatibility marker filename
_COMPAT_MARKER = "compat_version.json"

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


def ensure_python(progress_callback: ProgressCallback = None) -> Path:
    """Ensure a standalone Python interpreter is available for pip operations.

    Downloads python-build-standalone (Astral/indygreg) if not present.
    This Python is used exclusively for `pip install --target` to install
    on-demand packages — it is NOT the app's runtime Python.

    Returns:
        Path to the python3 binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    python_dir = get_managed_python_dir()
    python_bin = python_dir / "bin" / "python3"

    if python_bin.is_file() and os.access(python_bin, os.X_OK):
        return python_bin

    if platform.machine() != "arm64":
        raise RuntimeError("Automatic Python download only supports Apple Silicon (arm64)")

    if progress_callback:
        progress_callback(0.0, "Downloading Python 3.11...")

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "python.tar.gz"
        _download_file(_PYTHON_URL, tarball, progress_callback, "Python 3.11")

        if progress_callback:
            progress_callback(0.9, "Extracting Python 3.11...")

        # Extract tarball — python-build-standalone archives contain a
        # top-level "python/" directory with bin/, lib/, include/, etc.
        import tarfile

        python_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            # Strip the top-level "python/" prefix during extraction
            for member in tf.getmembers():
                # python/bin/python3 -> bin/python3
                parts = member.name.split("/", 1)
                if len(parts) > 1:
                    member.name = parts[1]
                else:
                    continue  # Skip the top-level directory itself
                tf.extract(member, python_dir)

    # Validate
    if not python_bin.is_file():
        raise RuntimeError("Python extraction failed — python3 binary not found")

    # Ensure pip is available
    result = subprocess.run(
        [str(python_bin), "-m", "pip", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        logger.warning(f"pip check failed: {result.stderr}")
        # Try to bootstrap pip
        subprocess.run(
            [str(python_bin), "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=60,
        )

    if progress_callback:
        progress_callback(1.0, "Python 3.11 installed")

    logger.info(f"Standalone Python installed: {python_bin}")
    return python_bin


def get_python_version() -> Optional[str]:
    """Get the version of the managed standalone Python, or None if not installed."""
    python_bin = get_managed_python_dir() / "bin" / "python3"
    if not python_bin.is_file():
        return None
    try:
        result = subprocess.run(
            [str(python_bin), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # "Python 3.11.11" -> "3.11.11"
            return result.stdout.strip().split()[-1]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _load_package_manifest() -> dict:
    """Load the package manifest from the bundled or source location.

    Returns:
        Parsed manifest dict, or empty dict on failure.
    """
    from core.paths import get_resource_path

    manifest_path = get_resource_path("core/package_manifest.json")
    if not manifest_path.exists():
        # Fallback: same directory as this module (source mode)
        manifest_path = Path(__file__).parent / "package_manifest.json"

    if not manifest_path.exists():
        logger.warning("Package manifest not found")
        return {}

    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load package manifest: {e}")
        return {}


def get_pip_specifier(package_name: str) -> Optional[str]:
    """Get the pip install specifier for a package from the manifest.

    Args:
        package_name: Package name as used in FEATURE_DEPS (e.g., "torch").

    Returns:
        pip specifier string (e.g., "torch>=2.0"), or None if not in manifest.
    """
    manifest = _load_package_manifest()
    pkg_info = manifest.get("packages", {}).get(package_name)
    if pkg_info:
        return pkg_info.get("pip_specifier", package_name)
    return package_name  # Fallback: use the name as-is


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
    pip install --target. Automatically downloads Python if not present.

    Args:
        specifier: pip install specifier (e.g., "torch>=2.4,<2.6").
        progress_callback: Optional progress callback.

    Returns:
        True if installation succeeded.

    Raises:
        RuntimeError: If Python download or pip install fails.
    """
    # Ensure standalone Python is available
    python_bin = ensure_python(progress_callback)

    packages_dir = get_managed_packages_dir()
    packages_dir.mkdir(parents=True, exist_ok=True)

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

    # Update ABI compatibility marker after successful install
    _write_compat_marker()

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


# ---------------------------------------------------------------------------
# ABI compatibility tracking
# ---------------------------------------------------------------------------

def _write_compat_marker() -> None:
    """Write a compatibility marker to the packages directory.

    Records the Python version and app version used to install packages,
    so we can detect ABI mismatches after updates.
    """
    packages_dir = get_managed_packages_dir()
    if not packages_dir.exists():
        return

    python_version = get_python_version() or "unknown"
    # App version from the frozen bundle's Info.plist, or fallback
    app_version = os.environ.get("APP_VERSION", "dev")

    marker = {
        "python_version": python_version,
        "app_version": app_version,
        "python_build_tag": _PYTHON_BUILD_TAG,
    }

    marker_path = packages_dir / _COMPAT_MARKER
    try:
        with open(marker_path, "w") as f:
            json.dump(marker, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to write compat marker: {e}")


def check_compat_marker() -> tuple[bool, str]:
    """Check if installed packages are ABI-compatible with current Python.

    Returns:
        Tuple of (compatible, message). If incompatible, message explains why.
    """
    packages_dir = get_managed_packages_dir()
    marker_path = packages_dir / _COMPAT_MARKER

    if not marker_path.exists():
        # No marker = no packages installed yet, or pre-marker install
        return True, ""

    try:
        with open(marker_path) as f:
            marker = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True, ""  # Can't read marker, assume OK

    current_python = get_python_version()
    if current_python is None:
        return True, ""  # No managed Python installed, nothing to check

    marker_python = marker.get("python_version", "")

    # Compare major.minor (3.11 vs 3.12 is an ABI break, 3.11.10 vs 3.11.11 is fine)
    current_minor = ".".join(current_python.split(".")[:2])
    marker_minor = ".".join(marker_python.split(".")[:2])

    if current_minor != marker_minor:
        return False, (
            f"Packages were installed with Python {marker_python} but the current "
            f"managed Python is {current_python}. Re-downloading packages is recommended "
            f"to avoid ABI compatibility issues."
        )

    return True, ""


def clear_packages(progress_callback: ProgressCallback = None) -> None:
    """Remove all managed packages (for ABI reset or disk cleanup).

    Args:
        progress_callback: Optional progress callback.
    """
    import shutil

    packages_dir = get_managed_packages_dir()
    if not packages_dir.exists():
        return

    if progress_callback:
        progress_callback(0.0, "Removing managed packages...")

    shutil.rmtree(packages_dir)
    packages_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(1.0, "Managed packages removed")

    logger.info("Cleared managed packages directory")
