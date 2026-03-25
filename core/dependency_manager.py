"""On-demand dependency manager for binary tools and Python packages.

Downloads and manages external dependencies that are not bundled in the app:
- FFmpeg / FFprobe (platform-specific static builds)
- yt-dlp (standalone binary)
- Deno (JavaScript runtime required by yt-dlp for modern YouTube extraction)
- Python 3.11 standalone (for pip install --target)
- Python packages via pip install --target (using standalone Python)

Managed files are stored in the platform-appropriate app support directory:
- macOS: ~/Library/Application Support/Scene Ripper/
- Windows: %LOCALAPPDATA%/Scene Ripper/
- Linux: ~/.local/share/scene-ripper/
"""

import hashlib
import importlib
import json
import logging
import os
import platform
import re
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

try:
    import certifi
except ImportError:  # pragma: no cover - optional during local development
    certifi = None

from core.binary_resolver import get_subprocess_kwargs
from core.paths import get_managed_bin_dir, get_managed_packages_dir, get_managed_python_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-specific download URLs
# ---------------------------------------------------------------------------

# macOS ARM64 binaries
_FFMPEG_URL_MACOS = "https://www.osxexperts.net/ffmpeg7arm.zip"
_FFPROBE_URL_MACOS = "https://www.osxexperts.net/ffprobe7arm.zip"
_YTDLP_URL_MACOS = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
_DENO_URL_MACOS = "https://github.com/denoland/deno/releases/latest/download/deno-aarch64-apple-darwin.zip"

# Windows x64 binaries
_FFMPEG_URL_WINDOWS = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
_YTDLP_URL_WINDOWS = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
_DENO_URL_WINDOWS = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip"

# Linux binaries (BtbN static builds)
_FFMPEG_URL_LINUX_X64 = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
_FFMPEG_URL_LINUX_ARM64 = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
_YTDLP_URL_LINUX_X64 = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux"
_YTDLP_URL_LINUX_ARM64 = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64"
_DENO_URL_LINUX_X64 = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip"
_DENO_URL_LINUX_ARM64 = "https://github.com/denoland/deno/releases/latest/download/deno-aarch64-unknown-linux-gnu.zip"


def _get_ffmpeg_url() -> str:
    """Get platform-appropriate FFmpeg download URL."""
    if sys.platform == "win32":
        return _FFMPEG_URL_WINDOWS
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _FFMPEG_URL_MACOS
    if sys.platform == "linux":
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return _FFMPEG_URL_LINUX_X64
        if machine == "aarch64":
            return _FFMPEG_URL_LINUX_ARM64
    raise RuntimeError(
        f"Automatic FFmpeg download not available for {sys.platform}/{platform.machine()}"
    )


def _get_ffprobe_url() -> str:
    """Get platform-appropriate FFprobe download URL."""
    if sys.platform == "win32":
        # BtbN Windows builds include both ffmpeg and ffprobe in the same zip
        return _FFMPEG_URL_WINDOWS
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _FFPROBE_URL_MACOS
    if sys.platform == "linux":
        # BtbN Linux builds include both ffmpeg and ffprobe in the same archive
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return _FFMPEG_URL_LINUX_X64
        if machine == "aarch64":
            return _FFMPEG_URL_LINUX_ARM64
    raise RuntimeError(
        f"Automatic FFprobe download not available for {sys.platform}/{platform.machine()}"
    )


def _get_ytdlp_url() -> str:
    """Get platform-appropriate yt-dlp download URL."""
    if sys.platform == "win32":
        return _YTDLP_URL_WINDOWS
    if sys.platform == "darwin":
        return _YTDLP_URL_MACOS
    if sys.platform == "linux":
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return _YTDLP_URL_LINUX_X64
        if machine == "aarch64":
            return _YTDLP_URL_LINUX_ARM64
    raise RuntimeError(
        f"Automatic yt-dlp download not available for {sys.platform}/{platform.machine()}"
    )


def _get_deno_url() -> str:
    """Get platform-appropriate Deno download URL."""
    if sys.platform == "win32":
        return _DENO_URL_WINDOWS
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _DENO_URL_MACOS
    if sys.platform == "linux":
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return _DENO_URL_LINUX_X64
        if machine == "aarch64":
            return _DENO_URL_LINUX_ARM64
    raise RuntimeError(
        f"Automatic Deno download not available for {sys.platform}/{platform.machine()}"
    )


def _get_binary_ext() -> str:
    """Get the platform-appropriate binary file extension."""
    return ".exe" if sys.platform == "win32" else ""


# Standalone Python 3.11 (python-build-standalone from Astral/indygreg)
# Self-contained tarball — no .pkg extraction or sudo needed
_PYTHON_VERSION = "3.11.11"
_PYTHON_BUILD_TAG = "20250317"
_PYTHON_BASE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{_PYTHON_BUILD_TAG}/cpython-{_PYTHON_VERSION}+{_PYTHON_BUILD_TAG}"
)


def _get_python_url() -> str:
    """Get platform-appropriate standalone Python download URL."""
    if sys.platform == "win32":
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return f"{_PYTHON_BASE_URL}-x86_64-pc-windows-msvc-install_only.tar.gz"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return f"{_PYTHON_BASE_URL}-aarch64-apple-darwin-install_only.tar.gz"
    if sys.platform == "linux":
        machine = platform.machine()
        if machine in ("x86_64", "AMD64"):
            return f"{_PYTHON_BASE_URL}-x86_64-unknown-linux-gnu-install_only.tar.gz"
        if machine == "aarch64":
            return f"{_PYTHON_BASE_URL}-aarch64-unknown-linux-gnu-install_only.tar.gz"
    raise RuntimeError(
        f"Automatic Python download not available for {sys.platform}/{platform.machine()}"
    )

# Minimum expected sizes to validate downloads (bytes)
_MIN_FFMPEG_SIZE = 10 * 1024 * 1024   # 10 MB
_MIN_FFPROBE_SIZE = 5 * 1024 * 1024   # 5 MB
_MIN_YTDLP_SIZE = 1 * 1024 * 1024     # 1 MB
_MIN_DENO_SIZE = 10 * 1024 * 1024     # 10 MB
_MIN_PYTHON_SIZE = 30 * 1024 * 1024   # 30 MB (extracted)

# SHA-256 checksums for download integrity verification.
# Update these when changing download URLs or versions.
# Set to None to skip verification (e.g., for yt-dlp /latest which changes).
_CHECKSUMS: dict[str, str | None] = {
    _FFMPEG_URL_MACOS: None,        # TODO: pin after verifying first download
    _FFPROBE_URL_MACOS: None,       # TODO: pin after verifying first download
    _YTDLP_URL_MACOS: None,         # /latest URL changes with each release
    _DENO_URL_MACOS: None,          # /latest URL changes with each release
    _FFMPEG_URL_WINDOWS: None,      # /latest URL changes
    _YTDLP_URL_WINDOWS: None,       # /latest URL changes
    _DENO_URL_WINDOWS: None,        # /latest URL changes
    _FFMPEG_URL_LINUX_X64: None,    # /latest URL changes
    _FFMPEG_URL_LINUX_ARM64: None,  # /latest URL changes
    _YTDLP_URL_LINUX_X64: None,    # /latest URL changes
    _YTDLP_URL_LINUX_ARM64: None,  # /latest URL changes
    _DENO_URL_LINUX_X64: None,      # /latest URL changes
    _DENO_URL_LINUX_ARM64: None,    # /latest URL changes
}

# ABI compatibility marker filename
_COMPAT_MARKER = "compat_version.json"

ProgressCallback = Optional[Callable[[float, str], None]]

_PIP_PROGRESS_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^(Collecting|Obtaining)\s+", re.IGNORECASE), 0.1),
    (re.compile(r"^(Using cached|Downloading)\s+", re.IGNORECASE), 0.35),
    (re.compile(r"^(Installing build dependencies|Getting requirements to build wheel|Preparing metadata)", re.IGNORECASE), 0.5),
    (re.compile(r"^Building wheel", re.IGNORECASE), 0.65),
    (re.compile(r"^Installing collected packages", re.IGNORECASE), 0.8),
    (re.compile(r"^Successfully installed", re.IGNORECASE), 0.95),
)


def _emit_progress(progress_callback: ProgressCallback, progress: float, message: str) -> None:
    """Send a bounded progress update if a callback was provided."""
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, progress)), message)


def _scaled_progress_callback(
    progress_callback: ProgressCallback,
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    """Map nested progress callbacks into a subrange of the overall progress bar."""
    span = max(0.0, end - start)

    def _callback(progress: float, message: str) -> None:
        clamped = max(0.0, min(1.0, progress))
        _emit_progress(progress_callback, start + (span * clamped), message)

    return _callback


def _pip_progress_from_output_line(line: str, specifier: str) -> tuple[float, str] | None:
    """Translate pip output into coarse but visible progress updates."""
    text = (line or "").strip()
    if not text:
        return None

    normalized_specifier = specifier.strip()
    for pattern, progress in _PIP_PROGRESS_PATTERNS:
        if pattern.search(text):
            return progress, text

    if normalized_specifier and normalized_specifier.lower() in text.lower():
        return 0.25, text

    return None


def _ensure_managed_packages_importable() -> Path:
    """Ensure the managed packages directory exists and is importable now."""
    packages_dir = get_managed_packages_dir()
    packages_dir.mkdir(parents=True, exist_ok=True)

    packages_str = str(packages_dir)
    if packages_str not in sys.path:
        sys.path.append(packages_str)

    importlib.invalidate_caches()
    return packages_dir


def _specifier_import_roots(specifiers: list[str]) -> list[str]:
    """Extract top-level import roots from pip specifiers."""
    roots: list[str] = []
    seen: set[str] = set()

    for specifier in specifiers:
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)", specifier)
        if not match:
            continue
        root = match.group(1).replace("-", "_").strip()
        if not root or root in seen:
            continue
        seen.add(root)
        roots.append(root)

    return roots


def _reset_imported_package_roots(package_roots: list[str]) -> None:
    """Drop cached modules for package roots that were just installed."""
    if not package_roots:
        importlib.invalidate_caches()
        return

    for module_name in list(sys.modules):
        for root in package_roots:
            if module_name == root or module_name.startswith(f"{root}."):
                sys.modules.pop(module_name, None)
                break

    importlib.invalidate_caches()


def clear_package_roots(package_names: list[str]) -> None:
    """Remove installed package roots and matching dist-info from managed packages."""
    import shutil

    packages_dir = get_managed_packages_dir()
    if not packages_dir.exists():
        return

    normalized = {name.replace("-", "_").lower() for name in package_names}
    for child in packages_dir.iterdir():
        child_name = child.name.lower()
        if child_name.endswith((".dist-info", ".data")):
            base_name = child_name.split("-", 1)[0].replace("-", "_")
        else:
            base_name = child.stem.replace("-", "_")

        if base_name not in normalized:
            continue

        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

    _reset_imported_package_roots(sorted(normalized))


def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA-256 checksum of a downloaded file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_hash:
        logger.error(f"Checksum mismatch for {file_path.name}: expected {expected_hash}, got {actual}")
        return False
    return True


def _get_ssl_context() -> ssl.SSLContext:
    """Return an SSL context for HTTPS downloads.

    Frozen Windows builds cannot rely on OpenSSL's default CA search paths
    being present inside the packaged app. Prefer certifi's bundled CA store
    when available so runtime downloads keep working after packaging.
    """
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _get_external_subprocess_env() -> dict[str, str]:
    """Return a sanitized environment for external helper programs.

    In frozen builds, external programs should not inherit PyInstaller-specific
    Python environment variables or PATH entries anchored in the bundled app.
    """
    env = os.environ.copy()

    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "__PYVENV_LAUNCHER__",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)

    meipass = getattr(sys, "_MEIPASS", None)
    raw_path = env.get("PATH", "")
    if meipass and raw_path:
        try:
            resolved_meipass = Path(meipass).resolve(strict=False)
            filtered_parts = []
            for part in raw_path.split(os.pathsep):
                if not part:
                    continue
                try:
                    resolved_part = Path(part).resolve(strict=False)
                except OSError:
                    filtered_parts.append(part)
                    continue
                if resolved_part == resolved_meipass or resolved_part.is_relative_to(resolved_meipass):
                    continue
                filtered_parts.append(part)
            env["PATH"] = os.pathsep.join(filtered_parts)
        except OSError:
            pass

    return env


@contextmanager
def _external_subprocess_runtime():
    """Temporarily disable PyInstaller's DLL search override for child processes."""
    restore_dir: str | None = None
    set_dll_directory = None

    if sys.platform == "win32" and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        try:
            import ctypes

            restore_dir = str(Path(sys._MEIPASS).resolve(strict=False))
            set_dll_directory = ctypes.windll.kernel32.SetDllDirectoryW
            set_dll_directory(None)
        except Exception:
            logger.debug("Failed to sanitize Windows DLL search path for external subprocess", exc_info=True)
            set_dll_directory = None
            restore_dir = None

    try:
        yield
    finally:
        if set_dll_directory is not None and restore_dir is not None:
            try:
                set_dll_directory(restore_dir)
            except Exception:
                logger.debug("Failed to restore Windows DLL search path after external subprocess", exc_info=True)


def _run_external_subprocess(cmd: list[str], **kwargs):
    """Run an external helper process with a sanitized frozen-app environment."""
    kwargs.setdefault("env", _get_external_subprocess_env())
    with _external_subprocess_runtime():
        return subprocess.run(cmd, **kwargs)


def _popen_external_subprocess(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Launch an external helper process with a sanitized frozen-app environment."""
    kwargs.setdefault("env", _get_external_subprocess_env())
    with _external_subprocess_runtime():
        return subprocess.Popen(cmd, **kwargs)


def _download_file(
    url: str,
    dest: Path,
    progress_callback: ProgressCallback = None,
    label: str = "",
) -> Path:
    """Download a file from a URL with progress reporting and optional hash verification.

    Args:
        url: URL to download from.
        dest: Destination file path.
        progress_callback: Optional (progress_0_to_1, message) callback.
        label: Human-readable label for progress messages.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails, file is too small, or checksum mismatches.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp file first, then move (atomic-ish)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        if progress_callback:
            progress_callback(0.0, f"Downloading {label}...")

        req = urllib.request.Request(url, headers={"User-Agent": "Scene-Ripper/1.0"})
        urlopen_kwargs: dict[str, object] = {"timeout": 120}
        if url.lower().startswith("https://"):
            urlopen_kwargs["context"] = _get_ssl_context()

        with urllib.request.urlopen(req, **urlopen_kwargs) as response:
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

        # Verify checksum if one is configured for this URL
        expected_hash = _CHECKSUMS.get(url)
        if expected_hash is not None:
            if not _verify_sha256(dest, expected_hash):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum verification failed for {label}")

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

    # Make executable (no-op on Windows)
    if sys.platform != "win32":
        final.chmod(final.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return final


def _extract_tar_binary(tar_path: Path, binary_name: str, dest_dir: Path) -> Path:
    """Extract a single binary from a tar archive (.tar.xz, .tar.gz, etc.).

    Args:
        tar_path: Path to the tar archive.
        binary_name: Name of the binary to extract (e.g., "ffmpeg").
        dest_dir: Directory to extract to.

    Returns:
        Path to the extracted binary.
    """
    import tarfile

    # Determine open mode from suffix
    name = tar_path.name
    if name.endswith(".tar.xz"):
        mode = "r:xz"
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        mode = "r:gz"
    elif name.endswith(".tar.bz2"):
        mode = "r:bz2"
    else:
        mode = "r:*"

    resolved_dest = dest_dir.resolve()

    with tarfile.open(tar_path, mode) as tf:
        # Find the binary in the archive (may be in a subdirectory)
        candidates = [m for m in tf.getmembers() if m.name.endswith("/" + binary_name) or m.name == binary_name]
        if not candidates:
            raise RuntimeError(f"{binary_name} not found in archive")

        target = candidates[0]

        # Path traversal protection
        target_path = (resolved_dest / target.name).resolve()
        if not target_path.is_relative_to(resolved_dest):
            raise RuntimeError(f"Suspicious tar member path: {target.name}")

        tf.extract(target, dest_dir)

        extracted = dest_dir / target.name
        final = dest_dir / binary_name

        # Move to dest_dir root if it was in a subdirectory
        if extracted != final:
            os.replace(extracted, final)

    # Make executable
    final.chmod(final.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return final


def _extract_binary(archive_path: Path, binary_name: str, dest_dir: Path) -> Path:
    """Extract a binary from an archive, dispatching to zip or tar extractor.

    Args:
        archive_path: Path to the archive file.
        binary_name: Name of the binary to extract.
        dest_dir: Directory to extract to.

    Returns:
        Path to the extracted binary.
    """
    name = archive_path.name
    if name.endswith(".zip"):
        return _extract_zip_binary(archive_path, binary_name, dest_dir)
    return _extract_tar_binary(archive_path, binary_name, dest_dir)


def ensure_ffmpeg(progress_callback: ProgressCallback = None) -> Path:
    """Ensure FFmpeg is available, downloading if needed.

    Returns:
        Path to the ffmpeg binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    bin_dir = get_managed_bin_dir()
    ext = _get_binary_ext()
    ffmpeg_path = bin_dir / f"ffmpeg{ext}"

    if ffmpeg_path.is_file():
        return ffmpeg_path

    url = _get_ffmpeg_url()

    # Derive archive extension from URL (.tar.xz for Linux, .zip for macOS/Windows)
    if ".tar.xz" in url:
        archive_ext = ".tar.xz"
    elif ".tar.gz" in url:
        archive_ext = ".tar.gz"
    else:
        archive_ext = ".zip"

    # Download and extract
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / f"ffmpeg{archive_ext}"
        _download_file(url, archive_path, _scaled_progress_callback(progress_callback, 0.0, 0.85), "FFmpeg")
        _emit_progress(progress_callback, 0.9, "Extracting FFmpeg...")
        _extract_binary(archive_path, f"ffmpeg{ext}", bin_dir)

    # Validate
    _emit_progress(progress_callback, 0.97, "Validating FFmpeg...")
    if not ffmpeg_path.is_file() or ffmpeg_path.stat().st_size < _MIN_FFMPEG_SIZE:
        ffmpeg_path.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg download appears corrupt (too small)")

    _emit_progress(progress_callback, 1.0, "FFmpeg installed")
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
    ext = _get_binary_ext()
    ffprobe_path = bin_dir / f"ffprobe{ext}"

    if ffprobe_path.is_file():
        return ffprobe_path

    url = _get_ffprobe_url()

    if ".tar.xz" in url:
        archive_ext = ".tar.xz"
    elif ".tar.gz" in url:
        archive_ext = ".tar.gz"
    else:
        archive_ext = ".zip"

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / f"ffprobe{archive_ext}"
        _download_file(url, archive_path, _scaled_progress_callback(progress_callback, 0.0, 0.85), "FFprobe")
        _emit_progress(progress_callback, 0.9, "Extracting FFprobe...")
        _extract_binary(archive_path, f"ffprobe{ext}", bin_dir)

    _emit_progress(progress_callback, 0.97, "Validating FFprobe...")
    if not ffprobe_path.is_file() or ffprobe_path.stat().st_size < _MIN_FFPROBE_SIZE:
        ffprobe_path.unlink(missing_ok=True)
        raise RuntimeError("FFprobe download appears corrupt (too small)")

    _emit_progress(progress_callback, 1.0, "FFprobe installed")
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
    ext = _get_binary_ext()
    ytdlp_path = bin_dir / f"yt-dlp{ext}"

    if ytdlp_path.is_file():
        return ytdlp_path

    url = _get_ytdlp_url()
    _download_file(url, ytdlp_path, _scaled_progress_callback(progress_callback, 0.0, 0.9), "yt-dlp")

    # Make executable (no-op on Windows)
    if sys.platform != "win32":
        ytdlp_path.chmod(ytdlp_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _emit_progress(progress_callback, 0.97, "Validating yt-dlp...")
    if ytdlp_path.stat().st_size < _MIN_YTDLP_SIZE:
        ytdlp_path.unlink(missing_ok=True)
        raise RuntimeError("yt-dlp download appears corrupt (too small)")

    _emit_progress(progress_callback, 1.0, "yt-dlp installed")
    logger.info(f"yt-dlp installed: {ytdlp_path}")
    return ytdlp_path


def ensure_deno(progress_callback: ProgressCallback = None) -> Path:
    """Ensure Deno is available, downloading if needed.

    Returns:
        Path to the Deno binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    bin_dir = get_managed_bin_dir()
    ext = _get_binary_ext()
    deno_path = bin_dir / f"deno{ext}"

    if deno_path.is_file():
        return deno_path

    url = _get_deno_url()
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "deno.zip"
        _download_file(url, archive_path, _scaled_progress_callback(progress_callback, 0.0, 0.85), "Deno")
        _emit_progress(progress_callback, 0.9, "Extracting Deno...")
        _extract_binary(archive_path, f"deno{ext}", bin_dir)

    if sys.platform != "win32":
        deno_path.chmod(deno_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _emit_progress(progress_callback, 0.97, "Validating Deno...")
    if not deno_path.is_file() or deno_path.stat().st_size < _MIN_DENO_SIZE:
        deno_path.unlink(missing_ok=True)
        raise RuntimeError("Deno download appears corrupt (too small)")

    _emit_progress(progress_callback, 1.0, "Deno installed")
    logger.info(f"Deno installed: {deno_path}")
    return deno_path


def update_yt_dlp(progress_callback: ProgressCallback = None) -> Path:
    """Force re-download of yt-dlp to get the latest version.

    Returns:
        Path to the updated yt-dlp binary.
    """
    bin_dir = get_managed_bin_dir()
    ext = _get_binary_ext()
    ytdlp_path = bin_dir / f"yt-dlp{ext}"

    # Remove existing to force re-download
    ytdlp_path.unlink(missing_ok=True)
    return ensure_yt_dlp(progress_callback)


def update_deno(progress_callback: ProgressCallback = None) -> Path:
    """Force re-download of Deno to get the latest version."""
    bin_dir = get_managed_bin_dir()
    ext = _get_binary_ext()
    deno_path = bin_dir / f"deno{ext}"

    deno_path.unlink(missing_ok=True)
    return ensure_deno(progress_callback)


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
    if sys.platform == "win32":
        python_bin = python_dir / "python.exe"
    else:
        python_bin = python_dir / "bin" / "python3"

    if python_bin.is_file() and (sys.platform == "win32" or os.access(python_bin, os.X_OK)):
        return python_bin

    python_url = _get_python_url()

    _emit_progress(progress_callback, 0.0, "Downloading Python 3.11...")

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "python.tar.gz"
        _download_file(python_url, tarball, _scaled_progress_callback(progress_callback, 0.0, 0.85), "Python 3.11")

        _emit_progress(progress_callback, 0.9, "Extracting Python 3.11...")

        # Extract tarball — python-build-standalone archives contain a
        # top-level "python/" directory with bin/, lib/, include/, etc.
        import tarfile

        python_dir.mkdir(parents=True, exist_ok=True)
        resolved_dest = python_dir.resolve()
        with tarfile.open(tarball, "r:gz") as tf:
            # Strip the top-level "python/" prefix during extraction
            for member in tf.getmembers():
                # python/bin/python3 -> bin/python3
                parts = member.name.split("/", 1)
                if len(parts) > 1:
                    member.name = parts[1]
                else:
                    continue  # Skip the top-level directory itself

                # Path traversal protection: ensure extraction stays within dest
                target_path = (resolved_dest / member.name).resolve()
                if not target_path.is_relative_to(resolved_dest):
                    logger.warning(f"Skipping suspicious tar member: {member.name}")
                    continue

                # Block symlinks pointing outside dest
                if member.issym() or member.islnk():
                    link_target = (resolved_dest / member.linkname).resolve()
                    if not link_target.is_relative_to(resolved_dest):
                        logger.warning(f"Skipping symlink escape: {member.name} -> {member.linkname}")
                        continue

                tf.extract(member, python_dir)

    # Validate
    if not python_bin.is_file():
        raise RuntimeError("Python extraction failed — python3 binary not found")

    # Ensure pip is available
    result = _run_external_subprocess(
        [str(python_bin), "-m", "pip", "--version"],
        capture_output=True, text=True, timeout=30,
        **get_subprocess_kwargs(),
    )
    if result.returncode != 0:
        logger.warning(f"pip check failed: {result.stderr}")
        # Try to bootstrap pip
        _run_external_subprocess(
            [str(python_bin), "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=60,
            **get_subprocess_kwargs(),
        )

    _emit_progress(progress_callback, 1.0, "Python 3.11 installed")

    logger.info(f"Standalone Python installed: {python_bin}")
    return python_bin


def get_python_version() -> Optional[str]:
    """Get the version of the managed standalone Python, or None if not installed."""
    if sys.platform == "win32":
        python_bin = get_managed_python_dir() / "python.exe"
    else:
        python_bin = get_managed_python_dir() / "bin" / "python3"
    if not python_bin.is_file():
        return None
    try:
        result = _run_external_subprocess(
            [str(python_bin), "--version"],
            capture_output=True, text=True, timeout=10,
            **get_subprocess_kwargs(),
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
        True if the binary exists in the managed bin directory.
    """
    ext = _get_binary_ext()
    path = get_managed_bin_dir() / f"{name}{ext}"
    return path.is_file()


def install_package(
    specifier: str,
    progress_callback: ProgressCallback = None,
) -> bool:
    """Install a Python package into the managed packages directory.

    Thin wrapper around install_packages() for legacy call sites.
    """
    return install_packages([specifier], progress_callback)


def install_packages(
    specifiers: list[str],
    progress_callback: ProgressCallback = None,
) -> bool:
    """Install Python packages into the managed packages directory.

    Uses the standalone managed Python (not the frozen app's Python) to run
    pip install --target. Automatically downloads Python if not present.
    Installing related packages together lets pip resolve a coherent set of
    dependencies instead of layering separate installs into one flat target dir.

    Args:
        specifiers: pip install specifiers (e.g., ["torch>=2.4,<2.6"]).
        progress_callback: Optional progress callback.

    Returns:
        True if installation succeeded.

    Raises:
        RuntimeError: If Python download or pip install fails.
    """
    normalized_specifiers = [specifier.strip() for specifier in specifiers if specifier.strip()]
    if not normalized_specifiers:
        return True

    _ensure_managed_packages_importable()

    # Ensure standalone Python is available
    python_bin = ensure_python(_scaled_progress_callback(progress_callback, 0.0, 0.2))

    packages_dir = get_managed_packages_dir()
    packages_dir.mkdir(parents=True, exist_ok=True)

    install_label = ", ".join(normalized_specifiers)
    _emit_progress(progress_callback, 0.2, f"Preparing install for {install_label}...")

    cmd = [
        str(python_bin),
        "-m", "pip", "install",
        "--target", str(packages_dir),
        "--upgrade",
        "--no-user",
        "--disable-pip-version-check",
        "--progress-bar", "off",
    ] + normalized_specifiers

    logger.info(f"Installing package: {' '.join(cmd)}")
    kwargs = get_subprocess_kwargs()
    output_lines: list[str] = []

    try:
        process = _popen_external_subprocess(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **kwargs,
        )
    except OSError as exc:
        logger.error("pip install failed to start: %s", exc)
        _emit_progress(progress_callback, 0.0, f"Failed to install {specifier}")
        return False

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            output_lines.append(line)
            parsed = _pip_progress_from_output_line(line, install_label)
            if parsed is not None:
                progress, message = parsed
                _emit_progress(progress_callback, 0.2 + (0.75 * progress), message)
        returncode = process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
        logger.error("pip install timed out for %s", install_label)
        _emit_progress(progress_callback, 0.0, f"Failed to install {install_label}")
        return False
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if returncode != 0:
        output = "\n".join(output_lines).strip()
        logger.error(f"pip install failed: {output}")
        _emit_progress(progress_callback, 0.0, f"Failed to install {install_label}")
        return False

    # Update ABI compatibility marker after successful install
    _write_compat_marker()
    _reset_imported_package_roots(_specifier_import_roots(normalized_specifiers))

    _emit_progress(progress_callback, 1.0, f"Installed {install_label}")

    logger.info(f"Package installed: {install_label}")
    return True


def is_package_available(module_name: str) -> bool:
    """Check if a Python package is importable (from managed packages or bundled).

    Uses importlib.util.find_spec() instead of __import__() to avoid
    actually loading heavy modules (torch, transformers) just to check availability.

    Args:
        module_name: Top-level module name (e.g., "torch", "ultralytics").

    Returns:
        True if the module can be imported.
    """
    import importlib.util
    _ensure_managed_packages_importable()
    return importlib.util.find_spec(module_name) is not None


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
    importlib.invalidate_caches()

    if progress_callback:
        progress_callback(1.0, "Managed packages removed")

    logger.info("Cleared managed packages directory")
