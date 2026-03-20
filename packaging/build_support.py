"""Build helpers for frozen desktop distribution assets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

WINDOWS_MPV_DLL_NAMES = ("mpv-2.dll", "libmpv-2.dll", "mpv-1.dll")
WINDOWS_WINSPARKLE_DLL_NAMES = ("WinSparkle.dll", "winsparkle.dll")


def _project_root_from_file(path: str) -> Path:
    return Path(path).resolve().parent.parent


def find_windows_mpv_runtime_dir(project_root: Path) -> Path | None:
    """Return the staged Windows mpv runtime directory, if available."""
    env_dir = os.environ.get("SCENE_RIPPER_MPV_DLL_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "packaging" / "runtime" / "mpv" / "windows",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            if any((candidate / dll_name).is_file() for dll_name in WINDOWS_MPV_DLL_NAMES):
                return candidate
    return None


def collect_windows_mpv_binaries(project_root: Path) -> list[tuple[str, str]]:
    """Collect staged Windows mpv runtime DLLs for PyInstaller."""
    runtime_dir = find_windows_mpv_runtime_dir(project_root)
    if runtime_dir is None:
        raise RuntimeError(
            "Windows mpv runtime not found. Stage mpv DLLs in "
            "packaging/runtime/mpv/windows or set SCENE_RIPPER_MPV_DLL_DIR."
        )

    binaries: list[tuple[str, str]] = []
    for dll_path in sorted(runtime_dir.glob("*.dll")):
        binaries.append((str(dll_path), "."))

    if not any(Path(src).name.lower() in WINDOWS_MPV_DLL_NAMES for src, _ in binaries):
        raise RuntimeError(f"No supported mpv runtime DLL found in {runtime_dir}")

    return binaries


def find_windows_winsparkle_runtime_dir(project_root: Path) -> Path | None:
    """Return the staged Windows WinSparkle runtime directory, if available."""
    env_dir = os.environ.get("SCENE_RIPPER_WINSPARKLE_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "packaging" / "runtime" / "winsparkle" / "windows",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            if any((candidate / dll_name).is_file() for dll_name in WINDOWS_WINSPARKLE_DLL_NAMES):
                return candidate
    return None


def collect_windows_winsparkle_binaries(project_root: Path) -> list[tuple[str, str]]:
    """Collect staged Windows WinSparkle binaries for PyInstaller if present."""
    runtime_dir = find_windows_winsparkle_runtime_dir(project_root)
    if runtime_dir is None:
        return []

    binaries: list[tuple[str, str]] = []
    for dll_path in sorted(runtime_dir.glob("*.dll")):
        binaries.append((str(dll_path), "."))
    return binaries


def _find_macos_libmpv(project_root: Path) -> Path | None:
    env_path = os.environ.get("SCENE_RIPPER_LIBMPV_DYLIB")
    candidates = [
        Path(env_path) if env_path else None,
        project_root / "packaging" / "runtime" / "mpv" / "macos" / "libmpv.dylib",
        Path("/opt/homebrew/lib/libmpv.dylib"),
        Path("/usr/local/lib/libmpv.dylib"),
    ]

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()

    brew = os.environ.get("HOMEBREW_PREFIX")
    if brew:
        candidate = Path(brew) / "lib" / "libmpv.dylib"
        if candidate.is_file():
            return candidate.resolve()

    try:
        result = subprocess.run(
            ["brew", "--prefix", "mpv"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    candidate = Path(result.stdout.strip()) / "lib" / "libmpv.dylib"
    if candidate.is_file():
        return candidate.resolve()
    return None


def _macos_dependency_paths(library_path: Path) -> list[Path]:
    """Return non-system Homebrew-style dylib dependencies for a Mach-O library."""
    try:
        result = subprocess.run(
            ["otool", "-L", str(library_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    dependencies: list[Path] = []
    for line in result.stdout.splitlines()[1:]:
        dep = line.strip().split(" (compatibility version", 1)[0]
        if dep.startswith(("/opt/homebrew/", "/usr/local/")):
            dep_path = Path(dep)
            if dep_path.is_file():
                dependencies.append(dep_path.resolve())
    return dependencies


def collect_macos_mpv_binaries(project_root: Path) -> list[tuple[str, str]]:
    """Collect libmpv and Homebrew dylib dependencies for PyInstaller."""
    libmpv = _find_macos_libmpv(project_root)
    if libmpv is None:
        raise RuntimeError(
            "libmpv.dylib not found. Install mpv via Homebrew or set "
            "SCENE_RIPPER_LIBMPV_DYLIB."
        )

    binaries: list[tuple[str, str]] = []
    to_visit = [libmpv]
    seen: set[Path] = set()

    while to_visit:
        current = to_visit.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        binaries.append((str(current), "."))
        to_visit.extend(_macos_dependency_paths(current))

    return sorted(binaries)


def find_macos_sparkle_runtime_dir(project_root: Path) -> Path | None:
    """Return the staged macOS Sparkle runtime directory, if available."""
    env_dir = os.environ.get("SCENE_RIPPER_SPARKLE_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "packaging" / "runtime" / "sparkle" / "macos",
    ]

    for candidate in candidates:
        if not candidate or not candidate.is_dir():
            continue

        expected_paths = (
            candidate / "Sparkle.framework",
            candidate / "sparkle.app",
            candidate / "bin" / "sparkle",
        )
        if any(path.exists() for path in expected_paths):
            return candidate
    return None


def _collect_runtime_files(root_dir: Path) -> list[tuple[str, str]]:
    """Collect all files in a runtime directory preserving their relative layout."""
    collected: list[tuple[str, str]] = []
    for file_path in sorted(path for path in root_dir.rglob("*") if path.is_file()):
        relative_parent = file_path.relative_to(root_dir).parent
        destination = "." if str(relative_parent) == "." else str(relative_parent)
        collected.append((str(file_path), destination))
    return collected


def collect_macos_sparkle_datas(project_root: Path) -> list[tuple[str, str]]:
    """Collect staged macOS Sparkle runtime assets for PyInstaller if present."""
    runtime_dir = find_macos_sparkle_runtime_dir(project_root)
    if runtime_dir is None:
        return []

    return _collect_runtime_files(runtime_dir)
