"""Build helpers for frozen desktop distribution assets."""

from __future__ import annotations

import base64
import binascii
import os
import re
import subprocess
from pathlib import Path

WINDOWS_MPV_DLL_NAMES = ("mpv-2.dll", "libmpv-2.dll", "mpv-1.dll")
WINDOWS_WINSPARKLE_DLL_NAMES = ("WinSparkle.dll", "winsparkle.dll")
WINDOWS_FFMPEG_BINARY_NAMES = ("ffmpeg.exe", "ffprobe.exe")
MACOS_FFMPEG_BINARY_NAMES = ("ffmpeg", "ffprobe")
REQUIREMENT_TO_IMPORT_TARGET = {
    "certifi": "certifi",
    "click": "click",
    "python-mpv": "mpv",
    "scenedetect": "scenedetect",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "google-api-python-client": "googleapiclient",
    "keyring": "keyring",
    "litellm": "litellm",
    "tenacity": "tenacity",
    "httpx": "httpx",
}
SUPPLEMENTAL_IMPORT_TARGETS = (
    "google_auth_httplib2",
    "google.auth",
    "google.oauth2",
    "httplib2",
    "scipy",
)
SUPPLEMENTAL_METADATA_TARGETS = (
    "google-auth",
    "google-auth-httplib2",
    "google-api-core",
    "googleapis-common-protos",
    "httplib2",
    "uritemplate",
    "scipy",
)
SUPPLEMENTAL_HIDDENIMPORTS = (
    # torch.package imports this stdlib module at runtime. The frozen base app
    # still needs it even when torch itself is installed on demand later.
    "pickletools",
    # transformers.dynamic_module_utils imports this stdlib module at runtime.
    # The frozen base app still needs it even when transformers is installed on demand.
    "filecmp",
    # transformers imports these stdlib modules indirectly in generation and
    # dynamic-module paths. The frozen base app still needs them even when
    # transformers is installed on demand later.
    "modulefinder",
    "cProfile",
)
PYINSTALLER_HANDLED_REQUIREMENTS = {
    "pyside6",
}
CURATED_PACKAGE_COLLECTIONS = {
    "litellm",
}
PACKAGE_DATA_EXCLUDES = {
    "litellm": (
        "**/proxy/**",
        "**/tests/**",
        "**/test/**",
        "**/benchmarks/**",
        "**/examples/**",
        "**/example*/**",
        "**/__pycache__/**",
        "**/*.md",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.gif",
        "**/*.svg",
    ),
}
PACKAGE_HIDDENIMPORT_EXCLUDES = {
    "litellm": (
        "litellm.proxy",
        "litellm.integrations.test_httpx",
        "litellm.responses.mcp.litellm_proxy_mcp_handler",
    ),
}


def _project_root_from_file(path: str) -> Path:
    return Path(path).resolve().parent.parent


def _normalize_requirement_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _extract_requirement_name(requirement_line: str) -> str:
    """Extract a normalized distribution name from a requirements.txt line."""
    wheel_line = requirement_line.split(";", 1)[0].strip()
    wheel_name = Path(wheel_line.split("#", 1)[0]).name
    wheel_match = re.match(
        r"(?P<name>.+)-\d+(?:\.\d+)*(?:[A-Za-z0-9_.+-]*)-[^-]+-[^-]+-[^-]+\.whl$",
        wheel_name,
    )
    if wheel_match:
        return _normalize_requirement_name(wheel_match.group("name"))

    requirement = wheel_line.split("@", 1)[0].strip()
    for separator in ("[", ">", "<", "=", "!", "~"):
        requirement = requirement.split(separator, 1)[0].strip()

    if requirement:
        return _normalize_requirement_name(requirement)

    return ""


def read_core_requirement_distributions(project_root: Path) -> tuple[str, ...]:
    """Return normalized distribution names from requirements-core.txt."""
    requirements_file = project_root / "requirements-core.txt"
    distributions: list[str] = []
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        requirement = _extract_requirement_name(line)
        if requirement:
            distributions.append(requirement)
    return tuple(distributions)


def get_core_pyinstaller_collect_targets(project_root: Path) -> tuple[str, ...]:
    """Return import-package roots that should be collected for frozen builds."""
    targets: list[str] = []
    for dist_name in read_core_requirement_distributions(project_root):
        if dist_name in PYINSTALLER_HANDLED_REQUIREMENTS:
            continue
        import_target = REQUIREMENT_TO_IMPORT_TARGET.get(dist_name)
        if import_target is None:
            raise RuntimeError(f"Missing PyInstaller import target mapping for {dist_name}")
        targets.append(import_target)

    targets.extend(SUPPLEMENTAL_IMPORT_TARGETS)
    return tuple(dict.fromkeys(targets))


def get_core_pyinstaller_metadata(project_root: Path) -> tuple[str, ...]:
    """Return distribution metadata names required by frozen desktop apps."""
    metadata = [
        dist_name
        for dist_name in read_core_requirement_distributions(project_root)
        if dist_name not in PYINSTALLER_HANDLED_REQUIREMENTS
    ]
    metadata.extend(SUPPLEMENTAL_METADATA_TARGETS)
    return tuple(dict.fromkeys(metadata))


def get_core_pyinstaller_hiddenimports() -> tuple[str, ...]:
    """Return extra hidden imports needed beyond collected package trees."""
    return SUPPLEMENTAL_HIDDENIMPORTS


def use_full_package_collection(module_name: str) -> bool:
    """Return whether PyInstaller should use collect_all() for this package."""
    return module_name not in CURATED_PACKAGE_COLLECTIONS


def get_pyinstaller_data_excludes(module_name: str) -> tuple[str, ...]:
    """Return package-specific data-file excludes for frozen builds."""
    return PACKAGE_DATA_EXCLUDES.get(module_name, ())


def get_pyinstaller_hiddenimport_excludes(module_name: str) -> tuple[str, ...]:
    """Return package-specific hidden import prefixes to skip."""
    return PACKAGE_HIDDENIMPORT_EXCLUDES.get(module_name, ())


def resolve_update_public_ed_key(
    explicit_public_key: str = "",
    private_key: str = "",
) -> str:
    """Return the updater public Ed25519 key, deriving it from the private key if needed.

    The private key is expected to be a base64-encoded raw Ed25519 seed, which is
    the format used by Sparkle/WinSparkle tooling. A PEM-encoded private key is
    also accepted.
    """
    public_key = explicit_public_key.strip()
    if public_key:
        return public_key

    private_key = private_key.strip()
    if not private_key:
        return ""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except Exception as exc:
        if private_key:
            raise RuntimeError(
                "Cannot derive updater public key because cryptography is unavailable."
            ) from exc
        return ""

    def _derive_from_private(private: object) -> str:
        if not isinstance(private, ed25519.Ed25519PrivateKey):
            return ""
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(public).decode("ascii")

    try:
        if private_key.startswith("-----BEGIN"):
            return _derive_from_private(
                serialization.load_pem_private_key(
                    private_key.encode("utf-8"),
                    password=None,
                )
            )

        try:
            raw_key = base64.b64decode(private_key.encode("utf-8"), validate=True)
        except binascii.Error:
            raw_key = b""

        if raw_key:
            if len(raw_key) == 64:
                raw_key = raw_key[:32]
            if len(raw_key) == 32:
                return _derive_from_private(
                    ed25519.Ed25519PrivateKey.from_private_bytes(raw_key)
                )
            try:
                return _derive_from_private(
                    serialization.load_der_private_key(raw_key, password=None)
                )
            except ValueError:
                pass

        hex_candidate = private_key.removeprefix("0x").removeprefix("0X")
        if len(hex_candidate) % 2 == 0:
            try:
                raw_hex = bytes.fromhex(hex_candidate)
            except ValueError:
                raw_hex = b""
            if raw_hex:
                if len(raw_hex) == 64:
                    raw_hex = raw_hex[:32]
                if len(raw_hex) == 32:
                    return _derive_from_private(
                        ed25519.Ed25519PrivateKey.from_private_bytes(raw_hex)
                    )
                try:
                    return _derive_from_private(
                        serialization.load_der_private_key(raw_hex, password=None)
                    )
                except ValueError:
                    pass

        return ""
    except (ValueError, TypeError, binascii.Error):
        return ""


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


def find_windows_ffmpeg_runtime_dir(project_root: Path) -> Path | None:
    """Return the staged Windows FFmpeg runtime directory, if available."""
    env_dir = os.environ.get("SCENE_RIPPER_FFMPEG_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "packaging" / "runtime" / "ffmpeg" / "windows",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            if all((candidate / name).is_file() for name in WINDOWS_FFMPEG_BINARY_NAMES):
                return candidate
    return None


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


def _prefix_runtime_destination(
    collected: list[tuple[str, str]],
    prefix: str,
) -> list[tuple[str, str]]:
    """Apply a destination prefix to collected runtime files."""
    prefixed: list[tuple[str, str]] = []
    for source_path, destination in collected:
        target = Path(prefix)
        if destination != ".":
            target = target / destination
        prefixed.append((source_path, str(target)))
    return prefixed


def collect_windows_ffmpeg_binaries(project_root: Path) -> list[tuple[str, str]]:
    """Collect staged Windows FFmpeg runtime files for PyInstaller."""
    runtime_dir = find_windows_ffmpeg_runtime_dir(project_root)
    if runtime_dir is None:
        raise RuntimeError(
            "Windows FFmpeg runtime not found. Stage FFmpeg in "
            "packaging/runtime/ffmpeg/windows or set SCENE_RIPPER_FFMPEG_DIR."
        )

    collected = _prefix_runtime_destination(_collect_runtime_files(runtime_dir), "bin")
    bundled_names = {Path(src).name.lower() for src, _ in collected}
    if not all(name.lower() in bundled_names for name in WINDOWS_FFMPEG_BINARY_NAMES):
        raise RuntimeError(f"Missing FFmpeg runtime binaries in {runtime_dir}")
    return collected


def find_macos_ffmpeg_runtime_dir(project_root: Path) -> Path | None:
    """Return the staged macOS FFmpeg runtime directory, if available."""
    env_dir = os.environ.get("SCENE_RIPPER_FFMPEG_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        project_root / "packaging" / "runtime" / "ffmpeg" / "macos",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            if all((candidate / name).is_file() for name in MACOS_FFMPEG_BINARY_NAMES):
                return candidate
    return None


def collect_macos_ffmpeg_binaries(project_root: Path) -> list[tuple[str, str]]:
    """Collect staged macOS FFmpeg runtime files for PyInstaller."""
    runtime_dir = find_macos_ffmpeg_runtime_dir(project_root)
    if runtime_dir is None:
        raise RuntimeError(
            "macOS FFmpeg runtime not found. Stage FFmpeg in "
            "packaging/runtime/ffmpeg/macos or set SCENE_RIPPER_FFMPEG_DIR."
        )

    collected = _prefix_runtime_destination(_collect_runtime_files(runtime_dir), "bin")
    bundled_names = {Path(src).name for src, _ in collected}
    if not all(name in bundled_names for name in MACOS_FFMPEG_BINARY_NAMES):
        raise RuntimeError(f"Missing FFmpeg runtime binaries in {runtime_dir}")
    return collected


def collect_macos_sparkle_datas(project_root: Path) -> list[tuple[str, str]]:
    """Collect the Sparkle framework for app bundling if a runtime is staged.

    The staged runtime directory may also contain helper apps and standalone CLI tools
    used by CI for signing and feed generation. Those should not be bundled into the
    shipped app because PyInstaller attempts to re-sign them as independent bundles.
    We also only bundle the framework's versioned contents and reconstruct the top-level
    symlink layout later so the embedded framework remains codesign-compatible.
    """
    runtime_dir = find_macos_sparkle_runtime_dir(project_root)
    if runtime_dir is None:
        return []

    framework_dir = runtime_dir / "Sparkle.framework"
    if not framework_dir.is_dir():
        return []

    versions_dir = framework_dir / "Versions"
    version_dir = next(
        (
            path
            for path in sorted(versions_dir.iterdir())
            if path.is_dir() and path.name != "Current"
        ),
        None,
    )
    if version_dir is None:
        return []

    collected = []
    for source_path, destination in _collect_runtime_files(version_dir):
        target = Path("Sparkle.framework") / "Versions" / version_dir.name
        if destination != ".":
            target = target / destination
        collected.append((source_path, str(target)))
    return collected
