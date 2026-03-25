"""Feature-to-dependency mapping for on-demand installation.

Maps application features to the binaries and Python packages they require,
enabling the UI to show "install required" prompts and the dependency manager
to download exactly what's needed.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from core.binary_resolver import find_binary
from core.dependency_manager import is_binary_available, is_package_available

logger = logging.getLogger(__name__)


def _scaled_progress_callback(
    progress_callback: Optional[Callable[[float, str], None]],
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    """Map dependency progress into a subrange of the overall feature install."""
    span = max(0.0, end - start)

    def _callback(progress: float, message: str) -> None:
        if progress_callback is None:
            return
        clamped = max(0.0, min(1.0, progress))
        progress_callback(start + (span * clamped), message)

    return _callback


def _validate_feature_runtime(name: str) -> None:
    """Run narrow runtime import checks for fragile on-demand features."""
    if name == "shot_classify":
        from core.analysis.shots import ensure_classification_runtime_available

        ensure_classification_runtime_available()


@dataclass
class FeatureDeps:
    """Dependencies required for a feature."""

    binaries: list[str]
    packages: list[str]
    size_estimate_mb: int  # Rough download size in MB


# Map feature names to their dependency requirements
FEATURE_DEPS: dict[str, FeatureDeps] = {
    "scene_detection": FeatureDeps(
        binaries=["ffmpeg"],
        packages=[],
        size_estimate_mb=0,
    ),
    "thumbnails": FeatureDeps(
        binaries=["ffmpeg"],
        packages=[],
        size_estimate_mb=0,
    ),
    "video_download": FeatureDeps(
        binaries=["yt-dlp", "deno"],
        packages=[],
        size_estimate_mb=80,
    ),
    "video_export": FeatureDeps(
        binaries=["ffmpeg"],
        packages=[],
        size_estimate_mb=0,
    ),
    "transcribe": FeatureDeps(
        binaries=["ffmpeg"],
        packages=["faster_whisper"],
        size_estimate_mb=150,
    ),
    "transcribe_cloud": FeatureDeps(
        binaries=["ffmpeg"],
        packages=[],
        size_estimate_mb=0,
    ),
    "transcribe_mlx": FeatureDeps(
        binaries=["ffmpeg"],
        packages=["lightning_whisper_mlx"],
        size_estimate_mb=100,
    ),
    "describe_local": FeatureDeps(
        binaries=[],
        packages=["mlx_vlm"],
        size_estimate_mb=200,
    ),
    "describe_local_cpu": FeatureDeps(
        binaries=[],
        packages=["torch", "transformers"],
        size_estimate_mb=450,
    ),
    "describe_cloud": FeatureDeps(
        binaries=[],
        packages=[],  # litellm is bundled in core
        size_estimate_mb=0,
    ),
    "shot_classify": FeatureDeps(
        binaries=[],
        packages=["torch", "torchvision", "transformers", "einops", "sentencepiece", "protobuf"],
        size_estimate_mb=450,
    ),
    "image_classify": FeatureDeps(
        binaries=[],
        packages=["torch", "torchvision"],
        size_estimate_mb=400,
    ),
    "object_detect": FeatureDeps(
        binaries=[],
        packages=["ultralytics"],
        size_estimate_mb=80,
    ),
    "ocr": FeatureDeps(
        binaries=[],
        packages=["paddleocr"],
        size_estimate_mb=300,
    ),
    "audio_analysis": FeatureDeps(
        binaries=[],
        packages=["librosa"],
        size_estimate_mb=80,
    ),
    "color_analysis": FeatureDeps(
        binaries=[],
        packages=[],  # opencv is bundled in core
        size_estimate_mb=0,
    ),
    "face_detect": FeatureDeps(
        binaries=[],
        packages=["insightface", "onnxruntime"],
        size_estimate_mb=300,
    ),
    "stem_separation": FeatureDeps(
        binaries=[],
        packages=["torch", "demucs_infer"],
        size_estimate_mb=2000,
    ),
}


def check_feature(name: str) -> tuple[bool, list[str]]:
    """Check if a feature's dependencies are satisfied.

    Args:
        name: Feature name from FEATURE_DEPS keys.

    Returns:
        Tuple of (all_available, list_of_missing_dep_names).
        Missing deps are prefixed with "binary:" or "package:" for clarity.
    """
    deps = FEATURE_DEPS.get(name)
    if deps is None:
        logger.warning(f"Unknown feature: {name}")
        return True, []

    missing = []

    for binary in deps.binaries:
        if find_binary(binary) is None:
            missing.append(f"binary:{binary}")

    for package in deps.packages:
        if not is_package_available(package):
            missing.append(f"package:{package}")

    return len(missing) == 0, missing


def check_feature_ready(name: str) -> tuple[bool, list[str]]:
    """Check whether a feature is both installed and runtime-usable."""
    available, missing = check_feature(name)
    if not available:
        return available, missing

    try:
        _validate_feature_runtime(name)
    except Exception as e:
        reason = str(e).strip() or e.__class__.__name__
        return False, [f"runtime:{reason}"]

    return True, []


def get_feature_size_estimate(name: str) -> int:
    """Get the estimated download size in MB for a feature's missing dependencies.

    Args:
        name: Feature name.

    Returns:
        Estimated download size in MB (0 if all deps are present).
    """
    deps = FEATURE_DEPS.get(name)
    if deps is None:
        return 0

    available, missing = check_feature_ready(name)
    if available:
        return 0

    return deps.size_estimate_mb


def get_all_feature_status() -> dict[str, tuple[bool, list[str]]]:
    """Get availability status for all registered features.

    Returns:
        Dict of feature_name -> (available, missing_deps).
    """
    return {name: check_feature(name) for name in FEATURE_DEPS}


def install_for_feature(
    name: str,
    progress_callback: Optional[Callable] = None,
) -> bool:
    """Install all missing dependencies for a feature.

    Ensures binaries are downloaded and Python packages are installed.
    Automatically downloads standalone Python if packages are needed.

    Args:
        name: Feature name from FEATURE_DEPS keys.
        progress_callback: Optional (progress_0_to_1, message) callback.

    Returns:
        True if all dependencies were successfully installed.

    Raises:
        ValueError: If feature name is unknown.
    """
    from core.dependency_manager import (
        clear_package_roots,
        ensure_ffmpeg,
        ensure_ffprobe,
        ensure_deno,
        ensure_yt_dlp,
        get_pip_specifier,
        install_packages,
    )

    deps = FEATURE_DEPS.get(name)
    if deps is None:
        raise ValueError(f"Unknown feature: {name}")

    runtime_repair = False

    available, missing = check_feature(name)
    if available:
        try:
            _validate_feature_runtime(name)
            logger.info(f"Feature '{name}' already has all dependencies")
            return True
        except Exception as e:
            logger.warning(
                "Feature '%s' appears installed but runtime validation failed; reinstalling packages: %s",
                name,
                e,
            )
            runtime_repair = True
            missing = [f"package:{package_name}" for package_name in deps.packages]

    success = True

    binary_installers = {
        "ffmpeg": ensure_ffmpeg,
        "ffprobe": ensure_ffprobe,
        "deno": ensure_deno,
        "yt-dlp": ensure_yt_dlp,
    }

    binary_steps: list[tuple[str, Callable]] = []
    package_names: list[str] = []
    for dep in missing:
        if dep.startswith("binary:"):
            binary_name = dep.split(":", 1)[1]
            installer = binary_installers.get(binary_name)
            if installer is not None:
                binary_steps.append((binary_name, installer))
        elif dep.startswith("package:"):
            package_name = dep.split(":", 1)[1]
            package_names.append(package_name)

    total_steps = len(binary_steps) + (1 if package_names else 0)
    if total_steps == 0:
        return success

    for index, (dep_name, installer) in enumerate(binary_steps):
        start = index / total_steps
        end = (index + 1) / total_steps
        scaled_callback = _scaled_progress_callback(progress_callback, start, end)
        try:
            installer(scaled_callback)
        except RuntimeError as e:
            logger.error(f"Failed to install {dep_name}: {e}")
            success = False

    if package_names:
        start = len(binary_steps) / total_steps
        scaled_callback = _scaled_progress_callback(progress_callback, start, 1.0)
        specifiers = [get_pip_specifier(dep_name) for dep_name in package_names]
        if runtime_repair:
            clear_package_roots(package_names)
        if not install_packages(specifiers, scaled_callback):
            success = False

    if success:
        try:
            _validate_feature_runtime(name)
        except Exception as e:
            logger.error(f"Runtime validation failed for {name}: {e}")
            success = False

    return success
