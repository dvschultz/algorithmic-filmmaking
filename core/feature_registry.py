"""Feature-to-dependency mapping for on-demand installation.

Maps application features to the binaries and Python packages they require,
enabling the UI to show "install required" prompts and the dependency manager
to download exactly what's needed.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from core.binary_resolver import find_binary
from core.dependency_manager import is_binary_available, is_package_available

logger = logging.getLogger(__name__)


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
        binaries=["yt-dlp"],
        packages=[],
        size_estimate_mb=20,
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
    "transcribe_mlx": FeatureDeps(
        binaries=[],
        packages=["lightning_whisper_mlx"],
        size_estimate_mb=100,
    ),
    "describe_local": FeatureDeps(
        binaries=[],
        packages=["mlx_vlm"],
        size_estimate_mb=200,
    ),
    "describe_cloud": FeatureDeps(
        binaries=[],
        packages=[],  # litellm is bundled in core
        size_estimate_mb=0,
    ),
    "shot_classify": FeatureDeps(
        binaries=[],
        packages=["torch", "transformers"],
        size_estimate_mb=450,
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

    available, missing = check_feature(name)
    if available:
        return 0

    return deps.size_estimate_mb


def get_all_feature_status() -> dict[str, tuple[bool, list[str]]]:
    """Get availability status for all registered features.

    Returns:
        Dict of feature_name -> (available, missing_deps).
    """
    return {name: check_feature(name) for name in FEATURE_DEPS}
