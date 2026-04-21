"""Feature-to-dependency mapping for on-demand installation.

Maps application features to the binaries and Python packages they require,
enabling the UI to show "install required" prompts and the dependency manager
to download exactly what's needed.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.binary_resolver import find_binary
from core.dependency_manager import is_binary_available, is_package_available

logger = logging.getLogger(__name__)

# Package names whose C extensions cannot be safely reinstalled in a running
# process. If already loaded in sys.modules, reinstalling them produces errors
# like "function '_has_torch_function' already has a docstring" because the
# new shared objects are layered on top of the already-loaded ones. Skip these
# from reinstall when they're already imported; install only what's missing.
# Native-code packages only — these either bundle C/Rust extensions or Metal/CUDA
# runtimes that break when reinstalled under a loaded interpreter. Pure-Python
# packages like transformers, mlx_vlm, ultralytics, and lightning_whisper_mlx
# must NOT be in this list: they are frequent version-pinned deps of VLMs and
# skipping them leaves users stuck on stale versions (e.g. a transformers that
# can't resolve Qwen3-VL's processor class).
_UNSAFE_TO_RELOAD_IF_LOADED = {
    "torch",
    "torchvision",
    "torchaudio",
    "onnxruntime",
    "insightface",
    "mlx",
    "paddleocr",
}

_FULL_PACKAGE_REPAIR_FEATURES = {
    "audio_analysis",
    "describe_local",
    "describe_local_cpu",
    "embeddings",
    "ocr",
    "image_classify",
    "object_detect",
    "face_detect",
    "gaze_detect",
    "shot_classify",
    "transcribe",
    "transcribe_mlx",
}


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
    if name == "describe_local":
        from core.analysis.description import ensure_local_description_runtime_available

        ensure_local_description_runtime_available()
    elif name == "describe_local_cpu":
        from core.analysis.description import ensure_local_cpu_description_runtime_available

        ensure_local_cpu_description_runtime_available()
    elif name == "shot_classify":
        from core.analysis.shots import ensure_classification_runtime_available

        ensure_classification_runtime_available()
    elif name == "image_classify":
        from core.analysis.classification import ensure_image_classification_runtime_available

        ensure_image_classification_runtime_available()
    elif name == "object_detect":
        from core.analysis.detection import ensure_object_detection_runtime_available

        ensure_object_detection_runtime_available()
    elif name == "ocr":
        from core.analysis.ocr import ensure_ocr_runtime_available

        ensure_ocr_runtime_available()
    elif name == "audio_analysis":
        from core.analysis.audio import ensure_audio_analysis_runtime_available

        ensure_audio_analysis_runtime_available()
    elif name == "face_detect":
        from core.analysis.faces import ensure_face_detection_runtime_available

        ensure_face_detection_runtime_available()
    elif name == "transcribe":
        from core.transcription import ensure_faster_whisper_runtime_available

        ensure_faster_whisper_runtime_available()
    elif name == "transcribe_mlx":
        from core.transcription import ensure_mlx_whisper_runtime_available

        ensure_mlx_whisper_runtime_available()
    elif name == "gaze_detect":
        from core.analysis.gaze import ensure_gaze_runtime_available

        ensure_gaze_runtime_available()
    elif name == "embeddings":
        try:
            from transformers.models.auto.image_processing_auto import AutoImageProcessor  # noqa: F401
            from transformers.models.auto.modeling_auto import AutoModel  # noqa: F401
        except Exception:
            from transformers import AutoImageProcessor, AutoModel  # noqa: F401


def requires_full_package_repair(name: str, missing: list[str]) -> bool:
    """Return True when a feature should reinstall its full package set."""
    if name not in _FULL_PACKAGE_REPAIR_FEATURES:
        return False
    return any(dep.startswith("package:") or dep.startswith("runtime:") for dep in missing)


def _filter_already_loaded_packages(package_names: list[str]) -> list[str]:
    """Remove packages whose top-level module is already imported in this process.

    Reinstalling a loaded C extension (e.g. torch) in-process corrupts its
    runtime state. We keep the loaded ones as-is and let pip install just
    the remaining packages.
    """
    remaining = []
    skipped = []
    for pkg_spec in package_names:
        # Strip version constraints: "torch>=2.4,<2.7" -> "torch"
        top = pkg_spec.split(">=")[0].split("<=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
        top_norm = top.replace("-", "_")
        if top_norm in _UNSAFE_TO_RELOAD_IF_LOADED and top_norm in sys.modules:
            skipped.append(top_norm)
        else:
            remaining.append(pkg_spec)
    if skipped:
        logger.info(
            "Skipping reinstall of already-loaded packages %s (would corrupt C-extension state)",
            skipped,
        )
    return remaining


@dataclass
class FeatureDeps:
    """Dependencies required for a feature."""

    binaries: list[str]
    packages: list[str]
    size_estimate_mb: int  # Rough download size in MB
    repair_packages: list[str] = field(default_factory=list)
    native_install: bool = False  # Use site-packages install for native extensions (e.g., mlx Metal)
    no_deps: bool = False  # Install with --no-deps to avoid pulling uncontrolled transitive deps
    needs_compiler: bool = False  # Requires C/C++ compiler (Xcode CLT on macOS)


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
        packages=["lightning_whisper_mlx", "tiktoken"],
        size_estimate_mb=300,
        repair_packages=["lightning_whisper_mlx", "mlx", "tiktoken"],
        native_install=True,  # mlx has Metal native extensions
    ),
    "describe_local": FeatureDeps(
        binaries=[],
        packages=["mlx_vlm", "torch", "torchvision", "transformers", "tokenizers"],
        size_estimate_mb=600,
        repair_packages=["mlx_vlm", "torch", "torchvision", "transformers", "tokenizers", "sentencepiece", "protobuf"],
        native_install=True,  # mlx has Metal native extensions that need proper site-packages
    ),
    "describe_local_cpu": FeatureDeps(
        binaries=[],
        packages=["torch", "torchvision", "transformers", "tokenizers"],
        size_estimate_mb=500,
        repair_packages=["torch", "torchvision", "transformers", "tokenizers"],
        native_install=True,  # torch/transformers need proper site-packages
    ),
    "describe_cloud": FeatureDeps(
        binaries=[],
        packages=[],  # litellm is bundled in core
        size_estimate_mb=0,
    ),
    "custom_query": FeatureDeps(
        binaries=[],
        packages=[],  # reuses describe infrastructure (litellm bundled, local VLM shared)
        size_estimate_mb=0,
    ),
    "shot_classify": FeatureDeps(
        binaries=[],
        packages=["torch", "transformers", "huggingface_hub", "sentencepiece", "protobuf"],
        size_estimate_mb=400,
        repair_packages=["torch", "transformers", "huggingface_hub", "tokenizers", "sentencepiece", "protobuf"],
        native_install=True,  # torch/transformers need proper site-packages to avoid circular imports
    ),
    "image_classify": FeatureDeps(
        binaries=[],
        packages=["torch", "torchvision"],
        size_estimate_mb=400,
        repair_packages=["torch", "torchvision"],
        native_install=True,  # torch needs proper site-packages
    ),
    "object_detect": FeatureDeps(
        binaries=[],
        packages=["torch", "ultralytics"],
        size_estimate_mb=430,
        repair_packages=["torch", "ultralytics"],
        native_install=True,  # ultralytics/torch need proper site-packages
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
        repair_packages=["insightface", "onnxruntime"],
        needs_compiler=True,  # insightface has Cython C++ extensions
        native_install=True,  # C++ extensions need proper site-packages, not --target
    ),
    "stem_separation": FeatureDeps(
        binaries=[],
        packages=["torch", "torchaudio", "demucs_infer", "librosa"],
        size_estimate_mb=2000,
        repair_packages=["torch", "torchaudio", "demucs_infer", "librosa"],
        native_install=True,  # torch has native extensions
    ),
    "gaze_detect": FeatureDeps(
        binaries=[],
        packages=["mediapipe"],
        size_estimate_mb=50,
        repair_packages=["mediapipe"],
    ),
    "embeddings": FeatureDeps(
        binaries=[],
        packages=["torch", "transformers"],
        size_estimate_mb=450,
        repair_packages=["torch", "transformers", "tokenizers"],
        native_install=True,
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
        install_native_packages,
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
            packages_to_reinstall = _filter_already_loaded_packages(deps.packages)
            missing = [f"package:{p}" for p in packages_to_reinstall]
    elif requires_full_package_repair(name, missing):
        logger.info(
            "Feature '%s' has partial package coverage; reinstalling the full package set",
            name,
        )
        runtime_repair = True
        packages_to_reinstall = _filter_already_loaded_packages(deps.packages)
        missing = [f"package:{p}" for p in packages_to_reinstall]

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
        repair_package_names = deps.repair_packages or deps.packages
        if runtime_repair:
            # Filter out already-loaded C extensions so we don't corrupt their
            # runtime state by reinstalling them on top of a live process.
            repair_package_names = _filter_already_loaded_packages(repair_package_names)
        package_batch_names = repair_package_names if runtime_repair else package_names
        specifiers = [get_pip_specifier(dep_name) for dep_name in package_batch_names]
        if runtime_repair and repair_package_names:
            clear_package_roots(repair_package_names)
        installer = install_native_packages if deps.native_install else install_packages
        if specifiers and not installer(specifiers, scaled_callback, no_deps=deps.no_deps):
            success = False

    if success:
        try:
            _validate_feature_runtime(name)
        except Exception as e:
            logger.error(f"Runtime validation failed for {name}: {e}")
            raise RuntimeError(str(e)) from e

    return success
