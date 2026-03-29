#!/usr/bin/env bash
# Build Scene Ripper macOS .app bundle locally.
#
# Usage:
#   ./packaging/macos/build.sh            # Build .app only
#   ./packaging/macos/build.sh --dmg      # Build .app + DMG
#   ./packaging/macos/build.sh --sign     # Build .app + ad-hoc code sign
#   ./packaging/macos/build.sh --dmg --sign  # Build all + sign
#
# Environment variables:
#   APP_VERSION       Version string (default: 0.2.4)
#   CODESIGN_IDENTITY Signing identity (default: ad-hoc "-")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_EXTERNAL_BUILD_ROOT="/Volumes/Lexar/scene-ripper-build"

APP_VERSION="${APP_VERSION:-0.2.4}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
BUILD_DMG=false
CODE_SIGN=false

for arg in "$@"; do
    case "$arg" in
        --dmg) BUILD_DMG=true ;;
        --sign) CODE_SIGN=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "==> Building Scene Ripper v${APP_VERSION}"
echo "    Project root: ${PROJECT_ROOT}"
echo "    Sign: ${CODE_SIGN} (identity: ${CODESIGN_IDENTITY})"
echo "    DMG: ${BUILD_DMG}"
echo ""

BUILD_ROOT="${SCENE_RIPPER_BUILD_ROOT:-}"
if [ -z "${BUILD_ROOT}" ] && [ -d "/Volumes/Lexar" ]; then
    BUILD_ROOT="${DEFAULT_EXTERNAL_BUILD_ROOT}"
fi
if [ -z "${BUILD_ROOT}" ]; then
    BUILD_ROOT="${PROJECT_ROOT}/build"
fi

MODEL_RUNTIME_DIR="${SCENE_RIPPER_MODEL_RUNTIME_DIR:-${BUILD_ROOT}/runtime/models/macos}"
PYINSTALLER_WORK_PATH="${BUILD_ROOT}/pyinstaller-work"
PYINSTALLER_DIST_PATH="${BUILD_ROOT}/dist"

mkdir -p "${BUILD_ROOT}" "${PYINSTALLER_WORK_PATH}" "${PYINSTALLER_DIST_PATH}"
export SCENE_RIPPER_BUILD_ROOT="${BUILD_ROOT}"
export SCENE_RIPPER_MODEL_RUNTIME_DIR="${MODEL_RUNTIME_DIR}"

# -------------------------------------------------------------------
# 0. Stage FFmpeg runtime
# -------------------------------------------------------------------
echo "==> Staging FFmpeg runtime..."
FFMPEG_RUNTIME_DIR="${PROJECT_ROOT}/packaging/runtime/ffmpeg/macos"
mkdir -p "${FFMPEG_RUNTIME_DIR}"

PROJECT_ROOT="$PROJECT_ROOT" python - <<'PY'
import shutil
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
runtime_dir = project_root / "packaging" / "runtime" / "ffmpeg" / "macos"
env_runtime_dir = os.environ.get("SCENE_RIPPER_FFMPEG_DIR", "").strip()
fallback_runtime_candidates = [
    Path(env_runtime_dir) if env_runtime_dir else None,
    project_root / "dist" / "Scene Ripper.app" / "Contents" / "Frameworks" / "bin",
    project_root / "dist" / "Scene Ripper" / "_internal" / "bin",
]
downloads = {
    "ffmpeg": "https://www.osxexperts.net/ffmpeg7arm.zip",
    "ffprobe": "https://www.osxexperts.net/ffprobe7arm.zip",
}

runtime_dir.mkdir(parents=True, exist_ok=True)

for candidate in fallback_runtime_candidates:
    if candidate is None or not candidate.is_dir():
        continue
    if all((candidate / name).is_file() for name in downloads):
        for binary_name in downloads:
            target_path = runtime_dir / binary_name
            shutil.copy2(candidate / binary_name, target_path)
            target_path.chmod(0o755)
        print(f"Reused staged FFmpeg runtime from {candidate}")
        raise SystemExit(0)


def _download_and_extract(binary_name: str, url: str, temp_path: Path) -> None:
    archive_path = temp_path / f"{binary_name}.zip"
    target_path = runtime_dir / binary_name

    for attempt in range(1, 5):
        archive_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = [name for name in archive.namelist() if name.endswith(binary_name)]
                if not members:
                    raise RuntimeError(f"{binary_name} not found in downloaded archive")
                extracted = Path(archive.extract(members[0], temp_path))
                shutil.move(str(extracted), target_path)
                target_path.chmod(0o755)
            return
        except (
            urllib.error.ContentTooShortError,
            urllib.error.URLError,
            zipfile.BadZipFile,
            OSError,
            RuntimeError,
        ) as exc:
            archive_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            if attempt == 4:
                raise
            delay = attempt * 2
            print(f"Retrying {binary_name} download after attempt {attempt}/4 failed: {exc}")
            time.sleep(delay)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    for binary_name, url in downloads.items():
        if (runtime_dir / binary_name).is_file():
            continue
        _download_and_extract(binary_name, url, temp_path)
PY

# -------------------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------------------
echo "==> Installing core + macOS ML dependencies..."
pip install -r "${PROJECT_ROOT}/requirements-core.txt" --quiet
pip install \
    'torch>=2.4,<2.7' \
    'torchaudio>=2.4,<2.7' \
    'torchvision>=0.19,<0.22' \
    'transformers>=4.50,<5' \
    'huggingface-hub>=0.34.0,<1.0' \
    'sentencepiece>=0.2.0,<1.0' \
    'protobuf>=4.25,<6' \
    'einops>=0.7.0,<1.0' \
    'ultralytics>=8.4.0,<9' \
    'insightface>=0.7.3,<1.0' \
    'onnxruntime>=1.16.0,<2.0' \
    'librosa>=0.10.0,<1.0' \
    'demucs-infer>=4.1.0,<5' \
    'mlx-vlm>=0.1.0,<1.0' \
    'paddleocr>=3.0.0,<4' \
    'paddlepaddle>=3.3.0,<4' \
    'rapidfuzz>=3.0.0' \
    --quiet
# lightning-whisper-mlx must be --no-deps to avoid tiktoken conflict with litellm
pip install 'lightning-whisper-mlx>=0.0.10,<1.0' --no-deps --quiet
pip install pyinstaller --quiet

# -------------------------------------------------------------------
# 1.5. Stage bundled local models
# -------------------------------------------------------------------
echo "==> Staging bundled local models..."
python "${SCRIPT_DIR}/stage_models.py"

# -------------------------------------------------------------------
# 2. Run PyInstaller
# -------------------------------------------------------------------
echo "==> Running PyInstaller..."
export APP_VERSION
cd "$PROJECT_ROOT"
pyinstaller "${SCRIPT_DIR}/scene_ripper.spec" \
    --distpath "${PYINSTALLER_DIST_PATH}" \
    --workpath "${PYINSTALLER_WORK_PATH}" \
    --noconfirm

APP_PATH="${PYINSTALLER_DIST_PATH}/Scene Ripper.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: .app bundle not found at ${APP_PATH}"
    exit 1
fi

echo "==> .app bundle created: ${APP_PATH}"
du -sh "$APP_PATH"
if ! find "$APP_PATH" -name "ffmpeg" | grep -q .; then
    echo "ERROR: bundled ffmpeg binary not found in ${APP_PATH}"
    exit 1
fi
if ! find "$APP_PATH" -name "ffprobe" | grep -q .; then
    echo "ERROR: bundled ffprobe binary not found in ${APP_PATH}"
    exit 1
fi

# -------------------------------------------------------------------
# 3. Code signing (optional)
# -------------------------------------------------------------------
if [ "$CODE_SIGN" = true ]; then
    echo "==> Code signing..."
    while IFS= read -r helper_binary; do
        codesign --force --options runtime \
            --sign "${CODESIGN_IDENTITY}" \
            "$helper_binary"
    done < <(find "$APP_PATH" -type f \( -name "ffmpeg" -o -name "ffprobe" \) | sort)

    codesign --force --deep --options runtime \
        --entitlements "${SCRIPT_DIR}/entitlements.plist" \
        --sign "${CODESIGN_IDENTITY}" \
        "$APP_PATH"

    echo "==> Verifying signature..."
    codesign --verify --deep --strict --verbose=2 "$APP_PATH"
    echo "    Signature OK"
fi

# -------------------------------------------------------------------
# 4. Create DMG (optional)
# -------------------------------------------------------------------
if [ "$BUILD_DMG" = true ]; then
    DMG_NAME="Scene-Ripper-${APP_VERSION}-arm64.dmg"
    DMG_PATH="${PYINSTALLER_DIST_PATH}/${DMG_NAME}"

    # Remove old DMG if it exists
    rm -f "$DMG_PATH"

    # Check if create-dmg is available
    if command -v create-dmg &>/dev/null; then
        echo "==> Creating branded DMG with create-dmg..."
        create-dmg \
            --volname "Scene Ripper" \
            --volicon "${PROJECT_ROOT}/assets/icon.icns" \
            --background "${PROJECT_ROOT}/assets/dmg-background.png" \
            --window-pos 200 120 \
            --window-size 660 400 \
            --icon-size 80 \
            --icon "Scene Ripper.app" 180 200 \
            --app-drop-link 480 200 \
            --no-internet-enable \
            "$DMG_PATH" \
            "$APP_PATH"
    else
        echo "==> create-dmg not found, creating simple DMG with hdiutil..."
        hdiutil create -volname "Scene Ripper" \
            -srcfolder "$APP_PATH" \
            -ov -format UDZO \
            "$DMG_PATH"
    fi

    echo "==> DMG created: ${DMG_PATH}"
    ls -lh "$DMG_PATH"
fi

# -------------------------------------------------------------------
# 5. Summary
# -------------------------------------------------------------------
echo ""
echo "==> Build complete!"
echo "    .app: ${APP_PATH}"
if [ "$BUILD_DMG" = true ]; then
    echo "    DMG:  ${PYINSTALLER_DIST_PATH}/${DMG_NAME}"
fi
echo ""
echo "To test the .app:"
echo "    open \"${APP_PATH}\""
