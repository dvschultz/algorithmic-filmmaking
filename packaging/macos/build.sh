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
#   APP_VERSION       Version string (default: 0.2.0)
#   CODESIGN_IDENTITY Signing identity (default: ad-hoc "-")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

APP_VERSION="${APP_VERSION:-0.2.0}"
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

# -------------------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------------------
echo "==> Installing core dependencies..."
pip install -r "${PROJECT_ROOT}/requirements-core.txt" --quiet
pip install pyinstaller --quiet

# -------------------------------------------------------------------
# 2. Run PyInstaller
# -------------------------------------------------------------------
echo "==> Running PyInstaller..."
export APP_VERSION
pyinstaller "${SCRIPT_DIR}/scene_ripper.spec" \
    --distpath "${PROJECT_ROOT}/dist" \
    --workpath "${PROJECT_ROOT}/build" \
    --noconfirm

APP_PATH="${PROJECT_ROOT}/dist/Scene Ripper.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: .app bundle not found at ${APP_PATH}"
    exit 1
fi

echo "==> .app bundle created: ${APP_PATH}"
du -sh "$APP_PATH"

# -------------------------------------------------------------------
# 3. Code signing (optional)
# -------------------------------------------------------------------
if [ "$CODE_SIGN" = true ]; then
    echo "==> Code signing..."
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
    DMG_PATH="${PROJECT_ROOT}/dist/${DMG_NAME}"

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
    echo "    DMG:  ${PROJECT_ROOT}/dist/${DMG_NAME}"
fi
echo ""
echo "To test the .app:"
echo "    open \"${APP_PATH}\""
