#!/bin/bash
#
# Build AppImage for Scene Ripper
#
# Prerequisites:
#   - appimage-builder installed (pip install appimage-builder)
#   - Docker (for AppImage testing)
#
# Usage:
#   ./build-appimage.sh [VERSION]
#
# Example:
#   ./build-appimage.sh 0.2.0
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:-0.1.0}"

echo "Building Scene Ripper AppImage v${VERSION}"
echo "Project root: ${PROJECT_ROOT}"

cd "$PROJECT_ROOT"

# Clean previous build
echo "Cleaning previous build..."
rm -rf AppDir *.AppImage

# Create AppDir structure
echo "Creating AppDir structure..."
mkdir -p AppDir/usr/src
mkdir -p AppDir/usr/share/applications
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps
mkdir -p AppDir/usr/share/icons/hicolor/scalable/apps

# Copy application source
echo "Copying application source..."
cp -r main.py core/ models/ ui/ AppDir/usr/src/
cp requirements-core.txt AppDir/usr/src/

# Copy desktop entry
echo "Copying desktop entry..."
cp "$SCRIPT_DIR/scene-ripper.desktop" AppDir/usr/share/applications/

# Copy icon (create placeholder if doesn't exist)
if [ -f "$PROJECT_ROOT/assets/icon.png" ]; then
    cp "$PROJECT_ROOT/assets/icon.png" AppDir/usr/share/icons/hicolor/256x256/apps/scene-ripper.png
else
    echo "Warning: No icon found at assets/icon.png, using placeholder"
    # Create a simple placeholder icon using ImageMagick if available
    if command -v convert &> /dev/null; then
        convert -size 256x256 xc:#4a90d9 -gravity center \
            -pointsize 48 -fill white -annotate 0 "SR" \
            AppDir/usr/share/icons/hicolor/256x256/apps/scene-ripper.png
    fi
fi

# Create AppRun script (fallback if appimage-builder doesn't create one)
cat > AppDir/AppRun << 'APPRUN'
#!/bin/bash
APPDIR="$(dirname "$(readlink -f "$0")")"
export PATH="${APPDIR}/usr/bin:${PATH}"
export PYTHONPATH="${APPDIR}/usr/lib/python3/site-packages:${PYTHONPATH}"
export QT_PLUGIN_PATH="${APPDIR}/usr/lib/x86_64-linux-gnu/qt6/plugins"
export GST_PLUGIN_PATH="${APPDIR}/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
export LD_LIBRARY_PATH="${APPDIR}/usr/lib:${APPDIR}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
exec "${APPDIR}/usr/bin/python3.11" "${APPDIR}/usr/src/main.py" "$@"
APPRUN
chmod +x AppDir/AppRun

# Build AppImage using appimage-builder
echo "Building AppImage with appimage-builder..."
export VERSION="$VERSION"
export APPDIR="$PROJECT_ROOT/AppDir"

# Inject version into recipe (appimage-builder !ENV tag is unreliable)
RECIPE="$PROJECT_ROOT/AppImageBuilder-resolved.yml"
sed "s/\!ENV \${VERSION:-0.1.0}/$VERSION/g" "$SCRIPT_DIR/AppImageBuilder.yml" > "$RECIPE"
appimage-builder --recipe "$RECIPE" --skip-test

# Rename output. appimage-builder derives the file name from the recipe's app
# name ("Scene Ripper"), so the produced file may use a space or an underscore
# and the old fixed "Scene_Ripper-*" pattern could miss it. Find whatever was
# written instead, and fail loudly when nothing was: the previous version
# swallowed the rename with "|| true" and then announced a build that did not
# exist, so the first sign of trouble was an unrelated step four minutes later.
TARGET="Scene_Ripper-${VERSION}-x86_64.AppImage"
PRODUCED="$(find . -maxdepth 2 -name '*.AppImage' -newer "$RECIPE" -print 2>/dev/null | head -n1)"
if [ -z "$PRODUCED" ]; then
    echo "ERROR: appimage-builder did not produce an .AppImage" >&2
    echo "Searched $PROJECT_ROOT (depth 2). Contents:" >&2
    find . -maxdepth 2 -name '*.AppImage*' -o -maxdepth 1 -name 'AppDir*' >&2 || true
    exit 1
fi
if [ "$PRODUCED" != "./$TARGET" ]; then
    mv "$PRODUCED" "$TARGET"
fi
test -s "$TARGET" || { echo "ERROR: $TARGET is empty" >&2; exit 1; }

echo ""
echo "Build complete!"
echo "Output: $TARGET ($(du -h "$TARGET" | cut -f1))"
echo ""
echo "To run:"
echo "  chmod +x Scene_Ripper-${VERSION}-x86_64.AppImage"
echo "  ./Scene_Ripper-${VERSION}-x86_64.AppImage"
