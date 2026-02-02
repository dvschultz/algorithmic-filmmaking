#!/bin/bash
# Download sample video clips from Mixkit (free, no login required)
# These are curated to represent different shot types

set -e

OUTPUT_DIR="${1:-test_clips}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading test clips to: $OUTPUT_DIR"
echo ""

# Function to download from Mixkit
download_mixkit() {
    local name="$1"
    local url="$2"
    local output="$OUTPUT_DIR/${name}.mp4"

    if [ -f "$output" ]; then
        echo "[SKIP] $name - already exists"
        return
    fi

    echo "[DOWNLOADING] $name"
    curl -L -o "$output" "$url" 2>/dev/null || wget -q -O "$output" "$url"
    local size=$(du -h "$output" | cut -f1)
    echo "  ✓ Downloaded ($size)"
}

echo "=== CLOSE-UP SHOTS ==="
# Extreme close-up - eye detail
download_mixkit "07_extreme_closeup_eye" \
    "https://assets.mixkit.co/videos/41180/41180-720.mp4"

# Close-up - face
download_mixkit "06_closeup_face" \
    "https://assets.mixkit.co/videos/34563/34563-720.mp4"

echo ""
echo "=== MEDIUM SHOTS ==="
# Medium close-up - head and shoulders
download_mixkit "05_medium_closeup" \
    "https://assets.mixkit.co/videos/42401/42401-720.mp4"

# Medium shot - waist up
download_mixkit "04_medium" \
    "https://assets.mixkit.co/videos/34487/34487-720.mp4"

echo ""
echo "=== WIDE/LONG SHOTS ==="
# Full shot - full body
download_mixkit "03_full_body" \
    "https://assets.mixkit.co/videos/34473/34473-720.mp4"

# Long shot - person in environment
download_mixkit "02_long" \
    "https://assets.mixkit.co/videos/42387/42387-720.mp4"

# Extreme wide - landscape/establishing
download_mixkit "01_extreme_long" \
    "https://assets.mixkit.co/videos/44426/44426-720.mp4"

echo ""
echo "=============================================="
echo "Download complete!"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "No files downloaded"
echo ""
echo "Test with VideoMAE:"
echo "  cd replicate/shot-classifier"
echo "  cog predict -i video=@../../$OUTPUT_DIR/06_closeup_face.mp4"
