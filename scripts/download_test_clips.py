#!/usr/bin/env python3
"""Download sample video clips for each shot type from free sources.

Uses Pexels API (free, no attribution required for testing).
Get a free API key at: https://www.pexels.com/api/

Usage:
    # Set your Pexels API key
    export PEXELS_API_KEY="your_key_here"

    # Run the script
    python scripts/download_test_clips.py
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# Shot type search queries optimized for Pexels
SHOT_TYPE_QUERIES = {
    "01_extreme_long": "landscape aerial drone wide establishing",
    "02_long": "person walking full body",
    "03_full": "person standing full body portrait",
    "04_medium": "person talking waist up conversation",
    "05_medium_closeup": "person face shoulders portrait",
    "06_closeup": "face close up portrait",
    "07_extreme_closeup": "eye close up detail macro face",
}

OUTPUT_DIR = Path(__file__).parent.parent / "test_clips"


def search_pexels_videos(query: str, api_key: str, per_page: int = 3) -> list:
    """Search Pexels for videos matching query."""
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}&orientation=landscape"

    req = urllib.request.Request(url)
    req.add_header("Authorization", api_key)

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("videos", [])
    except Exception as e:
        print(f"  Error searching: {e}")
        return []


def download_video(url: str, output_path: Path) -> bool:
    """Download video from URL."""
    try:
        print(f"  Downloading to {output_path.name}...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def main():
    api_key = os.environ.get("PEXELS_API_KEY")

    if not api_key:
        print("=" * 60)
        print("PEXELS API KEY REQUIRED")
        print("=" * 60)
        print()
        print("1. Get a free API key at: https://www.pexels.com/api/")
        print("2. Set it as environment variable:")
        print("   export PEXELS_API_KEY='your_key_here'")
        print("3. Run this script again")
        print()
        print("Alternative: Download manually from these sources:")
        print("- https://www.pexels.com/search/videos/close%20up%20face/")
        print("- https://www.pexels.com/search/videos/wide%20shot%20landscape/")
        print("- https://www.pexels.com/search/videos/medium%20shot%20person/")
        print("- https://pixabay.com/videos/")
        print("- https://mixkit.co/free-stock-video/")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading test clips to: {OUTPUT_DIR}")
    print()

    for shot_type, query in SHOT_TYPE_QUERIES.items():
        print(f"[{shot_type}] Searching: '{query}'")

        videos = search_pexels_videos(query, api_key, per_page=1)

        if not videos:
            print(f"  No videos found")
            continue

        video = videos[0]

        # Get the smallest video file (for faster testing)
        video_files = sorted(
            video.get("video_files", []),
            key=lambda x: x.get("width", 9999)
        )

        if not video_files:
            print(f"  No downloadable files")
            continue

        # Prefer SD quality for faster downloads
        video_file = None
        for vf in video_files:
            if vf.get("quality") == "sd" or vf.get("width", 0) <= 960:
                video_file = vf
                break

        if not video_file:
            video_file = video_files[0]

        download_url = video_file.get("link")
        output_path = OUTPUT_DIR / f"{shot_type}.mp4"

        if download_video(download_url, output_path):
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  ✓ Downloaded ({size_mb:.1f} MB)")

        print()

    print("=" * 60)
    print("Download complete!")
    print(f"Test clips saved to: {OUTPUT_DIR}")
    print()
    print("Next: Test your VideoMAE model with these clips:")
    print("  cd replicate/shot-classifier")
    print("  cog predict -i video=@../../test_clips/06_closeup.mp4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
