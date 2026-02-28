#!/usr/bin/env python3
"""Transcribe a video using OpenAI Whisper API.

Usage:
    python scripts/transcribe_whisper.py /path/to/video.mp4

Outputs transcript.txt and transcript.json in the same directory as the video.

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


# Whisper API limit is 25MB
MAX_FILE_SIZE = 25 * 1024 * 1024


def extract_audio(video_path: Path, output_path: Path) -> None:
    """Extract audio from video as mp3 using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                  # no video
        "-acodec", "libmp3lame",
        "-ab", "64k",           # 64kbps keeps file small
        "-ar", "16000",         # 16kHz is sufficient for speech
        "-ac", "1",             # mono
        "-y",                   # overwrite
        str(output_path),
    ]
    print(f"Extracting audio from {video_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        sys.exit(1)


def split_audio(audio_path: Path, chunk_dir: Path, chunk_minutes: int = 20) -> list[Path]:
    """Split audio into chunks if it exceeds the API size limit."""
    size = audio_path.stat().st_size
    if size <= MAX_FILE_SIZE:
        return [audio_path]

    print(f"Audio is {size / 1024 / 1024:.1f}MB (limit {MAX_FILE_SIZE // 1024 // 1024}MB), splitting into chunks...")
    chunk_dir.mkdir(exist_ok=True)

    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    chunk_seconds = chunk_minutes * 60
    chunks = []

    for i, start in enumerate(range(0, int(duration), chunk_seconds)):
        chunk_path = chunk_dir / f"chunk_{i:03d}.mp3"
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-acodec", "copy",
            "-y",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        chunks.append(chunk_path)
        print(f"  Created chunk {i + 1} ({start // 60}m - {min(start + chunk_seconds, int(duration)) // 60}m)")

    return chunks


def transcribe(audio_path: Path, client: OpenAI) -> dict:
    """Send audio to Whisper API and return response."""
    print(f"Transcribing {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.1f}MB)...")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return response


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe video using OpenAI Whisper API")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as video)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: {args.video} not found")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    output_dir = args.output_dir or args.video.parent
    stem = args.video.stem

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Extract audio
        audio_path = tmp_path / "audio.mp3"
        extract_audio(args.video, audio_path)
        print(f"Audio extracted: {audio_path.stat().st_size / 1024 / 1024:.1f}MB")

        # Split if needed
        chunks = split_audio(audio_path, tmp_path / "chunks")

        # Transcribe each chunk
        all_segments = []
        full_text_parts = []
        time_offset = 0.0

        for chunk_path in chunks:
            response = transcribe(chunk_path, client)

            # Accumulate segments with offset
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    all_segments.append({
                        "start": seg.start + time_offset,
                        "end": seg.end + time_offset,
                        "text": seg.text,
                    })
                # Update offset for next chunk
                time_offset = all_segments[-1]["end"]

            full_text_parts.append(response.text)

    # Write plain text with timestamps
    txt_path = output_dir / f"{stem}_transcript.txt"
    with open(txt_path, "w") as f:
        for seg in all_segments:
            ts = format_timestamp(seg["start"])
            f.write(f"[{ts}] {seg['text'].strip()}\n")
    print(f"\nTranscript saved: {txt_path}")

    # Write full JSON for reference
    json_path = output_dir / f"{stem}_transcript.json"
    with open(json_path, "w") as f:
        json.dump({
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "source": str(args.video),
        }, f, indent=2)
    print(f"JSON saved: {json_path}")

    # Print summary
    duration = all_segments[-1]["end"] if all_segments else 0
    print(f"\nDuration: {format_timestamp(duration)}")
    print(f"Segments: {len(all_segments)}")
    print(f"\nTo read the transcript:\n  cat {txt_path}")


if __name__ == "__main__":
    main()
