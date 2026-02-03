#!/usr/bin/env python3
"""
Prototype: Karaoke/Text-based Scene Detection

Detects scene cuts based on when on-screen text changes,
rather than visual scene changes.

Requires PaddleOCR 3.x (API changed significantly from 2.x).

Usage:
    python prototypes/karaoke_text_detector.py video.mp4
    python prototypes/karaoke_text_detector.py video.mp4 --output cuts.json
    python prototypes/karaoke_text_detector.py video.mp4 --preview

    # Split video into clips at detected cut points
    python prototypes/karaoke_text_detector.py video.mp4 --split
    python prototypes/karaoke_text_detector.py video.mp4 --split --split-dir ./clips

    # Restrict to bottom 25% (faster, for standard subtitles)
    python prototypes/karaoke_text_detector.py video.mp4 --roi-top 0.75

    # Use CPU instead of GPU
    python prototypes/karaoke_text_detector.py video.mp4 --device cpu

Dependencies:
    pip install paddlepaddle paddleocr rapidfuzz opencv-python numpy
    FFmpeg (for --split option)
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectionConfig:
    """Configuration for text-based scene detection."""

    roi_top: float = 0.0  # Top of ROI (0.0 = full frame, default)
    roi_bottom: float = 1.0  # Bottom of ROI
    roi_left: float = 0.0  # Left of ROI
    roi_right: float = 1.0  # Right of ROI
    similarity_threshold: float = 60.0  # Text similarity below this = new scene
    pixel_threshold: float = 0.02  # Pixel change to trigger OCR (lower = more sensitive)
    min_scene_frames: int = 15  # Min frames between cuts
    confirm_frames: int = 2  # Require N consecutive different frames to confirm cut
    cut_offset: int = 5  # Shift cuts backward by N frames (compensate for fade-in delay)
    device: str = "gpu:0"  # "gpu:0", "cpu", etc.
    frame_skip: int = 1  # Process every Nth frame (1 = all frames)


@dataclass
class SceneCut:
    """A detected scene cut."""

    frame: int
    timecode: str
    text_before: str
    text_after: str
    similarity: float


class KaraokeTextDetector:
    """Detect scene cuts based on text changes."""

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self._ocr = None
        self._last_roi = None
        self._last_confirmed_text = ""  # Text from last confirmed scene
        self._pending_text = ""  # Text we're waiting to confirm
        self._pending_frame = 0  # Frame where pending text was first seen
        self._confirm_count = 0  # How many frames we've seen pending_text
        self._frames_since_cut = 0
        self.cuts: list[SceneCut] = []
        self.fps = 30.0  # Updated when processing video
        self._ocr_calls = 0
        self._frames_processed = 0

    def _get_ocr(self):
        """Lazy-initialize PaddleOCR (v3.x API)."""
        if self._ocr is None:
            import os
            import logging

            # Suppress PaddleOCR/PaddlePaddle logging
            os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "WARNING")
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            logging.getLogger("paddle").setLevel(logging.WARNING)

            from paddleocr import PaddleOCR

            print("Initializing PaddleOCR (first run may download models)...")
            self._ocr = PaddleOCR(device=self.config.device)
        return self._ocr

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract region of interest from frame."""
        h, w = frame.shape[:2]
        top = int(h * self.config.roi_top)
        bottom = int(h * self.config.roi_bottom)
        left = int(w * self.config.roi_left)
        right = int(w * self.config.roi_right)
        return frame[top:bottom, left:right]

    def _pixels_changed(self, roi: np.ndarray) -> bool:
        """Check if ROI pixels changed significantly."""
        if self._last_roi is None:
            self._last_roi = roi.copy()
            return True

        if roi.shape != self._last_roi.shape:
            self._last_roi = roi.copy()
            return True

        diff = np.abs(roi.astype(float) - self._last_roi.astype(float))
        change_ratio = np.mean(diff) / 255.0

        self._last_roi = roi.copy()
        return change_ratio > self.config.pixel_threshold

    def _extract_text(self, roi: np.ndarray) -> str:
        """Extract text from ROI using OCR (PaddleOCR v3.x API)."""
        self._ocr_calls += 1
        ocr = self._get_ocr()

        # PaddleOCR v3.x uses predict() instead of ocr()
        result = ocr.predict(roi)

        if not result:
            return ""

        # v3.x returns a list of result objects
        texts = []
        for item in result:
            # Each item has 'rec_texts' attribute with recognized text
            if hasattr(item, "rec_texts"):
                texts.extend(item.rec_texts)
            # Fallback for dict-style results
            elif isinstance(item, dict) and "rec_texts" in item:
                texts.extend(item["rec_texts"])

        return " ".join(texts)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-100)."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0

        from rapidfuzz import fuzz

        return fuzz.ratio(text1.lower(), text2.lower())

    def _frame_to_timecode(self, frame: int) -> str:
        """Convert frame number to timecode string."""
        total_seconds = frame / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int((total_seconds % 1) * self.fps)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def process_frame(self, frame_num: int, frame: np.ndarray) -> Optional[SceneCut]:
        """Process a frame, return SceneCut if detected."""
        self._frames_processed += 1
        self._frames_since_cut += 1

        # Enforce minimum scene length
        if self._frames_since_cut < self.config.min_scene_frames:
            return None

        # Extract ROI
        roi = self._extract_roi(frame)

        # Fast path: skip if pixels haven't changed
        if not self._pixels_changed(roi):
            return None

        # Run OCR
        current_text = self._extract_text(roi)

        # Compare with last confirmed text
        similarity_to_confirmed = self._text_similarity(current_text, self._last_confirmed_text)

        # If text is similar to confirmed, reset any pending detection
        if similarity_to_confirmed >= self.config.similarity_threshold:
            self._pending_text = ""
            self._confirm_count = 0
            return None

        # Text is different from confirmed - check if it matches pending
        if self._pending_text:
            similarity_to_pending = self._text_similarity(current_text, self._pending_text)

            if similarity_to_pending >= self.config.similarity_threshold:
                # Same as pending text, increment confirmation
                self._confirm_count += 1

                # Check if we have enough confirmations
                if self._confirm_count >= self.config.confirm_frames:
                    # Confirmed! Register the cut
                    # Apply offset to shift cut backward (compensate for fade-in)
                    cut_frame = max(0, self._pending_frame - self.config.cut_offset)

                    cut = SceneCut(
                        frame=cut_frame,
                        timecode=self._frame_to_timecode(cut_frame),
                        text_before=self._last_confirmed_text,
                        text_after=self._pending_text,
                        similarity=self._text_similarity(self._pending_text, self._last_confirmed_text),
                    )

                    # Update state
                    self._last_confirmed_text = self._pending_text
                    self._pending_text = ""
                    self._confirm_count = 0
                    self._frames_since_cut = 0
                    self.cuts.append(cut)
                    return cut
            else:
                # Different from both confirmed and pending - new candidate
                self._pending_text = current_text
                self._pending_frame = frame_num
                self._confirm_count = 1
        else:
            # No pending text, start a new candidate
            self._pending_text = current_text
            self._pending_frame = frame_num
            self._confirm_count = 1

        return None

    def process_video(
        self, video_path: Path, progress_callback=None
    ) -> list[SceneCut]:
        """Process entire video and return list of cuts."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping for faster processing
            if frame_num % self.config.frame_skip == 0:
                self.process_frame(frame_num, frame)

            if progress_callback and frame_num % 30 == 0:
                progress_callback(frame_num, total_frames)

            frame_num += 1

        cap.release()

        return self.cuts

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "frames_processed": self._frames_processed,
            "ocr_calls": self._ocr_calls,
            "ocr_skip_ratio": (
                1 - (self._ocr_calls / self._frames_processed)
                if self._frames_processed > 0
                else 0
            ),
            "cuts_detected": len(self.cuts),
        }


def draw_roi_preview(frame: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Draw ROI rectangle on frame for preview."""
    h, w = frame.shape[:2]
    top = int(h * config.roi_top)
    bottom = int(h * config.roi_bottom)
    left = int(w * config.roi_left)
    right = int(w * config.roi_right)

    preview = frame.copy()

    # Determine if full frame
    is_full_frame = config.roi_top == 0.0 and config.roi_bottom == 1.0

    if is_full_frame:
        # Just add label for full frame mode
        cv2.putText(
            preview,
            "Full frame mode - detecting text anywhere",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            preview,
            "ROI (text region)",
            (left + 10, top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return preview


def split_video_at_cuts(
    video_path: Path,
    cuts: list[SceneCut],
    output_dir: Path,
    fps: float,
    total_frames: int,
    fast_mode: bool = False,
) -> list[Path]:
    """Split video into clips at detected cut points using FFmpeg.

    Args:
        video_path: Source video file
        cuts: List of detected scene cuts
        output_dir: Directory to save clips
        fps: Video framerate
        total_frames: Total frames in video
        fast_mode: If True, use stream copy (keyframe-only, fast but inaccurate)
                   If False, re-encode for frame-accurate cuts (slower)

    Returns:
        List of created clip file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of segments: (start_frame, end_frame)
    segments = []
    cut_frames = [0] + [c.frame for c in cuts] + [total_frames]

    for i in range(len(cut_frames) - 1):
        start = cut_frames[i]
        end = cut_frames[i + 1]
        if end > start:
            segments.append((start, end))

    created_files = []
    video_stem = video_path.stem

    mode_str = "fast (keyframe)" if fast_mode else "frame-accurate (re-encode)"
    print(f"\nSplitting video into {len(segments)} clips ({mode_str})...")

    for i, (start_frame, end_frame) in enumerate(segments, 1):
        # Convert frames to seconds
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        # Always output as mp4 for compatibility
        output_file = output_dir / f"{video_stem}_clip_{i:03d}.mp4"

        if fast_mode:
            # Fast but only cuts on keyframes
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_file),
            ]
        else:
            # Frame-accurate cutting with re-encode
            # Input seeking (-ss before -i) for speed, then precise trim
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",  # High quality
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(output_file),
            ]

        print(f"  [{i}/{len(segments)}] {output_file.name} ({duration:.2f}s)")

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            created_files.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"    Error: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        except FileNotFoundError:
            print("    Error: FFmpeg not found. Please install FFmpeg.")
            break

    return created_files


def main():
    parser = argparse.ArgumentParser(
        description="Detect scene cuts based on text changes (karaoke-style)"
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--roi-top",
        type=float,
        default=0.0,
        help="Top of text region (0.0=full frame, 0.75=bottom 25%%, default: 0.0)",
    )
    parser.add_argument(
        "--roi-bottom",
        type=float,
        default=1.0,
        help="Bottom of text region (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Text similarity threshold (0-100, default: 60). Lower = more sensitive.",
    )
    parser.add_argument(
        "--min-scene",
        type=int,
        default=15,
        help="Minimum frames between cuts (default: 15)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=2,
        help="Require N consecutive frames with new text to confirm cut (default: 2). Higher = fewer false positives.",
    )
    parser.add_argument(
        "--cut-offset",
        type=int,
        default=5,
        help="Shift cuts backward by N frames (default: 5). Compensates for fade-in detection delay.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="Device for OCR: 'gpu:0', 'cpu', etc. (default: gpu:0)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file for cuts"
    )
    parser.add_argument(
        "--split", action="store_true", help="Split video into clips at cut points (frame-accurate, re-encodes)"
    )
    parser.add_argument(
        "--split-fast",
        action="store_true",
        help="Use fast splitting (keyframe-only, may be inaccurate)",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Output directory for clips (default: ./[video_name]_clips)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview window with ROI"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detected text"
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    config = DetectionConfig(
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        similarity_threshold=args.threshold,
        min_scene_frames=args.min_scene,
        confirm_frames=args.confirm_frames,
        cut_offset=args.cut_offset,
        frame_skip=args.frame_skip,
        device=args.device,
    )

    print(f"Processing: {args.video}")
    if config.roi_top == 0.0 and config.roi_bottom == 1.0:
        print("ROI: Full frame (text can be anywhere)")
    else:
        print(f"ROI: top={config.roi_top}, bottom={config.roi_bottom}")
    print(f"Threshold: {config.similarity_threshold}% (lower = more sensitive)")
    print(f"Min scene: {config.min_scene_frames} frames")
    print(f"Confirm frames: {config.confirm_frames} (higher = fewer false positives)")
    print(f"Cut offset: -{config.cut_offset} frames (compensates for fade-in)")
    if config.frame_skip > 1:
        print(f"Frame skip: every {config.frame_skip} frames")
    print()

    detector = KaraokeTextDetector(config)

    def progress(current, total):
        pct = (current / total) * 100 if total > 0 else 0
        print(
            f"\rProgress: {pct:.1f}% ({current}/{total} frames)", end="", flush=True
        )

    # Optional: show preview of first frame with ROI
    if args.preview:
        cap = cv2.VideoCapture(str(args.video))
        ret, frame = cap.read()
        if ret:
            preview = draw_roi_preview(frame, config)
            cv2.imshow("ROI Preview (press any key to continue)", preview)
            print("Showing ROI preview. Press any key to start processing...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()

    # Process video
    cuts = detector.process_video(args.video, progress_callback=progress)
    print()  # newline after progress

    # Print stats
    stats = detector.get_stats()
    print(f"\nProcessing stats:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  OCR calls: {stats['ocr_calls']}")
    print(f"  OCR skip ratio: {stats['ocr_skip_ratio']:.1%}")

    # Print results
    print(f"\nDetected {len(cuts)} scene cuts:\n")
    for i, cut in enumerate(cuts, 1):
        print(f"Cut {i}: Frame {cut.frame} ({cut.timecode})")
        print(f"  Similarity: {cut.similarity:.1f}%")
        if args.verbose:
            before = (
                f"{cut.text_before[:50]}..."
                if len(cut.text_before) > 50
                else cut.text_before
            )
            after = (
                f"{cut.text_after[:50]}..."
                if len(cut.text_after) > 50
                else cut.text_after
            )
            print(f"  Before: {before or '(empty)'}")
            print(f"  After:  {after or '(empty)'}")
        print()

    # Save to JSON if requested
    if args.output:
        output_data = {
            "video": str(args.video),
            "config": asdict(config),
            "fps": detector.fps,
            "stats": stats,
            "cuts": [asdict(c) for c in cuts],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved cuts to: {args.output}")

    # Split video if requested
    if args.split or args.split_fast:
        if not cuts:
            print("\nNo cuts detected, nothing to split.")
        else:
            # Get total frames for the last segment
            cap = cv2.VideoCapture(str(args.video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Determine output directory
            split_dir = args.split_dir
            if split_dir is None:
                split_dir = args.video.parent / f"{args.video.stem}_clips"

            created = split_video_at_cuts(
                args.video,
                cuts,
                split_dir,
                detector.fps,
                total_frames,
                fast_mode=args.split_fast,
            )
            print(f"\nCreated {len(created)} clips in: {split_dir}")


if __name__ == "__main__":
    main()
