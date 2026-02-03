"""Scene detection module wrapping PySceneDetect."""

import logging
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from scenedetect import detect, AdaptiveDetector, ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.video_stream import VideoStream
from scenedetect.backends.opencv import VideoStreamCv2

from models.clip import Clip, Source

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for scene detection."""

    # Sensitivity: lower = more sensitive (more scenes detected)
    # Range: 1.0 (very sensitive) to 10.0 (less sensitive)
    threshold: float = 3.0

    # Minimum scene length in frames
    min_scene_length: int = 15

    # Use adaptive detector (better for dynamic footage)
    use_adaptive: bool = True

    # Minimum content value before a cut is detected (AdaptiveDetector only)
    # Lower values = more sensitive to soft transitions like crossfades
    min_content_val: float = 15.0

    # Number of frames to average together (AdaptiveDetector only)
    # Higher values = better for slow fades, but may miss quick cuts
    window_width: int = 2

    @classmethod
    def default(cls) -> "DetectionConfig":
        """Standard detection settings for hard cuts."""
        return cls()

    @classmethod
    def crossfade(cls) -> "DetectionConfig":
        """Optimized for footage with crossfades and soft transitions.

        Uses lower thresholds and wider averaging window to catch
        gradual scene changes that standard detection might miss.
        """
        return cls(
            threshold=1.8,
            min_scene_length=15,
            use_adaptive=True,
            min_content_val=10.0,
            window_width=4,
        )

    @classmethod
    def sensitive(cls) -> "DetectionConfig":
        """Very sensitive detection for maximum scene granularity."""
        return cls(
            threshold=1.5,
            min_scene_length=10,
            use_adaptive=True,
            min_content_val=8.0,
            window_width=2,
        )


@dataclass
class KaraokeDetectionConfig:
    """Configuration for karaoke/text-based scene detection.

    Detects scene cuts based on when on-screen text changes rather than
    visual scene changes. Optimized for karaoke videos, lyrics overlays,
    and subtitle-based content.
    """

    # Region of interest (0.0 = top, 1.0 = bottom)
    roi_top_percent: float = 0.0  # Full frame by default (text position varies)
    roi_bottom_percent: float = 1.0
    roi_left_percent: float = 0.0
    roi_right_percent: float = 1.0

    # Text comparison threshold (0-100, below this = new scene)
    text_similarity_threshold: float = 60.0

    # Pixel change threshold to trigger OCR (0.0-1.0)
    pixel_change_threshold: float = 0.02

    # Minimum frames between cuts (~0.5 sec at 30fps)
    min_scene_frames: int = 15

    # Confirmation frames: require N consecutive frames with same new text
    # Reduces false positives from OCR jitter during fade-in/out
    confirm_frames: int = 3

    # Cut offset: shift cuts backward to catch fade-in starts
    # OCR detects text mid-fade, so we shift back to get the actual start
    cut_offset: int = 5

    # PaddleOCR 3.x device parameter: "gpu:0", "gpu:1", or "cpu"
    device: str = "gpu:0"

    # OCR language (PaddleOCR language code)
    language: str = "en"

    # Frame skip for faster processing (1 = every frame, 2 = every other, etc.)
    frame_skip: int = 1


class KaraokeTextDetector:
    """Detect scene changes based on on-screen text changes.

    Optimized for karaoke-style videos where text overlays change
    but the visual background remains consistent.

    Uses PaddleOCR for text extraction and RapidFuzz for text comparison.
    Includes pixel pre-filter to skip redundant OCR calls.
    """

    def __init__(self, config: Optional[KaraokeDetectionConfig] = None):
        self.config = config or KaraokeDetectionConfig()
        self._ocr = None  # Lazy init
        self._last_roi = None
        self._last_text = ""
        self._frames_since_cut = 0
        self._scene_cuts = []
        # Confirmation frame tracking
        self._pending_text = ""
        self._pending_count = 0
        # Stats
        self._frames_processed = 0
        self._ocr_calls = 0

    def _get_ocr(self):
        """Lazy-initialize PaddleOCR 3.x."""
        if self._ocr is None:
            import os
            os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "WARNING")
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            logging.getLogger("paddle").setLevel(logging.WARNING)

            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(device=self.config.device)
            logger.info(f"Initialized PaddleOCR with device={self.config.device}")
        return self._ocr

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region of interest from frame."""
        h, w = frame.shape[:2]
        top = int(h * self.config.roi_top_percent)
        bottom = int(h * self.config.roi_bottom_percent)
        left = int(w * self.config.roi_left_percent)
        right = int(w * self.config.roi_right_percent)
        return frame[top:bottom, left:right]

    def _pixels_changed(self, roi: np.ndarray) -> bool:
        """Check if ROI pixels changed significantly from last frame."""
        if self._last_roi is None:
            self._last_roi = roi.copy()
            return True

        # Handle size mismatch (shouldn't happen, but be safe)
        if roi.shape != self._last_roi.shape:
            self._last_roi = roi.copy()
            return True

        # Calculate normalized pixel difference
        diff = np.abs(roi.astype(float) - self._last_roi.astype(float))
        change_ratio = np.mean(diff) / 255.0

        self._last_roi = roi.copy()
        return change_ratio > self.config.pixel_change_threshold

    def _extract_text(self, roi: np.ndarray) -> str:
        """Extract text from ROI using PaddleOCR 3.x."""
        self._ocr_calls += 1
        ocr = self._get_ocr()

        try:
            result = ocr.predict(roi)  # PaddleOCR 3.x API
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return ""

        if not result:
            return ""

        # PaddleOCR 3.x returns list of dicts with 'rec_texts' key
        texts = []
        for item in result:
            if isinstance(item, dict) and "rec_texts" in item:
                texts.extend(item["rec_texts"])
            elif isinstance(item, list):
                # Fallback for different format
                for line in item:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        if isinstance(text, str):
                            texts.append(text)

        return " ".join(texts)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-100)."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0

        from rapidfuzz import fuzz
        return fuzz.ratio(text1.lower(), text2.lower())

    def process_frame(self, frame_num: int, frame: np.ndarray) -> bool:
        """Process a frame and return True if scene cut detected.

        Args:
            frame_num: Current frame number
            frame: BGR image array

        Returns:
            True if this frame starts a new scene
        """
        self._frames_processed += 1
        self._frames_since_cut += 1

        # Enforce minimum scene length
        if self._frames_since_cut < self.config.min_scene_frames:
            return False

        # Extract ROI
        roi = self._extract_roi(frame)

        # Fast path: skip if pixels haven't changed
        if not self._pixels_changed(roi):
            return False

        # Run OCR
        current_text = self._extract_text(roi)

        # Compare with previous text
        similarity = self._text_similarity(current_text, self._last_text)

        # Detect scene cut if text changed significantly
        if similarity < self.config.text_similarity_threshold:
            # Confirmation frames: require N consecutive frames with same new text
            if current_text == self._pending_text:
                self._pending_count += 1
            else:
                self._pending_text = current_text
                self._pending_count = 1

            # Only register cut after confirmation threshold reached
            if self._pending_count >= self.config.confirm_frames:
                self._last_text = current_text
                self._frames_since_cut = 0
                # Apply cut offset to catch fade-in starts
                actual_cut_frame = max(0, frame_num - self.config.cut_offset)
                self._scene_cuts.append(actual_cut_frame)
                # Reset confirmation state
                self._pending_text = ""
                self._pending_count = 0
                logger.debug(f"Cut detected at frame {actual_cut_frame}: '{current_text[:50]}...'")
                return True
        else:
            # Text is similar to last confirmed text, reset pending state
            self._pending_text = ""
            self._pending_count = 0
            self._last_text = current_text

        return False

    def get_scene_list(self) -> list[int]:
        """Return list of frame numbers where scenes start."""
        return self._scene_cuts.copy()

    def get_stats(self) -> dict:
        """Return processing statistics."""
        skip_ratio = 1 - (self._ocr_calls / max(1, self._frames_processed))
        return {
            "frames_processed": self._frames_processed,
            "ocr_calls": self._ocr_calls,
            "ocr_skip_ratio": skip_ratio,
            "cuts_detected": len(self._scene_cuts),
        }

    def reset(self):
        """Reset detector state for new video."""
        self._last_roi = None
        self._last_text = ""
        self._frames_since_cut = 0
        self._scene_cuts = []
        self._pending_text = ""
        self._pending_count = 0
        self._frames_processed = 0
        self._ocr_calls = 0


class SceneDetector:
    """Detects scenes in video files using PySceneDetect."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect_scenes(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> tuple[Source, list[Clip]]:
        """
        Detect scenes in a video file.

        Args:
            video_path: Path to the video file
            progress_callback: Optional callback receiving progress (0.0 to 1.0)

        Returns:
            Tuple of (Source metadata, list of detected Clips)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video to get metadata
        video = VideoStreamCv2(str(video_path))
        source = Source(
            file_path=video_path,
            duration_seconds=video.duration.get_seconds(),
            fps=video.frame_rate,
            width=video.frame_size[0],
            height=video.frame_size[1],
        )

        # Create detector based on config
        if self.config.use_adaptive:
            detector = AdaptiveDetector(
                adaptive_threshold=self.config.threshold,
                min_scene_len=self.config.min_scene_length,
                min_content_val=self.config.min_content_val,
                window_width=self.config.window_width,
            )
        else:
            detector = ContentDetector(
                threshold=self.config.threshold * 9,  # Scale for ContentDetector
                min_scene_len=self.config.min_scene_length,
            )

        # Detect scenes
        scene_list = detect(
            str(video_path),
            detector,
            show_progress=False,
        )

        # Convert to Clip objects
        clips = []
        for start_timecode, end_timecode in scene_list:
            clip = Clip(
                source_id=source.id,
                start_frame=start_timecode.get_frames(),
                end_frame=end_timecode.get_frames(),
            )
            clips.append(clip)

        # Fallback: if no scenes detected, create a single clip spanning the entire video
        if not clips:
            total_frames = source.total_frames
            fallback_clip = Clip(
                source_id=source.id,
                start_frame=0,
                end_frame=total_frames,
            )
            clips = [fallback_clip]
            logger.info(f"No scenes detected in {video_path.name}, created full-video clip")

        return source, clips

    def detect_scenes_with_progress(
        self,
        video_path: Path,
        progress_callback: Callable[[float, str], None],
    ) -> tuple[Source, list[Clip]]:
        """
        Detect scenes with detailed progress reporting.

        Args:
            video_path: Path to the video file
            progress_callback: Callback receiving (progress 0.0-1.0, status message)

        Returns:
            Tuple of (Source metadata, list of detected Clips)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        progress_callback(0.0, "Opening video...")

        # Open video to get metadata
        video = VideoStreamCv2(str(video_path))
        source = Source(
            file_path=video_path,
            duration_seconds=video.duration.get_seconds(),
            fps=video.frame_rate,
            width=video.frame_size[0],
            height=video.frame_size[1],
        )

        progress_callback(0.1, "Analyzing frames...")

        # Create detector
        if self.config.use_adaptive:
            detector = AdaptiveDetector(
                adaptive_threshold=self.config.threshold,
                min_scene_len=self.config.min_scene_length,
                min_content_val=self.config.min_content_val,
                window_width=self.config.window_width,
            )
        else:
            detector = ContentDetector(
                threshold=self.config.threshold * 9,
                min_scene_len=self.config.min_scene_length,
            )

        # Create scene manager for manual processing with progress
        scene_manager = SceneManager()
        scene_manager.add_detector(detector)

        # Open new video stream for processing
        video_stream = VideoStreamCv2(str(video_path))
        total_frames = video_stream.duration.get_frames()

        # Define progress callback for frame-by-frame updates
        last_logged_percent = -1

        def frame_callback(frame_img, frame_num):
            nonlocal last_logged_percent
            # Calculate progress (10% to 90% range for detection phase)
            progress = 0.1 + (frame_num / total_frames) * 0.8
            percent = int((frame_num / total_frames) * 100)

            # Log every 10% and update progress callback every 1%
            if percent >= last_logged_percent + 10:
                logger.info(f"Scene detection: {percent}% ({frame_num:,}/{total_frames:,} frames)")
                last_logged_percent = percent

            # Update progress callback every 1% to avoid too many UI updates
            if percent % 1 == 0:
                progress_callback(progress, f"Analyzing frames... {percent}%")

        scene_manager.detect_scenes(video_stream, show_progress=False, callback=frame_callback)

        progress_callback(0.9, "Extracting scenes...")

        # Get detected scenes
        scene_list = scene_manager.get_scene_list()

        # Convert to Clip objects
        clips = []
        for start_timecode, end_timecode in scene_list:
            clip = Clip(
                source_id=source.id,
                start_frame=start_timecode.get_frames(),
                end_frame=end_timecode.get_frames(),
            )
            clips.append(clip)

        # Fallback: if no scenes detected, create a single clip spanning the entire video
        if not clips:
            fallback_clip = Clip(
                source_id=source.id,
                start_frame=0,
                end_frame=total_frames,
            )
            clips = [fallback_clip]
            logger.info(f"No scenes detected in {video_path.name}, created full-video clip")
            progress_callback(1.0, "No scenes found - using full video as single clip")
        else:
            progress_callback(1.0, f"Found {len(clips)} scenes")

        return source, clips

    def detect_karaoke_scenes_with_progress(
        self,
        video_path: Path,
        progress_callback: Callable[[float, str], None],
        karaoke_config: Optional[KaraokeDetectionConfig] = None,
    ) -> tuple[Source, list[Clip]]:
        """
        Detect scenes based on text changes with progress reporting.

        Uses OCR to detect when on-screen text changes, optimized for
        karaoke videos, lyrics overlays, and subtitle-based content.

        Args:
            video_path: Path to the video file
            progress_callback: Callback receiving (progress 0.0-1.0, status message)
            karaoke_config: Optional karaoke detection configuration

        Returns:
            Tuple of (Source metadata, list of detected Clips)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        progress_callback(0.0, "Opening video...")

        # Open video with cv2 to get metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        source = Source(
            file_path=video_path,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
        )

        progress_callback(0.05, "Initializing text detector...")

        # Create karaoke detector
        config = karaoke_config or KaraokeDetectionConfig()
        detector = KaraokeTextDetector(config)

        progress_callback(0.1, "Analyzing text changes...")

        # Process frames
        frame_num = 0
        frame_skip = config.frame_skip
        last_progress = 0.1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply frame skip
            if frame_num % frame_skip == 0:
                detector.process_frame(frame_num, frame)

            frame_num += 1

            # Update progress periodically (every 30 frames)
            if frame_num % 30 == 0:
                progress = 0.1 + (0.8 * frame_num / total_frames)
                if progress > last_progress + 0.01:  # Update at least 1% increments
                    stats = detector.get_stats()
                    progress_callback(
                        progress,
                        f"Frame {frame_num}/{total_frames} ({stats['cuts_detected']} cuts)"
                    )
                    last_progress = progress

        cap.release()

        progress_callback(0.9, "Building clips...")

        # Get stats for logging
        stats = detector.get_stats()
        logger.info(
            f"Karaoke detection complete: {stats['frames_processed']} frames, "
            f"{stats['ocr_calls']} OCR calls ({stats['ocr_skip_ratio']:.1%} skipped), "
            f"{stats['cuts_detected']} cuts"
        )

        # Convert cut points to scene ranges (clips)
        cuts = [0] + detector.get_scene_list()
        # Add final frame if not already included
        if not cuts or cuts[-1] < total_frames - 1:
            cuts.append(total_frames)

        # Remove duplicates and sort
        cuts = sorted(set(cuts))

        clips = []
        for i in range(len(cuts) - 1):
            start_frame = cuts[i]
            end_frame = cuts[i + 1] - 1
            if end_frame > start_frame:  # Ensure valid clip
                clip = Clip(
                    source_id=source.id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                clips.append(clip)

        # Fallback: if no scenes detected, create a single clip
        if not clips:
            fallback_clip = Clip(
                source_id=source.id,
                start_frame=0,
                end_frame=total_frames,
            )
            clips = [fallback_clip]
            logger.info(f"No text changes detected in {video_path.name}, created full-video clip")
            progress_callback(1.0, "No text changes found - using full video as single clip")
        else:
            progress_callback(1.0, f"Found {len(clips)} scenes based on text changes")

        return source, clips
