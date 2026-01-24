"""Scene detection module wrapping PySceneDetect."""

from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

from scenedetect import detect, AdaptiveDetector, ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.video_stream import VideoStream
from scenedetect.backends.opencv import VideoStreamCv2

from models.clip import Clip, Source


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

        scene_manager.detect_scenes(video_stream, show_progress=False)

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

        progress_callback(1.0, f"Found {len(clips)} scenes")

        return source, clips
