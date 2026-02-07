"""Unified input for analysis operations on Clips or Frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.clip import Clip, Source
    from models.frame import Frame


@dataclass
class AnalysisTarget:
    """Unified input for analysis operations - either a Clip or a Frame.

    Workers accept a list of AnalysisTarget objects as an alternative to
    their existing clip-based inputs. The target_type field determines
    how the worker should load the image data:

    - "clip": Extract a representative frame from source video (existing path)
    - "frame": Load the image directly from image_path (simpler, no extraction)
    """

    target_type: str  # "clip" or "frame"
    id: str  # clip_id or frame_id
    image_path: Optional[Path] = None  # Direct image path (for frames)
    video_path: Optional[Path] = None  # Source video path (for clips)
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    fps: Optional[float] = None
    # Pre-existing analysis results for skip_existing checks
    dominant_colors: Optional[list] = None
    shot_type: Optional[str] = None
    description: Optional[str] = None
    detected_objects: Optional[list] = None
    object_labels: Optional[list] = None
    extracted_texts: Optional[list] = None
    cinematography: object = None

    @classmethod
    def from_clip(cls, clip: "Clip", source: "Source") -> "AnalysisTarget":
        """Create target from a Clip and its Source.

        Args:
            clip: The Clip object to analyze
            source: The Source video the clip belongs to
        """
        return cls(
            target_type="clip",
            id=clip.id,
            image_path=clip.thumbnail_path,
            video_path=source.file_path if source else None,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            fps=source.fps if source else None,
            dominant_colors=clip.dominant_colors,
            shot_type=clip.shot_type,
            description=clip.description,
            detected_objects=clip.detected_objects,
            object_labels=clip.object_labels,
            extracted_texts=clip.extracted_texts,
            cinematography=clip.cinematography,
        )

    @classmethod
    def from_frame(cls, frame: "Frame") -> "AnalysisTarget":
        """Create target from a Frame.

        Args:
            frame: The Frame object to analyze
        """
        return cls(
            target_type="frame",
            id=frame.id,
            image_path=frame.file_path,
            video_path=None,
            start_frame=None,
            end_frame=None,
            fps=None,
            dominant_colors=frame.dominant_colors,
            shot_type=frame.shot_type,
            description=frame.description,
            detected_objects=frame.detected_objects,
            object_labels=None,  # Frame model doesn't have object_labels
            extracted_texts=frame.extracted_texts,
            cinematography=frame.cinematography,
        )
