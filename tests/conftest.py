"""Shared test fixtures and helpers for all tests."""

import sys
from pathlib import Path
from typing import Optional

import pytest

# Platform markers (use as @pytest.mark.windows_only / @pytest.mark.unix_only)
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
unix_only = pytest.mark.skipif(sys.platform == "win32", reason="Unix only")

from core.project import Project
from core.transcription import TranscriptSegment
from models.clip import Source, Clip


@pytest.fixture
def test_project() -> Project:
    """Create a project with test data.

    Contains two sources: a 1080p video and a 720p video.
    """
    project = Project.new(name="Test Project")

    source1 = Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    source2 = Source(
        id="src-2",
        file_path=Path("/test/video2.mp4"),
        duration_seconds=60.0,
        fps=24.0,
        width=1280,
        height=720,
    )
    project.add_source(source1)
    project.add_source(source2)

    return project


@pytest.fixture
def test_source() -> Source:
    """Create a single test source."""
    return Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


def make_test_clip(
    clip_id: str,
    source_id: str = "src-1",
    start_frame: int = 0,
    end_frame: int = 90,
    shot_type: Optional[str] = None,
    transcript_text: Optional[str] = None,
    dominant_colors: Optional[list] = None,
    object_labels: Optional[list] = None,
    detected_objects: Optional[list] = None,
    person_count: Optional[int] = None,
    description: Optional[str] = None,
) -> Clip:
    """Create a test clip with optional attributes.

    This is a factory function (not a fixture) so tests can create
    multiple clips with different configurations.

    Args:
        clip_id: Unique identifier for the clip
        source_id: ID of the source video (default: "src-1")
        start_frame: Starting frame number
        end_frame: Ending frame number
        shot_type: Shot type classification (e.g., "close-up", "wide")
        transcript_text: If provided, creates a TranscriptSegment
        dominant_colors: List of dominant colors
        object_labels: List of detected object labels
        detected_objects: List of detection dictionaries
        person_count: Number of detected people
        description: Text description of the clip

    Returns:
        Configured Clip instance
    """
    transcript = None
    if transcript_text:
        transcript = [
            TranscriptSegment(
                start_time=0.0,
                end_time=3.0,
                text=transcript_text,
                confidence=0.95,
            )
        ]

    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        shot_type=shot_type,
        transcript=transcript,
        dominant_colors=dominant_colors,
        object_labels=object_labels,
        detected_objects=detected_objects,
        person_count=person_count,
        description=description,
    )
