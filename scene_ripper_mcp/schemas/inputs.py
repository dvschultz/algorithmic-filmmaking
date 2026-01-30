"""Pydantic input models for MCP tools."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional


class ProjectPathInput(BaseModel):
    """Input requiring a project path."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(
        ...,
        description="Absolute path to project file (.sceneripper or .json)",
        min_length=1,
    )

    @field_validator("project_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not (v.endswith(".sceneripper") or v.endswith(".json")):
            raise ValueError("Project path must end with .sceneripper or .json")
        return v


class DetectScenesInput(BaseModel):
    """Input for scene detection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    video_path: str = Field(..., description="Absolute path to video file")
    output_project: str = Field(..., description="Path for output project JSON")
    sensitivity: float = Field(
        default=3.0,
        description="Detection sensitivity (1.0=more scenes, 10.0=fewer)",
        ge=1.0,
        le=10.0,
    )
    min_scene_length: float = Field(
        default=0.5,
        description="Minimum scene length in seconds",
        ge=0.1,
    )


class YouTubeSearchInput(BaseModel):
    """Input for YouTube search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=200)
    max_results: int = Field(default=25, ge=1, le=50)


class DownloadVideoInput(BaseModel):
    """Input for video download."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(..., description="YouTube or video URL")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory (defaults to settings.download_dir)"
    )


class DownloadVideosInput(BaseModel):
    """Input for bulk video download."""

    model_config = ConfigDict(str_strip_whitespace=True)

    urls: list[str] = Field(..., description="List of YouTube or video URLs", min_length=1, max_length=10)
    output_dir: Optional[str] = Field(
        default=None, description="Output directory (defaults to settings.download_dir)"
    )


class FilterClipsInput(BaseModel):
    """Input for filtering clips."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    shot_type: Optional[str] = Field(default=None, description="Filter by shot type")
    has_speech: Optional[bool] = Field(
        default=None, description="Filter by speech presence"
    )
    min_duration: Optional[float] = Field(
        default=None, description="Minimum duration (seconds)"
    )
    max_duration: Optional[float] = Field(
        default=None, description="Maximum duration (seconds)"
    )
    tags: Optional[list[str]] = Field(
        default=None, description="Filter by tags (any match)"
    )


class ClipTagsInput(BaseModel):
    """Input for adding/removing clip tags."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    clip_id: str = Field(..., description="ID of the clip to modify")
    tags: list[str] = Field(..., description="Tags to add or remove", min_length=1)


class ClipNoteInput(BaseModel):
    """Input for adding a clip note."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    clip_id: str = Field(..., description="ID of the clip to modify")
    note: str = Field(..., description="Note text to set")


class ClipMetadataInput(BaseModel):
    """Input for getting clip metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    clip_id: str = Field(..., description="ID of the clip")


class SequenceClipsInput(BaseModel):
    """Input for sequence operations with clip IDs."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    clip_ids: list[str] = Field(..., description="List of clip IDs", min_length=1)


class ExportClipsInput(BaseModel):
    """Input for exporting clips."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    output_dir: str = Field(..., description="Directory for exported clip files")
    clip_ids: Optional[list[str]] = Field(
        default=None, description="Specific clip IDs to export (None = all)"
    )


class ExportSequenceInput(BaseModel):
    """Input for exporting sequence as video."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    output_path: str = Field(..., description="Path for output video file")


class ExportEDLInput(BaseModel):
    """Input for EDL export."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    output_path: str = Field(..., description="Path for output EDL file")


class ExportDatasetInput(BaseModel):
    """Input for dataset export."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    output_path: str = Field(..., description="Path for output JSON file")


class ListProjectsInput(BaseModel):
    """Input for listing projects in a directory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    directory: str = Field(..., description="Directory to search for project files (.sceneripper or .json)")


class CreateProjectInput(BaseModel):
    """Input for creating a new project."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Project name", min_length=1, max_length=100)
    output_path: str = Field(..., description="Path for the new project file")


class ImportVideoInput(BaseModel):
    """Input for importing a video into a project."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    video_path: str = Field(..., description="Path to video file to import")


class RemoveSourceInput(BaseModel):
    """Input for removing a source from a project."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    source_id: str = Field(..., description="ID of the source to remove")


class AnalyzeColorsInput(BaseModel):
    """Input for color analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    num_colors: int = Field(default=5, description="Number of dominant colors to extract", ge=1, le=10)


class AnalyzeShotsInput(BaseModel):
    """Input for shot type analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")


class TranscribeInput(BaseModel):
    """Input for transcription."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    model: str = Field(
        default="small.en",
        description="Whisper model size: tiny.en, small.en, medium.en, large-v3",
    )
    language: str = Field(default="en", description="Language code (en, auto, etc.)")
