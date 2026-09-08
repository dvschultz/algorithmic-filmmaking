"""Verified analysis fixtures produced through the shared operation boundary."""

from pathlib import Path
from unittest.mock import patch

from core.project import Project
from models.clip import Clip, Source


def verify_clip_analysis(clip: Clip, directory: Path, *, embeddings: bool = False) -> None:
    from core.spine.analyze import analyze_colors, embeddings as analyze_embeddings

    media = directory / f"{clip.id}.mp4"
    media.write_bytes(b"source fixture")
    thumbnail = directory / f"{clip.id}.jpg"
    thumbnail.write_bytes(b"thumbnail fixture")
    clip.thumbnail_path = thumbnail
    source = Source(id=clip.source_id, file_path=media, fps=30)
    project = Project(sources=[source], clips=[clip])
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(10, 20, 30)]):
        analyze_colors(project)
    if embeddings:
        with patch("core.analysis.embeddings.extract_clip_embeddings_batch", return_value=[[0.1] * 768]), patch("core.analysis.embeddings.unload_model"):
            analyze_embeddings(project)
