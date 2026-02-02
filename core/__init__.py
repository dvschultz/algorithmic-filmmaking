"""Core video processing modules."""

from core.scene_detect import SceneDetector
from core.ffmpeg import FFmpegProcessor
from core.thumbnail import ThumbnailGenerator
from core.downloader import VideoDownloader
from core.scene_report import (
    generate_sequence_report,
    generate_clips_report,
    report_to_html,
    REPORT_SECTIONS,
    DEFAULT_SECTIONS,
)

__all__ = [
    "SceneDetector",
    "FFmpegProcessor",
    "ThumbnailGenerator",
    "VideoDownloader",
    # Scene reports
    "generate_sequence_report",
    "generate_clips_report",
    "report_to_html",
    "REPORT_SECTIONS",
    "DEFAULT_SECTIONS",
]
