"""Core video processing modules."""

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


def __getattr__(name: str):
    if name == "SceneDetector":
        from core.scene_detect import SceneDetector

        return SceneDetector
    if name == "FFmpegProcessor":
        from core.ffmpeg import FFmpegProcessor

        return FFmpegProcessor
    if name == "ThumbnailGenerator":
        from core.thumbnail import ThumbnailGenerator

        return ThumbnailGenerator
    if name == "VideoDownloader":
        from core.downloader import VideoDownloader

        return VideoDownloader
    if name in {
        "generate_sequence_report",
        "generate_clips_report",
        "report_to_html",
        "REPORT_SECTIONS",
        "DEFAULT_SECTIONS",
    }:
        from core import scene_report

        return getattr(scene_report, name)
    raise AttributeError(f"module 'core' has no attribute {name!r}")
