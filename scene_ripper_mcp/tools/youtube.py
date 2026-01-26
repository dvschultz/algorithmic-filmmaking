"""YouTube search and download MCP tools."""

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_path

logger = logging.getLogger(__name__)


def _get_youtube_api_key() -> Optional[str]:
    """Get YouTube API key from environment or keyring."""
    # Check environment first
    if api_key := os.environ.get("YOUTUBE_API_KEY"):
        return api_key

    # Try keyring
    try:
        from core.settings import _get_api_key_from_keyring

        return _get_api_key_from_keyring() or None
    except ImportError:
        return None


@mcp.tool()
async def search_youtube(
    query: Annotated[str, "Search query"],
    max_results: Annotated[int, "Maximum number of results (1-50)"] = 25,
    video_duration: Annotated[Optional[str], "Filter by duration: short, medium, long"] = None,
    ctx: Context = None,
) -> str:
    """Search YouTube for videos matching a query.

    Requires YOUTUBE_API_KEY environment variable or keyring credential.

    Args:
        query: Search query string
        max_results: Number of results to return (1-50)
        video_duration: Optional filter (short=<4min, medium=4-20min, long=>20min)

    Returns:
        JSON with video results (id, title, channel, duration, thumbnail)
    """
    api_key = _get_youtube_api_key()
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "YouTube API key not configured. Set YOUTUBE_API_KEY environment variable or configure in settings.",
            }
        )

    try:
        from core.youtube_api import YouTubeSearchClient

        client = YouTubeSearchClient(api_key)
        result = client.search(
            query=query,
            max_results=min(max(1, max_results), 50),
            video_duration=video_duration,
        )

        videos = [
            {
                "video_id": v.video_id,
                "title": v.title,
                "channel": v.channel_title,
                "duration": v.duration_str,
                "duration_seconds": v.duration.total_seconds() if v.duration else None,
                "thumbnail": v.thumbnail_url,
                "url": v.youtube_url,
                "view_count": v.view_count,
                "definition": v.definition,
            }
            for v in result.videos
        ]

        return json.dumps(
            {
                "success": True,
                "query": query,
                "count": len(videos),
                "total_results": result.total_results,
                "videos": videos,
            }
        )
    except Exception as e:
        logger.exception("YouTube search failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def download_video(
    url: Annotated[str, "YouTube or Vimeo video URL"],
    output_dir: Annotated[Optional[str], "Output directory (defaults to settings.download_dir)"] = None,
    ctx: Context = None,
) -> str:
    """Download a video from YouTube or Vimeo.

    Uses yt-dlp for downloading. Only YouTube and Vimeo URLs are supported.

    Args:
        url: Video URL (YouTube or Vimeo)
        output_dir: Optional output directory

    Returns:
        JSON with download result and file path
    """
    # Validate output directory if provided
    if output_dir:
        valid, error, output_path = validate_path(output_dir, must_be_dir=True)
        if not valid:
            return json.dumps({"success": False, "error": error})
    else:
        from core.settings import load_settings

        settings = load_settings()
        output_path = settings.download_dir

    try:
        if ctx:
            await ctx.report_progress(0.0, "Starting download...")

        from core.downloader import VideoDownloader

        downloader = VideoDownloader(download_dir=output_path)

        # Validate URL first
        valid, error = downloader.is_valid_url(url)
        if not valid:
            return json.dumps({"success": False, "error": error})

        # Download with progress callback
        async def progress_callback(progress: float, message: str):
            if ctx:
                await ctx.report_progress(progress / 100.0, message)

        result = downloader.download(url)

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        if result.success:
            return json.dumps(
                {
                    "success": True,
                    "file_path": str(result.file_path),
                    "title": result.title,
                    "duration": result.duration,
                }
            )
        else:
            return json.dumps({"success": False, "error": result.error})
    except Exception as e:
        logger.exception("Video download failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def download_videos(
    urls: Annotated[list[str], "List of video URLs to download"],
    output_dir: Annotated[Optional[str], "Output directory (defaults to settings.download_dir)"] = None,
    ctx: Context = None,
) -> str:
    """Download multiple videos from YouTube or Vimeo.

    Downloads videos sequentially, continuing even if some fail.

    Args:
        urls: List of video URLs (max 10)
        output_dir: Optional output directory

    Returns:
        JSON with download results for each video
    """
    if len(urls) > 10:
        return json.dumps({"success": False, "error": "Maximum 10 URLs allowed per batch"})

    if not urls:
        return json.dumps({"success": False, "error": "No URLs provided"})

    # Validate output directory if provided
    if output_dir:
        valid, error, output_path = validate_path(output_dir, must_be_dir=True)
        if not valid:
            return json.dumps({"success": False, "error": error})
    else:
        from core.settings import load_settings

        settings = load_settings()
        output_path = settings.download_dir

    try:
        from core.downloader import VideoDownloader

        downloader = VideoDownloader(download_dir=output_path)

        results = []
        successful = 0
        failed = 0

        for i, url in enumerate(urls):
            if ctx:
                await ctx.report_progress(i / len(urls), f"Downloading {i + 1}/{len(urls)}...")

            # Validate URL
            valid, error = downloader.is_valid_url(url)
            if not valid:
                results.append({"url": url, "success": False, "error": error})
                failed += 1
                continue

            # Download
            result = downloader.download(url)

            if result.success:
                results.append(
                    {
                        "url": url,
                        "success": True,
                        "file_path": str(result.file_path),
                        "title": result.title,
                    }
                )
                successful += 1
            else:
                results.append({"url": url, "success": False, "error": result.error})
                failed += 1

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        return json.dumps(
            {
                "success": successful > 0,
                "total": len(urls),
                "successful": successful,
                "failed": failed,
                "results": results,
            }
        )
    except Exception as e:
        logger.exception("Bulk download failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def get_video_info(
    url: Annotated[str, "YouTube or Vimeo video URL"],
    ctx: Context = None,
) -> str:
    """Get video metadata without downloading.

    Args:
        url: Video URL (YouTube or Vimeo)

    Returns:
        JSON with video metadata (title, duration, uploader, thumbnail)
    """
    try:
        from core.downloader import VideoDownloader

        downloader = VideoDownloader()

        # Validate URL
        valid, error = downloader.is_valid_url(url)
        if not valid:
            return json.dumps({"success": False, "error": error})

        info = downloader.get_video_info(url)

        return json.dumps(
            {
                "success": True,
                "url": url,
                "title": info["title"],
                "duration": info["duration"],
                "uploader": info["uploader"],
                "thumbnail": info["thumbnail"],
            }
        )
    except Exception as e:
        logger.exception("Failed to get video info")
        return json.dumps({"success": False, "error": str(e)})
