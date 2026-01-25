"""YouTube search and download commands."""

from pathlib import Path
from typing import Optional

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_table, output_success, output_info
from cli.utils.progress import create_progress_callback


@click.command()
@click.argument("query")
@click.option(
    "--max-results",
    "-n",
    type=int,
    default=None,
    help="Maximum number of results (default: 25, max: 50)",
)
@click.option(
    "--order",
    type=click.Choice(["relevance", "date", "rating", "viewCount", "title"]),
    default="relevance",
    help="Sort order (default: relevance)",
)
@click.option(
    "--duration",
    type=click.Choice(["short", "medium", "long"]),
    default=None,
    help="Filter by duration: short (<4min), medium (4-20min), long (>20min)",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    max_results: Optional[int],
    order: str,
    duration: Optional[str],
) -> None:
    """Search YouTube for videos.

    Requires a YouTube Data API key. Set via:
    - Environment variable: YOUTUBE_API_KEY
    - Config file: ~/.config/scene-ripper/config.json

    \b
    Examples:
        scene_ripper search "Soviet animation 1980s"
        scene_ripper search "nature documentary" --duration long
        scene_ripper search "film noir" --max-results 10 --order date
    """
    try:
        from core.youtube_api import (
            YouTubeSearchClient,
            InvalidAPIKeyError,
            QuotaExceededError,
            YouTubeAPIError,
        )
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Check for API key
    api_key = config.youtube_api_key
    if not api_key:
        exit_with(
            ExitCode.VALIDATION_ERROR,
            "YouTube API key not configured. Set YOUTUBE_API_KEY environment variable "
            "or add 'youtube_api_key' to ~/.config/scene-ripper/config.json",
        )

    # Use config default for max results
    if max_results is None:
        max_results = config.youtube_results_count

    try:
        client = YouTubeSearchClient(api_key=api_key)
        result = client.search(
            query=query,
            max_results=max_results,
            order=order,
            video_duration=duration,
        )
    except InvalidAPIKeyError:
        exit_with(ExitCode.VALIDATION_ERROR, "Invalid YouTube API key")
    except QuotaExceededError:
        exit_with(ExitCode.NETWORK_ERROR, "YouTube API quota exceeded. Try again tomorrow.")
    except YouTubeAPIError as e:
        exit_with(ExitCode.NETWORK_ERROR, f"YouTube API error: {e}")

    as_json = ctx.obj.get("json", False)

    if as_json:
        output_data = {
            "query": query,
            "total_results": result.total_results,
            "returned": len(result.videos),
            "results": [
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "channel": v.channel_title,
                    "duration": v.duration_str,
                    "views": v.view_count,
                    "url": v.youtube_url,
                    "thumbnail": v.thumbnail_url,
                    "definition": v.definition,
                }
                for v in result.videos
            ],
        }
        if result.next_page_token:
            output_data["next_page_token"] = result.next_page_token
        output_result(output_data, as_json=True)
    else:
        click.echo(f"Found {result.total_results} results for '{query}'")
        click.echo()

        # Table output
        headers = ["#", "Title", "Duration", "Channel"]
        rows = []
        for i, video in enumerate(result.videos, 1):
            title = video.title[:50] + "..." if len(video.title) > 50 else video.title
            channel = video.channel_title[:20] if video.channel_title else ""
            rows.append([i, title, video.duration_str, channel])

        output_table(headers, rows)

        click.echo()
        click.echo("To download, copy the number and run:")
        click.echo(f"  scene_ripper download https://youtube.com/watch?v=VIDEO_ID")


@click.command()
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for downloaded video",
)
@click.option(
    "--detect",
    "-d",
    is_flag=True,
    help="Automatically run scene detection after download",
)
@click.option(
    "--sensitivity",
    "-s",
    type=float,
    default=None,
    help="Detection sensitivity if --detect is used (1.0-10.0)",
)
@click.pass_context
def download(
    ctx: click.Context,
    url: str,
    output_dir: Optional[Path],
    detect: bool,
    sensitivity: Optional[float],
) -> None:
    """Download a video from YouTube or Vimeo.

    Downloads the video in best available quality.
    Optionally runs scene detection automatically.

    \b
    Examples:
        scene_ripper download https://youtube.com/watch?v=dQw4w9WgXcQ
        scene_ripper download https://vimeo.com/123456789 -o ./videos/
        scene_ripper download https://youtube.com/watch?v=xyz --detect
    """
    try:
        from core.downloader import VideoDownloader, DownloadResult
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Determine output directory
    if output_dir is None:
        output_dir = config.download_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize downloader
    try:
        downloader = VideoDownloader(download_dir=output_dir)
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    # Validate URL
    valid, error = downloader.is_valid_url(url)
    if not valid:
        exit_with(ExitCode.VALIDATION_ERROR, error)

    # Get video info first
    output_info("Getting video info...")
    try:
        info = downloader.get_video_info(url)
        output_info(f"Title: {info['title']}")
        output_info(f"Duration: {info['duration']}s")
    except Exception as e:
        exit_with(ExitCode.NETWORK_ERROR, f"Failed to get video info: {e}")

    # Download with progress
    progress = create_progress_callback("Downloading")

    result = downloader.download(
        url=url,
        progress_callback=progress,
    )

    if not result.success:
        exit_with(ExitCode.NETWORK_ERROR, result.error or "Download failed")

    output_data = {
        "success": True,
        "title": result.title,
        "file_path": str(result.file_path),
        "duration": result.duration,
    }

    # Run scene detection if requested
    if detect and result.file_path:
        output_info("Running scene detection...")

        try:
            from core.scene_detect import SceneDetector, DetectionConfig
            from core.project import save_project

            # Use default or specified sensitivity
            if sensitivity is None:
                sensitivity = config.default_sensitivity

            detection_config = DetectionConfig(
                threshold=sensitivity,
                min_scene_length=int(0.5 * 30),  # Will be updated
                use_adaptive=True,
            )

            detector = SceneDetector(config=detection_config)
            detect_progress = create_progress_callback("Detecting scenes")

            source, clips = detector.detect_scenes_with_progress(
                video_path=result.file_path,
                progress_callback=detect_progress,
            )

            # Save project
            project_path = result.file_path.with_suffix(".json")
            save_project(
                filepath=project_path,
                sources=[source],
                clips=clips,
                sequence=None,
            )

            output_data["detected_clips"] = len(clips)
            output_data["project_file"] = str(project_path)

        except Exception as e:
            output_info(f"Scene detection failed: {e}")
            output_data["detection_error"] = str(e)

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(output_data, as_json=True)
    else:
        output_success(f"Downloaded: {result.title}")
        click.echo(f"File: {result.file_path}")
        if "detected_clips" in output_data:
            click.echo(f"Detected {output_data['detected_clips']} scenes")
            click.echo(f"Project: {output_data['project_file']}")
