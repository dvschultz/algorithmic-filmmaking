"""Main application window."""

import logging
import os
import re
import sys
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QStatusBar,
    QInputDialog,
    QTabWidget,
    QLabel,
    QDockWidget,
    QDialog,
)
from PySide6.QtCore import Qt, Signal, QThread, QUrl, QTimer, Slot
from PySide6.QtGui import QDesktopServices, QKeySequence, QAction, QDragEnterEvent, QDropEvent, QUndoStack

from models.clip import Source, Clip
from core.scene_detect import SceneDetector, DetectionConfig, KaraokeDetectionConfig
from core.thumbnail import ThumbnailGenerator
from core.downloader import (
    VideoDownloader,
    YTDLP_COOKIE_HELP_URL,
    DOWNLOAD_ERROR_COOKIES_REQUIRED,
    DOWNLOAD_ERROR_JS_RUNTIME_REQUIRED,
    classify_download_error_message,
)
from core.sequence_export import SequenceExporter, ExportConfig
from core.sequence_preview import (
    SequencePreviewSettings,
    compute_sequence_preview_signature,
    get_sequence_preview_path,
)
from core.dataset_export import export_dataset, DatasetExportConfig
from core.edl_export import export_edl, EDLExportConfig
from core.srt_export import export_srt, SRTExportConfig
from core.ffmpeg import FFmpegProcessor
from core.settings import (
    load_settings,
    save_settings,
    migrate_from_qsettings,
    validate_download_dir,
    get_default_download_dir,
    is_download_dir_from_env,
)
from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeSearchResult,
    YouTubeVideo,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)
from core.project import (
    Project,
    ProjectMetadata,
    ProjectLoadError,
)
from ui.project_adapter import ProjectSignalAdapter
from ui.settings_dialog import SettingsDialog
from ui.tabs import CollectTab, CutTab, AnalyzeTab, FramesTab, SequenceTab, RenderTab
from ui.theme import theme, Spacing
from ui.chat_panel import ChatPanel
from ui.chat_worker import ChatAgentWorker
from ui.clip_details_sidebar import ClipDetailsSidebar
from ui.dialogs import IntentionImportDialog, AnalysisPickerDialog, URLImportDialog
from ui.log_viewer import LogViewerWidget, get_in_app_log_bridge
from core.analysis_dependencies import get_operation_feature_candidates
from core.analysis_operations import (
    OPERATIONS_BY_KEY,
    PHASE_ORDER,
)
from ui.workers.base import CancellableWorker, summarize_messages
from ui.workers.cinematography_worker import CinematographyWorker
from ui.workers.color_worker import ColorAnalysisWorker
from ui.workers.shot_type_worker import ShotTypeWorker
from ui.workers.transcription_worker import TranscriptionWorker
from ui.workers.classification_worker import ClassificationWorker
from ui.workers.object_detection_worker import ObjectDetectionWorker
from ui.workers.face_detection_worker import FaceDetectionWorker
from ui.workers.gaze_worker import GazeAnalysisWorker
from ui.workers.embedding_worker import EmbeddingAnalysisWorker
from ui.workers.description_worker import DescriptionWorker
from ui.workers.custom_query_worker import CustomQueryWorker
from ui.workers.export_worker import ExportBundleWorker
from ui.workers.sequence_preview_worker import SequencePreviewWorker
from core.gui_state import GUIState
from core.plan_controller import PlanController
from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
from core.app_version import get_app_version, get_release_channel
from core.update_service import UpdateService

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _timeline_frame_to_source_seconds(seq_clip, timeline_frame: int, source_fps: float) -> float:
    """Map timeline frame position to source-video seconds for a sequence clip."""
    frame_in_clip = max(0, timeline_frame - seq_clip.start_frame)
    source_frame = seq_clip.in_point + frame_in_clip
    source_frame = max(seq_clip.in_point, min(source_frame, seq_clip.out_point))
    return source_frame / source_fps


def _resolve_playback_source(seq_clip, source, timeline_frame: int):
    """Resolve the file and timing info for playing a sequence clip.

    Returns (file_to_load, clip_start_seconds, clip_end_seconds, source_seconds).
    """
    prerendered = getattr(seq_clip, "prerendered_path", None)
    if prerendered and Path(prerendered).exists():
        file_to_load = Path(prerendered)
        clip_start_seconds = 0.0
        clip_end_seconds = (seq_clip.out_point - seq_clip.in_point) / source.fps
        frame_in_clip = max(0, timeline_frame - seq_clip.start_frame)
        source_seconds = frame_in_clip / source.fps
    else:
        file_to_load = source.file_path
        source_seconds = _timeline_frame_to_source_seconds(seq_clip, timeline_frame, source.fps)
        clip_start_seconds = seq_clip.in_point / source.fps
        clip_end_seconds = seq_clip.out_point / source.fps
    return file_to_load, clip_start_seconds, clip_end_seconds, source_seconds


def _source_ms_to_timeline_seconds(
    seq_clip,
    position_ms: int,
    source_fps: float,
    timeline_fps: float,
) -> float:
    """Map source-video playback position to timeline seconds for a sequence clip."""
    source_frame = int((position_ms / 1000.0) * source_fps)
    frame_offset = source_frame - seq_clip.in_point
    timeline_frame = seq_clip.start_frame + frame_offset
    min_frame = seq_clip.start_frame
    max_frame = max(seq_clip.start_frame, seq_clip.end_frame() - 1)
    timeline_frame = max(min_frame, min(timeline_frame, max_frame))
    return timeline_frame / timeline_fps


class DetectionWorker(CancellableWorker):
    """Background worker for scene detection.

    Supports both visual detection (adaptive/content) and text-based
    detection (karaoke mode).
    """

    progress = Signal(float, str)  # progress (0-1), status message
    detection_completed = Signal(object, list)  # source, clips (renamed from 'finished' to avoid shadowing QThread.finished)

    def __init__(
        self,
        video_path: Path,
        config: DetectionConfig = None,
        mode: str = "adaptive",
        karaoke_config: KaraokeDetectionConfig = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.config = config or DetectionConfig()
        self.mode = mode
        self.karaoke_config = karaoke_config

    def run(self):
        self._log_start()
        try:
            if self.is_cancelled():
                self._log_cancelled()
                return

            detector = SceneDetector(self.config)

            if self.mode == "karaoke":
                # Use karaoke (text-based) detection
                source, clips = detector.detect_karaoke_scenes_with_progress(
                    self.video_path,
                    lambda p, m: self.progress.emit(p, m),
                    self.karaoke_config,
                )
            else:
                # Use visual detection (adaptive or content)
                source, clips = detector.detect_scenes_with_progress(
                    self.video_path,
                    lambda p, m: self.progress.emit(p, m),
                )

            if self.is_cancelled():
                self._log_cancelled()
                return
            self.detection_completed.emit(source, clips)
            self._log_complete()
        except Exception as e:
            if not self.is_cancelled():
                self._log_error(str(e))
                self.error.emit(str(e))


class ThumbnailWorker(QThread):
    """Background worker for thumbnail generation."""

    progress = Signal(int, int)  # current, total
    thumbnail_ready = Signal(str, str)  # clip_id, thumbnail_path
    # Note: Don't override QThread.finished - use the built-in signal instead

    def __init__(
        self,
        source: Source,
        clips: list[Clip],
        cache_dir: Path = None,
        sources_by_id: dict[str, Source] = None,
    ):
        super().__init__()
        self.source = source
        self.clips = clips
        self.cache_dir = cache_dir
        self.sources_by_id = sources_by_id or {}
        logger.debug("ThumbnailWorker created")

    def run(self):
        logger.info("ThumbnailWorker.run() STARTING")
        logger.info(f"ThumbnailWorker: {len(self.clips)} clips to process")
        logger.info(f"ThumbnailWorker: sources_by_id has {len(self.sources_by_id)} entries: {list(self.sources_by_id.keys())}")
        logger.info(f"ThumbnailWorker: default source: {self.source.id if self.source else None}")
        generator = ThumbnailGenerator(cache_dir=self.cache_dir)
        total = len(self.clips)

        for i, clip in enumerate(self.clips):
            try:
                # Use clip's source if available, fall back to default source
                source = self.sources_by_id.get(clip.source_id, self.source)
                logger.info(f"ThumbnailWorker: clip {clip.id[:8]} source_id={clip.source_id}, found source: {source.id if source else None}")
                if not source:
                    logger.warning(f"No source found for clip {clip.id} (source_id={clip.source_id})")
                    continue

                logger.info(f"ThumbnailWorker: generating thumbnail for clip {clip.id}, video: {source.file_path}")
                thumb_path = generator.generate_clip_thumbnail(
                    video_path=source.file_path,
                    start_seconds=clip.start_time(source.fps),
                    end_seconds=clip.end_time(source.fps),
                )
                clip.thumbnail_path = thumb_path
                logger.info(f"ThumbnailWorker: emitting thumbnail_ready for clip {clip.id}, path={thumb_path}")
                self.thumbnail_ready.emit(clip.id, str(thumb_path))
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for clip {clip.id}: {e}")

            self.progress.emit(i + 1, total)

        logger.info("ThumbnailWorker.run() COMPLETED")
        # QThread's built-in finished signal will be emitted after run() returns


class DownloadWorker(CancellableWorker):
    """Background worker for video downloads."""

    progress = Signal(float, str)  # progress (0-100), status message
    download_completed = Signal(object)  # DownloadResult (renamed from 'finished' to avoid shadowing QThread.finished)

    def __init__(self, url: str, resolution: Optional[str] = None):
        super().__init__()
        self.url = url
        self.resolution = resolution

    def run(self):
        try:
            downloader = VideoDownloader()
            result = downloader.download(
                self.url,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self.is_cancelled(),
                resolution=self.resolution,
            )
            if result.success:
                self.download_completed.emit(result)
            else:
                self.error.emit(result.error or "Download failed")
        except Exception as e:
            self.error.emit(str(e))


def _calculate_download_timeout(duration_seconds: float, height: int | None) -> int:
    """Calculate download timeout based on video duration and resolution.

    Args:
        duration_seconds: Video duration in seconds
        height: Video height in pixels (e.g., 1080 for 1080p), or None if unknown

    Returns:
        Timeout in seconds
    """
    # Seconds of timeout per minute of video, by resolution
    # Higher resolutions = larger files = more download time needed
    TIMEOUT_MULTIPLIERS = {
        4320: 180,  # 8K: 3 min timeout per video minute
        2160: 120,  # 4K: 2 min timeout per video minute
        1440: 90,   # 1440p: 1.5 min timeout per video minute
        1080: 60,   # 1080p: 1 min timeout per video minute
        720: 45,    # 720p: 45 sec timeout per video minute
        480: 30,    # 480p: 30 sec timeout per video minute
        360: 20,    # 360p: 20 sec timeout per video minute
    }
    MIN_TIMEOUT = 120   # 2 minutes minimum
    MAX_TIMEOUT = 3600  # 1 hour cap
    DEFAULT_MULTIPLIER = 60  # Default to 1080p assumption

    # Find the appropriate multiplier based on resolution
    if height is None:
        multiplier = DEFAULT_MULTIPLIER
    else:
        # Find closest resolution tier (round down to nearest tier)
        multiplier = DEFAULT_MULTIPLIER
        for tier_height, tier_multiplier in sorted(TIMEOUT_MULTIPLIERS.items()):
            if height >= tier_height:
                multiplier = tier_multiplier

    duration_minutes = duration_seconds / 60
    timeout = int(duration_minutes * multiplier)
    return max(MIN_TIMEOUT, min(timeout, MAX_TIMEOUT))


class URLBulkDownloadWorker(CancellableWorker):
    """Background worker for downloading multiple videos from URLs in parallel."""

    progress = Signal(int, int, str)  # current, total, message
    video_finished = Signal(str, object)  # url, DownloadResult
    all_finished = Signal(list)  # list of result dicts

    MAX_WORKERS = 3  # Parallel download limit

    def __init__(self, urls: list[str], download_dir: Path):
        super().__init__()
        self.urls = urls
        self.download_dir = download_dir
        self._results = []
        self._completed_count = 0
        self._lock = None  # Initialized in run()

    def _download_single(self, url: str) -> dict:
        """Download a single URL (called from thread pool)."""
        downloader = VideoDownloader(download_dir=self.download_dir)

        try:
            valid, error = downloader.is_valid_url(url)
            if not valid:
                return {"url": url, "success": False, "error": error, "result": None}

            # Get video info first to calculate appropriate timeout
            try:
                info = downloader.get_video_info(url, include_format_details=True)
                duration = info.get("duration", 0) or 0
                height = info.get("height")
                timeout = _calculate_download_timeout(duration, height)
                logger.debug(f"Download timeout for {url}: {timeout}s (duration={duration}s, height={height})")
            except Exception as e:
                # If we can't get info, use a generous default
                logger.warning(f"Could not get video info for timeout calc: {e}, using 10 min default")
                timeout = 600

            result = downloader.download(url, max_download_seconds=timeout)

            if result.success:
                return {
                    "url": url,
                    "success": True,
                    "file_path": str(result.file_path) if result.file_path else None,
                    "title": result.title,
                    "duration": result.duration,
                    "result": result,
                }
            else:
                return {
                    "url": url,
                    "success": False,
                    "error": result.error or "Download failed",
                    "result": None,
                }

        except Exception as e:
            return {"url": url, "success": False, "error": str(e), "result": None}

    def run(self):
        """Download videos in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        import time

        total = len(self.urls)
        self._lock = threading.Lock()
        self._completed_count = 0
        start_time = time.time()

        logger.info(f"URLBulkDownloadWorker starting: {total} URLs")
        self.progress.emit(0, total, f"Starting {total} downloads...")

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_url = {executor.submit(self._download_single, url): url for url in self.urls}

            # Process results as they complete
            for future in as_completed(future_to_url):
                if self.is_cancelled():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                url = future_to_url[future]
                try:
                    result_dict = future.result()

                    # Remove the internal 'result' object before storing
                    download_result = result_dict.pop("result", None)

                    with self._lock:
                        self._results.append(result_dict)
                        self._completed_count += 1
                        count = self._completed_count

                    # Emit progress from main QThread (safe)
                    self.progress.emit(count, total, f"Downloaded {count}/{total}...")

                    # Emit video_finished for successful downloads
                    if result_dict["success"] and download_result:
                        self.video_finished.emit(url, download_result)

                except Exception as e:
                    with self._lock:
                        self._results.append({"url": url, "success": False, "error": str(e)})
                        self._completed_count += 1

        elapsed = time.time() - start_time
        success_count = sum(1 for r in self._results if r.get("success"))
        logger.info(f"URLBulkDownloadWorker finished: {success_count}/{total} succeeded in {elapsed:.1f}s")
        self.progress.emit(total, total, "Downloads complete")
        self.all_finished.emit(self._results)


class SequenceExportWorker(QThread):
    """Background worker for sequence export."""

    progress = Signal(float, str)  # progress (0-1), status message
    export_completed = Signal(object)  # output path (Path) (renamed from 'finished' to avoid shadowing QThread.finished)
    error = Signal(str)

    def __init__(self, sequence, sources, clips, config):
        super().__init__()
        self.sequence = sequence
        self.sources = sources
        self.clips = clips
        self.config = config

    def run(self):
        try:
            exporter = SequenceExporter()
            success = exporter.export(
                sequence=self.sequence,
                sources=self.sources,
                clips=self.clips,
                config=self.config,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            if success:
                self.export_completed.emit(self.config.output_path)
            else:
                self.error.emit("Export failed")
        except Exception as e:
            self.error.emit(str(e))


class YouTubeSearchWorker(QThread):
    """Background worker for YouTube search."""

    search_completed = Signal(object)  # YouTubeSearchResult (renamed from 'finished' to avoid shadowing QThread.finished)
    error = Signal(str)

    def __init__(self, client: YouTubeSearchClient, query: str, max_results: int = 25, video_duration: str = "", video_definition: str = ""):
        super().__init__()
        self.client = client
        self.query = query
        self.max_results = max_results
        self.video_duration = video_duration or None
        self.video_definition = video_definition or None

    def run(self):
        try:
            result = self.client.search(self.query, self.max_results, video_duration=self.video_duration, video_definition=self.video_definition)
            self.search_completed.emit(result)
        except QuotaExceededError as e:
            self.error.emit(str(e))
        except InvalidAPIKeyError as e:
            self.error.emit(str(e))
        except YouTubeAPIError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Search failed: {e}")


class InternetArchiveSearchWorker(QThread):
    """Background worker for Internet Archive search."""

    search_completed = Signal(list)  # list of InternetArchiveVideo
    error = Signal(str)

    def __init__(self, query: str, max_results: int = 25):
        super().__init__()
        self.query = query
        self.max_results = max_results

    def run(self):
        try:
            from core.internet_archive_api import InternetArchiveClient, InternetArchiveError
            client = InternetArchiveClient()
            results = client.search(self.query, self.max_results)
            self.search_completed.emit(results)
        except InternetArchiveError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Search failed: {e}")


class BulkDownloadWorker(QThread):
    """Background worker for parallel bulk downloads."""

    progress = Signal(int, int, str)  # current, total, message
    video_finished = Signal(object)  # DownloadResult
    video_error = Signal(str, str)  # video_id, error message
    all_finished = Signal()

    def __init__(self, videos: list, download_dir: Path, max_parallel: int = 2):
        super().__init__()
        self.videos = videos
        self.download_dir = download_dir
        self.max_parallel = max_parallel
        import threading
        self._cancel_event = threading.Event()
        self._completed = 0

    def cancel(self):
        """Request cancellation."""
        self._cancel_event.set()

    def run(self):
        """Run parallel downloads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(self.videos)
        logger.info(f"BulkDownloadWorker starting download of {total} videos")
        self.progress.emit(0, total, f"Starting download of {total} videos...")

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all downloads
            future_to_video = {
                executor.submit(self._download_one, video): video
                for video in self.videos
            }

            # Process completions
            for future in as_completed(future_to_video):
                if self._cancel_event.is_set():
                    logger.info("BulkDownloadWorker cancelled")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                video = future_to_video[future]
                try:
                    result = future.result()
                    if result.success:
                        logger.info(f"Download succeeded: {video.title}")
                        self.video_finished.emit(result)
                    else:
                        error_msg = result.error or "Download failed (no error message)"
                        logger.error(f"Download failed for '{video.title}': {error_msg}")
                        self.video_error.emit(video.video_id, error_msg)
                except Exception as e:
                    logger.exception(f"Download exception for '{video.title}': {e}")
                    self.video_error.emit(video.video_id, str(e))

                self._completed += 1
                self.progress.emit(
                    self._completed, total, f"Downloaded {self._completed}/{total}"
                )

        logger.info(f"BulkDownloadWorker finished: {self._completed}/{total} completed")
        self.all_finished.emit()

    def _download_one(self, video):
        """Download a single video (YouTube or Internet Archive)."""
        logger.debug(f"Starting download: {video.title} ({video.video_id}) to {self.download_dir}")
        downloader = VideoDownloader(download_dir=self.download_dir)

        # Get the appropriate URL based on video type
        if hasattr(video, 'youtube_url'):
            url = video.youtube_url
        elif hasattr(video, 'download_url'):
            url = video.download_url
        else:
            # Fallback - should not happen
            raise ValueError(f"Unknown video type: {type(video)}")

        result = downloader.download(
            url,
            cancel_check=self._cancel_event.is_set,
        )
        if result.success:
            logger.debug(f"Download finished: {video.title} -> {result.file_path}")
        else:
            logger.debug(f"Download returned failure: {video.title} - {result.error}")
        return result


class MainWindow(QMainWindow):
    """Main application window with drag-drop, detection, and preview."""

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a filename to prevent path traversal and invalid characters."""
        # Remove path separators and other dangerous/invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        # Strip leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized or "video"

    def _ensure_video_download_available(self) -> bool:
        """Prompt to install yt-dlp if the video download feature is missing."""
        from ui.widgets.dependency_widgets import prompt_feature_download

        return prompt_feature_download("video_download", self)

    def _ensure_analysis_operation_available(
        self,
        op_key: str,
        *,
        prompt_cache: Optional[dict[str, bool]] = None,
        description_tier: Optional[str] = None,
    ) -> bool:
        """Prompt for installable analysis dependencies when required."""
        feature_candidates = get_operation_feature_candidates(
            op_key,
            self.settings,
            description_tier=description_tier,
        )
        if not feature_candidates:
            return True

        from core.feature_registry import check_feature_ready

        preferred_feature = feature_candidates[0]
        alternate_available = False
        for feature_name in feature_candidates:
            available, _ = check_feature_ready(feature_name)
            if available:
                if feature_name == preferred_feature:
                    if prompt_cache is not None:
                        prompt_cache[feature_name] = True
                    return True
                alternate_available = True

        if prompt_cache is not None and preferred_feature in prompt_cache:
            return prompt_cache[preferred_feature] or alternate_available

        from ui.widgets.dependency_widgets import prompt_feature_download

        available = prompt_feature_download(preferred_feature, self)
        if prompt_cache is not None:
            prompt_cache[preferred_feature] = available
        return available or alternate_available

    def _filter_available_analysis_operations(
        self,
        operations: list[str],
        *,
        description_tier: Optional[str] = None,
    ) -> list[str]:
        """Return only analysis operations whose dependencies are available."""
        prompt_cache: dict[str, bool] = {}
        available_ops: list[str] = []
        skipped_ops: list[str] = []

        for op_key in operations:
            if self._ensure_analysis_operation_available(
                op_key,
                prompt_cache=prompt_cache,
                description_tier=description_tier,
            ):
                available_ops.append(op_key)
            else:
                skipped_ops.append(op_key)

        if skipped_ops:
            logger.info(
                "Skipping analysis operations with unavailable dependencies: %s",
                skipped_ops,
            )

        return available_ops

    # Class-level counter to track instances
    _instance_count = 0

    def __init__(self):
        super().__init__()
        MainWindow._instance_count += 1
        self._instance_id = MainWindow._instance_count
        logger.info(f"=== MAINWINDOW INIT START (instance #{self._instance_id}) ===")
        self.setWindowTitle("Scene Ripper - Algorithmic Filmmaking")
        self.setMinimumSize(1200, 800)
        # Start at screen size
        screen = self.screen().availableGeometry()
        self.resize(screen.width(), screen.height())
        self.setAcceptDrops(True)

        # Migrate QSettings to JSON on first launch (if needed)
        if migrate_from_qsettings():
            logger.info("Settings migrated from QSettings to JSON")

        # Load settings
        self.settings = load_settings()
        logger.info(f"Loaded settings: sensitivity={self.settings.default_sensitivity}")
        self._log_bridge = get_in_app_log_bridge()
        self._log_bridge.install()

        # Apply theme preference from settings
        theme().set_preference(self.settings.theme_preference)

        # Project state - single source of truth
        self.project = Project.new()
        self._project_adapter = ProjectSignalAdapter(self.project, self)

        # Connect project adapter signals for view synchronization
        self._project_adapter.clips_updated.connect(self._on_clips_updated)
        self._project_adapter.clips_added.connect(self._on_clips_added)
        self._project_adapter.clips_removed.connect(self._on_clips_removed)
        self._project_adapter.source_added.connect(self._on_source_added)
        self._project_adapter.source_updated.connect(self._on_source_updated)
        self._project_adapter.frames_removed.connect(self._on_frames_removed)
        self._project_adapter.sequence_changed.connect(lambda _: self._refresh_timeline_from_project())
        self._project_adapter.audio_sources_changed.connect(self._on_audio_sources_changed)

        # Active audio import workers (kept alive while running)
        self._active_audio_imports: set = set()
        self._active_audio_transcribes: set = set()

        # UI state (not part of Project - these are GUI-specific selections)
        self.current_source: Optional[Source] = None  # Currently active/selected source
        self._analyze_queue: deque[Source] = deque()  # Queue for batch analysis (O(1) popleft)
        self._analyze_queue_total: int = 0  # Total count for progress display
        self._detection_start_time: Optional[float] = None  # When batch detection started
        self._detection_current_progress: float = 0.0  # Current video progress (0-1)
        self.detection_worker: Optional[DetectionWorker] = None
        self.thumbnail_worker: Optional[ThumbnailWorker] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.url_bulk_download_worker: Optional[URLBulkDownloadWorker] = None
        self.export_worker: Optional[SequenceExportWorker] = None
        self.sequence_preview_worker: Optional[SequencePreviewWorker] = None
        self._rendered_sequence_preview_path: Optional[Path] = None
        self._rendered_sequence_preview_signature: Optional[str] = None
        self._rendered_sequence_preview_profile: str = SequencePreviewSettings().profile_label
        self._sequence_preview_play_after_render_frame: Optional[int] = None
        self._using_rendered_sequence_preview = False
        self.color_worker: Optional[ColorAnalysisWorker] = None
        self.shot_type_worker: Optional[ShotTypeWorker] = None
        self.transcription_worker: Optional[TranscriptionWorker] = None
        self.classification_worker: Optional[ClassificationWorker] = None
        self.detection_worker_yolo: Optional[ObjectDetectionWorker] = None
        self.description_worker: Optional[DescriptionWorker] = None
        self.custom_query_worker: Optional[CustomQueryWorker] = None
        self._custom_query_text: Optional[str] = None
        self._active_custom_query_text: Optional[str] = None
        self.youtube_search_worker: Optional[YouTubeSearchWorker] = None
        self.ia_search_worker: Optional[InternetArchiveSearchWorker] = None
        self.bulk_download_worker: Optional[BulkDownloadWorker] = None
        self.export_bundle_worker: Optional[ExportBundleWorker] = None
        self.youtube_client: Optional[YouTubeSearchClient] = None

        # Intention-first workflow coordinator and dialog
        self.intention_workflow: Optional[IntentionWorkflowCoordinator] = None
        self.intention_import_dialog: Optional[IntentionImportDialog] = None

        # Guards to prevent duplicate signal handling
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._classification_finished_handled = False
        self._object_detection_finished_handled = False
        self._description_finished_handled = False
        self._shot_type_finished_handled = False
        self._color_run_error: Optional[str] = None
        self._classification_run_error: Optional[str] = None
        self._object_detection_run_error: Optional[str] = None
        self._description_run_error: Optional[str] = None
        self._description_run_errors: list[str] = []
        self._text_extraction_run_error: Optional[str] = None
        self._cinematography_run_error: Optional[str] = None
        self._shot_type_run_error: Optional[str] = None
        self._transcription_run_error: Optional[str] = None
        self._transcription_run_errors: list[str] = []
        self._transcription_finished_handled = False
        self._text_extraction_finished_handled = False
        self._cinematography_finished_handled = False
        self._gaze_finished_handled = False

        # Suppression flag: prevents _on_clips_added from starting its own ThumbnailWorker
        # when _on_detection_finished is managing the full detection→thumbnail pipeline
        self._suppress_clips_added_thumbnails = False

        # Generation IDs for workers - used to ignore stale signals from cancelled workers
        # Incremented each time a new worker starts; signals with old generation are ignored
        self._detection_generation: int = 0
        self._thumbnail_generation: int = 0

        # State for analysis pipeline (phase-based execution)
        self._analysis_selected_ops: list[str] = []  # Operations to run
        self._analysis_clips: list = []  # Clips being analyzed
        self._analysis_current_phase: str = ""  # "local"|"sequential"|"cloud"|""
        self._analysis_phase_remaining: int = 0  # Concurrent worker counter
        self._analysis_completed_ops: list[str] = []  # Ops finished so far
        self._analysis_pending_phases: list[str] = []  # Phases still to run
        self._analysis_sequential_queue: list[str] = []  # Queue for sequential ops
        self._transcription_source_queue: list = []  # Queue for multi-source transcription

        # Agent tool waiting state - tracks when agent is waiting for worker completion
        self._pending_agent_detection = False
        self._pending_agent_color_analysis = False
        self._pending_agent_shot_analysis = False
        self._pending_agent_transcription = False
        self._pending_agent_classification = False
        self._pending_agent_object_detection = False
        self._pending_agent_description = False
        self._pending_agent_analyze_all = False
        self._pending_agent_export = False
        self._pending_agent_export_bundle = False
        self._pending_agent_download = False
        self._agent_color_clips: list = []
        self._agent_shot_clips: list = []
        self._agent_transcription_clips: list = []
        self._agent_classification_clips: list = []
        self._agent_object_detection_clips: list = []
        self._agent_description_clips: list = []
        self._agent_download_results: list = []  # Results for bulk download
        self._pending_agent_tool_call_id: Optional[str] = None
        self._pending_agent_tool_name: Optional[str] = None

        # Plan execution state
        self._pending_plan_tool_call_id: Optional[str] = None

        # GUI state tracking for agent context awareness
        self._gui_state = GUIState()

        # Cached PlanController for agent tool access
        self.plan_controller = PlanController(self._gui_state)

        # Project path tracking (for backwards compatibility - delegates to project)
        # Note: self.project.path is the actual storage

        self.undo_stack = QUndoStack(self)

        logger.info("Setting up UI...")
        self._setup_ui()
        logger.info("Connecting signals...")
        self._connect_signals()

        # Set up chat panel
        logger.info("Setting up chat panel...")
        self._setup_chat_panel()

        # Set up clip details sidebar
        logger.info("Setting up clip details sidebar...")
        self._setup_clip_details_sidebar()

        logger.info("Setting up log viewer...")
        self._setup_log_dock()

        # Playback state (must be after _setup_ui so tabs are initialized)
        logger.info("Setting up playback state...")
        self._is_playing = False
        self._current_playback_clip = None  # Currently playing SequenceClip
        self._preview_sync_clip = None  # Clip used for direct preview (non-timeline playback)
        self._sequence_preview_source_id = None  # Currently loaded source in sequence preview
        self._sequence_preview_loading = False  # Waiting for sequence preview source to load
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None  # (start_seconds, end_seconds)
        self._pending_sequence_preview_seek_seconds = None
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None  # (start_seconds, end_seconds)
        self._syncing_timeline_from_video = False
        self._playback_timer = QTimer(self)  # Parent to self for proper lifecycle
        self._playback_timer.setInterval(33)  # ~30fps update rate
        self._playback_timer.timeout.connect(self._on_playback_tick)

        # Auto-save timer for silent saves after tool execution
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.timeout.connect(self._do_auto_save)

        # Launch update checker in background (frozen mode only)
        self._update_check_worker = None
        self._manual_update_check = False
        self._update_service = UpdateService(get_app_version(), self.settings)
        self._start_update_check()

        logger.info(f"=== MAINWINDOW INIT COMPLETE (instance #{self._instance_id}) ===")

    # --- Property delegates to Project for backward compatibility ---

    @property
    def sources(self) -> list[Source]:
        """All source videos in the project (delegates to Project)."""
        return self.project.sources

    @property
    def sources_by_id(self) -> dict[str, Source]:
        """Source lookup by ID (delegates to Project)."""
        return self.project.sources_by_id

    @property
    def clips(self) -> list[Clip]:
        """All clips from all analyzed sources (delegates to Project)."""
        return self.project.clips

    @property
    def clips_by_id(self) -> dict[str, Clip]:
        """Clip lookup by ID (delegates to Project)."""
        return self.project.clips_by_id

    @property
    def clips_by_source(self) -> dict[str, list[Clip]]:
        """Clips organized by source ID (delegates to Project)."""
        return self.project.clips_by_source

    @property
    def current_project_path(self) -> Optional[Path]:
        """Current project file path (delegates to Project)."""
        return self.project.path

    @current_project_path.setter
    def current_project_path(self, value: Optional[Path]) -> None:
        """Set current project file path (delegates to Project)."""
        self.project.path = value

    @property
    def project_metadata(self) -> Optional[ProjectMetadata]:
        """Project metadata (delegates to Project)."""
        return self.project.metadata

    @project_metadata.setter
    def project_metadata(self, value: Optional[ProjectMetadata]) -> None:
        """Set project metadata (delegates to Project)."""
        if value is not None:
            self.project.metadata = value

    @property
    def _is_dirty(self) -> bool:
        """Whether project has unsaved changes (delegates to Project)."""
        return self.project.is_dirty

    # --- Sequence tab synchronization ---

    def _refresh_sequence_tab_clips(self):
        """Refresh the Sequence tab with all available clips from all sources.

        This ensures _available_clips contains clips from ALL sources, not just
        the most recently detected source. Called when switching to Sequence tab
        and after detection completes.
        """
        if not self.clips:
            return

        # Build (Clip, Source) tuples for all enabled clips
        all_clips = []
        for clip in self.clips:
            if clip.disabled:
                continue
            source = self.sources_by_id.get(clip.source_id)
            if source:
                all_clips.append((clip, source))

        # Update Sequence tab with all clips via public method
        self.sequence_tab.set_available_clips(all_clips, self.clips, self.sources_by_id)

        logger.debug(f"Refreshed Sequence tab with {len(all_clips)} clips from {len(self.sources_by_id)} sources")

    # --- Worker lifecycle management ---

    def _stop_worker_safely(self, worker: Optional[QThread], name: str, timeout_ms: int = 3000) -> None:
        """Safely stop a running QThread worker.

        Prevents 'QThread: Destroyed while thread is still running' crashes by
        ensuring the worker thread has stopped before the object is garbage collected.

        Args:
            worker: The QThread worker to stop (can be None)
            name: Worker name for logging
            timeout_ms: Maximum time to wait for graceful shutdown
        """
        if worker is None:
            return

        if not worker.isRunning():
            return

        logger.info(f"Stopping {name} worker before creating new one...")

        # Try graceful cancellation if supported
        if hasattr(worker, 'cancel'):
            worker.cancel()

        # Wait for worker to finish
        if not worker.wait(timeout_ms):
            logger.warning(f"{name} worker did not stop gracefully, terminating...")
            worker.terminate()
            worker.wait(1000)

        logger.info(f"{name} worker stopped")

    def _stop_all_workers(self):
        """Stop all running workers for clean project reset.

        Called when clearing project state to ensure no stale workers
        continue processing data from the old project.

        MAINTAINABILITY NOTE: When adding new QThread workers to MainWindow,
        you MUST add them to the workers_to_stop list below. Otherwise the
        worker may continue running after New Project, causing state leakage
        or "QThread: Destroyed while thread is still running" crashes.
        """
        # Stop chat worker if running
        if self._chat_worker and self._chat_worker.isRunning():
            self._chat_worker.stop()
            self._chat_worker.wait(1000)

        # Stop all analysis workers
        workers_to_stop = [
            (getattr(self, 'thumbnail_worker', None), "Thumbnail"),
            (getattr(self, 'detection_worker', None), "Detection"),
            (getattr(self, 'color_worker', None), "Color"),
            (getattr(self, 'shot_type_worker', None), "ShotType"),
            (getattr(self, 'transcription_worker', None), "Transcription"),
            (getattr(self, 'classification_worker', None), "Classification"),
            (getattr(self, 'description_worker', None), "Description"),
            (getattr(self, 'text_extraction_worker', None), "TextExtraction"),
            (getattr(self, 'cinematography_worker', None), "Cinematography"),
            (getattr(self, 'download_worker', None), "Download"),
            (getattr(self, 'bulk_download_worker', None), "BulkDownload"),
            (getattr(self, 'url_bulk_download_worker', None), "URLBulkDownload"),
            (getattr(self, 'youtube_search_worker', None), "YouTubeSearch"),
            (getattr(self, 'ia_search_worker', None), "InternetArchiveSearch"),
            (getattr(self, 'export_worker', None), "Export"),
            (getattr(self, 'sequence_preview_worker', None), "SequencePreview"),
            (getattr(self, '_gaze_worker', None), "Gaze"),
        ]

        for worker, name in workers_to_stop:
            self._stop_worker_safely(worker, name, timeout_ms=2000)

    def _setup_ui(self):
        """Set up the user interface."""
        # Create menu bar
        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Dependency banner (shown when FFmpeg is missing in frozen mode)
        self._dependency_banner = None
        self._update_banner = None
        self._setup_dependency_banners(layout)

        # Tab widget for workflow pages
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Shared clip-filter state — Cut and Analyze point at the same
        # FilterState so filter values are shared across tabs by default
        # (see docs/plans/2026-04-21-001-feat-comprehensive-clip-filter-system-plan.md).
        from core.filter_state import FilterState
        self._filter_state = FilterState()

        # Create tabs
        self.collect_tab = CollectTab()
        self.cut_tab = CutTab(filter_state=self._filter_state)
        self.analyze_tab = AnalyzeTab(filter_state=self._filter_state)
        self.frames_tab = FramesTab()
        self.sequence_tab = SequenceTab()
        self.render_tab = RenderTab()

        # Add tabs
        self.tab_widget.addTab(self.collect_tab, "Collect")
        self.tab_widget.addTab(self.cut_tab, "Cut")
        self.tab_widget.addTab(self.analyze_tab, "Analyze")
        self.tab_widget.addTab(self.frames_tab, "Frames")
        self.tab_widget.addTab(self.sequence_tab, "Sequence")
        self.tab_widget.addTab(self.render_tab, "Render")

        # Set up Analyze tab lookups (it uses references, not copies)
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        # Restore filter-sidebar visibility from settings (per tab)
        try:
            self.cut_tab.set_filter_sidebar_visible(self.settings.cut_filter_sidebar_visible)
            self.analyze_tab.set_filter_sidebar_visible(self.settings.analyze_filter_sidebar_visible)
        except AttributeError:
            # If settings object doesn't have the fields yet (migration), leave defaults.
            pass

        # Set up Frames tab project reference
        self.frames_tab.set_project(self.project)

        # Set up Sequence tab GUI state and project references
        self.sequence_tab.set_gui_state(self._gui_state)
        self.sequence_tab.set_project(self.project)

        layout.addWidget(self.tab_widget)

        # Bottom: Progress bar (global, below tabs)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Drop a video file to begin")

        # Queue indicator (permanent widget on right side)
        self.queue_label = QLabel("")
        self.queue_label.setStyleSheet(f"color: {theme().text_secondary}; padding-right: {Spacing.MD}px;")
        self.queue_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.queue_label)

    def _setup_dependency_banners(self, layout):
        """Show banners for missing dependencies when running as a frozen .app."""
        from core.paths import is_frozen
        from core.binary_resolver import find_binary

        if not is_frozen():
            return

        # Check if FFmpeg is missing
        if find_binary("ffmpeg") is None:
            from ui.widgets.dependency_widgets import DependencyBanner
            self._dependency_banner = DependencyBanner(
                message="Scene Ripper needs FFmpeg to process videos.",
                button_text="Download FFmpeg (~150 MB)",
                dep_name="ffmpeg",
            )
            self._dependency_banner.download_requested.connect(self._on_ffmpeg_download_requested)
            layout.addWidget(self._dependency_banner)

    def _on_ffmpeg_download_requested(self, dep_name: str):
        """Handle FFmpeg download request from the dependency banner."""
        from core.dependency_manager import ensure_ffmpeg, ensure_ffprobe
        from ui.widgets.dependency_widgets import DependencyDownloadDialog

        def install_both(progress_callback):
            ensure_ffmpeg(progress_callback)
            ensure_ffprobe(progress_callback)

        dialog = DependencyDownloadDialog(
            title="Download FFmpeg",
            message="Downloading FFmpeg and FFprobe (~150 MB total).\nThis is required for video processing.",
            install_func=install_both,
            parent=self,
        )
        dialog.download_completed.connect(self._on_ffmpeg_downloaded)
        dialog.exec()

    def _on_ffmpeg_downloaded(self):
        """Called when FFmpeg download completes successfully."""
        # Dismiss the banner
        if self._dependency_banner:
            self._dependency_banner.setVisible(False)

        # Refresh already-instantiated components so they pick up the new binaries
        if hasattr(self, 'ffmpeg') and self.ffmpeg is not None:
            self.ffmpeg.refresh_binaries()
        if hasattr(self, 'thumbnail_generator') and self.thumbnail_generator is not None:
            self.thumbnail_generator.ffmpeg_path = __import__('core.binary_resolver', fromlist=['find_binary']).find_binary("ffmpeg")
            self.thumbnail_generator.ffmpeg_available = self.thumbnail_generator.ffmpeg_path is not None

        # Notify status bar
        self.status_bar.showMessage("FFmpeg installed successfully!", 5000)
        logger.info("FFmpeg downloaded — features enabled")

    def _show_update_banner(self, version: str, release_url: str):
        """Show an update-available banner at the top of the window."""
        from ui.widgets.dependency_widgets import UpdateBanner

        if self._update_banner:
            self._update_banner.setVisible(False)
            self._update_banner.deleteLater()

        self._update_banner = UpdateBanner(version, release_url)
        self._update_banner.download_clicked.connect(
            lambda url: QDesktopServices.openUrl(QUrl(url))
        )

        # Insert at position 0 in the central layout (above dependency banner and tabs)
        central_layout = self.centralWidget().layout()
        central_layout.insertWidget(0, self._update_banner)

    def _start_update_check(self):
        """Launch a background update check (frozen mode only, throttled)."""
        from core.paths import is_frozen
        if not is_frozen():
            return
        if not self.settings.check_for_updates:
            return
        self._launch_update_check(interactive=False)

    def _launch_update_check(self, interactive: bool) -> None:
        """Start an update check if one is not already running."""
        if self._update_check_worker and self._update_check_worker.isRunning():
            if interactive:
                self.status_bar.showMessage("Already checking for updates...", 3000)
            return

        from core.update_checker import UpdateCheckWorker

        self._manual_update_check = interactive
        self._update_service = UpdateService(get_app_version(), self.settings)
        self._update_check_worker = UpdateCheckWorker(
            get_app_version(),
            self.settings,
            interactive=interactive,
        )
        self._update_check_worker.update_available.connect(self._show_update_banner)
        if interactive:
            self._update_check_worker.update_available.connect(self._on_manual_update_available)
            self._update_check_worker.up_to_date.connect(self._on_manual_update_up_to_date)
            self._update_check_worker.check_failed.connect(self._on_manual_update_failed)
            self.status_bar.showMessage("Checking for updates...", 3000)
        self._update_check_worker.finished.connect(self._on_update_check_done)
        if hasattr(self, "check_for_updates_action") and self.check_for_updates_action is not None:
            self.check_for_updates_action.setEnabled(False)
        self._update_check_worker.start()

    def _on_report_bug(self) -> None:
        """Show the bug report dialog."""
        from ui.dialogs.bug_report_dialog import BugReportDialog

        dialog = BugReportDialog(parent=self)
        dialog.exec()

    def _on_manual_check_for_updates(self) -> None:
        """Run an explicit user-triggered update check."""
        from core.paths import is_frozen

        if sys.platform == "darwin":
            from core.macos_updater import start_interactive_update_check

            native_status = start_interactive_update_check(
                update_channel=getattr(self.settings, "update_channel", get_release_channel()),
            )
            if native_status.launched:
                self._update_service.record_native_check_started()
                try:
                    save_settings(self.settings)
                except Exception:
                    logger.debug("Failed to persist native update launch state", exc_info=True)
                self.status_bar.showMessage("Checking for updates...", 3000)
                return

            if is_frozen() and native_status.reason:
                QMessageBox.information(
                    self,
                    "Native Updates Unavailable",
                    f"{native_status.reason}\n\nFalling back to browser-based update checks.",
                )

        if sys.platform == "win32":
            from core.windows_updater import start_interactive_update_check

            native_status = start_interactive_update_check(
                update_channel=getattr(self.settings, "update_channel", get_release_channel()),
                automatically_check=bool(getattr(self.settings, "check_for_updates", True)),
            )
            if native_status.launched:
                self._update_service.record_native_check_started()
                try:
                    save_settings(self.settings)
                except Exception:
                    logger.debug("Failed to persist native update launch state", exc_info=True)
                self.status_bar.showMessage("Checking for updates...", 3000)
                return

            if is_frozen() and native_status.reason:
                QMessageBox.information(
                    self,
                    "Native Updates Unavailable",
                    f"{native_status.reason}\n\nFalling back to browser-based update checks.",
                )

        self._launch_update_check(interactive=True)

    def _on_manual_update_available(self, version: str, release_url: str) -> None:
        """Prompt the user to open the release page for a newer version."""
        current_version = get_app_version()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Update Available")
        msg.setText(f"Scene Ripper {version} is available.")
        msg.setInformativeText(
            f"You're currently running {current_version}. Open the release page to download the update?"
        )
        open_button = msg.addButton("Open Release Page", QMessageBox.AcceptRole)
        skip_button = msg.addButton("Skip This Version", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()

        if msg.clickedButton() == open_button:
            QDesktopServices.openUrl(QUrl(release_url))
        elif msg.clickedButton() == skip_button:
            self._update_service.skip_version(version)
            try:
                save_settings(self.settings)
            except Exception:
                logger.debug("Failed to persist skipped update version", exc_info=True)

    def _on_manual_update_up_to_date(self, version: str, _release_url: str) -> None:
        """Notify the user when no newer release is available."""
        QMessageBox.information(
            self,
            "You're Up to Date",
            f"Scene Ripper {get_app_version()} is up to date.\nLatest release: {version}",
        )

    def _on_manual_update_failed(self, message: str) -> None:
        """Report update check failures for manual checks."""
        QMessageBox.warning(self, "Update Check Failed", message)

    def _on_update_check_done(self):
        """Save settings after update check (records last_update_check timestamp)."""
        try:
            save_settings(self.settings)
        except Exception:
            pass  # Non-critical
        if hasattr(self, "check_for_updates_action") and self.check_for_updates_action is not None:
            self.check_for_updates_action.setEnabled(True)
        self._manual_update_check = False
        self._update_check_worker = None

    def _on_tab_changed(self, index: int):
        """Handle tab switching."""
        # Get all tabs
        tabs = [
            self.collect_tab,
            self.cut_tab,
            self.analyze_tab,
            self.frames_tab,
            self.sequence_tab,
            self.render_tab,
        ]

        # Track active tab for agent context
        tab_names = ["collect", "cut", "analyze", "frames", "sequence", "render"]
        if 0 <= index < len(tab_names):
            active_tab = tab_names[index]
            self._gui_state.active_tab = active_tab

            # Sync selection state for the new tab
            if active_tab == "cut":
                selected = self.cut_tab.clip_browser.get_selected_clips()
                self._gui_state.selected_clip_ids = [c.id for c in selected]
            elif active_tab == "analyze":
                selected = self.analyze_tab.clip_browser.get_selected_clips()
                self._gui_state.selected_clip_ids = [c.id for c in selected]
            elif active_tab == "frames":
                selected = self.frames_tab.frame_browser.get_selected_frame_ids()
                self._gui_state.selected_frame_ids = selected

        # Notify tabs of activation/deactivation
        for i, tab in enumerate(tabs):
            if i == index:
                tab.on_tab_activated()
            else:
                tab.on_tab_deactivated()

        # Refresh Sequence tab with all clips when switching to it
        if index == 4:  # Sequence tab
            self._refresh_sequence_tab_clips()

        # Update Render tab with current sequence info when switching to it
        if index == 5:  # Render tab
            self._update_render_tab_sequence_info()

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # Project actions
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.New)  # Cmd+N
        new_project_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project...", self)
        open_project_action.setShortcut(QKeySequence.Open)  # Cmd+O
        open_project_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_project_action)

        self.save_project_action = QAction("&Save Project", self)
        self.save_project_action.setShortcut(QKeySequence.Save)  # Cmd+S
        self.save_project_action.triggered.connect(self._on_save_project)
        file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction("Save Project &As...", self)
        self.save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.save_project_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(self.save_project_as_action)

        # Recent Projects submenu
        self.recent_projects_menu = file_menu.addMenu("Recent Pro&jects")
        self._update_recent_projects_menu()

        file_menu.addSeparator()

        # Import submenu
        import_menu = file_menu.addMenu("&Import")

        import_video_action = QAction("&Video...", self)
        import_video_action.setShortcut(QKeySequence("Ctrl+I"))
        import_video_action.triggered.connect(self._on_import_click)
        import_menu.addAction(import_video_action)

        import_folder_action = QAction("&Folder...", self)
        import_folder_action.setShortcut(QKeySequence("Ctrl+Shift+I"))
        import_folder_action.triggered.connect(self._on_import_folder_click)
        import_menu.addAction(import_folder_action)

        import_url_action = QAction("From &URL...", self)
        import_url_action.triggered.connect(self._on_import_url_click)
        import_menu.addAction(import_url_action)

        file_menu.addSeparator()

        # Export EDL action
        self.export_edl_action = QAction("Export &EDL...", self)
        self.export_edl_action.setToolTip("Export timeline as Edit Decision List for NLE import")
        self.export_edl_action.setEnabled(False)
        self.export_edl_action.triggered.connect(self._on_export_edl_click)
        file_menu.addAction(self.export_edl_action)

        # Export Project Bundle action
        self.export_bundle_action = QAction("Export Project &Bundle...", self)
        self.export_bundle_action.setToolTip("Export project as self-contained bundle folder")
        self.export_bundle_action.triggered.connect(self._on_export_bundle_click)
        file_menu.addAction(self.export_bundle_action)

        file_menu.addSeparator()

        # Settings action (Preferences on macOS)
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.setMenuRole(QAction.PreferencesRole)  # macOS standard
        settings_action.triggered.connect(self._on_settings_click)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        # Quit action
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.setMenuRole(QAction.QuitRole)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")

        undo_action = self.undo_stack.createUndoAction(self, "&Undo")
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)

        redo_action = self.undo_stack.createRedoAction(self, "&Redo")
        redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction("Select &All", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)  # Cmd+A
        select_all_action.triggered.connect(self._on_select_all)
        edit_menu.addAction(select_all_action)

        deselect_all_action = QAction("&Deselect All", self)
        deselect_all_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        deselect_all_action.triggered.connect(self._on_deselect_all)
        edit_menu.addAction(deselect_all_action)

        # View menu with tab shortcuts
        view_menu = menu_bar.addMenu("&View")

        tab_names = ["&Collect", "C&ut", "&Analyze", "&Frames", "&Sequence", "&Render"]
        for i, name in enumerate(tab_names):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            action.triggered.connect(lambda checked, idx=i: self.tab_widget.setCurrentIndex(idx))
            view_menu.addAction(action)

        view_menu.addSeparator()

        # Store reference to view menu for chat panel toggle
        self._view_menu = view_menu

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        self.check_for_updates_action = QAction("Check for &Updates...", self)
        self.check_for_updates_action.triggered.connect(self._on_manual_check_for_updates)
        help_menu.addAction(self.check_for_updates_action)

        help_menu.addSeparator()

        report_bug_action = QAction("Report &Bug...", self)
        report_bug_action.setToolTip("Submit a bug report via GitHub, Gmail, or email")
        report_bug_action.triggered.connect(self._on_report_bug)
        help_menu.addAction(report_bug_action)

        help_menu.addSeparator()

        from core.update_checker import _GITHUB_OWNER, _GITHUB_REPO
        _docs_base = f"https://github.com/{_GITHUB_OWNER}/{_GITHUB_REPO}/blob/main/docs/user-guide"

        _help_docs = [
            ("&Chat Agent Guide", "agent.md", "What the chat agent can do"),
            ("Agent &Tools Reference", "agent-tools.md", "All 103 agent tools with examples"),
            ("&Analysis Guide", "analysis.md", "How each analysis type works"),
            ("&Sequencer Guide", "sequencers.md", "Sequencer algorithm documentation"),
            ("&Local AI Models", "local-models.md", "Local models used for analysis"),
            ("API &Keys Guide", "api-keys.md", "API key setup for cloud features"),
            ("&Prompt Reference", "prompts.md", "System prompts used by analysis"),
            ("&Reporting Bugs", "reporting-bugs.md", "How to report issues"),
        ]
        for label, filename, tooltip in _help_docs:
            action = QAction(label, self)
            action.setToolTip(tooltip)
            action.triggered.connect(
                lambda checked=False, f=filename: QDesktopServices.openUrl(
                    QUrl(f"{_docs_base}/{f}")
                )
            )
            help_menu.addAction(action)

    def _is_any_worker_running(self) -> bool:
        """Check if any background worker is currently running."""
        workers = [
            self.detection_worker,
            self.thumbnail_worker,
            self.download_worker,
            self.export_worker,
            self.sequence_preview_worker,
            self.color_worker,
            self.shot_type_worker,
            self.transcription_worker,
            self.classification_worker,
            self.detection_worker_yolo,
            self.youtube_search_worker,
            self.ia_search_worker,
            self.bulk_download_worker,
            self.export_bundle_worker,
        ]
        return any(w and w.isRunning() for w in workers)

    def _on_settings_click(self):
        """Open the settings dialog."""
        # Disable path settings if background operations are running
        paths_disabled = self._is_any_worker_running()
        dialog = SettingsDialog(self.settings, paths_disabled=paths_disabled, parent=self)
        if dialog.exec() == SettingsDialog.Accepted:
            self.settings = dialog.get_settings()
            save_settings(self.settings)
            self._apply_settings()
            self.status_bar.showMessage("Settings saved")
            logger.info("Settings updated and saved")

    def _apply_settings(self):
        """Apply current settings to the UI and components."""
        # Apply theme preference
        theme().set_preference(self.settings.theme_preference)

        # Update chat panel provider and model from settings
        if hasattr(self, 'chat_panel'):
            self.chat_panel.set_provider(self.settings.llm_provider)
            self.chat_panel.update_provider_availability()

        logger.info(
            f"Settings applied: sensitivity={self.settings.default_sensitivity}, "
            f"quality={self.settings.export_quality}, "
            f"theme={self.settings.theme_preference}"
        )

    def _connect_signals(self):
        """Connect UI signals."""
        # Collect tab signals
        self.collect_tab.videos_added.connect(self._on_videos_added)
        self.collect_tab.audio_files_added.connect(self._on_audio_files_added)
        self.collect_tab.audio_remove_requested.connect(self._on_audio_remove_requested)
        self.collect_tab.audio_transcribe_requested.connect(self._on_audio_transcribe_requested)
        self.collect_tab.analyze_requested.connect(self._on_analyze_requested)
        self.collect_tab.source_selected.connect(self._on_source_selected)
        self.collect_tab.download_requested.connect(self._on_download_requested_from_tab)
        self.collect_tab.delete_sources_requested.connect(self._on_delete_sources_requested)

        # Video search panel signals (YouTube and Internet Archive)
        self.collect_tab.youtube_search_panel.search_requested.connect(
            self._on_video_search
        )
        self.collect_tab.youtube_search_panel.download_requested.connect(
            self._on_bulk_download
        )

        # Cut tab signals
        self.cut_tab.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)
        self.cut_tab.analyze_selected_requested.connect(self._on_analyze_selected_from_cut)
        self.cut_tab.selection_changed.connect(self._on_cut_selection_changed)
        self.cut_tab.clip_browser.filters_changed.connect(self._on_cut_filters_changed)
        self.cut_tab.clip_browser.view_details_requested.connect(self.show_clip_details)
        self.cut_tab.clip_browser.export_requested.connect(self._on_clip_export_requested)

        # Analyze tab signals
        self.analyze_tab.quick_run_requested.connect(self._on_quick_run_from_tab)
        self.analyze_tab.analyze_picker_requested.connect(self._on_analyze_picker_from_tab)
        self.analyze_tab.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)
        self.analyze_tab.selection_changed.connect(self._on_analyze_selection_changed)
        self.analyze_tab.clips_changed.connect(self._on_analyze_clips_changed)
        self.analyze_tab.clip_browser.filters_changed.connect(self._on_analyze_filters_changed)
        self.analyze_tab.clip_browser.view_details_requested.connect(self.show_clip_details)
        self.analyze_tab.clip_browser.export_requested.connect(self._on_clip_export_requested)

        # Sequence tab signals
        self.sequence_tab.playback_requested.connect(self._on_playback_requested)
        self.sequence_tab.stop_requested.connect(self._on_stop_requested)
        self.sequence_tab.export_requested.connect(self._on_sequence_export_click)
        self.sequence_tab.render_preview_requested.connect(self._on_render_sequence_preview_requested)
        self.sequence_tab.edl_export_requested.connect(self._on_sequence_edl_export_requested)
        self.sequence_tab.all_edl_export_requested.connect(self._on_all_sequence_edl_export_requested)
        # Intention-first workflow trigger
        self.sequence_tab.intention_import_requested.connect(self._on_intention_import_requested)
        # Description analysis request from Storyteller
        self.sequence_tab.description_analysis_requested.connect(self._on_description_analysis_requested)
        # Update Render tab when sequence changes (clips added/removed/generated)
        self.sequence_tab.timeline.sequence_changed.connect(self._update_render_tab_sequence_info)
        self.sequence_tab.timeline.sequence_changed.connect(self._invalidate_sequence_preview)
        # Update EDL export menu item when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._on_sequence_changed)
        # Update agent context when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._on_sequence_ids_changed)
        # Mark project dirty when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._mark_dirty)
        # Persist auto-computed clip metadata (brightness, volume, embeddings)
        self.sequence_tab.clips_data_changed.connect(lambda clips: self.project.update_clips(clips))
        self.sequence_tab.chromatic_bar_setting_changed.connect(self._on_chromatic_bar_setting_changed)

        # Frames tab signals
        self.frames_tab.extract_frames_requested.connect(self._on_extract_frames_requested)
        self.frames_tab.import_images_requested.connect(self._on_import_images_requested)
        self.frames_tab.analyze_frames_requested.connect(self._on_analyze_frames_requested)
        self.frames_tab.add_to_sequence_requested.connect(self._on_add_frames_to_sequence)
        self.frames_tab.frames_selected.connect(self._on_frames_selection_changed)

        # Render tab signals
        self.render_tab.export_sequence_requested.connect(self._on_sequence_export_click)
        self.render_tab.export_clips_requested.connect(self._on_export_click)
        self.render_tab.export_all_clips_requested.connect(self._on_export_all_click)
        self.render_tab.export_dataset_requested.connect(self._on_export_dataset_click)
        self.render_tab.export_srt_requested.connect(self._on_export_srt_click)

        # Sequence tab timeline signals for playback sync
        self.sequence_tab.timeline.playhead_changed.connect(self._on_timeline_playhead_changed)

        # Sequence tab video player signals for playback sync
        self.sequence_tab.video_player.position_updated.connect(self._on_video_position_updated)
        self.sequence_tab.video_player.playback_state_changed.connect(self._on_video_state_changed)
        self.sequence_tab.video_player.media_loaded.connect(self._on_sequence_video_loaded)
        self.sequence_tab.video_player.media_load_failed.connect(self._on_sequence_video_load_failed)

    def _setup_chat_panel(self):
        """Initialize the chat panel dock widget."""
        # Initialize chat state
        self._chat_history: list[dict] = []
        self._last_user_message: str = ""
        self._current_chat_bubble = None
        self._current_tool_indicator = None
        self._chat_worker: Optional[ChatAgentWorker] = None

        # Create chat panel
        self.chat_panel = ChatPanel()

        # Set initial provider from settings
        self.chat_panel.set_provider(self.settings.llm_provider)

        # Create dock widget
        self.chat_dock = QDockWidget("Agent Chat", self)
        self.chat_dock.setWidget(self.chat_panel)
        self.chat_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)

        # Start hidden by default
        self.chat_dock.setVisible(False)

        # Add toggle action to View menu
        self.chat_toggle_action = self.chat_dock.toggleViewAction()
        self.chat_toggle_action.setText("Show Agent Chat")
        self.chat_toggle_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._view_menu.addAction(self.chat_toggle_action)

        # Connect chat panel signals
        self.chat_panel.message_sent.connect(self._on_chat_message)
        self.chat_panel.cancel_requested.connect(self._on_chat_cancel)
        self.chat_panel.provider_changed.connect(self._on_chat_provider_changed)
        self.chat_panel.clear_requested.connect(self._on_chat_clear)
        self.chat_panel.export_requested.connect(self._on_chat_export)
        self.chat_panel.status_requested.connect(self._on_chat_status)

        # Connect plan signals
        self.chat_panel.plan_confirmed.connect(self._on_plan_confirmed)
        self.chat_panel.plan_cancelled.connect(self._on_plan_cancelled)
        self.chat_panel.plan_retry_requested.connect(self._on_plan_retry_requested)
        self.chat_panel.plan_stop_requested.connect(self._on_plan_stop_requested)

    def _setup_clip_details_sidebar(self):
        """Initialize the clip details sidebar dock widget."""
        # Create sidebar
        self.clip_details_sidebar = ClipDetailsSidebar(self)

        # Add as dock widget on left side
        self.addDockWidget(Qt.LeftDockWidgetArea, self.clip_details_sidebar)

        # Start hidden by default
        self.clip_details_sidebar.setVisible(False)

        # Connect clip edited signal to update project
        self.clip_details_sidebar.clip_edited.connect(self._on_clip_edited)

        # Add toggle action to View menu
        self.clip_details_toggle = self.clip_details_sidebar.toggleViewAction()
        self.clip_details_toggle.setText("Show Clip Details")
        self.clip_details_toggle.setShortcut(QKeySequence("Ctrl+D"))
        self._view_menu.addAction(self.clip_details_toggle)

    def _setup_log_dock(self):
        """Initialize the global in-app log viewer dock widget."""
        self.log_viewer = LogViewerWidget(self._log_bridge, self)

        self.log_dock = QDockWidget("Logs", self)
        self.log_dock.setWidget(self.log_viewer)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setVisible(False)

        self.log_toggle_action = self.log_dock.toggleViewAction()
        self.log_toggle_action.setText("Show Logs")
        self.log_toggle_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        self._view_menu.addAction(self.log_toggle_action)

    def show_log_viewer(self):
        """Reveal and focus the in-app log viewer."""
        self.log_dock.setVisible(True)
        self.log_dock.raise_()
        self.log_dock.activateWindow()

    def show_clip_details(self, clip: Clip, source: Source):
        """Show the clip details sidebar for a clip.

        Args:
            clip: The clip to display
            source: The source video
        """
        if hasattr(self, 'clip_details_sidebar'):
            self.clip_details_sidebar.show_clip(clip, source)

    @Slot(object)
    def _on_clip_edited(self, clip: Clip):
        """Handle clip edited from sidebar.

        Args:
            clip: The edited clip
        """
        # Notify project of clip update (triggers observers including ClipBrowser)
        self.project.update_clips([clip])
        logger.debug(f"Clip {clip.id} updated from sidebar")

    @Slot(list)
    def _on_clips_updated(self, clips: list):
        """Handle clips updated signal from project.

        Forwards clip updates to both tab clip browsers for display refresh.
        Also auto-includes any newly-analyzed clips in the Analyze tab.

        Args:
            clips: List of updated clips
        """
        # Update clip browsers in both tabs
        if hasattr(self, 'cut_tab') and hasattr(self.cut_tab, 'clip_browser'):
            self.cut_tab.clip_browser.update_clips(clips)
        if hasattr(self, 'analyze_tab') and hasattr(self.analyze_tab, 'clip_browser'):
            self.analyze_tab.clip_browser.update_clips(clips)
            self._auto_include_analyzed_clips(clips)

    def _auto_include_analyzed_clips(self, clips: list) -> None:
        """Add any clip with at least one analysis result to the Analyze tab.

        Called from clips_added (project load brings analyzed clips into a
        fresh app session) and clips_updated (analysis pipeline writes
        results back). Disabled clips are still included if they were
        analyzed previously — they render as disabled in the browser.
        """
        if not hasattr(self, 'analyze_tab'):
            return
        existing = self.analyze_tab.get_clip_ids()
        to_add = [c.id for c in clips if c.has_any_analysis() and c.id not in existing]
        if to_add:
            self.analyze_tab.add_clips(to_add)

    @Slot(list)
    def _on_clips_added(self, clips: list):
        """Handle clips added signal from project.

        Refreshes lookups, sets current source, and generates thumbnails for new clips.
        Clips are added to Cut tab via _on_thumbnail_ready when thumbnails complete.

        Args:
            clips: List of added clips
        """
        logger.info(f"Project clips_added event: {len(clips)} clips")

        # Refresh Analyze tab lookups (cached properties may have been invalidated)
        if hasattr(self, 'analyze_tab'):
            self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)
            self._auto_include_analyzed_clips(clips)

        # Determine which source these clips belong to
        clip_source_id = clips[0].source_id if clips else None
        clip_source = self.sources_by_id.get(clip_source_id) if clip_source_id else None

        # Set current source if not already set (needed for _on_thumbnail_ready)
        if clip_source and not self.current_source:
            self.current_source = clip_source
            logger.info(f"Set current_source to {clip_source.id}")

        # Set source in Cut tab (prepares UI state, clips added via _on_thumbnail_ready)
        if hasattr(self, 'cut_tab') and clip_source:
            self.cut_tab.set_source(clip_source)

        # Generate thumbnails - _on_thumbnail_ready will add clips to Cut tab
        # Skip if _on_detection_finished is managing the full pipeline (avoid double generation)
        if self._suppress_clips_added_thumbnails:
            logger.info("Detection handler managing thumbnails, skipping clips_added thumbnail generation")
            return

        # Skip if intention workflow is running (it manages its own thumbnail generation)
        if hasattr(self, 'intention_workflow') and self.intention_workflow and self.intention_workflow.is_running:
            logger.info("Intention workflow running, skipping automatic thumbnail generation")
            return

        clips_needing_thumbnails = [c for c in clips if not c.thumbnail_path or not c.thumbnail_path.exists()]
        if clips_needing_thumbnails:
            logger.info(f"Starting thumbnail generation for {len(clips_needing_thumbnails)} clips")
            # Don't start if another thumbnail worker is running
            if self.thumbnail_worker and self.thumbnail_worker.isRunning():
                logger.warning("ThumbnailWorker already running, queueing clips for later")
                # Store clips for later processing
                if not hasattr(self, '_pending_thumbnail_clips'):
                    self._pending_thumbnail_clips = []
                self._pending_thumbnail_clips.extend(clips_needing_thumbnails)
            else:
                # Get default source for thumbnails (first clip's source)
                default_source = None
                if clips_needing_thumbnails:
                    first_clip = clips_needing_thumbnails[0]
                    default_source = self.sources_by_id.get(first_clip.source_id)

                if default_source:
                    # Safely stop any running worker (shouldn't happen due to check above, but defensive)
                    self._stop_worker_safely(self.thumbnail_worker, "thumbnail")
                    self.thumbnail_worker = ThumbnailWorker(
                        default_source,
                        clips_needing_thumbnails,
                        self.settings.thumbnail_cache_dir,
                        sources_by_id=self.sources_by_id,
                    )
                    # Connect to _on_thumbnail_ready - this adds clips to Cut tab!
                    self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
                    self.thumbnail_worker.finished.connect(self._on_agent_thumbnails_finished, Qt.UniqueConnection)
                    logger.info("Starting ThumbnailWorker for agent-added clips...")
                    self.thumbnail_worker.start()

    @Slot()
    def _on_agent_thumbnails_finished(self):
        """Handle thumbnails completed for agent-added clips."""
        logger.info("Agent thumbnail generation finished")

        # Process any pending thumbnail clips
        if hasattr(self, '_pending_thumbnail_clips') and self._pending_thumbnail_clips:
            pending = self._pending_thumbnail_clips
            self._pending_thumbnail_clips = []
            logger.info(f"Processing {len(pending)} pending thumbnail clips")
            clips_still_needing = [c for c in pending if not c.thumbnail_path or not c.thumbnail_path.exists()]
            if clips_still_needing:
                default_source = self.sources_by_id.get(clips_still_needing[0].source_id)
                if default_source:
                    # Safely stop worker (should be done already since we're in finished handler)
                    self._stop_worker_safely(self.thumbnail_worker, "thumbnail")
                    self.thumbnail_worker = ThumbnailWorker(
                        default_source,
                        clips_still_needing,
                        self.settings.thumbnail_cache_dir,
                        sources_by_id=self.sources_by_id,
                    )
                    # Connect to _on_thumbnail_ready - this adds clips to Cut tab!
                    self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
                    self.thumbnail_worker.finished.connect(self._on_agent_thumbnails_finished, Qt.UniqueConnection)
                    self.thumbnail_worker.start()

    @Slot(object)
    def _on_source_added(self, source):
        """Handle source added signal from project.

        Refreshes lookups for tabs that need to resolve source references.

        Args:
            source: The added Source object
        """
        logger.info(f"Project source_added event: {source.id}")
        # Refresh Analyze tab lookups (cached properties may have been invalidated)
        if hasattr(self, 'analyze_tab'):
            self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

    def _on_delete_sources_requested(self, source_ids: list[str]):
        """Handle delete request from Collect tab. Guards against sequence usage."""
        if not source_ids:
            return

        deletable = []
        blocked = {}  # source_id -> list of sequence names

        for source_id in source_ids:
            seq_names = self.project.source_in_sequences(source_id)
            if seq_names:
                source = self.project.sources_by_id.get(source_id)
                name = source.filename if source else source_id
                blocked[name] = seq_names
            else:
                deletable.append(source_id)

        # Show error for blocked sources
        if blocked:
            lines = []
            for name, seqs in blocked.items():
                lines.append(f"  {name}: used in {', '.join(seqs)}")
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "The following sources have clips in sequences:\n"
                + "\n".join(lines)
                + "\n\nDelete those sequences first.",
            )

        if not deletable:
            return

        # Count clips for confirmation message
        total_clips = sum(
            len([c for c in self.project.clips if c.source_id == sid])
            for sid in deletable
        )
        source_names = []
        for sid in deletable:
            source = self.project.sources_by_id.get(sid)
            source_names.append(source.filename if source else sid)

        if len(deletable) == 1:
            msg = f"Delete \"{source_names[0]}\" and its {total_clips} clips?\nThis cannot be undone."
        else:
            msg = f"Delete {len(deletable)} sources and {total_clips} clips?\nThis cannot be undone."

        result = QMessageBox.question(
            self, "Delete Sources", msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return

        # Perform deletion
        for source_id in deletable:
            self.project.remove_source(source_id)
            self.collect_tab.remove_source(source_id)

        # Refresh lookups
        if hasattr(self, "analyze_tab"):
            self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        self._update_window_title()
        self.status_bar.showMessage(
            f"Deleted {len(deletable)} source(s) and {total_clips} clips"
        )

    @Slot(list)
    def _on_clips_removed(self, clips: list):
        """Handle clips removed signal from project.

        Removes clips from both Cut and Analyze tab clip browsers.
        """
        clip_ids = [c.id for c in clips]
        logger.info(f"Project clips_removed event: {len(clips)} clips")
        if hasattr(self, 'cut_tab') and hasattr(self.cut_tab, 'clip_browser'):
            self.cut_tab.clip_browser.remove_clips_by_ids(clip_ids)
        if hasattr(self, 'analyze_tab') and hasattr(self.analyze_tab, 'clip_browser'):
            self.analyze_tab.clip_browser.remove_clips_by_ids(clip_ids)

    @Slot(list)
    def _on_frames_removed(self, frames: list):
        """Handle frames removed signal from project."""
        logger.info(f"Project frames_removed event: {len(frames)} frames")
        if hasattr(self, 'frames_tab'):
            self.frames_tab.update_frame_browser()

    @Slot(object)
    def _on_source_updated(self, source):
        """Handle source updated signal from project.

        Refreshes the source browser in the Collect tab.
        """
        logger.info(f"Project source_updated event: {source.id}")
        if hasattr(self, 'collect_tab') and hasattr(self.collect_tab, 'source_browser'):
            self.collect_tab.source_browser.update()

    def _on_chat_message(self, message: str):
        """Handle user message from chat panel."""
        from core.llm_client import ProviderConfig, ProviderType
        from core.settings import (
            get_anthropic_api_key, get_openai_api_key,
            get_gemini_api_key, get_openrouter_api_key
        )

        # Store message for history
        self._last_user_message = message

        # Update pending action with user's response (if any)
        # This ensures the agent sees the actual user input in context
        if self._gui_state.pending_action:
            self._gui_state.update_pending_action_response(message)

        # Check for chat-based plan confirmation
        if self.chat_panel.has_pending_plan():
            if self._is_plan_confirmation(message):
                # User confirmed plan via chat message
                plan = self.chat_panel.get_current_plan()
                if plan:
                    self._on_plan_confirmed(plan)
                    return

        # Cancel any existing worker
        if self._chat_worker and self._chat_worker.isRunning():
            self._chat_worker.stop()
            self._chat_worker.wait(1000)

        # Get current provider config
        provider_key = self.chat_panel.get_provider()

        # Get API key for the selected provider (not from disk settings)
        api_key_getters = {
            "anthropic": get_anthropic_api_key,
            "openai": get_openai_api_key,
            "gemini": get_gemini_api_key,
            "openrouter": get_openrouter_api_key,
        }
        api_key = api_key_getters.get(provider_key, lambda: "")()

        # Build provider config
        config = ProviderConfig(
            provider=ProviderType(provider_key),
            model=self.settings.llm_model,
            api_key=api_key or None,
            api_base=self.settings.llm_api_base or None,
            temperature=self.settings.llm_temperature,
        )

        # Build message history
        messages = self._chat_history + [{"role": "user", "content": message}]

        # Create busy check callback
        def check_busy(tool_name: str) -> bool:
            """Check if a conflicting worker is running."""
            worker_map = {
                "detect_scenes": "detection_worker",
                "download_video": "download_worker",
            }
            attr = worker_map.get(tool_name)
            if attr and hasattr(self, attr):
                worker = getattr(self, attr)
                return worker is not None and worker.isRunning()
            return False

        # Start worker with GUI state context
        self._chat_worker = ChatAgentWorker(
            config=config,
            messages=messages,
            project=self.project,
            busy_check=check_busy,
            gui_state_context=self._gui_state.to_context_string(project=self.project),
        )
        # Clear last error after it's been consumed by the agent context
        self._gui_state.clear_last_error()

        # Connect signals
        bubble = self.chat_panel.start_streaming_response()
        self._current_chat_bubble = bubble

        self._chat_worker.text_chunk.connect(self.chat_panel.on_stream_chunk)
        self._chat_worker.clear_current_bubble.connect(self.chat_panel.on_clear_bubble)
        self._chat_worker.tool_called.connect(self._on_chat_tool_called)
        self._chat_worker.tool_result.connect(self._on_chat_tool_result)
        self._chat_worker.gui_tool_requested.connect(self._on_gui_tool_requested)
        self._chat_worker.gui_tool_cancelled.connect(self._on_gui_tool_cancelled)
        self._chat_worker.complete.connect(self._on_chat_complete)
        self._chat_worker.error.connect(self._on_chat_error)

        # Workflow progress for compound operations
        self._chat_worker.workflow_progress.connect(self._on_workflow_progress)

        # GUI sync signals - update GUI components when agent performs actions
        self._chat_worker.youtube_search_completed.connect(self._on_agent_youtube_search)
        self._chat_worker.video_download_completed.connect(self._on_agent_video_downloaded)

        self._chat_worker.start()

    def _on_chat_tool_called(self, name: str, args: dict):
        """Handle tool execution start."""
        logger.info(f"Chat agent calling tool: {name}")
        self._current_tool_indicator = self.chat_panel.add_tool_indicator(name)

    def _on_chat_tool_result(self, name: str, result: dict, success: bool):
        """Handle tool execution completion."""
        logger.info(f"Chat tool {name} completed: success={success}")
        if self._current_tool_indicator:
            self._current_tool_indicator.set_complete(success)

        # Auto-save check: if tool modifies project state and succeeded, schedule save
        if success:
            from core.chat_tools import tools as tool_registry
            tool_def = tool_registry.get(name)
            if tool_def and tool_def.modifies_project_state:
                self._schedule_auto_save()

    def _schedule_auto_save(self):
        """Schedule a debounced auto-save after tool execution.

        Only saves if project has a path (was saved before) and is dirty.
        Uses 300ms debounce to coalesce rapid consecutive tool calls.
        """
        if not self.project or not self.project.path:
            return
        if not self.project.is_dirty:
            return

        # Debounce: restart timer on each call
        self._auto_save_timer.stop()
        self._auto_save_timer.start(300)  # 300ms debounce

    def _do_auto_save(self):
        """Execute the auto-save."""
        if not self.project or not self.project.path:
            return
        if not self.project.is_dirty:
            return

        try:
            self.project.save()
            logger.info(f"Auto-saved project to {self.project.path}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            # Keep project dirty so user can manually save

    @Slot(str)
    def _on_gui_tool_cancelled(self, tool_name: str):
        """Cancel any running worker when agent tool times out.

        This prevents orphaned background threads from continuing to run
        after the agent has given up waiting for them.

        Args:
            tool_name: Name of the tool that timed out
        """
        logger.warning(f"Agent tool '{tool_name}' timed out. Attempting to cancel worker.")

        # Map tool names to their worker attributes
        worker_map = {
            "detect_scenes_live": "detection_worker",
            "download_video": "download_worker",
        }

        worker_attr = worker_map.get(tool_name)
        if worker_attr:
            worker = getattr(self, worker_attr, None)
            if worker and worker.isRunning() and hasattr(worker, "cancel"):
                logger.info(f"Cancelling {worker_attr} due to timeout")
                worker.cancel()

        # Clear pending tool state
        if self._pending_agent_tool_name == tool_name:
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None

    @Slot(str, int, int)
    def _on_workflow_progress(self, step_name: str, current: int, total: int):
        """Handle workflow progress updates for compound operations.

        Args:
            step_name: Name of the current step/tool
            current: Current step number (1-indexed)
            total: Total number of steps
        """
        logger.info(f"Workflow progress: {step_name} ({current}/{total})")
        # Update status bar with progress
        self.statusBar().showMessage(f"Processing: {step_name} ({current}/{total})")

    @Slot(str, dict, str)
    def _on_gui_tool_requested(self, tool_name: str, args: dict, tool_call_id: str):
        """Execute a GUI-modifying tool on the main thread.

        This slot is called by ChatAgentWorker when a tool that modifies
        GUI state needs to be executed. The tool runs on the main thread
        to ensure thread safety with Qt.

        For tools that start background workers (detect_scenes_live, etc.),
        the result is deferred until the worker completes.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            tool_call_id: ID for tracking the tool call
        """
        from core.chat_tools import tools as tool_registry
        import inspect

        logger.info(f"Executing GUI tool on main thread: {tool_name}")

        tool = tool_registry.get(tool_name)
        if not tool or not tool.modifies_gui_state:
            # Invalid tool request
            result = {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "success": False,
                "error": f"Tool '{tool_name}' is not a valid GUI tool"
            }
        else:
            try:
                # Check what parameters the tool accepts
                sig = inspect.signature(tool.func)
                params = sig.parameters

                # Inject project if tool accepts it
                if "project" in params:
                    args["project"] = self.project

                # Inject gui_state if tool accepts it
                if "gui_state" in params:
                    args["gui_state"] = self._gui_state

                # Inject main_window if tool accepts it (for project load/new)
                if "main_window" in params:
                    args["main_window"] = self

                tool_result = tool.func(**args)

                # Check if tool needs to wait for async worker completion
                if isinstance(tool_result, dict) and tool_result.get("_wait_for_worker"):
                    wait_type = tool_result["_wait_for_worker"]
                    logger.info(f"GUI tool {tool_name} waiting for worker: {wait_type}")
                    # Store tool_call_id for when worker completes
                    self._pending_agent_tool_call_id = tool_call_id
                    self._pending_agent_tool_name = tool_name

                    # Start the appropriate worker based on wait_type
                    started = self._start_worker_for_tool(wait_type, tool_result)
                    if not started:
                        # Worker couldn't start - likely already running
                        is_running = False
                        worker_attr = {
                            "description": "description_worker",
                            "classification": "classification_worker",
                            "object_detection": "detection_worker_yolo",
                            "color_analysis": "color_worker",
                            "shot_analysis": "shot_type_worker",
                            "transcription": "transcription_worker",
                        }.get(wait_type)
                        
                        if worker_attr and hasattr(self, worker_attr):
                            w = getattr(self, worker_attr)
                            is_running = w and w.isRunning()

                        error_msg = f"Failed to start {wait_type} worker."
                        if is_running:
                            error_msg = (
                                f"An analysis task ({wait_type}) is already in progress. "
                                "Please wait for it to complete. Note: First-time runs may "
                                "take several minutes while model weights are downloaded."
                            )

                        result = {
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "success": False,
                            "error": error_msg
                        }
                        self._pending_agent_tool_call_id = None
                        self._pending_agent_tool_name = None
                        if self._chat_worker:
                            self._chat_worker.set_gui_tool_result(result)
                    # Don't call set_gui_tool_result yet - worker handler will do it
                    return

                # Check if tool wants to display a plan widget
                if isinstance(tool_result, dict) and tool_result.get("_display_plan"):
                    logger.info(f"GUI tool {tool_name} displaying plan widget")
                    self._handle_display_plan(tool_result, tool_call_id)
                    return

                # Handle special GUI actions based on tool results
                self._apply_gui_tool_side_effects(tool_name, args, tool_result)

                result = {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "success": True,
                    "result": tool_result
                }
                logger.info(f"GUI tool {tool_name} completed successfully")
            except Exception as e:
                logger.exception(f"GUI tool execution failed: {tool_name}")
                result = {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "success": False,
                    "error": str(e)
                }

        # Send result back to worker thread
        if self._chat_worker:
            self._chat_worker.set_gui_tool_result(result)

    def _apply_gui_tool_side_effects(self, tool_name: str, args: dict, result: dict):
        """Apply GUI side effects after tool execution.

        Some tools modify GUIState but also need to trigger actual UI updates.
        This method handles those cases.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
            result: Tool result
        """
        if not result.get("success", True):
            return

        if tool_name == "navigate_to_tab":
            # Actually switch the tab in the UI
            tab_name = args.get("tab_name", "")
            tab_names = ["collect", "cut", "analyze", "frames", "sequence", "render"]
            tab_aliases = {"generate": "sequence"}
            tab_name = tab_aliases.get(tab_name, tab_name)
            if tab_name in tab_names:
                index = tab_names.index(tab_name)
                self.tab_widget.setCurrentIndex(index)

        elif tool_name == "select_clips":
            # Update clip browser selection in the active tab
            selected_ids = result.get("selected", [])
            active_tab = self._gui_state.active_tab

            if active_tab == "cut":
                self.cut_tab.clip_browser.set_selection(selected_ids)
            elif active_tab == "analyze":
                self.analyze_tab.clip_browser.set_selection(selected_ids)

        elif tool_name == "navigate_to_frames_tab":
            # Actually switch the tab in the UI
            tab_names = ["collect", "cut", "analyze", "frames", "sequence", "render"]
            index = tab_names.index("frames")
            self.tab_widget.setCurrentIndex(index)

        elif tool_name == "send_to_analyze":
            # Switch to Analyze tab after sending clips
            self._switch_to_tab("analyze")

        elif tool_name in ("apply_filters", "clear_filters", "set_clip_sort_order"):
            # Force clip browser to repaint after filter/sort changes
            browser = self.get_active_clip_browser()
            if browser:
                browser.update()

        elif tool_name == "show_clip_details":
            # Open the clip details sidebar for the specified clip
            clip_id = result.get("clip_id")
            source_id = result.get("source_id")
            if clip_id and source_id:
                clip = self.project.clips_by_id.get(clip_id)
                source = self.project.sources_by_id.get(source_id)
                if clip and source:
                    self.show_clip_details(clip, source)

    def _refresh_timeline_from_project(self):
        """Refresh the sequence tab timeline from the project's sequence."""
        if not self.project.sequence:
            self.sequence_tab.timeline_preview.clear()
            self.sequence_tab._set_state(self.sequence_tab.STATE_CARDS)
            return

        # Build sources dict from project
        sources = {s.id: s for s in self.project.sources}

        # Load sequence into timeline
        self.sequence_tab.timeline.load_sequence(
            self.project.sequence,
            sources,
            self.project.clips
        )

        # Update sequence tab's internal state
        self.sequence_tab._sources.update(sources)
        if self.project.clips:
            # Build (Clip, Source) tuples for all clips
            self.sequence_tab._available_clips = [
                (clip, sources.get(clip.source_id))
                for clip in self.project.clips
                if sources.get(clip.source_id)
            ]
            self.sequence_tab._clips = self.project.clips

        sequence_clip_ids = [
            seq_clip.source_clip_id
            for seq_clip in self.project.sequence.get_all_clips()
            if seq_clip.source_clip_id
        ]
        sequence_preview = [
            (self.project.clips_by_id[clip_id], sources.get(self.project.clips_by_id[clip_id].source_id))
            for clip_id in sequence_clip_ids
            if clip_id in self.project.clips_by_id
            and sources.get(self.project.clips_by_id[clip_id].source_id) is not None
        ]

        has_timeline_clips = any(track.clips for track in self.project.sequence.tracks)

        # Update state based on timeline content
        if has_timeline_clips:
            # Show timeline state (2-state model: CARDS vs TIMELINE)
            self.sequence_tab._set_state(self.sequence_tab.STATE_TIMELINE)
            # Update timeline preview
            self.sequence_tab.timeline_preview.set_clips(sequence_preview, sources)
        else:
            self.sequence_tab.timeline_preview.clear()
            self.sequence_tab._set_state(self.sequence_tab.STATE_CARDS)

        self.sequence_tab.sync_sequence_metadata(self.project.sequence)
        self._update_sequence_chromatic_bar()

        # Zoom to fit the content
        self.sequence_tab.timeline._on_zoom_fit()

        logger.info(
            "Refreshed timeline with %s clips",
            len(self.project.sequence.get_all_clips()),
        )

    def _on_chat_complete(self, response: str, tool_history: list[dict]):
        """Handle chat completion with full history."""
        logger.info("Chat response complete")
        self.chat_panel.on_stream_complete()

        # Add user message to history
        self._chat_history.append({"role": "user", "content": self._last_user_message})

        # Add tool interactions to history
        for msg in tool_history:
            self._chat_history.append(msg)

        # Add final assistant response
        if response and response != "*Cancelled*":
            self._chat_history.append({"role": "assistant", "content": response})

    def _on_chat_error(self, error: str):
        """Handle chat error."""
        logger.error(f"Chat error: {error}")
        self.chat_panel.on_stream_error(error)

    # =========================================================================
    # Plan Execution Flow
    # =========================================================================

    def _handle_display_plan(self, tool_result: dict, tool_call_id: str):
        """Handle the present_plan tool result by showing the plan widget.

        Args:
            tool_result: Tool result containing plan data
            tool_call_id: ID for tracking the tool call
        """
        from models.plan import Plan

        # Create Plan from the tool result
        steps = tool_result.get("steps", [])
        summary = tool_result.get("summary", "")
        plan = Plan.from_steps(steps, summary)
        plan.id = tool_result.get("plan_id", plan.id)

        # Store plan in GUI state
        self._gui_state.set_plan(plan)

        # Store pending tool call ID for when user confirms
        self._pending_plan_tool_call_id = tool_call_id

        # Display plan widget in chat panel
        self.chat_panel.show_plan_widget(plan)

        # Return result to worker - plan is displayed, awaiting confirmation
        if self._chat_worker:
            result = {
                "tool_call_id": tool_call_id,
                "name": "present_plan",
                "success": True,
                "result": {
                    "plan_id": plan.id,
                    "step_count": len(steps),
                    "status": "awaiting_confirmation",
                    "message": "Plan displayed. Waiting for user to confirm or edit."
                }
            }
            self._chat_worker.set_gui_tool_result(result)

    def _is_plan_confirmation(self, message: str) -> bool:
        """Check if a message is a plan confirmation.

        Detects phrases like "confirm", "run it", "execute", "go ahead",
        "looks good", "start", "do it".

        Args:
            message: User message to check

        Returns:
            True if message appears to be plan confirmation
        """
        message_lower = message.lower().strip()

        # Direct confirmation phrases
        confirmation_phrases = [
            "confirm",
            "confirmed",
            "run it",
            "run the plan",
            "execute",
            "execute it",
            "execute the plan",
            "go ahead",
            "go for it",
            "looks good",
            "lgtm",
            "start",
            "start it",
            "do it",
            "proceed",
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "let's go",
            "lets go",
            "let's do it",
            "lets do it",
        ]

        # Check for exact match or phrase at start
        for phrase in confirmation_phrases:
            if message_lower == phrase:
                return True
            # Also check if it starts with the phrase followed by punctuation
            if message_lower.startswith(phrase) and (
                len(message_lower) == len(phrase)
                or message_lower[len(phrase)] in "!.,;: "
            ):
                return True

        # Check for negation - don't confirm if user says "no", "don't", "cancel", etc.
        negation_phrases = [
            "no",
            "nope",
            "don't",
            "dont",
            "cancel",
            "stop",
            "wait",
            "hold on",
            "not yet",
            "actually",
            "change",
            "edit",
            "modify",
        ]
        for phrase in negation_phrases:
            if phrase in message_lower:
                return False

        return False

    @Slot(object)
    def _on_plan_confirmed(self, plan):
        """Handle plan confirmation from chat panel.

        Args:
            plan: Confirmed Plan object (may have edited steps)
        """
        logger.info(f"Plan confirmed: {plan.summary} with {len(plan.steps)} steps")

        # Update GUI state with confirmed plan
        self._gui_state.set_plan(plan)

        # Set plan as executing in the widget
        self.chat_panel.set_plan_executing(True)
        plan.start_execution()

        # Send a message to the agent to start executing
        execution_message = (
            f"User confirmed the plan. Execute the following {len(plan.steps)} steps in order:\n\n"
        )
        for i, step in enumerate(plan.steps):
            execution_message += f"{i+1}. {step.description}\n"

        execution_message += "\nExecute each step one at a time. Report progress after each step."

        # Simulate a user message to trigger execution
        self._on_chat_message(execution_message)

    @Slot()
    def _on_plan_cancelled(self):
        """Handle plan cancellation from chat panel."""
        logger.info("Plan cancelled by user")

        # Clear plan from GUI state
        self._gui_state.clear_plan_state()

        # Add a message indicating cancellation
        self.chat_panel.add_assistant_message("*Plan cancelled*")

    @Slot(int)
    def _on_plan_retry_requested(self, step_index: int):
        """Handle retry request for a failed plan step.

        Args:
            step_index: Index of the step to retry
        """
        plan = self._gui_state.current_plan
        if not plan or step_index >= len(plan.steps):
            return

        logger.info(f"Retrying plan step {step_index + 1}: {plan.steps[step_index].description}")

        # Reset step status
        plan.retry_current_step()
        self.chat_panel.update_plan_step_status(step_index, "running")

        # Send message to agent to retry this step
        step = plan.steps[step_index]
        retry_message = f"Retry step {step_index + 1}: {step.description}"
        self._on_chat_message(retry_message)

    @Slot()
    def _on_plan_stop_requested(self):
        """Handle stop request after plan step failure."""
        plan = self._gui_state.current_plan
        if not plan:
            return

        logger.info("Plan execution stopped by user")

        # Mark plan as failed
        plan.stop_on_failure()

        # Update widget
        self.chat_panel.set_plan_executing(False)

        # Generate summary of what was completed
        completed = sum(1 for s in plan.steps if s.status == "completed")
        failed = sum(1 for s in plan.steps if s.status == "failed")
        pending = sum(1 for s in plan.steps if s.status == "pending")

        summary = "**Plan stopped**\n\n"
        summary += f"- Completed: {completed}/{len(plan.steps)} steps\n"
        if failed > 0:
            summary += f"- Failed: {failed} step(s)\n"
        if pending > 0:
            summary += f"- Skipped: {pending} step(s)\n"

        self.chat_panel.add_assistant_message(summary)

        # Clear plan from GUI state
        self._gui_state.clear_plan_state()

    def update_plan_step_progress(self, step_index: int, status: str, error: str = None, summary: str = None):
        """Update a plan step's progress during execution.

        Called by the agent execution flow to update step status.

        Args:
            step_index: Index of the step
            status: New status (running, completed, failed)
            error: Error message if failed
            summary: Result summary if completed
        """
        plan = self._gui_state.current_plan
        if not plan or step_index >= len(plan.steps):
            return

        # Update plan model
        step = plan.steps[step_index]
        step.status = status
        if error:
            step.error = error
        if summary:
            step.result_summary = summary

        # Update widget
        self.chat_panel.update_plan_step_status(step_index, status, error)

        # Check if plan is complete
        if status == "completed" and step_index == len(plan.steps) - 1:
            # All steps completed
            plan.status = "completed"
            self.chat_panel.set_plan_executing(False)
            self.chat_panel.mark_plan_completed()

            # Generate completion summary
            self.chat_panel.add_assistant_message(
                f"**Plan complete!** {plan.get_progress_summary()}"
            )
            self._gui_state.clear_plan_state()

    @Slot(str, list)
    def _on_agent_youtube_search(self, query: str, videos: list[dict]):
        """Sync agent YouTube search results to the GUI.

        When the chat agent performs a YouTube search, this updates the
        Collect tab's YouTube panel to display the results.

        Args:
            query: Search query that was executed
            videos: List of video dicts from search_youtube tool
        """
        logger.info(f"Agent YouTube search completed: '{query}' with {len(videos)} results")

        # Convert dicts to YouTubeVideo objects
        video_objects = []
        for v in videos:
            # Parse duration string (e.g., "5:23" or "1:02:30") to timedelta
            duration = None
            duration_str = v.get("duration", "")
            if duration_str:
                parts = duration_str.split(":")
                try:
                    if len(parts) == 2:
                        duration = timedelta(minutes=int(parts[0]), seconds=int(parts[1]))
                    elif len(parts) == 3:
                        duration = timedelta(
                            hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2])
                        )
                except (ValueError, IndexError):
                    pass

            video_objects.append(YouTubeVideo(
                video_id=v.get("video_id", ""),
                title=v.get("title", ""),
                description="",  # Not included in tool output
                channel_title=v.get("channel", ""),
                thumbnail_url=v.get("thumbnail", ""),
                duration=duration,
                view_count=v.get("view_count"),
            ))

        # Update YouTube panel with results
        self.collect_tab.youtube_search_panel.search_input.setText(query)
        self.collect_tab.youtube_search_panel.display_results(video_objects)

        # Expand panel if collapsed
        if not self.collect_tab.youtube_search_panel._expanded:
            self.collect_tab.youtube_search_panel.toggle_btn.click()

        # Switch to Collect tab so user sees the results
        self._switch_to_tab("collect")

        # Update status bar
        self.status_bar.showMessage(f"Agent found {len(videos)} videos for '{query}'")

        # Update GUI state for agent context
        self._gui_state.update_from_search(query, videos)

    @Slot(str, dict)
    def _on_agent_video_downloaded(self, url: str, result: dict):
        """Sync agent video download to the GUI.

        When the chat agent downloads a video, this adds it to the library.

        Args:
            url: URL that was downloaded
            result: Download result dict containing file_path
        """
        file_path = result.get("file_path")
        if not file_path:
            logger.warning(f"Agent download completed but no file_path in result: {result}")
            return

        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Agent downloaded file not found: {path}")
            return

        logger.info(f"Agent downloaded video: {path.name}")

        # Add to library using existing method
        self._load_video(path)

        # Update status bar
        self.status_bar.showMessage(f"Agent downloaded: {path.name}")

    def _switch_to_tab(self, tab_name: str):
        """Switch to a specific tab by name.

        Args:
            tab_name: Tab name ("collect", "cut", "analyze", "frames", "sequence", "render")
        """
        tab_map = {
            "collect": 0,
            "cut": 1,
            "analyze": 2,
            "frames": 3,
            "sequence": 4,
            "generate": 4,
            "render": 5,
        }
        index = tab_map.get(tab_name.lower())
        if index is not None and hasattr(self, 'tab_widget'):
            self.tab_widget.setCurrentIndex(index)

    def get_active_clip_browser(self):
        """Get the clip browser for the currently active tab.

        Returns:
            ClipBrowser instance or None if not available
        """
        active_tab = self._gui_state.active_tab if self._gui_state else "cut"
        return self.get_clip_browser(active_tab)

    def get_clip_browser(self, tab_name: str = None):
        """Get the clip browser for a specific tab.

        Args:
            tab_name: Tab name ('cut' or 'analyze'), or None for active tab

        Returns:
            ClipBrowser instance or None if not available
        """
        if tab_name is None:
            tab_name = self._gui_state.active_tab if self._gui_state else "cut"

        if tab_name == "cut" and hasattr(self, 'cut_tab'):
            return self.cut_tab.clip_browser
        elif tab_name == "analyze" and hasattr(self, 'analyze_tab'):
            return self.analyze_tab.clip_browser
        return None

    def get_video_player(self):
        """Get the video player from the sequence tab.

        Returns:
            VideoPlayer instance or None if not available
        """
        if hasattr(self, 'sequence_tab') and hasattr(self.sequence_tab, 'video_player'):
            return self.sequence_tab.video_player
        return None

    def get_selected_clips(self):
        """Get selected clips from the active tab's clip browser.

        Returns:
            List of selected Clip objects, or empty list if none
        """
        clip_browser = self.get_active_clip_browser()
        if clip_browser:
            return clip_browser.get_selected_clips()
        return []

    def get_analyze_clip_count(self) -> int:
        """Get the number of clips in the Analyze tab.

        Returns:
            Number of clips, or 0 if tab not available
        """
        if hasattr(self, 'analyze_tab'):
            return len(self.analyze_tab.get_clip_ids())
        return 0

    def remove_source_from_library(self, source_id: str) -> bool:
        """Remove a source from the library browser.

        Args:
            source_id: ID of the source to remove

        Returns:
            True if removed, False if not found
        """
        if hasattr(self, 'collect_tab') and hasattr(self.collect_tab, 'source_browser'):
            self.collect_tab.source_browser.remove_source(source_id)
            return True
        return False

    def _on_chat_cancel(self):
        """Handle chat cancellation."""
        if self._chat_worker and self._chat_worker.isRunning():
            logger.info("Cancelling chat worker")
            self._chat_worker.stop()

    def _on_chat_clear(self):
        """Handle chat clear request - reset conversation history."""
        logger.info("Clearing chat history")
        self._chat_history.clear()
        self._last_user_message = ""

    def _on_chat_status(self):
        """Handle /status slash command - display project summary in chat."""
        from core.chat_tools import get_project_summary

        if not self.project:
            self.chat_panel.add_assistant_message(
                "No project loaded. Import a video to get started."
            )
            return

        result = get_project_summary(self.project)
        if result.get("success"):
            self.chat_panel.add_assistant_message(result["summary"])
        else:
            self.chat_panel.add_assistant_message(
                "Could not retrieve project status."
            )

    def _on_chat_export(self):
        """Handle chat export request - show export dialog and save files."""
        from pathlib import Path
        import subprocess
        import sys

        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from core.chat_export import export_chat
        from ui.export_chat_dialog import ExportChatDialog

        if not self._chat_history:
            QMessageBox.information(
                self,
                "No Messages",
                "There are no messages to export."
            )
            return

        # Show export options dialog
        dialog = ExportChatDialog(len(self._chat_history), self)
        if dialog.exec() != ExportChatDialog.Accepted:
            return

        # Get export folder
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            str(Path.home())
        )
        if not output_dir:
            return  # User cancelled

        # Get config with user selections
        project_name = self.project.metadata.name if self.project else ""
        config = dialog.get_config(Path(output_dir), project_name)

        # Perform export
        success, created_files, error = export_chat(self._chat_history, config)

        if success and created_files:
            # Build success message
            file_list = "\n".join(f"  • {Path(f).name}" for f in created_files)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Export Complete")
            msg.setText(f"Chat exported successfully:\n\n{file_list}")
            msg.setStandardButtons(QMessageBox.Ok)

            # Add "Open Folder" button
            open_folder_btn = msg.addButton("Open Folder", QMessageBox.ActionRole)
            msg.exec()

            # Handle "Open Folder" click
            if msg.clickedButton() == open_folder_btn:
                # Open folder in system file manager
                if sys.platform == "darwin":
                    subprocess.run(["open", output_dir], check=False)
                elif sys.platform == "win32":
                    subprocess.run(["explorer", output_dir], check=False)
                else:
                    subprocess.run(["xdg-open", output_dir], check=False)

            logger.info(f"Chat exported to: {created_files}")
        else:
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export chat:\n\n{error}"
            )
            logger.error(f"Chat export failed: {error}")

    def _on_chat_provider_changed(self, provider: str):
        """Handle provider selection change."""
        logger.info(f"Chat provider changed to: {provider}")
        # Update settings
        self.settings.llm_provider = provider
        # Use the user's configured model for this provider, not the default
        self.settings.llm_model = self.settings.get_model_for_provider(provider)
        logger.info(f"Chat model set to: {self.settings.llm_model}")
        # Note: Not auto-saving to allow temporary changes during session

    def _update_chat_project_state(self):
        """Update chat panel with current project state for context-aware prompts."""
        if not hasattr(self, 'chat_panel'):
            return

        has_sources = len(self.project.sources) > 0
        clip_count = len(self.project.clips)

        # Check if any clips have been analyzed (shot_type or transcript)
        has_analyzed = any(
            clip.shot_type or clip.transcript
            for clip in self.project.clips
        )

        sequence_length = (
            len(self.project.sequence.tracks[0].clips)
            if self.project.sequence and self.project.sequence.tracks
            else 0
        )

        self.chat_panel.update_project_state(
            has_sources=has_sources,
            clip_count=clip_count,
            has_analyzed=has_analyzed,
            sequence_length=sequence_length
        )

    def _on_download_requested_from_tab(self, url: str, resolution: str):
        """Handle download request from Collect tab."""
        self._download_video(url, resolution=resolution)

    def _on_videos_added(self, paths: list[Path]):
        """Handle multiple videos added from Collect tab."""
        for path in paths:
            self._add_video_to_library(path)

    def _on_analyze_requested(self, source_ids: list[str]):
        """Handle analyze request from Collect tab.

        Args:
            source_ids: List of source IDs to analyze. If empty, analyze all unanalyzed.
        """
        if source_ids:
            sources_to_analyze = [
                self.sources_by_id[sid]
                for sid in source_ids
                if sid in self.sources_by_id and not self.sources_by_id[sid].analyzed
            ]
        else:
            sources_to_analyze = [s for s in self.sources if not s.analyzed]

        if not sources_to_analyze:
            self.status_bar.showMessage("All videos are already analyzed")
            return

        # Queue all sources for batch analysis
        self._analyze_queue = deque(sources_to_analyze)
        self._analyze_queue_total = len(sources_to_analyze)
        self._detection_start_time = time.time()  # Track when batch started
        self._detection_current_progress = 0.0
        self._start_next_analysis()

    def _start_next_analysis(self):
        """Start analyzing the next source in the queue."""
        if not self._analyze_queue:
            self.status_bar.showMessage("Batch analysis complete")
            self.queue_label.setVisible(False)
            self._analyze_queue_total = 0
            self._detection_start_time = None  # Reset tracking
            self._detection_current_progress = 0.0
            return

        # Pop the next source from the queue (O(1) with deque)
        source = self._analyze_queue.popleft()
        remaining = len(self._analyze_queue)
        current = self._analyze_queue_total - remaining

        # Update queue indicator
        if self._analyze_queue_total > 1:
            self.queue_label.setText(f"Processing {current} of {self._analyze_queue_total}")
            self.queue_label.setVisible(True)

        self._select_source(source)
        self._start_detection("adaptive", {"threshold": self.settings.default_sensitivity})
        self.tab_widget.setCurrentIndex(1)  # Switch to Cut tab

    def _on_source_selected(self, source: Source):
        """Handle source selection from Collect tab."""
        self._select_source(source)

    def _select_source(self, source: Source):
        """Select a source as the current active source."""
        if source.id == getattr(self.current_source, 'id', None):
            return  # Already selected

        self.current_source = source

        # Update Cut tab with source info (keeps existing clips visible)
        self.cut_tab.set_source(source)

        # Update Sequence tab
        self.sequence_tab.set_source(source)

        self._update_window_title()
        self.status_bar.showMessage(f"Selected: {source.filename}")

    def _create_source_with_metadata(self, path: Path) -> Source:
        """Create a Source with metadata extracted via FFprobe.

        Args:
            path: Path to the video file

        Returns:
            Source with duration, fps, width, height populated
        """
        source = Source(file_path=path)

        # Try to extract metadata using FFprobe
        try:
            processor = FFmpegProcessor()
            info = processor.get_video_info(path)
            source.duration_seconds = info.get("duration", 0.0)
            source.fps = info.get("fps", 30.0)
            source.width = info.get("width", 0)
            source.height = info.get("height", 0)
            logger.debug(f"Extracted metadata for {path.name}: {source.duration_seconds:.1f}s, {source.fps:.1f}fps, {source.width}x{source.height}")
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {path.name}: {e}")
            # Source will have default values (duration=0, fps=30, etc.)

        return source

    def _add_video_to_library(self, path: Path):
        """Add a video file to the library without making it active."""
        # Check if already in library
        for source in self.sources:
            if source.file_path == path:
                self.status_bar.showMessage(f"Video already in library: {path.name}")
                return

        # Create new source with metadata and add to project
        source = self._create_source_with_metadata(path)
        self.project.add_source(source)

        # Add to CollectTab grid
        self.collect_tab.add_source(source)

        # Generate thumbnail for the source
        self._generate_source_thumbnail(source)

        # Update chat panel with project state (new source added)
        self._update_chat_project_state()

        self.status_bar.showMessage(f"Added to library: {path.name}")

    def _on_audio_files_added(self, paths: list[Path]):
        """Spawn an AudioImportWorker for each picked audio file."""
        from ui.workers.audio_import_worker import AudioImportWorker

        # Reject duplicates by file path (matches video import behavior)
        existing_paths = {a.file_path for a in self.project.audio_sources}

        for path in paths:
            if path in existing_paths:
                self.status_bar.showMessage(f"Audio already in library: {path.name}")
                continue
            existing_paths.add(path)

            worker = AudioImportWorker(path, parent=self)
            self._active_audio_imports.add(worker)

            worker.audio_ready.connect(self._on_audio_imported)
            worker.error.connect(self._on_audio_import_error)
            worker.finished_signal.connect(
                lambda w=worker: self._active_audio_imports.discard(w)
            )
            worker.start()

    def _on_audio_imported(self, audio):
        """Handle a successful audio import — add to project."""
        self.project.add_audio_source(audio)
        self._update_chat_project_state()
        self.status_bar.showMessage(f"Audio added: {audio.filename}")

    def _on_audio_import_error(self, message: str):
        """Show audio import errors in the status bar."""
        self.status_bar.showMessage(f"Audio import failed: {message}")

    def _on_audio_remove_requested(self, audio_source_id: str):
        """Remove an audio source from the project."""
        removed = self.project.remove_audio_source(audio_source_id)
        if removed is not None:
            self._update_chat_project_state()
            self.status_bar.showMessage(f"Audio removed: {removed.filename}")

    def _on_audio_sources_changed(self, audio_sources):
        """Refresh the Collect tab's audio library when project state changes."""
        self.collect_tab.set_audio_sources(audio_sources)

    def _on_audio_transcribe_requested(self, audio_source_id: str):
        """Run Whisper transcription on the selected audio source."""
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        audio = self.project.get_audio_source(audio_source_id)
        if audio is None:
            self.status_bar.showMessage(f"Audio source not found: {audio_source_id}")
            return
        if audio.transcript:
            self.status_bar.showMessage(f"Already transcribed: {audio.filename}")
            return

        worker = AudioTranscribeWorker(audio, parent=self)
        self._active_audio_transcribes.add(worker)

        worker.transcript_ready.connect(self._on_audio_transcript_ready)
        worker.error.connect(self._on_audio_transcribe_error)
        worker.finished_signal.connect(
            lambda w=worker: self._active_audio_transcribes.discard(w)
        )
        self.status_bar.showMessage(f"Transcribing {audio.filename}…")
        worker.start()

    def _on_audio_transcript_ready(self, audio_source_id: str, segments: list):
        """Persist the transcript on the AudioSource and notify observers."""
        audio = self.project.get_audio_source(audio_source_id)
        if audio is None:
            return
        audio.transcript = segments
        self.project._dirty = True
        self.project._notify_observers(
            "audio_sources_changed", self.project.audio_sources
        )
        self.status_bar.showMessage(
            f"Transcribed {audio.filename}: {len(segments)} segment(s)"
        )
        self._update_chat_project_state()

    def _on_audio_transcribe_error(self, message: str):
        self.status_bar.showMessage(f"Transcription failed: {message}")

    def _generate_source_thumbnail(self, source: Source):
        """Generate a thumbnail for a source video (first frame)."""
        from core.thumbnail import ThumbnailGenerator

        generator = ThumbnailGenerator()
        try:
            thumb_path = generator.generate_first_frame(source.file_path, source.id)
            if thumb_path:
                source.thumbnail_path = thumb_path
                self.collect_tab.update_source_thumbnail(source.id, thumb_path)
        except Exception as e:
            logger.warning(f"Failed to generate source thumbnail: {e}")

    def _on_cut_selection_changed(self, clip_ids: list[str]):
        """Handle selection change in Cut tab."""
        self._gui_state.selected_clip_ids = clip_ids
        self._gui_state.cut_selected_ids = clip_ids
        logger.debug(f"GUI State updated: {len(clip_ids)} clips selected in Cut tab")

    def _on_analyze_selection_changed(self, clip_ids: list[str]):
        """Handle selection change in Analyze tab."""
        self._gui_state.selected_clip_ids = clip_ids
        self._gui_state.analyze_selected_ids = clip_ids
        logger.debug(f"GUI State updated: {len(clip_ids)} clips selected in Analyze tab")

    def _on_cut_filters_changed(self):
        """Handle filter change in Cut tab."""
        filters = self.cut_tab.get_active_filters()
        self._gui_state.update_active_filters(filters)
        logger.debug("GUI State updated: Cut tab filters changed")

    def _on_analyze_filters_changed(self):
        """Handle filter change in Analyze tab."""
        filters = self.analyze_tab.get_active_filters()
        self._gui_state.update_active_filters(filters)
        logger.debug("GUI State updated: Analyze tab filters changed")

    def _on_analyze_clips_changed(self, clip_ids: list[str]):
        """Handle clip collection change in Analyze tab."""
        self._gui_state.analyze_tab_ids = clip_ids
        logger.debug(f"GUI State updated: {len(clip_ids)} clips in Analyze tab")

    def _on_sequence_ids_changed(self):
        """Handle sequence change and update context."""
        sequence = self.sequence_tab.timeline.get_sequence()
        clip_ids = [c.source_clip_id for c in sequence.get_all_clips()]
        self._gui_state.sequence_ids = clip_ids
        logger.debug(f"GUI State updated: {len(clip_ids)} clips in sequence")

    def _on_analyze_selected_from_cut(self, clip_ids: list[str]):
        """Handle clips being sent from Cut tab for analysis - opens picker modal."""
        if not clip_ids:
            return

        clips = [self.project.clips_by_id[cid] for cid in clip_ids if cid in self.project.clips_by_id]
        if not clips:
            return

        dialog = AnalysisPickerDialog(
            len(clips), "selected clips", self.settings, self, clips=clips
        )
        if dialog.exec() == QDialog.Accepted:
            operations = dialog.selected_operations()
            if not operations:
                return
            # Save settings with new selection
            save_settings(self.settings)
            # Add clips to Analyze tab and switch
            self.analyze_tab.add_clips(clip_ids)
            self.tab_widget.setCurrentWidget(self.analyze_tab)
            self._run_analysis_pipeline(clips, operations)

    # ------------------------------------------------------------------
    # Analysis Pipeline: Quick Run / Picker / Phase-Based Engine
    # ------------------------------------------------------------------

    def _on_quick_run_from_tab(self, op_key: str):
        """Handle quick-run dropdown from Analyze tab (single operation, immediate)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, [op_key])

    def _on_analyze_picker_from_tab(self):
        """Handle 'Analyze...' button from Analyze tab - open picker modal."""
        clips = self.analyze_tab.get_clips()
        if not clips:
            return
        dialog = AnalysisPickerDialog(
            len(clips), "clips in Analyze tab", self.settings, self, clips=clips
        )
        if dialog.exec() == QDialog.Accepted:
            operations = dialog.selected_operations()
            if operations:
                save_settings(self.settings)
                self._run_analysis_pipeline(clips, operations)

    def _run_analysis_pipeline(self, clips: list, operations: list[str]):
        """Central entry point for running analysis operations.

        Organizes operations into phases (local → sequential → cloud) and
        executes them with smart concurrency. Called from:
        - Analyze tab "Analyze..." dialog result
        - Analyze tab dropdown "Quick Run"
        - Cut tab "Analyze Selected" dialog result
        - Agent tool analyze_all_live

        Args:
            clips: List of Clip objects to analyze
            operations: List of operation keys to run
        """
        if not clips or not operations:
            return

        # Disabled clips are excluded from analysis the same way they're
        # excluded from sequence/export. They keep any prior analysis fields
        # so the Analyze tab can still surface them as disabled.
        enabled_clips = [c for c in clips if not getattr(c, "disabled", False)]
        skipped = len(clips) - len(enabled_clips)
        if skipped:
            logger.info("Skipping %d disabled clip(s) from analysis pipeline", skipped)
        if not enabled_clips:
            return
        clips = enabled_clips

        # Custom Query needs query text — prompt if not already set (e.g., from agent tool)
        if "custom_query" in operations and not self._custom_query_text:
            query_text, ok = QInputDialog.getText(
                self, "Custom Visual Query",
                "What are you looking for? (e.g., 'blue flower', 'person wearing a hat')"
            )
            if ok and query_text.strip():
                self._custom_query_text = query_text.strip()
            else:
                # User cancelled — remove custom_query from operations
                operations = [op for op in operations if op != "custom_query"]
                if not operations:
                    return

        # Validate operation keys
        valid_ops = [op for op in operations if op in OPERATIONS_BY_KEY]
        valid_ops = self._filter_available_analysis_operations(valid_ops)
        if not valid_ops:
            return

        for op_key in valid_ops:
            self._reset_analysis_run_error(op_key)

        logger.info(f"Starting analysis pipeline: {valid_ops} on {len(clips)} clips")
        self._gui_state.set_processing("analysis", f"{', '.join(valid_ops)} on {len(clips)} clips")

        # Store state
        self._analysis_clips = clips
        self._analysis_selected_ops = valid_ops
        self._analysis_completed_ops = []
        self._analysis_current_phase = ""
        self._analysis_phase_remaining = 0
        self._analysis_sequential_queue = []

        # Build phase queue: only phases that have selected operations
        self._analysis_pending_phases = [
            phase for phase in PHASE_ORDER
            if any(OPERATIONS_BY_KEY[op].phase == phase for op in valid_ops)
        ]

        # Update UI state
        self.analyze_tab.set_analyzing(True, "pipeline")

        # Start first phase
        self._start_next_analysis_phase()

    def _start_next_analysis_phase(self):
        """Start the next phase in the analysis pipeline."""
        if not self._analysis_pending_phases:
            # All phases done
            self._on_analysis_pipeline_complete()
            return

        phase = self._analysis_pending_phases.pop(0)
        self._analysis_current_phase = phase

        # Get operations for this phase
        phase_ops = [
            op for op in self._analysis_selected_ops
            if OPERATIONS_BY_KEY[op].phase == phase
        ]

        if not phase_ops:
            # No ops in this phase, skip
            self._start_next_analysis_phase()
            return

        clips = self._analysis_clips
        logger.info(f"Starting analysis phase '{phase}': {phase_ops} on {len(clips)} clips")

        if phase == "local":
            # Local ops run concurrently
            self._analysis_phase_remaining = len(phase_ops)
            for op_key in phase_ops:
                self._launch_analysis_worker(op_key, clips)
        elif phase == "sequential":
            # Sequential ops run one at a time (e.g., transcription is memory-heavy)
            self._analysis_phase_remaining = len(phase_ops)
            # Store ordered queue; pop first and launch it
            self._analysis_sequential_queue = list(phase_ops[1:])
            self._launch_analysis_worker(phase_ops[0], clips)
        elif phase == "cloud":
            # Cloud ops run concurrently (I/O-bound API calls)
            self._analysis_phase_remaining = len(phase_ops)
            for op_key in phase_ops:
                self._launch_analysis_worker(op_key, clips)

    def _launch_analysis_worker(self, op_key: str, clips: list):
        """Launch a worker for a specific analysis operation.

        Args:
            op_key: Operation key (e.g., "colors", "shots")
            clips: List of clips to process
        """
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        op = OPERATIONS_BY_KEY[op_key]
        self.status_bar.showMessage(f"{op.label}: processing {len(clips)} clips...")

        if op_key == "colors":
            self._launch_colors_worker(clips)
        elif op_key == "shots":
            self._launch_shots_worker(clips)
        elif op_key == "classify":
            self._launch_classification_worker(clips)
        elif op_key == "detect_objects":
            self._launch_object_detection_worker(clips)
        elif op_key == "face_embeddings":
            self._launch_face_detection_worker(clips)
        elif op_key == "extract_text":
            self._launch_text_extraction_worker(clips)
        elif op_key == "transcribe":
            self._launch_transcription_worker(clips)
        elif op_key == "describe":
            self._launch_description_worker(clips)
        elif op_key == "cinematography":
            self._launch_cinematography_worker(clips)
        elif op_key == "custom_query":
            self._launch_custom_query_worker(clips)
        elif op_key == "gaze":
            self._launch_gaze_worker(clips)
        elif op_key == "embeddings":
            self._launch_embeddings_worker(clips)
        else:
            logger.warning(f"Unknown analysis operation: {op_key}")
            self._on_analysis_phase_worker_finished(op_key)

    def _launch_colors_worker(self, clips: list):
        """Launch color analysis worker."""
        self._color_analysis_finished_handled = False
        self._reset_analysis_run_error("colors")
        logger.info(f"Creating ColorAnalysisWorker (pipeline) for {len(clips)} clips...")
        self.color_worker = ColorAnalysisWorker(clips, parallelism=self.settings.color_analysis_parallelism, sources_by_id=self.project.sources_by_id)
        self.color_worker.progress.connect(self._on_color_progress)
        self.color_worker.color_ready.connect(self._on_color_ready)
        self.color_worker.error.connect(self._on_color_error)
        self.color_worker.analysis_completed.connect(
            self._on_pipeline_colors_finished, Qt.UniqueConnection
        )
        self.color_worker.finished.connect(self.color_worker.deleteLater)
        self.color_worker.finished.connect(lambda: setattr(self, 'color_worker', None))
        self.color_worker.start()

    def _launch_shots_worker(self, clips: list):
        """Launch shot type classification worker."""
        self._shot_type_finished_handled = False
        self._shot_type_run_error = None
        logger.info(f"Creating ShotTypeWorker (pipeline) for {len(clips)} clips...")
        self.shot_type_worker = ShotTypeWorker(clips, self.project.sources_by_id, parallelism=self.settings.local_model_parallelism)
        self.shot_type_worker.progress.connect(self._on_shot_type_progress)
        self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
        self.shot_type_worker.error.connect(self._on_shot_type_error)
        self.shot_type_worker.analysis_completed.connect(
            self._on_pipeline_shots_finished, Qt.UniqueConnection
        )
        self.shot_type_worker.finished.connect(self.shot_type_worker.deleteLater)
        self.shot_type_worker.finished.connect(lambda: setattr(self, 'shot_type_worker', None))
        self.shot_type_worker.start()

    def _launch_classification_worker(self, clips: list):
        """Launch content classification worker."""
        self._classification_finished_handled = False
        self._reset_analysis_run_error("classify")
        logger.info(f"Creating ClassificationWorker (pipeline) for {len(clips)} clips...")
        self.classification_worker = ClassificationWorker(clips, parallelism=self.settings.local_model_parallelism)
        self.classification_worker.progress.connect(self._on_classification_progress)
        self.classification_worker.labels_ready.connect(self._on_classification_ready)
        self.classification_worker.error.connect(self._on_classification_error)
        self.classification_worker.classification_completed.connect(
            self._on_pipeline_classify_finished, Qt.UniqueConnection
        )
        self.classification_worker.finished.connect(self.classification_worker.deleteLater)
        self.classification_worker.finished.connect(lambda: setattr(self, 'classification_worker', None))
        self.classification_worker.start()

    def _launch_object_detection_worker(self, clips: list):
        """Launch object detection worker."""
        self._object_detection_finished_handled = False
        self._reset_analysis_run_error("detect_objects")
        logger.info(f"Creating ObjectDetectionWorker (pipeline) for {len(clips)} clips...")
        self.detection_worker_yolo = ObjectDetectionWorker(clips, parallelism=self.settings.local_model_parallelism)
        self.detection_worker_yolo.progress.connect(self._on_object_detection_progress)
        self.detection_worker_yolo.objects_ready.connect(self._on_objects_ready)
        self.detection_worker_yolo.error.connect(self._on_object_detection_error)
        self.detection_worker_yolo.detection_completed.connect(
            self._on_pipeline_detect_objects_finished, Qt.UniqueConnection
        )
        self.detection_worker_yolo.finished.connect(self.detection_worker_yolo.deleteLater)
        self.detection_worker_yolo.finished.connect(lambda: setattr(self, 'detection_worker_yolo', None))
        self.detection_worker_yolo.start()

    def _launch_face_detection_worker(self, clips: list):
        """Launch face detection worker."""
        self._face_detection_finished_handled = False
        sources_by_id = {s.id: s for s in self.sources}
        logger.info(f"Creating FaceDetectionWorker (pipeline) for {len(clips)} clips...")
        self.face_detection_worker = FaceDetectionWorker(
            clips, sources_by_id=sources_by_id,
        )
        self.face_detection_worker.progress.connect(self._on_face_detection_progress)
        self.face_detection_worker.detection_completed.connect(
            self._on_pipeline_face_detection_finished, Qt.UniqueConnection
        )
        self.face_detection_worker.finished.connect(self.face_detection_worker.deleteLater)
        self.face_detection_worker.finished.connect(lambda: setattr(self, 'face_detection_worker', None))
        self.face_detection_worker.start()

    def _launch_gaze_worker(self, clips: list):
        """Launch gaze direction analysis worker."""
        self._gaze_finished_handled = False
        sources_by_id = {s.id: s for s in self.sources}
        logger.info(f"Creating GazeAnalysisWorker (pipeline) for {len(clips)} clips...")
        self._gaze_worker = GazeAnalysisWorker(
            clips, sources_by_id=sources_by_id,
        )
        self._gaze_worker.progress.connect(self._on_gaze_progress)
        self._gaze_worker.gaze_ready.connect(self._on_gaze_ready)
        self._gaze_worker.detection_completed.connect(
            self._on_pipeline_gaze_finished, Qt.UniqueConnection
        )
        self._gaze_worker.error.connect(self._on_gaze_error)
        self._gaze_worker.finished.connect(self._gaze_worker.deleteLater)
        self._gaze_worker.finished.connect(lambda: setattr(self, '_gaze_worker', None))
        self._gaze_worker.start()

    def _launch_embeddings_worker(self, clips: list):
        """Launch DINOv2 embedding extraction worker.

        The pipeline has already filtered this operation via
        _filter_available_analysis_operations (which calls check_feature_ready
        for torch+transformers), so no additional dependency guard is needed
        here — if the user declined install, this launcher is never reached.
        """
        self._embeddings_finished_handled = False
        logger.info(f"Creating EmbeddingAnalysisWorker (pipeline) for {len(clips)} clips...")
        self._embeddings_worker = EmbeddingAnalysisWorker(clips)
        self._embeddings_worker.progress.connect(self._on_embeddings_progress)
        self._embeddings_worker.embedding_ready.connect(self._on_embedding_ready)
        self._embeddings_worker.analysis_completed.connect(
            self._on_pipeline_embeddings_finished, Qt.UniqueConnection
        )
        self._embeddings_worker.error.connect(self._on_embeddings_error)
        self._embeddings_worker.finished.connect(self._embeddings_worker.deleteLater)
        self._embeddings_worker.finished.connect(lambda: setattr(self, '_embeddings_worker', None))
        self._embeddings_worker.start()

    def _launch_text_extraction_worker(self, clips: list):
        """Launch text extraction worker."""
        self._text_extraction_finished_handled = False
        self._reset_analysis_run_error("extract_text")
        sources_by_id = {s.id: s for s in self.sources}

        # Filter to clips needing extraction
        clips_to_process = [c for c in clips if not c.extracted_texts]
        if not clips_to_process:
            logger.info("All clips already have extracted text, skipping")
            self._on_analysis_phase_worker_finished("extract_text")
            return

        from ui.workers.text_extraction_worker import TextExtractionWorker

        method = self.settings.text_extraction_method
        vlm_only = (method == "vlm")
        use_vlm = (method in ("vlm", "hybrid"))
        vlm_model = self.settings.text_extraction_vlm_model if use_vlm else None

        logger.info(f"Creating TextExtractionWorker (pipeline) for {len(clips_to_process)} clips")
        self.text_extraction_worker = TextExtractionWorker(
            clips=clips_to_process,
            sources_by_id=sources_by_id,
            num_keyframes=3,
            use_vlm_fallback=use_vlm,
            vlm_model=vlm_model,
            vlm_only=vlm_only,
        )
        self.text_extraction_worker.progress.connect(self._on_text_extraction_progress)
        self.text_extraction_worker.clip_completed.connect(self._on_text_extraction_clip_ready)
        self.text_extraction_worker.finished.connect(
            self._on_pipeline_extract_text_finished
        )
        self.text_extraction_worker.error.connect(self._on_text_extraction_error)
        self.text_extraction_worker.finished.connect(self.text_extraction_worker.deleteLater)
        self.text_extraction_worker.finished.connect(
            lambda results: setattr(self, 'text_extraction_worker', None)
        )
        self.text_extraction_worker.start()

    def _launch_transcription_worker(self, clips: list):
        """Launch transcription worker (handles multi-source sequentially)."""
        self._transcription_finished_handled = False

        # Pipeline entry points should prompt before launch, but keep a
        # non-interactive guard here so internal calls fail safely.
        from core.feature_registry import check_feature_ready

        feature_candidates = get_operation_feature_candidates("transcribe", self.settings)
        if feature_candidates and not any(check_feature_ready(name)[0] for name in feature_candidates):
            logger.warning("Transcription skipped: dependencies unavailable")
            self.status_bar.showMessage(
                "Transcription unavailable - install dependencies in Settings > Dependencies"
            )
            self._on_analysis_phase_worker_finished("transcribe")
            return

        # Group clips by source_id for multi-source transcription
        clips_by_source: dict = {}
        for clip in clips:
            if clip.source_id not in clips_by_source:
                clips_by_source[clip.source_id] = []
            clips_by_source[clip.source_id].append(clip)

        logger.info(f"Transcription: {len(clips)} clips from {len(clips_by_source)} sources")

        # Queue transcription for each source
        self._transcription_source_queue = list(clips_by_source.items())
        self._start_next_source_transcription_pipeline()

    def _start_next_source_transcription_pipeline(self):
        """Start transcription for the next source in the pipeline queue."""
        if not self._transcription_source_queue:
            logger.info("All source transcriptions complete")
            self._on_analysis_phase_worker_finished("transcribe")
            return

        source_id, clips = self._transcription_source_queue.pop(0)
        source = self.sources_by_id.get(source_id)

        if not source:
            logger.warning(f"Source {source_id} not found, skipping {len(clips)} clips")
            self._start_next_source_transcription_pipeline()
            return

        self._transcription_finished_handled = False

        remaining = len(self._transcription_source_queue)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(
            f"Transcribing {len(clips)} clips ({remaining + 1} sources remaining)..."
        )

        self._stop_worker_safely(self.transcription_worker, "Transcription")

        logger.info(f"Creating TranscriptionWorker for source {source_id} ({len(clips)} clips)")
        self.transcription_worker = TranscriptionWorker(
            clips,
            source,
            self.settings.transcription_model,
            self.settings.transcription_language,
            parallelism=self.settings.transcription_parallelism,
            backend=self.settings.transcription_backend,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
        self.transcription_worker.transcription_completed.connect(
            self._on_pipeline_source_transcription_finished, Qt.UniqueConnection
        )
        self.transcription_worker.error.connect(self._on_transcription_error)
        self.transcription_worker.finished.connect(self.transcription_worker.deleteLater)
        self.transcription_worker.finished.connect(lambda: setattr(self, 'transcription_worker', None))
        self.transcription_worker.start()

    @Slot()
    def _on_pipeline_source_transcription_finished(self):
        """Handle transcription completion for one source in pipeline flow."""
        logger.info("=== PIPELINE: SOURCE TRANSCRIPTION FINISHED ===")
        if self._transcription_finished_handled:
            return
        self._transcription_finished_handled = True
        self._start_next_source_transcription_pipeline()

    def _launch_description_worker(self, clips: list):
        """Launch description worker."""
        self._description_finished_handled = False
        self._reset_description_run_errors()
        tier = self.settings.description_model_tier
        sources = self.project.sources_by_id

        logger.info(f"Creating DescriptionWorker (pipeline) with tier={tier}...")
        self.description_worker = DescriptionWorker(
            clips, tier=tier, sources=sources,
            parallelism=self.settings.description_parallelism,
        )
        self.description_worker.progress.connect(self._on_description_progress)
        self.description_worker.description_ready.connect(self._on_description_ready)
        self.description_worker.error.connect(self._on_description_error)
        self.description_worker.description_completed.connect(
            self._on_pipeline_describe_finished, Qt.UniqueConnection
        )
        self.description_worker.finished.connect(self.description_worker.deleteLater)
        self.description_worker.finished.connect(lambda: setattr(self, 'description_worker', None))
        self.description_worker.start()

    def _launch_custom_query_worker(self, clips: list):
        """Launch custom query worker."""
        query = self._custom_query_text
        if not query:
            logger.warning("Custom query requested but no query text set")
            self._on_analysis_phase_worker_finished("custom_query")
            return

        # Clear immediately to prevent stale reuse
        self._custom_query_text = None
        self._active_custom_query_text = query

        self._custom_query_finished_handled = False
        tier = self.settings.description_model_tier
        sources = self.project.sources_by_id
        parallelism = 3 if tier == "cloud" else 1

        logger.info(f"Creating CustomQueryWorker for '{query}' on {len(clips)} clips, tier={tier}")
        self.custom_query_worker = CustomQueryWorker(
            clips, query=query, sources_by_id=sources,
            tier=tier, parallelism=parallelism,
        )
        self.custom_query_worker.progress.connect(self._on_custom_query_progress)
        self.custom_query_worker.query_result_ready.connect(self._on_custom_query_ready)
        self.custom_query_worker.error.connect(self._on_custom_query_error)
        self.custom_query_worker.analysis_completed.connect(
            self._on_pipeline_custom_query_finished, Qt.UniqueConnection
        )
        self.custom_query_worker.finished.connect(self.custom_query_worker.deleteLater)
        self.custom_query_worker.finished.connect(lambda: setattr(self, 'custom_query_worker', None))
        self.custom_query_worker.start()

    @Slot(int, int)
    def _on_custom_query_progress(self, current: int, total: int):
        """Handle custom query progress updates."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.status_bar.showMessage(f"Custom query: {current}/{total} clips...")

    @Slot(str, str, bool, float, str)
    def _on_custom_query_ready(self, clip_id: str, query: str, match: bool, confidence: float, model: str):
        """Handle custom query result for a single clip."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            if clip.custom_queries is None:
                clip.custom_queries = []
            clip.custom_queries.append({
                "query": query,
                "match": match,
                "confidence": round(confidence, 4),
                "model": model,
            })
            if hasattr(self, "analyze_tab"):
                self.analyze_tab.update_clip_custom_queries(clip_id, clip.custom_queries)
            if hasattr(self, "cut_tab") and hasattr(self.cut_tab, "clip_browser"):
                self.cut_tab.clip_browser.update_clip_custom_queries(clip_id, clip.custom_queries)
            if hasattr(self, "clip_details_sidebar"):
                self.clip_details_sidebar.refresh_custom_queries_if_showing(
                    clip_id,
                    clip.custom_queries,
                )
            self._mark_dirty()
            logger.debug(f"Custom query '{query}' for {clip_id}: match={match} ({confidence:.0%})")

    @Slot(str)
    def _on_custom_query_error(self, error_msg: str):
        """Handle custom query error."""
        logger.error(f"Custom query error: {error_msg}")

    def _launch_cinematography_worker(self, clips: list):
        """Launch cinematography analysis worker."""
        self._cinematography_finished_handled = False
        self._reset_analysis_run_error("cinematography")
        sources_by_id = {s.id: s for s in self.sources}
        mode = self.settings.cinematography_input_mode
        model = self.settings.cinematography_model
        parallelism = self.settings.cinematography_batch_parallelism

        logger.info(f"Creating CinematographyWorker (pipeline) for {len(clips)} clips")
        self.cinematography_worker = CinematographyWorker(
            clips=clips,
            sources_by_id=sources_by_id,
            mode=mode,
            model=model,
            parallelism=parallelism,
            skip_existing=True,
        )
        self.cinematography_worker.progress.connect(self._on_cinematography_progress)
        self.cinematography_worker.clip_completed.connect(self._on_cinematography_clip_ready)
        self.cinematography_worker.finished.connect(
            self._on_pipeline_cinematography_finished
        )
        self.cinematography_worker.error.connect(self._on_cinematography_error)
        self.cinematography_worker.finished.connect(self.cinematography_worker.deleteLater)
        self.cinematography_worker.finished.connect(
            lambda results: setattr(self, 'cinematography_worker', None)
        )
        self.cinematography_worker.start()

    # Named slots for pipeline phase completion (Qt.UniqueConnection requires
    # pointer-to-member, not lambdas — lambdas silently fail to connect).
    @Slot()
    def _on_pipeline_colors_finished(self):
        self._on_analysis_phase_worker_finished("colors")

    @Slot()
    def _on_pipeline_shots_finished(self):
        self._on_analysis_phase_worker_finished("shots")

    @Slot()
    def _on_pipeline_classify_finished(self):
        self._on_analysis_phase_worker_finished("classify")

    @Slot()
    def _on_pipeline_detect_objects_finished(self):
        self._on_analysis_phase_worker_finished("detect_objects")

    @Slot()
    def _on_pipeline_face_detection_finished(self):
        from core.analysis.faces import unload_model
        unload_model()
        self._on_analysis_phase_worker_finished("face_embeddings")

    @Slot()
    def _on_pipeline_gaze_finished(self):
        if self._gaze_finished_handled:
            return
        self._gaze_finished_handled = True
        self._on_analysis_phase_worker_finished("gaze")

    @Slot(str)
    def _on_gaze_error(self, msg):
        """Handle gaze analysis errors — log and ensure pipeline advances."""
        logger.error("Gaze analysis error: %s", msg)
        self.statusBar().showMessage(f"Gaze analysis failed: {msg}", 5000)

    @Slot()
    def _on_pipeline_extract_text_finished(self):
        self._on_analysis_phase_worker_finished("extract_text")

    @Slot()
    def _on_pipeline_describe_finished(self):
        self._on_analysis_phase_worker_finished("describe")

    @Slot()
    def _on_pipeline_cinematography_finished(self):
        self._on_analysis_phase_worker_finished("cinematography")

    @Slot()
    def _on_pipeline_custom_query_finished(self):
        self._on_analysis_phase_worker_finished("custom_query")

    def _build_custom_query_agent_summary(
        self,
        clips: list,
        query: str | None,
    ) -> dict | None:
        """Build a structured summary for the chat agent after custom query analysis."""
        if not query:
            return None

        matches = []
        non_matches = []
        missing_result_ids = []

        for clip in clips:
            latest_result = None
            for query_result in reversed(getattr(clip, "custom_queries", None) or []):
                if str(query_result.get("query") or "").strip() == query:
                    latest_result = query_result
                    break

            if latest_result is None:
                missing_result_ids.append(clip.id)
                continue

            source = self.project.sources_by_id.get(clip.source_id)
            row = {
                "clip_id": clip.id,
                "source_name": source.filename if source else None,
                "match": bool(latest_result.get("match")),
                "confidence": latest_result.get("confidence"),
                "model": latest_result.get("model"),
                "description": getattr(clip, "description", None),
            }
            if row["match"]:
                matches.append(row)
            else:
                non_matches.append(row)

        matches.sort(
            key=lambda row: (
                row["confidence"]
                if isinstance(row.get("confidence"), (int, float))
                else -1.0
            ),
            reverse=True,
        )

        return {
            "query": query,
            "checked_count": len(clips),
            "matched_count": len(matches),
            "non_match_count": len(non_matches),
            "missing_result_count": len(missing_result_ids),
            "missing_result_ids": missing_result_ids[:20],
            "matches": matches[:20],
            "response_guidance": (
                "Summarize these actual VLM custom visual query results. "
                "Do not describe this as a metadata search or sorting algorithm. "
                "Format each matched clip as one bullet containing clip_id, "
                "confidence, and description/source if present; do not split a "
                "single match across separate numbered-list items."
            ),
        }

    def _build_agent_clip_context(self, clip) -> dict:
        source = self.project.sources_by_id.get(clip.source_id)
        return {
            "clip_id": clip.id,
            "source_name": source.filename if source else None,
            "description": getattr(clip, "description", None),
        }

    def _build_agent_detected_clip_summary(self, clips: list) -> list[dict]:
        rows = []
        for clip in clips[:20]:
            source = self.project.sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0
            rows.append(
                {
                    "clip_id": clip.id,
                    "source_name": source.filename if source else None,
                    "start_seconds": round(clip.start_time(fps), 3),
                    "end_seconds": round(clip.end_time(fps), 3),
                    "duration_seconds": round(clip.duration_seconds(fps), 3),
                }
            )
        return rows

    @staticmethod
    def _rgb_to_hex(color) -> str | None:
        try:
            r, g, b = color
            return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _truncate_for_agent(value: str | None, limit: int = 500) -> str | None:
        if not value:
            return None
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _summarize_detections_for_agent(detections: list[dict] | None) -> list[dict]:
        rows = []
        for detection in detections or []:
            row = {
                "label": detection.get("label") or detection.get("class") or detection.get("name"),
            }
            confidence = detection.get("confidence")
            if isinstance(confidence, (int, float)):
                row["confidence"] = round(float(confidence), 4)
            rows.append(row)
        return rows[:10]

    def _build_agent_analysis_summary(
        self,
        clips: list,
        completed_ops: list[str],
    ) -> dict:
        """Build operation-specific analysis results for chat-agent summaries."""
        summaries: dict[str, dict] = {}

        if "colors" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.dominant_colors:
                    continue
                row = self._build_agent_clip_context(clip)
                row["dominant_colors_rgb"] = [list(color) for color in clip.dominant_colors[:5]]
                row["dominant_colors_hex"] = [
                    hex_color
                    for hex_color in (
                        self._rgb_to_hex(color) for color in clip.dominant_colors[:5]
                    )
                    if hex_color is not None
                ]
                per_clip.append(row)
            summaries["colors"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "shots" in completed_ops:
            distribution = {}
            per_clip = []
            for clip in clips:
                if not clip.shot_type:
                    continue
                distribution[clip.shot_type] = distribution.get(clip.shot_type, 0) + 1
                row = self._build_agent_clip_context(clip)
                row["shot_type"] = clip.shot_type
                per_clip.append(row)
            summaries["shots"] = {
                "analyzed_count": len(per_clip),
                "distribution": distribution,
                "clips": per_clip[:20],
            }

        if "classify" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.object_labels:
                    continue
                row = self._build_agent_clip_context(clip)
                row["labels"] = list(clip.object_labels[:10])
                per_clip.append(row)
            summaries["classify"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "detect_objects" in completed_ops:
            per_clip = []
            total_people = 0
            for clip in clips:
                if clip.detected_objects is None and clip.person_count is None:
                    continue
                total_people += clip.person_count or 0
                row = self._build_agent_clip_context(clip)
                row["person_count"] = clip.person_count or 0
                row["objects"] = self._summarize_detections_for_agent(clip.detected_objects)
                per_clip.append(row)
            summaries["detect_objects"] = {
                "analyzed_count": len(per_clip),
                "total_people": total_people,
                "clips": per_clip[:20],
            }

        if "face_embeddings" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.face_embeddings:
                    continue
                row = self._build_agent_clip_context(clip)
                row["face_count"] = len(clip.face_embeddings)
                confidences = [
                    entry.get("confidence")
                    for entry in clip.face_embeddings
                    if isinstance(entry.get("confidence"), (int, float))
                ]
                if confidences:
                    row["max_confidence"] = round(max(confidences), 4)
                per_clip.append(row)
            summaries["face_embeddings"] = {
                "analyzed_count": len(per_clip),
                "total_faces": sum(row["face_count"] for row in per_clip),
                "clips": per_clip[:20],
            }

        if "extract_text" in completed_ops:
            per_clip = []
            for clip in clips:
                text = clip.combined_text
                if not text:
                    continue
                row = self._build_agent_clip_context(clip)
                row["text"] = self._truncate_for_agent(text)
                row["text_result_count"] = len(clip.extracted_texts or [])
                per_clip.append(row)
            summaries["extract_text"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "transcribe" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.transcript:
                    continue
                row = self._build_agent_clip_context(clip)
                row["transcript_excerpt"] = self._truncate_for_agent(clip.get_transcript_text())
                row["segment_count"] = len(clip.transcript)
                per_clip.append(row)
            summaries["transcribe"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "describe" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.description:
                    continue
                row = self._build_agent_clip_context(clip)
                row["model"] = clip.description_model
                per_clip.append(row)
            summaries["describe"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "cinematography" in completed_ops:
            per_clip = []
            for clip in clips:
                if not clip.cinematography:
                    continue
                data = clip.cinematography.to_dict()
                row = self._build_agent_clip_context(clip)
                row["cinematography"] = {
                    key: data.get(key)
                    for key in (
                        "shot_size",
                        "shot_size_confidence",
                        "camera_angle",
                        "camera_movement",
                        "subject_position",
                        "lighting_style",
                        "emotional_intensity",
                        "suggested_pacing",
                        "analysis_model",
                    )
                    if data.get(key) is not None
                }
                per_clip.append(row)
            summaries["cinematography"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if "custom_query" in completed_ops:
            custom_query_summary = self._build_custom_query_agent_summary(
                clips,
                self._active_custom_query_text,
            )
            if custom_query_summary is not None:
                summaries["custom_query"] = custom_query_summary

        if "gaze" in completed_ops:
            distribution = {}
            per_clip = []
            for clip in clips:
                if not clip.gaze_category:
                    continue
                distribution[clip.gaze_category] = distribution.get(clip.gaze_category, 0) + 1
                row = self._build_agent_clip_context(clip)
                row["gaze_category"] = clip.gaze_category
                row["gaze_yaw"] = clip.gaze_yaw
                row["gaze_pitch"] = clip.gaze_pitch
                per_clip.append(row)
            summaries["gaze"] = {
                "analyzed_count": len(per_clip),
                "distribution": distribution,
                "clips": per_clip[:20],
            }

        if "embeddings" in completed_ops:
            per_clip = []
            for clip in clips:
                if clip.embedding is None:
                    continue
                row = self._build_agent_clip_context(clip)
                row["embedding_model"] = clip.embedding_model
                row["embedding_dimensions"] = len(clip.embedding)
                row["has_boundary_embeddings"] = (
                    clip.first_frame_embedding is not None
                    and clip.last_frame_embedding is not None
                )
                per_clip.append(row)
            summaries["embeddings"] = {
                "analyzed_count": len(per_clip),
                "clips": per_clip[:20],
            }

        if summaries:
            summaries["response_guidance"] = (
                "Summarize only facts present in these structured analysis results. "
                "Do not invent clip descriptions, labels, counts, or matches that are "
                "not present in the tool output. Group each clip's result together."
            )

        return summaries

    def _build_agent_analysis_result(
        self,
        clips: list,
        completed_ops: list[str],
        message: str,
        extra: dict | None = None,
    ) -> dict:
        """Build the full agent-facing payload for analysis completion."""
        result = {
            "success": True,
            "clip_count": len(clips),
            "clip_ids": [clip.id for clip in clips],
            "operations_completed": completed_ops,
            "message": message,
        }
        if extra:
            result.update(extra)

        analysis_results = self._build_agent_analysis_summary(clips, completed_ops)
        if analysis_results:
            result["analysis_results"] = analysis_results
            if "custom_query" in analysis_results:
                result["custom_visual_query"] = analysis_results["custom_query"]
        return result

    def _on_analysis_phase_worker_finished(self, op_key: str):
        """Handle completion of one worker in the analysis pipeline.

        Decrements phase counter and advances to next phase when all workers
        in the current phase are done. For sequential phases, launches the
        next queued operation before checking phase completion.
        """
        self._analysis_completed_ops.append(op_key)
        self._analysis_phase_remaining -= 1
        logger.info(
            f"=== PIPELINE: {op_key} finished "
            f"({self._analysis_phase_remaining} remaining in phase '{self._analysis_current_phase}') ==="
        )

        # For sequential phase, launch the next queued op if any remain
        if (
            self._analysis_current_phase == "sequential"
            and self._analysis_sequential_queue
        ):
            next_op = self._analysis_sequential_queue.pop(0)
            logger.info(f"PIPELINE: launching next sequential op '{next_op}'")
            self._launch_analysis_worker(next_op, self._analysis_clips)
            return

        if self._analysis_phase_remaining <= 0:
            self._start_next_analysis_phase()

    def _on_analysis_pipeline_complete(self):
        """Handle completion of the entire analysis pipeline."""
        clips = self._analysis_clips
        completed = self._analysis_completed_ops
        clip_count = len(clips)

        logger.info(f"Analysis pipeline complete: {completed} on {clip_count} clips")
        self._gui_state.clear_processing("analysis")

        self.analyze_tab.set_analyzing(False)
        self.progress_bar.setVisible(False)
        error_labels = self._get_completed_analysis_error_labels(completed)
        if error_labels:
            self.status_bar.showMessage(
                f"Analysis finished with errors ({', '.join(error_labels)}) - {clip_count} clips ({', '.join(completed)})"
            )
            self._show_completed_analysis_error_dialog(completed)
        else:
            self.status_bar.showMessage(
                f"Analysis complete - {clip_count} clips ({', '.join(completed)})"
            )

        # Mark affected sources as having analysis data
        analyzed_source_ids = set()
        for clip in clips:
            if clip.source_id:
                analyzed_source_ids.add(clip.source_id)
        for source_id in analyzed_source_ids:
            source = self.project.sources_by_id.get(source_id)
            if source:
                source.has_analysis = True
            self.collect_tab.update_source_has_analysis(source_id, True)

        # Save project
        if self.project.path:
            self.project.save()

        # Update chat panel
        self._update_chat_project_state()

        # If agent was waiting for analyze_all, send result back
        if self._pending_agent_analyze_all and self._chat_worker:
            self._pending_agent_analyze_all = False

            shot_types = {}
            transcribed_count = 0
            for clip in clips:
                if clip.shot_type:
                    shot_types[clip.shot_type] = shot_types.get(clip.shot_type, 0) + 1
                if clip.transcript:
                    transcribed_count += 1

            agent_result = self._build_agent_analysis_result(
                clips,
                completed,
                f"Analyzed {clip_count} clips ({', '.join(completed)})",
                {
                    "shot_type_summary": shot_types,
                    "transcribed_count": transcribed_count,
                },
            )

            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": agent_result,
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent analysis result to agent: {clip_count} clips")

        if "custom_query" in completed:
            self._active_custom_query_text = None

        self._analysis_clips = []
        self._analysis_selected_ops = []

    # ------------------------------------------------------------------
    # Individual analysis handlers (kept for manual standalone + backward compat)
    # ------------------------------------------------------------------

    def _on_analyze_colors_from_tab(self):
        """Handle color extraction request from Analyze tab (standalone)."""
        clips = self.analyze_tab.get_clips()
        if not clips:
            return
        self._run_analysis_pipeline(clips, ["colors"])

    def _on_analyze_shots_from_tab(self):
        """Handle shot type classification request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["shots"])

    def _on_transcribe_from_tab(self):
        """Handle transcription request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["transcribe"])

    def _on_transcription_error(self, error: str):
        """Handle transcription error."""
        logger.error(f"Transcription error: {error}")
        if error not in self._transcription_run_errors:
            self._transcription_run_errors.append(error)
        self._transcription_run_error = self._summarize_transcription_errors()
        self._gui_state.set_last_error(f"Transcription error: {error}")
        self.status_bar.showMessage("Transcription finished with errors", 5000)

    @Slot(str)
    def _on_shot_type_error(self, error_msg: str):
        """Handle shot type classification errors."""
        logger.warning(f"Shot type classification error: {error_msg}")
        self._shot_type_run_error = error_msg
        self._gui_state.set_last_error(
            f"Shot type classification error: {error_msg}"
        )
        self.status_bar.showMessage(
            "Shot type classification finished with errors",
            5000,
        )

    def _on_classify_from_tab(self):
        """Handle classification request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["classify"])

    def _on_detect_objects_from_tab(self):
        """Handle object detection request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["detect_objects"])

    def _on_describe_from_tab(self):
        """Handle description request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["describe"])

    @Slot(str, str)
    def _on_description_error(self, clip_id: str, error_msg: str):
        """Handle description error for a single clip."""
        logger.warning(f"Description error for clip {clip_id}: {error_msg}")
        summary_line = f"{clip_id}: {error_msg}"
        self._description_run_errors.append(summary_line)
        self._description_run_error = self._summarize_description_errors()
        self._gui_state.set_last_error(f"Description error: {summary_line}")
        self.status_bar.showMessage(
            "Description generation finished with errors",
            5000,
        )

    @Slot(str)
    def _on_color_error(self, error_msg: str):
        """Handle aggregated color extraction errors."""
        logger.warning(f"Color extraction error: {error_msg}")
        self._record_analysis_run_error(
            "_color_run_error",
            "Color extraction",
            error_msg,
        )

    @Slot(str)
    def _on_classification_error(self, error_msg: str):
        """Handle aggregated content classification errors."""
        logger.warning(f"Content classification error: {error_msg}")
        self._record_analysis_run_error(
            "_classification_run_error",
            "Content classification",
            error_msg,
        )

    @Slot(str)
    def _on_object_detection_error(self, error_msg: str):
        """Handle aggregated object detection errors."""
        logger.warning(f"Object detection error: {error_msg}")
        self._record_analysis_run_error(
            "_object_detection_run_error",
            "Object detection",
            error_msg,
        )

    def _reset_description_run_errors(self) -> None:
        """Clear accumulated description errors for a new run."""
        self._description_run_error = None
        self._description_run_errors = []

    def _reset_transcription_run_errors(self) -> None:
        """Clear accumulated transcription errors for a new run."""
        self._transcription_run_error = None
        self._transcription_run_errors = []

    def _summarize_description_errors(self) -> Optional[str]:
        """Return a compact summary of accumulated description errors."""
        if not self._description_run_errors:
            return None
        return summarize_messages(
            self._description_run_errors,
            header=f"Description failed for {len(self._description_run_errors)} clips:",
        )

    def _summarize_transcription_errors(self) -> Optional[str]:
        """Return a compact summary of accumulated transcription errors."""
        if not self._transcription_run_errors:
            return None
        return summarize_messages(
            self._transcription_run_errors,
            header=(
                f"Transcription failed in {len(self._transcription_run_errors)} "
                "batches:"
            ),
        )

    def _reset_analysis_run_error(self, op_key: str) -> None:
        """Clear the stored error summary for an analysis operation."""
        attr_map = {
            "colors": "_color_run_error",
            "classify": "_classification_run_error",
            "detect_objects": "_object_detection_run_error",
            "extract_text": "_text_extraction_run_error",
            "cinematography": "_cinematography_run_error",
        }
        attr_name = attr_map.get(op_key)
        if attr_name:
            setattr(self, attr_name, None)
        elif op_key == "describe":
            self._reset_description_run_errors()
        elif op_key == "transcribe":
            self._reset_transcription_run_errors()
        elif op_key == "shots":
            self._shot_type_run_error = None

    def _record_analysis_run_error(
        self,
        attr_name: str,
        ui_label: str,
        error_msg: str,
    ) -> None:
        """Persist an analysis error for the end-of-run summary dialog."""
        setattr(self, attr_name, error_msg)
        self._gui_state.set_last_error(f"{ui_label} error: {error_msg}")
        self.status_bar.showMessage(f"{ui_label} finished with errors", 5000)

    def _get_completed_analysis_error_labels(self, completed_ops: list[str]) -> list[str]:
        """Return user-facing labels for analysis operations that finished with errors."""
        return [
            label
            for _, _, label in self._get_completed_analysis_error_details(completed_ops)
        ]

    def _get_completed_analysis_error_details(
        self,
        completed_ops: list[str],
    ) -> list[tuple[str, str, str]]:
        """Return (op_key, error, label) for completed operations with errors."""
        details: list[tuple[str, str, str]] = []
        op_errors = [
            ("colors", self._color_run_error, "colors"),
            ("shots", self._shot_type_run_error, "shot type"),
            ("classify", self._classification_run_error, "classification"),
            ("detect_objects", self._object_detection_run_error, "object detection"),
            ("extract_text", self._text_extraction_run_error, "text extraction"),
            ("transcribe", self._transcription_run_error, "transcription"),
            ("describe", self._description_run_error, "description"),
            ("cinematography", self._cinematography_run_error, "cinematography"),
        ]
        for op_key, error_value, label in op_errors:
            if error_value and op_key in completed_ops:
                details.append((op_key, error_value, label))
        return details

    def _show_completed_analysis_error_dialog(self, completed_ops: list[str]) -> None:
        """Show one aggregated error dialog after an analysis run finishes."""
        details = self._get_completed_analysis_error_details(completed_ops)
        if not details:
            return

        sections = [
            f"{label.title()}:\n{error_value}"
            for _, error_value, label in details
        ]
        QMessageBox.warning(
            self,
            "Analysis Finished With Errors",
            "Some analysis operations failed:\n\n" + "\n\n".join(sections),
        )

    def _on_description_analysis_requested(self, clip_ids: list):
        """Handle description analysis request from Sequence tab (Storyteller).

        Navigates to Analyze tab, adds the specified clips, and runs description analysis.

        Args:
            clip_ids: List of clip IDs that need description analysis
        """
        if not clip_ids:
            return

        logger.info(f"Description analysis requested for {len(clip_ids)} clips")

        # Get clips from project
        clips = [self.clips_by_id.get(cid) for cid in clip_ids if cid in self.clips_by_id]
        if not clips:
            QMessageBox.warning(
                self,
                "No Clips Found",
                "Could not find the specified clips for analysis."
            )
            return

        # Navigate to Analyze tab
        self.tab_widget.setCurrentWidget(self.analyze_tab)

        # Add clips to Analyze tab
        self.analyze_tab.add_clips(clip_ids)

        # Wait a moment for UI to update, then trigger description analysis
        QTimer.singleShot(100, self._on_describe_from_tab)

        self.status_bar.showMessage(f"Running description analysis on {len(clips)} clips...")

    def _on_extract_text_from_tab(self):
        """Handle text extraction request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["extract_text"])

    @Slot(int, int, str)
    def _on_text_extraction_progress(self, current: int, total: int, clip_id: str):
        """Handle text extraction progress update."""
        if current == 0:
            self.status_bar.showMessage(
                "Text Extraction: loading OCR model..."
            )
        else:
            self.status_bar.showMessage(f"Extracting text: {current}/{total} clips...")
        self.progress_bar.setValue(int((current / total) * 100))

    @Slot(str, list)
    def _on_text_extraction_clip_ready(self, clip_id: str, texts: list):
        """Handle text extracted for single clip or frame."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.extracted_texts = texts
            self.analyze_tab.update_clip_extracted_text(clip_id, texts)
            self._mark_dirty()
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            self.project.update_frame(clip_id, extracted_texts=texts)
            self._mark_dirty()

    @Slot(str)
    def _on_text_extraction_error(self, error_msg: str):
        """Handle text extraction error."""
        logger.warning(f"Text extraction error: {error_msg}")
        self._record_analysis_run_error(
            "_text_extraction_run_error",
            "Text extraction",
            error_msg,
        )

    def _on_cinematography_from_tab(self):
        """Handle cinematography analysis request (standalone redirect to pipeline)."""
        clips = self.analyze_tab.get_clips()
        if clips:
            self._run_analysis_pipeline(clips, ["cinematography"])

    @Slot(int, int, str)
    def _on_cinematography_progress(self, current: int, total: int, clip_id: str):
        """Handle cinematography analysis progress update."""
        if current == 0:
            self.status_bar.showMessage(
                "Rich Analysis: loading model..."
            )
        else:
            self.status_bar.showMessage(f"Analyzing cinematography: {current}/{total} clips...")
        self.progress_bar.setValue(int((current / total) * 100))

    @Slot(str, object)
    def _on_cinematography_clip_ready(self, clip_id: str, cinematography):
        """Handle cinematography analyzed for single clip or frame."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.cinematography = cinematography
            # Also update shot_type for compatibility with existing filtering
            clip.shot_type = cinematography.get_simple_shot_type()
            self.analyze_tab.update_clip_cinematography(clip_id, cinematography)
            self.analyze_tab.update_clip_shot_type(clip_id, clip.shot_type)
            # Refresh sidebar if it's showing this clip
            if hasattr(self, 'clip_details_sidebar'):
                self.clip_details_sidebar.refresh_shot_type_if_showing(clip_id, clip.shot_type)
            self._mark_dirty()
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            shot_type = cinematography.get_simple_shot_type()
            self.project.update_frame(
                clip_id,
                cinematography=cinematography,
                shot_type=shot_type,
            )
            self._mark_dirty()

    @Slot(str)
    def _on_cinematography_error(self, error_msg: str):
        """Handle cinematography analysis error."""
        logger.warning(f"Cinematography analysis error: {error_msg}")
        self._record_analysis_run_error(
            "_cinematography_run_error",
            "Cinematography analysis",
            error_msg,
        )

    # Agent-triggered analysis completion handlers
    # These are separate from manual handlers to allow independent tracking

    @Slot()
    def _on_agent_color_analysis_finished(self):
        """Handle color analysis completion when triggered by agent."""
        logger.info("=== AGENT COLOR ANALYSIS FINISHED ===")

        # Guard against duplicate calls
        if self._color_analysis_finished_handled:
            logger.warning("_on_agent_color_analysis_finished already handled, ignoring duplicate")
            return
        self._color_analysis_finished_handled = True

        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)

        clips = self._agent_color_clips
        clip_count = len(clips)
        if self._color_run_error:
            self.status_bar.showMessage(
                f"Color extraction finished with errors - {clip_count} clips"
            )
        else:
            self.status_bar.showMessage(f"Color extraction complete - {clip_count} clips")

        # Send result back to agent
        if self._pending_agent_color_analysis and self._chat_worker:
            self._pending_agent_color_analysis = False
            agent_result = self._build_agent_analysis_result(
                clips,
                ["colors"],
                f"Extracted colors from {clip_count} clips",
            )
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": agent_result,
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._agent_color_clips = []
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent color analysis result to agent: {clip_count} clips")

    @Slot()
    def _on_agent_shot_analysis_finished(self):
        """Handle shot type classification completion when triggered by agent."""
        logger.info("=== AGENT SHOT ANALYSIS FINISHED ===")

        # Guard against duplicate calls
        if self._shot_type_finished_handled:
            logger.warning("_on_agent_shot_analysis_finished already handled, ignoring duplicate")
            return
        self._shot_type_finished_handled = True

        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)

        clips = self._agent_shot_clips
        clip_count = len(clips)
        if self._shot_type_run_error:
            self.status_bar.showMessage(
                f"Shot type classification finished with errors - {clip_count} clips"
            )
        else:
            self.status_bar.showMessage(
                f"Shot type classification complete - {clip_count} clips"
            )

        # Build shot type summary
        shot_types = {}
        for clip in clips:
            st = clip.shot_type or "unknown"
            shot_types[st] = shot_types.get(st, 0) + 1

        # Send result back to agent
        if self._pending_agent_shot_analysis and self._chat_worker:
            self._pending_agent_shot_analysis = False
            agent_result = self._build_agent_analysis_result(
                clips,
                ["shots"],
                f"Classified shot types for {clip_count} clips",
                {"shot_type_summary": shot_types},
            )
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": agent_result,
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._agent_shot_clips = []
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent shot analysis result to agent: {clip_count} clips")

        # Update chat panel with project state
        self._update_chat_project_state()

    @Slot()
    def _on_agent_transcription_finished(self):
        """Handle transcription completion when triggered by agent."""
        logger.info("=== AGENT TRANSCRIPTION FINISHED ===")

        # Guard against duplicate calls
        if self._transcription_finished_handled:
            logger.warning("_on_agent_transcription_finished already handled, ignoring duplicate")
            return
        self._transcription_finished_handled = True

        # Check if there are more sources to process
        if self._agent_transcription_source_queue:
            next_source, next_clips = self._agent_transcription_source_queue.pop(0)
            remaining = len(self._agent_transcription_source_queue)
            total_sources = remaining + 2  # +1 for current, +1 for just-completed
            current_source_num = total_sources - remaining

            logger.info(f"Continuing transcription with next source: {next_source.filename} ({current_source_num}/{total_sources})")

            # Reset guard for next source
            self._transcription_finished_handled = False

            # Update status
            self.status_bar.showMessage(
                f"Transcribing {len(next_clips)} clips (source {current_source_num}/{total_sources})..."
            )

            # Safely stop previous worker before creating new one
            self._stop_worker_safely(self.transcription_worker, "Transcription")

            # Start worker for next source
            self.transcription_worker = TranscriptionWorker(
                next_clips,
                next_source,
                self.settings.transcription_model,
                self.settings.transcription_language,
                parallelism=self.settings.transcription_parallelism,
                backend=self.settings.transcription_backend,
            )
            self.transcription_worker.progress.connect(self._on_transcription_progress)
            self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
            self.transcription_worker.transcription_completed.connect(self._on_agent_transcription_finished, Qt.UniqueConnection)
            self.transcription_worker.error.connect(self._on_transcription_error)
            # Clean up thread safely after it finishes
            self.transcription_worker.finished.connect(self.transcription_worker.deleteLater)
            self.transcription_worker.finished.connect(lambda: setattr(self, 'transcription_worker', None))
            self.transcription_worker.start()
            return

        # All sources processed - finalize
        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)

        clips = self._agent_transcription_clips
        clip_count = len(clips)
        self.status_bar.showMessage(f"Transcription complete - {clip_count} clips")

        # Build transcript summary
        transcribed_count = sum(1 for c in clips if c.transcript)

        # Send result back to agent
        if self._pending_agent_transcription and self._chat_worker:
            self._pending_agent_transcription = False
            agent_result = self._build_agent_analysis_result(
                clips,
                ["transcribe"],
                f"Transcribed {transcribed_count} of {clip_count} clips",
                {"transcribed_count": transcribed_count},
            )
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": agent_result,
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._agent_transcription_clips = []
            self._agent_transcription_source_queue = []
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent transcription result to agent: {transcribed_count}/{clip_count} clips")

        # Update chat panel with project state
        self._update_chat_project_state()

    # Drag and drop handlers
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    self._load_video(path)
                    return

    # Action handlers
    def _on_import_click(self):
        """Handle import button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
        )
        if file_path:
            self._load_video(Path(file_path))

    def _on_import_folder_click(self):
        """Handle import folder menu action."""
        from ui.source_browser import scan_folder_for_videos, AddVideoCard

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Import Video Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if folder_path:
            videos = scan_folder_for_videos(
                Path(folder_path), AddVideoCard.VIDEO_EXTENSIONS
            )
            if videos:
                self._on_videos_added(videos)

    def _load_video(self, path: Path):
        """Load a video file - adds to library and selects it."""
        # Check if already in library
        existing = None
        for source in self.sources:
            if source.file_path == path:
                existing = source
                break

        if existing:
            # Already in library, just select it
            self._select_source(existing)
        else:
            # Add to library with metadata
            source = self._create_source_with_metadata(path)
            self.project.add_source(source)
            self.collect_tab.add_source(source)

            # Generate thumbnail
            self._generate_source_thumbnail(source)

            # Select it
            self._select_source(source)

    def _on_import_url_click(self):
        """Handle import URL button click."""
        url, resolution = URLImportDialog.get_import_request(self)
        if url and resolution:
            self._download_video(url, resolution=resolution)

    def _download_video(self, url: str, resolution: Optional[str] = None):
        """Start downloading a video from URL."""
        if not self._ensure_video_download_available():
            return

        # Update UI state
        self.collect_tab.set_downloading(True)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        self.download_worker = DownloadWorker(url, resolution=resolution)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.download_completed.connect(self._on_download_finished)
        self.download_worker.error.connect(self._on_download_error)
        # Clean up thread safely after it finishes to prevent "QThread: Destroyed while running" crash
        self.download_worker.finished.connect(self.download_worker.deleteLater)
        self.download_worker.finished.connect(lambda: setattr(self, 'download_worker', None))
        self._gui_state.set_processing("download", url[:60])
        self.download_worker.start()

    def _on_download_progress(self, progress: float, message: str):
        """Handle download progress update."""
        self.progress_bar.setValue(int(progress))
        self.status_bar.showMessage(message)

    def _on_download_finished(self, result):
        """Handle download completion."""
        self._gui_state.clear_processing("download")
        self.progress_bar.setVisible(False)
        self.collect_tab.set_downloading(False)

        if result.file_path and result.file_path.exists():
            self._load_video(result.file_path)
            self.status_bar.showMessage(f"Downloaded: {result.title}")
        else:
            QMessageBox.warning(self, "Download Error", "Download completed but file not found")

    def _show_download_error_dialog(
        self,
        error: str,
        *,
        failed_titles: Optional[list[str]] = None,
    ) -> None:
        """Show a recovery-oriented dialog for download failures."""
        error_kind = classify_download_error_message(error)

        if error_kind == DOWNLOAD_ERROR_COOKIES_REQUIRED:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(
                "YouTube Cookies Required"
                if not failed_titles
                else "Some Downloads Need Cookies"
            )
            if failed_titles:
                title_list = "\n".join(f"• {title}" for title in failed_titles)
                msg.setText(
                    "YouTube requires browser authentication cookies for some selected videos."
                )
                msg.setInformativeText(
                    "Sign in to YouTube in your browser, export cookies for yt-dlp, "
                    "then retry these videos.\n\nAffected videos:\n"
                    f"{title_list}"
                )
            else:
                msg.setText("YouTube requires browser authentication cookies for this video.")
                msg.setInformativeText(
                    "Sign in to YouTube in your browser, export cookies for yt-dlp, "
                    "then retry the download."
                )
            msg.setDetailedText(error)
            open_button = msg.addButton("Open Cookie Instructions", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Close)
            msg.exec()
            if msg.clickedButton() == open_button:
                QDesktopServices.openUrl(QUrl(YTDLP_COOKIE_HELP_URL))
            return

        if error_kind == DOWNLOAD_ERROR_JS_RUNTIME_REQUIRED:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("JavaScript Runtime Required")
            msg.setText("YouTube downloads require a JavaScript runtime.")
            msg.setInformativeText(
                "Install Deno from Settings > Dependencies, then retry the download."
            )
            msg.setDetailedText(error)
            msg.exec()
            return

        QMessageBox.critical(self, "Download Error", error)

    def _on_download_error(self, error: str):
        """Handle download error."""
        self._gui_state.clear_processing("download")
        self._gui_state.set_last_error(f"Download error: {error}")
        self.progress_bar.setVisible(False)
        self.collect_tab.set_downloading(False)
        self._show_download_error_dialog(error)

    # Video search handlers (YouTube and Internet Archive)
    @Slot(str, str)
    def _on_video_search(self, source: str, query: str, video_duration: str = "", video_definition: str = ""):
        """Route search request to appropriate handler based on source."""
        if source == "youtube":
            self._on_youtube_search(query, video_duration, video_definition)
        elif source == "internet_archive":
            self._on_internet_archive_search(query)
        else:
            logger.warning(f"Unknown search source: {source}")

    def _on_youtube_search(self, query: str, video_duration: str = "", video_definition: str = ""):
        """Handle YouTube search request."""
        if not self.settings.youtube_api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please configure your YouTube API key in Settings > API Keys.",
            )
            return

        # Initialize client if needed
        if not self.youtube_client:
            try:
                self.youtube_client = YouTubeSearchClient(
                    self.settings.youtube_api_key
                )
            except InvalidAPIKeyError as e:
                QMessageBox.critical(self, "Invalid API Key", str(e))
                return

        self.collect_tab.youtube_search_panel.set_searching(True)

        # Run search in thread
        self.youtube_search_worker = YouTubeSearchWorker(
            self.youtube_client, query, self.settings.youtube_results_count,
            video_duration=video_duration,
            video_definition=video_definition,
        )
        self.youtube_search_worker.search_completed.connect(self._on_youtube_search_finished)
        self.youtube_search_worker.error.connect(self._on_youtube_search_error)
        # Clean up thread safely after it finishes to prevent "QThread: Destroyed while running" crash
        self.youtube_search_worker.finished.connect(self.youtube_search_worker.deleteLater)
        self.youtube_search_worker.finished.connect(lambda: setattr(self, 'youtube_search_worker', None))
        self.youtube_search_worker.start()

    @Slot(object)
    def _on_youtube_search_finished(self, result: YouTubeSearchResult):
        """Handle YouTube search completion."""
        self.collect_tab.youtube_search_panel.set_searching(False)
        self.collect_tab.youtube_search_panel.display_results(result.videos)
        self.status_bar.showMessage(f"Found {len(result.videos)} videos")

        # Update GUI state for agent context
        query = self.collect_tab.youtube_search_panel.get_search_query()
        self._gui_state.update_from_search(query, [
            {
                "video_id": v.video_id,
                "title": v.title,
                "duration": v.duration_str,
                "channel": v.channel_title,
                "thumbnail": v.thumbnail_url,
            }
            for v in result.videos
        ])

    @Slot(str)
    def _on_youtube_search_error(self, error: str):
        """Handle YouTube search error."""
        self.collect_tab.youtube_search_panel.set_searching(False)
        QMessageBox.critical(self, "Search Failed", error)

    def _on_internet_archive_search(self, query: str):
        """Handle Internet Archive search request."""
        self.collect_tab.youtube_search_panel.set_searching(True)

        # Run search in thread
        self.ia_search_worker = InternetArchiveSearchWorker(
            query, self.settings.youtube_results_count  # Reuse the same count setting
        )
        self.ia_search_worker.search_completed.connect(self._on_ia_search_finished)
        self.ia_search_worker.error.connect(self._on_ia_search_error)
        # Clean up thread safely after it finishes
        self.ia_search_worker.finished.connect(self.ia_search_worker.deleteLater)
        self.ia_search_worker.finished.connect(lambda: setattr(self, 'ia_search_worker', None))
        self.ia_search_worker.start()

    @Slot(list)
    def _on_ia_search_finished(self, videos: list):
        """Handle Internet Archive search completion."""
        self.collect_tab.youtube_search_panel.set_searching(False)
        self.collect_tab.youtube_search_panel.display_results(videos)
        self.status_bar.showMessage(f"Found {len(videos)} videos on Internet Archive")

        # Update GUI state for agent context
        query = self.collect_tab.youtube_search_panel.get_search_query()
        self._gui_state.update_from_search(query, [
            {
                "video_id": v.video_id,
                "title": v.title,
                "duration": v.duration_str,
                "channel": v.channel_title,  # This returns creator for IA videos
                "thumbnail": v.thumbnail_url,
            }
            for v in videos
        ])

    @Slot(str)
    def _on_ia_search_error(self, error: str):
        """Handle Internet Archive search error."""
        self.collect_tab.youtube_search_panel.set_searching(False)
        QMessageBox.critical(self, "Search Failed", error)

    @Slot(list)
    def _on_bulk_download(self, videos: list):
        """Start bulk download of selected videos."""
        if not videos:
            return

        if not self._ensure_video_download_available():
            return

        # Track download results for summary
        self._bulk_download_total = len(videos)
        self._bulk_download_success = 0
        self._bulk_download_errors: list[tuple[str, str]] = []  # (title, error)
        self._bulk_video_titles = {v.video_id: v.title for v in videos}

        self.collect_tab.youtube_search_panel.set_downloading(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(videos))
        self.progress_bar.setValue(0)

        self.bulk_download_worker = BulkDownloadWorker(
            videos,
            download_dir=self.settings.download_dir,
            max_parallel=self.settings.youtube_parallel_downloads,
        )
        self.bulk_download_worker.progress.connect(self._on_bulk_progress)
        self.bulk_download_worker.video_finished.connect(self._on_bulk_video_finished)
        self.bulk_download_worker.video_error.connect(self._on_bulk_video_error)
        self.bulk_download_worker.all_finished.connect(self._on_bulk_finished)
        # Clean up thread safely after it finishes to prevent "QThread: Destroyed while running" crash
        self.bulk_download_worker.finished.connect(self.bulk_download_worker.deleteLater)
        self.bulk_download_worker.finished.connect(lambda: setattr(self, 'bulk_download_worker', None))
        self.bulk_download_worker.start()

    @Slot(int, int, str)
    def _on_bulk_progress(self, current: int, total: int, message: str):
        """Update bulk download progress."""
        self.progress_bar.setValue(current)
        self.status_bar.showMessage(message)

    @Slot(object)
    def _on_bulk_video_finished(self, result):
        """Handle single video download completion."""
        self._bulk_download_success += 1
        if result.file_path and result.file_path.exists():
            self._load_video(result.file_path)

    @Slot(str, str)
    def _on_bulk_video_error(self, video_id: str, error: str):
        """Log individual video download error."""
        title = self._bulk_video_titles.get(video_id, video_id)
        logger.warning(f"Failed to download '{title}' ({video_id}): {error}")
        self._bulk_download_errors.append((title, error))

    @Slot()
    def _on_bulk_finished(self):
        """Handle bulk download completion."""
        self.progress_bar.setVisible(False)
        self.collect_tab.youtube_search_panel.set_downloading(False)

        total = self._bulk_download_total
        success = self._bulk_download_success
        errors = self._bulk_download_errors

        if errors:
            # Show summary with error count
            self.status_bar.showMessage(
                f"Download complete: {success}/{total} succeeded, {len(errors)} failed"
            )

            cookie_failures = [
                (title, error)
                for title, error in errors
                if classify_download_error_message(error) == DOWNLOAD_ERROR_COOKIES_REQUIRED
            ]
            if cookie_failures and len(cookie_failures) == len(errors):
                self._show_download_error_dialog(
                    cookie_failures[0][1],
                    failed_titles=[title for title, _error in cookie_failures],
                )
            else:
                error_details = "\n".join(
                    f"• {title}: {error}" for title, error in errors
                )
                QMessageBox.warning(
                    self,
                    "Some Downloads Failed",
                    f"Successfully downloaded {success} of {total} videos.\n\n"
                    f"Failed downloads:\n{error_details}",
                )
        else:
            self.status_bar.showMessage(f"Downloaded {success} videos successfully")

    def _start_detection(self, mode: str = "adaptive", config_dict: dict = None):
        """Start scene detection with given mode and config.

        Args:
            mode: Detection mode ('adaptive', 'content', or 'karaoke')
            config_dict: Configuration dictionary with mode-specific parameters
        """
        logger.info(f"=== START DETECTION (mode={mode}) ===")
        if not self.current_source:
            return

        config_dict = config_dict or {}

        # Guard against concurrent detection
        if self.detection_worker and self.detection_worker.isRunning():
            logger.warning("Detection already in progress, ignoring request")
            return

        # Reset guards for new detection run
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._shot_type_finished_handled = False
        self._transcription_finished_handled = False

        # Build configuration based on mode
        if mode == "karaoke":
            # Karaoke mode uses KaraokeDetectionConfig
            karaoke_config = KaraokeDetectionConfig(
                roi_top_percent=config_dict.get("roi_top_percent", 0.0),
                text_similarity_threshold=config_dict.get("text_similarity_threshold", 60.0),
                confirm_frames=config_dict.get("confirm_frames", 3),
                cut_offset=config_dict.get("cut_offset", 5),
            )
            visual_config = None
        else:
            # Visual mode uses DetectionConfig
            visual_config = DetectionConfig(
                threshold=config_dict.get("threshold", 3.0),
                use_adaptive=(mode == "adaptive"),
                luma_only=config_dict.get("luma_only"),
            )
            karaoke_config = None

        # Start detection in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        logger.info("Creating DetectionWorker...")
        self.detection_worker = DetectionWorker(
            self.current_source.file_path,
            config=visual_config,
            mode=mode,
            karaoke_config=karaoke_config,
        )
        self.detection_worker.progress.connect(self._on_detection_progress)
        self.detection_worker.detection_completed.connect(self._on_detection_finished, Qt.UniqueConnection)
        self.detection_worker.error.connect(self._on_detection_error)
        # Clean up thread safely after it finishes to prevent "QThread: Destroyed while running" crash
        self.detection_worker.finished.connect(self.detection_worker.deleteLater)
        self.detection_worker.finished.connect(lambda: setattr(self, 'detection_worker', None))
        logger.info("Starting DetectionWorker...")
        self._gui_state.set_processing("scene_detection", f"running on {self.current_source.filename}")
        self.detection_worker.start()
        logger.info("DetectionWorker started")

    def _on_detection_progress(self, progress: float, message: str):
        """Handle detection progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)
        self._detection_current_progress = progress  # Track for agent status checks

    @Slot(object, list)
    def _on_detection_finished(self, source: Source, clips: list[Clip]):
        """Handle detection completion."""
        logger.info("=== DETECTION FINISHED ===")
        self._gui_state.clear_processing("scene_detection")

        # Guard against duplicate calls
        if self._detection_finished_handled:
            logger.warning("_on_detection_finished already handled, ignoring duplicate call")
            return
        self._detection_finished_handled = True

        logger.info(f"Detection worker running: {self.detection_worker.isRunning() if self.detection_worker else 'None'}")

        # Update the existing source in the library with detected metadata
        if self.current_source and self.current_source.id in self.sources_by_id:
            # Update metadata from detection
            self.current_source.duration_seconds = source.duration_seconds
            self.current_source.fps = source.fps
            self.current_source.width = source.width
            self.current_source.height = source.height
            self.current_source.analyzed = True

            # Update CollectTab to show analyzed badge
            self.collect_tab.update_source_analyzed(self.current_source.id, True)
        else:
            # Source not in library yet (shouldn't happen normally)
            self.current_source = source
            source.analyzed = True

        # Update clips to reference the existing source ID (detection creates a new Source object)
        # This ensures clips_by_source lookups work correctly
        logger.info(f"Updating {len(clips)} clips to use source_id={self.current_source.id}")
        for clip in clips:
            clip.source_id = self.current_source.id

        # Add new clips to the collection (don't replace existing clips from other sources)
        # This replaces any existing clips from this source (handles re-analysis case)
        # Flag prevents _on_clips_added from also starting thumbnail generation.
        # This works because replace_source_clips fires clips_added synchronously
        # (same-thread direct connection). If notification delivery ever becomes
        # asynchronous, this flag would be cleared before the handler runs.
        self._suppress_clips_added_thumbnails = True
        try:
            self.project.replace_source_clips(self.current_source.id, clips)
        finally:
            self._suppress_clips_added_thumbnails = False
        self._update_window_title()

        # Remove old clips for this source from the Cut tab UI (handles re-analysis case)
        self.cut_tab.remove_clips_for_source(self.current_source.id)

        # Remove orphaned clips from Analyze tab (clips that no longer exist)
        valid_clip_ids = set(self.clips_by_id.keys())
        removed_count = self.analyze_tab.remove_orphaned_clips(valid_clip_ids)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} orphaned clips from Analyze tab")

        self.status_bar.showMessage(f"Found {len(clips)} scenes. Generating thumbnails...")

        # Start thumbnail generation - safely stop any running worker first
        self._stop_worker_safely(self.thumbnail_worker, "thumbnail")
        logger.info("Creating ThumbnailWorker...")
        self.thumbnail_worker = ThumbnailWorker(
            source, clips, cache_dir=self.settings.thumbnail_cache_dir
        )
        self.thumbnail_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_worker.finished.connect(self._on_thumbnails_finished, Qt.UniqueConnection)
        logger.info("Starting ThumbnailWorker...")
        self.thumbnail_worker.start()
        logger.info("ThumbnailWorker started")

    def _on_detection_error(self, error: str):
        """Handle detection error."""
        logger.error(f"=== DETECTION ERROR: {error} ===")
        self._gui_state.clear_processing("scene_detection")
        self._gui_state.set_last_error(f"Detection error: {error}")
        self.progress_bar.setVisible(False)

        # If agent was waiting for detection, send error result
        if self._pending_agent_detection and self._chat_worker:
            self._pending_agent_detection = False
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": False,
                "error": error
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent detection error to agent: {error}")
        else:
            # Only show dialog for manual detection
            QMessageBox.critical(self, "Detection Error", error)

    def _on_thumbnail_progress(self, current: int, total: int):
        """Handle thumbnail generation progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
        """Handle individual thumbnail completion."""
        logger.info(f"_on_thumbnail_ready called: clip_id={clip_id}, thumb_path={thumb_path}")
        clip = self.clips_by_id.get(clip_id)
        if clip:
            # Look up the clip's actual source, not current_source which may have changed
            clip_source = self.sources_by_id.get(clip.source_id)
            if clip_source:
                logger.info(f"Found clip {clip_id}, source={clip_source.id}, adding to cut_tab")
                # Add to Cut tab (primary clip browser for detection)
                self.cut_tab.add_clip(clip, clip_source)
            else:
                logger.warning(f"Source not found for clip {clip_id}: source_id={clip.source_id}")
        else:
            logger.warning(f"Clip not found in clips_by_id: {clip_id}")
            logger.warning(f"Available clip IDs: {list(self.clips_by_id.keys())[:5]}...")
            logger.warning(f"Total clips in project: {len(self.project.clips)}")

    @Slot()
    def _on_thumbnails_finished(self):
        """Handle all thumbnails completed."""
        logger.info("=== THUMBNAILS FINISHED ===")

        # Guard against duplicate calls
        if self._thumbnails_finished_handled:
            logger.warning("_on_thumbnails_finished already handled, ignoring duplicate call")
            return
        self._thumbnails_finished_handled = True

        logger.info(f"Thumbnail worker running: {self.thumbnail_worker.isRunning() if self.thumbnail_worker else 'None'}")

        # Refresh Analyze tab lookups (cached properties may have been invalidated)
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        # Update Cut tab with ALL clips from ALL sources (clips accumulate as sources are detected)
        # This ensures the clip count label and _clips list match the browser contents
        logger.info(f"_on_thumbnails_finished: total {len(self.clips)} clips from {len(self.sources)} sources")
        self.cut_tab.set_clips(self.clips)

        # Refresh sequence tab with ALL clips from ALL sources
        self._refresh_sequence_tab_clips()

        # Update Render tab with total clip count from all sources
        self.render_tab.set_detected_clips_count(len(self.clips))

        # Detection complete - ready for user to manually analyze clips
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Ready - {len(self.clips)} scenes detected")

        # Update chat panel with project state for context-aware prompts
        self._update_chat_project_state()

        # If agent was waiting for detection, send result back
        if self._pending_agent_detection and self._chat_worker:
            self._pending_agent_detection = False
            if self.current_source:
                current_source_clips = self.project.clips_by_source.get(
                    self.current_source.id, []
                )
            else:
                current_source_clips = []
            clip_ids = [c.id for c in current_source_clips]
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "source_id": self.current_source.id if self.current_source else None,
                    "source_name": self.current_source.filename if self.current_source else None,
                    "clip_count": len(clip_ids),
                    "clip_ids": clip_ids,
                    "detected_clips": self._build_agent_detected_clip_summary(
                        current_source_clips
                    ),
                    "response_guidance": (
                        "Summarize the scene detection using only these clip IDs, "
                        "source names, and timing ranges. Do not invent descriptions "
                        "for detected clips."
                    ),
                    "message": f"Detected {len(clip_ids)} scenes"
                }
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent detection result to agent: {len(clip_ids)} clips")

        # Continue with next source in batch queue (deferred to let worker cleanup)
        QTimer.singleShot(0, self._start_next_analysis)

    def _on_color_progress(self, current: int, total: int):
        """Handle color analysis progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_color_ready(self, clip_id: str, colors: list):
        """Handle color extraction complete for a clip or frame."""
        # Try clip first
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.dominant_colors = colors
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_colors(clip_id, colors)
            self.analyze_tab.update_clip_colors(clip_id, colors)
            self._mark_dirty()
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            self.project.update_frame(clip_id, dominant_colors=colors)
            self._mark_dirty()

    def _on_shot_type_progress(self, current: int, total: int):
        """Handle shot type classification progress."""
        if current == 0:
            self.status_bar.showMessage(
                "Classify Shots: downloading model (first run, ~400 MB)..."
            )
        else:
            self.status_bar.showMessage(
                f"Classify Shots: processing clip {current}/{total}..."
            )
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_shot_type_ready(self, clip_id: str, shot_type: str, confidence: float):
        """Handle shot type classification complete for a clip or frame."""
        # Try clip first
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.shot_type = shot_type
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_shot_type(clip_id, shot_type)
            self.analyze_tab.update_clip_shot_type(clip_id, shot_type)
            # Refresh sidebar if it's showing this clip
            if hasattr(self, 'clip_details_sidebar'):
                self.clip_details_sidebar.refresh_shot_type_if_showing(clip_id, shot_type)
            self._mark_dirty()
            logger.debug(f"Clip {clip_id}: {shot_type} ({confidence:.2f})")
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            self.project.update_frame(clip_id, shot_type=shot_type)
            self._mark_dirty()
            logger.debug(f"Frame {clip_id}: {shot_type} ({confidence:.2f})")

    def _on_transcription_progress(self, current: int, total: int):
        """Handle transcription progress."""
        if current == 0:
            self.status_bar.showMessage(
                "Transcribe: downloading Whisper model (first run)..."
            )
        else:
            self.status_bar.showMessage(f"Transcribing: {current}/{total} clips...")
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_transcript_ready(self, clip_id: str, segments: list):
        """Handle transcript ready for a clip."""
        # Update the clip model
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.transcript = segments
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_transcript(clip_id, segments)
            self.analyze_tab.update_clip_transcript(clip_id, segments)
            self._mark_dirty()
            logger.debug(f"Clip {clip_id}: transcribed {len(segments)} segments")

    def _on_clip_dragged_to_timeline(self, clip: Clip):
        """Handle clip dragged from browser to timeline."""
        if self.current_source:
            # Add to Sequence tab timeline (primary source of truth)
            self.sequence_tab.add_clip_to_timeline(clip, self.current_source)
            self.status_bar.showMessage("Added clip to timeline")
            self._mark_dirty()
            # Update Render tab with new sequence info
            self._update_render_tab_sequence_info()

    def _on_timeline_playhead_changed(self, time_seconds: float):
        """Handle timeline playhead position change."""
        # Don't seek during timeline-driven playback or while we're applying
        # a playhead update that originated from the video player.
        if self._is_playing or self._syncing_timeline_from_video:
            return

        if self._has_ready_sequence_preview():
            preview_path = self._rendered_sequence_preview_path
            preview_key = f"preview:{preview_path}"
            self._preview_sync_clip = None
            self.sequence_tab.video_player.set_chromatic_color_bar(None)
            if self._sequence_preview_source_id != preview_key:
                self._sequence_preview_source_id = preview_key
                self._using_rendered_sequence_preview = True
                self.sequence_tab.video_player.clear_clip_range()
                self.sequence_tab.video_player.load_video(preview_path)
            self.sequence_tab.video_player.seek_to(time_seconds)
            return

        seq_clip, _, source = self.sequence_tab.timeline.get_clip_at_playhead()
        if not seq_clip or not source:
            # Fallback behavior when no sequence clip is under playhead.
            self._preview_sync_clip = None
            self._pending_sequence_playback_source_id = None
            self._pending_sequence_playback_range = None
            self._pending_sequence_preview_source_id = None
            self._pending_sequence_preview_clip_range = None
            self._pending_sequence_preview_seek_seconds = None
            self.sequence_tab.video_player.clear_clip_range()
            self.sequence_tab.video_player.seek_to(time_seconds)
            self._update_sequence_chromatic_bar(None)
            return

        sequence = self.sequence_tab.timeline.get_sequence()
        timeline_frame = int(time_seconds * sequence.fps)

        file_to_load, clip_start_seconds, clip_end_seconds, source_seconds = (
            _resolve_playback_source(seq_clip, source, timeline_frame)
        )

        # Determine the source ID for tracking loaded sources
        preview_source_key = str(file_to_load)

        # Keep preview source aligned to the clip under playhead.
        if self._sequence_preview_source_id != preview_source_key:
            self._preview_sync_clip = seq_clip
            self._update_sequence_chromatic_bar(seq_clip)
            self._sequence_preview_source_id = preview_source_key
            self._sequence_preview_loading = True
            self._pending_sequence_preview_source_id = preview_source_key
            self._pending_sequence_preview_clip_range = (clip_start_seconds, clip_end_seconds)
            self._pending_sequence_preview_seek_seconds = source_seconds
            self.sequence_tab.video_player.load_video(file_to_load)
            return

        self._preview_sync_clip = seq_clip
        if self._sequence_preview_loading:
            # Source is still loading; defer seek/range until media_loaded.
            self._pending_sequence_preview_source_id = preview_source_key
            self._pending_sequence_preview_clip_range = (clip_start_seconds, clip_end_seconds)
            self._pending_sequence_preview_seek_seconds = source_seconds
            return

        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None
        self.sequence_tab.video_player.set_clip_range(clip_start_seconds, clip_end_seconds)
        self.sequence_tab.video_player.seek_to(source_seconds)
        self._update_sequence_chromatic_bar(seq_clip)

    @Slot()
    def _on_sequence_video_loaded(self):
        """Apply deferred sequence preview sync after a source finishes loading."""
        if getattr(self, "_using_rendered_sequence_preview", False):
            self._sequence_preview_loading = False
            return

        self._sequence_preview_loading = False

        if self._is_playing:
            # During timeline playback, prefer deferred playback ranges.
            if (
                self._pending_sequence_playback_source_id is not None
                and self._pending_sequence_playback_source_id == self._sequence_preview_source_id
                and self._pending_sequence_playback_range is not None
            ):
                start_seconds, end_seconds = self._pending_sequence_playback_range
                self._pending_sequence_playback_source_id = None
                self._pending_sequence_playback_range = None
                self.sequence_tab.video_player.play_range(start_seconds, end_seconds)
                self._playback_timer.start()
            self._pending_sequence_preview_source_id = None
            self._pending_sequence_preview_clip_range = None
            self._pending_sequence_preview_seek_seconds = None
            return

        if (
            self._pending_sequence_preview_source_id is None
            or self._pending_sequence_preview_source_id != self._sequence_preview_source_id
        ):
            return

        clip_range = self._pending_sequence_preview_clip_range
        seek_seconds = self._pending_sequence_preview_seek_seconds
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None

        if clip_range:
            self.sequence_tab.video_player.set_clip_range(clip_range[0], clip_range[1])
        if seek_seconds is not None:
            self.sequence_tab.video_player.seek_to(seek_seconds)

    @Slot()
    def _on_sequence_video_load_failed(self):
        """Handle video load failure — clear loading state to prevent deadlock."""
        logger.warning("Sequence video load failed — clearing loading state")
        self._sequence_preview_loading = False
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        # Invalidate the source ID so the next attempt reloads
        self._sequence_preview_source_id = None
        if self._is_playing:
            self._stop_playback()

    def _on_sequence_changed(self):
        """Handle sequence modification."""
        # Update Export EDL menu item state
        if self.project and self.project.sequences:
            has_clips = any(sequence.duration_frames > 0 for sequence in self.project.sequences)
        else:
            sequence = self.sequence_tab.timeline.get_sequence()
            has_clips = sequence.duration_frames > 0
        self.export_edl_action.setEnabled(has_clips)

        # Update chat panel with project state (sequence may have changed)
        self._update_chat_project_state()

    @Slot(bool)
    def _on_chromatic_bar_setting_changed(self, enabled: bool):
        """Apply chromatic bar visibility changes to sequence preview."""
        self._invalidate_sequence_preview()
        if not enabled:
            self.sequence_tab.video_player.set_chromatic_color_bar(None)
            return
        self._update_sequence_chromatic_bar()

    def _get_sequence_preview_inputs(self):
        """Return sequence render inputs from the active timeline."""
        sequence = self.sequence_tab.get_sequence()
        sources = self.sequence_tab.timeline.get_sources_lookup()
        clips = self.sequence_tab.timeline.get_clips_lookup()

        if not sources and self.sources_by_id:
            sources = dict(self.sources_by_id)
        if not clips:
            for clip in self.clips:
                source = self.sources_by_id.get(clip.source_id)
                if source:
                    clips[clip.id] = (clip, source)

        frames = self.project.frames_by_id if self.project else {}
        return sequence, sources, clips, frames

    def _get_sequence_preview_cache_entry(self):
        """Return (signature, path, settings) for the active sequence preview."""
        sequence, sources, clips, frames = self._get_sequence_preview_inputs()
        settings = SequencePreviewSettings()
        signature = compute_sequence_preview_signature(
            sequence=sequence,
            sources=sources,
            clips=clips,
            frames=frames,
            settings=settings,
        )
        path = get_sequence_preview_path(sequence, signature)
        return signature, path, settings

    def _has_ready_sequence_preview(self) -> bool:
        """Whether the active sequence has a valid cached preview file."""
        sequence = self.sequence_tab.get_sequence()
        if not sequence.get_all_clips():
            return False
        try:
            signature, path, settings = self._get_sequence_preview_cache_entry()
        except Exception:
            return False
        if path.exists() and path.stat().st_size > 0:
            self._rendered_sequence_preview_signature = signature
            self._rendered_sequence_preview_path = path
            self._rendered_sequence_preview_profile = settings.profile_label
            return True
        return False

    @Slot()
    def _invalidate_sequence_preview(self):
        """Mark the rendered sequence preview stale after timeline edits."""
        if getattr(self, "sequence_preview_worker", None) and self.sequence_preview_worker.isRunning():
            return
        self._rendered_sequence_preview_path = None
        self._rendered_sequence_preview_signature = None
        self._sequence_preview_play_after_render_frame = None
        self._using_rendered_sequence_preview = False
        if hasattr(self, "sequence_tab"):
            sequence = self.sequence_tab.get_sequence()
            if sequence.get_all_clips():
                self.sequence_tab.set_sequence_preview_status(
                    "Stale",
                    SequencePreviewSettings().profile_label,
                )
            else:
                self.sequence_tab.set_sequence_preview_status("Not rendered")

    @Slot()
    def _on_render_sequence_preview_requested(self):
        """Render the active sequence preview explicitly from the header button."""
        self._start_sequence_preview_render(play_after_frame=None)

    def _start_sequence_preview_render(self, play_after_frame: Optional[int]):
        """Start preview rendering, optionally continuing playback afterwards."""
        if self.sequence_preview_worker and self.sequence_preview_worker.isRunning():
            self._sequence_preview_play_after_render_frame = play_after_frame
            return

        sequence, sources, clips, frames = self._get_sequence_preview_inputs()
        if not sequence.get_all_clips():
            QMessageBox.information(self, "Render Preview", "No clips in timeline to preview")
            return

        signature, path, settings = self._get_sequence_preview_cache_entry()
        if path.exists() and path.stat().st_size > 0:
            self._rendered_sequence_preview_signature = signature
            self._rendered_sequence_preview_path = path
            self._rendered_sequence_preview_profile = settings.profile_label
            self.sequence_tab.set_sequence_preview_status("Ready", settings.profile_label)
            if play_after_frame is not None:
                self._start_rendered_sequence_preview_playback(play_after_frame)
            return

        self._sequence_preview_play_after_render_frame = play_after_frame
        self.sequence_tab.set_sequence_preview_status(
            "Rendering",
            settings.profile_label,
            rendering=True,
        )
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Rendering sequence preview...")

        self.sequence_preview_worker = SequencePreviewWorker(
            sequence=sequence,
            sources=sources,
            clips=clips,
            frames=frames,
            settings=settings,
        )
        self.sequence_preview_worker.progress.connect(self._on_sequence_preview_progress)
        self.sequence_preview_worker.preview_completed.connect(self._on_sequence_preview_finished)
        self.sequence_preview_worker.error.connect(self._on_sequence_preview_error)
        self.sequence_preview_worker.finished.connect(self.sequence_preview_worker.deleteLater)
        self.sequence_preview_worker.finished.connect(
            lambda: setattr(self, "sequence_preview_worker", None)
        )
        self.sequence_preview_worker.start()

    @Slot(float, str)
    def _on_sequence_preview_progress(self, progress: float, message: str):
        """Update UI while a sequence preview render is running."""
        value = int(max(0.0, min(1.0, progress)) * 100)
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
        self.sequence_tab.set_sequence_preview_status(
            f"Rendering {value}%",
            self._rendered_sequence_preview_profile,
            rendering=True,
        )

    @Slot(object, str, str, bool)
    def _on_sequence_preview_finished(
        self,
        path,
        signature: str,
        profile_label: str,
        from_cache: bool,
    ):
        """Handle successful sequence preview render."""
        try:
            current_signature, _, _ = self._get_sequence_preview_cache_entry()
        except Exception:
            current_signature = None
        if current_signature != signature:
            self._sequence_preview_play_after_render_frame = None
            self.progress_bar.setVisible(False)
            self.sequence_tab.set_sequence_preview_status(
                "Stale",
                SequencePreviewSettings().profile_label,
            )
            self.status_bar.showMessage("Sequence preview is stale after timeline changes", 3000)
            return

        self._rendered_sequence_preview_path = Path(path)
        self._rendered_sequence_preview_signature = signature
        self._rendered_sequence_preview_profile = profile_label
        self.progress_bar.setVisible(False)
        self.sequence_tab.set_sequence_preview_status("Ready", profile_label)
        self.status_bar.showMessage(
            "Sequence preview ready" if not from_cache else "Using cached sequence preview",
            3000,
        )

        play_after = self._sequence_preview_play_after_render_frame
        self._sequence_preview_play_after_render_frame = None
        if play_after is not None:
            self._start_rendered_sequence_preview_playback(play_after)

    @Slot(str)
    def _on_sequence_preview_error(self, message: str):
        """Handle failed sequence preview render."""
        self._sequence_preview_play_after_render_frame = None
        self.progress_bar.setVisible(False)
        self.sequence_tab.set_sequence_preview_status(
            "Failed",
            SequencePreviewSettings().profile_label,
        )
        self.status_bar.showMessage("Sequence preview failed", 5000)
        QMessageBox.warning(self, "Render Preview", f"Preview render failed:\n{message}")

    def _resolve_sequence_clip_bar_color(self, seq_clip) -> Optional[tuple[int, int, int]]:
        """Resolve the dominant color for a sequence clip."""
        if not seq_clip:
            return None
        clip_data = self.sequence_tab.timeline._clip_lookup.get(seq_clip.source_clip_id)
        if not clip_data:
            return None
        clip, _ = clip_data
        if not clip.dominant_colors:
            return None
        color = clip.dominant_colors[0]
        if len(color) < 3:
            return None
        r, g, b = color[0], color[1], color[2]
        return (
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
        )

    def _update_sequence_chromatic_bar(self, seq_clip=None):
        """Update preview's full-width chromatic bar for the active sequence clip."""
        if not self.sequence_tab.should_show_chromatic_color_bar():
            self.sequence_tab.video_player.set_chromatic_color_bar(None)
            return

        active_clip = seq_clip
        if active_clip is None:
            active_clip, _, _ = self.sequence_tab.timeline.get_clip_at_playhead()
        color = self._resolve_sequence_clip_bar_color(active_clip) or (0, 0, 0)
        self.sequence_tab.video_player.set_chromatic_color_bar(color)

    # --- Playback methods ---

    def _start_rendered_sequence_preview_playback(self, start_frame: int):
        """Play the active sequence through its continuous rendered preview."""
        if not self._has_ready_sequence_preview() or not self._rendered_sequence_preview_path:
            self._start_sequence_preview_render(play_after_frame=start_frame)
            return

        sequence = self.sequence_tab.timeline.get_sequence()
        if sequence.duration_frames == 0:
            return

        start_seconds = max(0.0, start_frame / sequence.fps)
        self._is_playing = True
        self._using_rendered_sequence_preview = True
        self._current_playback_clip = None
        self._preview_sync_clip = None
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None
        self._sequence_preview_loading = False
        self.sequence_tab.video_player.set_playing(True)
        self.sequence_tab.video_player.set_loop(False)
        self.sequence_tab.video_player.set_speed_control_enabled(False)
        self.sequence_tab.video_player.set_chromatic_color_bar(None)
        self.clip_details_sidebar.video_player.mute = True

        preview_key = f"preview:{self._rendered_sequence_preview_path}"
        if self._sequence_preview_source_id != preview_key:
            self._sequence_preview_source_id = preview_key
            self.sequence_tab.video_player.clear_clip_range()
            self.sequence_tab.video_player.load_video(self._rendered_sequence_preview_path)
        self.sequence_tab.video_player.seek_to(start_seconds)
        self.sequence_tab.video_player.play()

    def _on_playback_requested(self, start_frame: int):
        """Start sequence playback from given frame."""
        if self._is_playing:
            # Toggle pause
            self._pause_playback()
            return

        sequence = self.sequence_tab.timeline.get_sequence()
        if sequence.duration_frames == 0:
            return  # Nothing to play

        if self._has_ready_sequence_preview():
            self._start_rendered_sequence_preview_playback(start_frame)
            return

        self._start_sequence_preview_render(play_after_frame=start_frame)
        return

    def _play_clip_at_frame(self, frame: int):
        """Load and play the clip at given timeline frame."""
        sequence = self.sequence_tab.timeline.get_sequence()

        # Check if we're past the end of sequence
        if frame >= sequence.duration_frames:
            self._stop_playback()
            # Reset playhead to beginning
            self.sequence_tab.timeline.set_playhead_time(0)
            return

        # Get clip at current position
        seq_clip, clip, source = self.sequence_tab.timeline.get_clip_at_playhead()

        if not seq_clip:
            # No clip at this position (gap) - show black and advance via timer
            self._current_playback_clip = None
            self._pending_sequence_playback_source_id = None
            self._pending_sequence_playback_range = None
            self.sequence_tab.video_player.stop()  # Shows black
            self._update_sequence_chromatic_bar(None)
            self._playback_timer.start()
            return

        self._current_playback_clip = seq_clip
        self._preview_sync_clip = seq_clip
        self._update_sequence_chromatic_bar(seq_clip)

        file_to_load, clip_start_seconds, clip_end_seconds, source_seconds = (
            _resolve_playback_source(seq_clip, source, frame)
        )

        preview_source_key = str(file_to_load)
        end_seconds = clip_end_seconds

        # Load source and play range.
        # If source changes (or is still loading), defer play_range until media_loaded.
        if self._sequence_preview_source_id != preview_source_key:
            self._sequence_preview_source_id = preview_source_key
            self._sequence_preview_loading = True
            self._pending_sequence_playback_source_id = preview_source_key
            self._pending_sequence_playback_range = (source_seconds, end_seconds)
            self.sequence_tab.video_player.load_video(file_to_load)
            return

        if self._sequence_preview_loading:
            self._pending_sequence_playback_source_id = preview_source_key
            self._pending_sequence_playback_range = (source_seconds, end_seconds)
            return

        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        self.sequence_tab.video_player.play_range(source_seconds, end_seconds)

        # Start timer to monitor for clip transitions
        self._playback_timer.start()

    def _on_playback_tick(self):
        """Called during playback to check for clip transitions and advance playhead in gaps."""
        if not self._is_playing:
            self._playback_timer.stop()
            return
        if getattr(self, "_using_rendered_sequence_preview", False):
            self._playback_timer.stop()
            return

        sequence = self.sequence_tab.timeline.get_sequence()
        current_time = self.sequence_tab.timeline.get_playhead_time()
        current_frame = int(current_time * sequence.fps)

        # Check if we're past the end of sequence
        if current_frame >= sequence.duration_frames:
            self._stop_playback()
            self.sequence_tab.timeline.set_playhead_time(0)
            return

        if self._current_playback_clip:
            # Playing a clip - check if we've moved past it
            if current_frame >= self._current_playback_clip.end_frame():
                # Move to next position
                next_frame = self._current_playback_clip.end_frame()
                self.sequence_tab.timeline.set_playhead_time(next_frame / sequence.fps)
                self._play_clip_at_frame(next_frame)
        else:
            # In a gap - advance playhead manually
            # Advance by ~1 frame worth of time (33ms at 30fps)
            new_time = current_time + (self._playback_timer.interval() / 1000.0)
            new_frame = int(new_time * sequence.fps)

            # Check if we've reached a clip
            self.sequence_tab.timeline.set_playhead_time(new_time)
            seq_clip, _, _ = self.sequence_tab.timeline.get_clip_at_playhead()

            if seq_clip:
                # Found a clip - start playing it
                self._play_clip_at_frame(new_frame)

    def _on_video_position_updated(self, position_ms: int):
        """Sync timeline playhead to video position during playback."""
        # Update GUI state for agent context
        if self._gui_state:
            clip_id = self._current_playback_clip.source_clip_id if self._current_playback_clip else None
            speed = self.sequence_tab.video_player.playback_speed
            self._gui_state.update_playback_state(
                position_ms=position_ms, clip_id=clip_id, speed=speed,
            )

        if getattr(self, "_using_rendered_sequence_preview", False):
            sequence = self.sequence_tab.timeline.get_sequence()
            timeline_seconds = position_ms / 1000.0
            if sequence.duration_seconds > 0:
                timeline_seconds = max(
                    0.0,
                    min(timeline_seconds, sequence.duration_seconds),
                )
            self._syncing_timeline_from_video = True
            try:
                self.sequence_tab.timeline.set_playhead_time(timeline_seconds)
            finally:
                self._syncing_timeline_from_video = False
            return

        # Case 1: Timeline-driven playback (existing behavior)
        if self._is_playing and self._current_playback_clip:
            seq_clip = self._current_playback_clip
            clip_data = self.sequence_tab.timeline._clip_lookup.get(seq_clip.source_clip_id)
            if not clip_data:
                return

            _, source = clip_data

            # Convert video position to source frame
            source_seconds = position_ms / 1000.0
            source_frame = int(source_seconds * source.fps)

            # Calculate timeline frame
            # timeline_frame = start_frame + (source_frame - in_point)
            frame_offset = source_frame - seq_clip.in_point
            timeline_frame = seq_clip.start_frame + frame_offset
            timeline_seconds = timeline_frame / self.sequence_tab.timeline.sequence.fps

            # Update playhead position
            self.sequence_tab.timeline.set_playhead_time(timeline_seconds)
            return

        # Case 2: Direct video playback (when user clicks play on VideoPlayer)
        # Only sync if video is actually playing
        if not self.sequence_tab.video_player.is_playing:
            return

        # Only sync when in timeline state (not cards state)
        if self.sequence_tab._current_state != self.sequence_tab.STATE_TIMELINE:
            return

        # Prefer clip-aware mapping when we know which sequence clip this preview
        # was synced to (e.g. after timeline scrubbing).
        seq_clip = self._preview_sync_clip
        if seq_clip:
            clip_data = self.sequence_tab.timeline._clip_lookup.get(seq_clip.source_clip_id)
            if clip_data:
                _, source = clip_data
                timeline_seconds = _source_ms_to_timeline_seconds(
                    seq_clip,
                    position_ms,
                    source.fps,
                    self.sequence_tab.timeline.sequence.fps,
                )
                self._syncing_timeline_from_video = True
                try:
                    self.sequence_tab.timeline.set_playhead_time(timeline_seconds)
                finally:
                    self._syncing_timeline_from_video = False
                return

        # Fallback when no clip context is available.
        video_seconds = position_ms / 1000.0
        self._syncing_timeline_from_video = True
        try:
            self.sequence_tab.timeline.set_playhead_time(video_seconds)
        finally:
            self._syncing_timeline_from_video = False

    def _on_video_state_changed(self, playing: bool):
        """Handle video player state changes.

        Args:
            playing: True if playing, False if paused/stopped
        """
        logger.debug(f"Video state changed: playing={playing}, is_playing: {self._is_playing}")

        # Update GUI state for agent context
        if self._gui_state:
            self._gui_state.update_playback_state(is_playing=playing)

        if not self._is_playing:
            return

        if getattr(self, "_using_rendered_sequence_preview", False):
            if not playing:
                self._is_playing = False
                self.sequence_tab.video_player.set_playing(False)
                self.clip_details_sidebar.video_player.mute = False
                self.sequence_tab.video_player.set_speed_control_enabled(True)
            return

        # Ignore pause events during source loading — mpv.pause=True is set
        # synchronously in load_video(), which fires this callback before the
        # new file is ready. Without this guard, we'd prematurely advance to
        # the next clip and overwrite pending playback state.
        if self._sequence_preview_loading:
            return

        if not playing:
            # Clip ended naturally (stopped/paused) - check if we should continue to next
            if self._current_playback_clip:
                next_frame = self._current_playback_clip.end_frame()
                self.sequence_tab.timeline.set_playhead_time(
                    next_frame / self.sequence_tab.timeline.sequence.fps
                )
                self._play_clip_at_frame(next_frame)

    def _pause_playback(self):
        """Pause playback."""
        self._is_playing = False
        self._using_rendered_sequence_preview = False
        self._playback_timer.stop()
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        self.sequence_tab.video_player.pause()
        self.sequence_tab.video_player.set_playing(False)
        # Restore sidebar audio and speed control
        self.clip_details_sidebar.video_player.mute = False
        self.sequence_tab.video_player.set_speed_control_enabled(True)
        self._update_sequence_chromatic_bar()

    def _on_stop_requested(self):
        """Handle stop request from video player."""
        self._stop_playback()

    def _stop_playback(self):
        """Stop playback and reset state."""
        self._is_playing = False
        self._using_rendered_sequence_preview = False
        self._playback_timer.stop()
        self._current_playback_clip = None
        self._preview_sync_clip = None
        self._pending_sequence_playback_source_id = None
        self._pending_sequence_playback_range = None
        self._pending_sequence_preview_source_id = None
        self._pending_sequence_preview_clip_range = None
        self._pending_sequence_preview_seek_seconds = None
        self.sequence_tab.video_player.stop()
        self.sequence_tab.video_player.set_playing(False)
        # Restore sidebar audio and speed control
        self.clip_details_sidebar.video_player.mute = False
        self.sequence_tab.video_player.set_speed_control_enabled(True)
        self._update_sequence_chromatic_bar()

    def _on_export_click(self):
        """Export selected clips."""
        selected = self.cut_tab.clip_browser.get_selected_clips()
        if not selected:
            QMessageBox.information(self, "Export", "No clips selected")
            return
        self._export_clips(selected)

    def _on_export_all_click(self):
        """Export all enabled clips."""
        enabled = [c for c in self.clips if not c.disabled]
        if not enabled:
            return
        self._export_clips(enabled)

    def _on_export_dataset_click(self):
        """Export clip metadata to JSON file."""
        if not self.current_source or not self.clips:
            QMessageBox.information(self, "Export Dataset", "No clips available to export")
            return

        # Get default filename
        source_name = self._sanitize_filename(self.current_source.file_path.stem)
        default_name = f"{source_name}_dataset.json"

        # Show save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Dataset",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        # Configure and export
        config = DatasetExportConfig(
            output_path=output_path,
            include_thumbnails=True,
            pretty_print=True,
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        def on_progress(progress: float, message: str):
            self.progress_bar.setValue(int(progress * 100))
            self.status_bar.showMessage(message)

        enabled_clips = [c for c in self.clips if not c.disabled]
        success = export_dataset(self.current_source, enabled_clips, config, on_progress)

        self.progress_bar.setVisible(False)

        if success:
            self.status_bar.showMessage(f"Dataset exported to {output_path.name}")
            # Open containing folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))
        else:
            QMessageBox.critical(self, "Export Error", "Failed to export dataset")

    def _on_export_srt_click(self):
        """Export sequence metadata as SRT subtitle file."""
        sequence = self.sequence_tab.get_sequence()
        all_clips = sequence.get_all_clips()

        if not all_clips:
            QMessageBox.information(self, "Export SRT", "No clips in timeline to export")
            return

        # Build lookups
        sources = self.sequence_tab.timeline.get_sources_lookup()
        clips_lookup = {}
        for clip in self.clips:
            clips_lookup[clip.id] = clip

        # Fallback to project data
        if not sources and self.sources_by_id:
            sources = dict(self.sources_by_id)

        # Get default filename from sequence name
        default_name = f"{sequence.name}.srt"
        if sequence.name == "Untitled Sequence" and self.project_metadata:
            default_name = f"{self.project_metadata.name}.srt"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export SRT",
            default_name,
            "SRT Subtitle Files (*.srt);;All Files (*)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".srt")

        config = SRTExportConfig(output_path=output_path)
        success, exported, skipped = export_srt(
            sequence, clips_lookup, sources, config
        )

        if success:
            if exported == 0:
                # All clips skipped - warn user
                QMessageBox.warning(
                    self,
                    "Export SRT",
                    f"No clips have the required metadata for this sequence type.\n"
                    f"An empty SRT file was created.\n\n"
                    f"Skipped: {skipped} clips",
                )
            else:
                self.status_bar.showMessage(
                    f"SRT exported: {exported} entries ({skipped} skipped)"
                )
        else:
            QMessageBox.critical(self, "Export Error", "Failed to export SRT file")

    # ------------------------------------------------------------------
    # Frames tab handlers
    # ------------------------------------------------------------------

    def _on_extract_frames_requested(self, source_id: str, mode: str, interval: int):
        """Launch frame extraction worker for the given source."""
        source = self.sources_by_id.get(source_id)
        if not source:
            self.status_bar.showMessage("Source not found")
            return

        # Output dir for extracted frames
        output_dir = self.project.path.parent / "frames" / source_id if self.project.path else Path.home() / ".cache" / "scene-ripper" / "frames" / source_id

        from ui.workers.frame_extraction_worker import FrameExtractionWorker

        worker = FrameExtractionWorker(
            source=source,
            clip=None,
            mode=mode,
            interval=interval,
            output_dir=output_dir,
        )
        worker.progress.connect(
            lambda cur, tot: self.status_bar.showMessage(
                f"Extracting frames: {cur}/{tot}"
            )
        )
        worker.extraction_completed.connect(
            lambda frames: self._on_frames_extracted(frames, source_id)
        )
        worker.error.connect(
            lambda msg: self.status_bar.showMessage(f"Extraction error: {msg}")
        )

        self._frame_extraction_worker = worker
        worker.start()
        self.status_bar.showMessage("Extracting frames...")

    def _on_frames_extracted(self, frames: list, source_id: str):
        """Handle completed frame extraction."""
        if not frames:
            self.status_bar.showMessage("No frames extracted")
            return

        self.project.add_frames(frames)
        self.frames_tab.update_frame_browser()
        self.status_bar.showMessage(f"Extracted {len(frames)} frames")
        self._mark_dirty()

    def _on_import_images_requested(self, paths: list):
        """Import image files as Frame objects."""
        from models.frame import Frame
        from core.thumbnail import generate_image_thumbnail

        frames = []
        thumb_dir = self.project.path.parent / "thumbnails" if self.project.path else Path.home() / ".cache" / "scene-ripper" / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        for image_path in paths:
            frame = Frame(file_path=image_path)

            # Generate thumbnail
            thumb_path = thumb_dir / f"{frame.id}_thumb.jpg"
            try:
                generate_image_thumbnail(image_path, thumb_path)
                frame.thumbnail_path = thumb_path
            except Exception:
                pass

            frames.append(frame)

        if frames:
            self.project.add_frames(frames)
            self.frames_tab.update_frame_browser()
            self.status_bar.showMessage(f"Imported {len(frames)} images")
            self._mark_dirty()

    def _on_analyze_frames_requested(self, frame_ids: list):
        """Analyze selected frames using AnalysisTarget-based pipeline.

        Converts Frame objects to AnalysisTarget instances and runs them
        through the same workers used for clip analysis. Excludes
        transcription (no audio in still images).
        """
        from core.analysis_target import AnalysisTarget

        if not frame_ids:
            return

        frames = [
            self.project.frames_by_id.get(fid)
            for fid in frame_ids
            if fid in self.project.frames_by_id
        ]
        if not frames:
            self.status_bar.showMessage("No valid frames found for analysis")
            return

        targets = [AnalysisTarget.from_frame(f) for f in frames]

        # Run all frame-compatible operations (exclude transcribe - no audio)
        frame_ops = ["colors", "shots", "classify", "detect_objects", "extract_text"]
        self._run_frame_analysis(targets, frame_ops)

    def _run_frame_analysis(self, targets: list, operations: list[str]):
        """Run analysis operations on AnalysisTarget objects.

        Launches workers with the analysis_targets parameter instead of clips.
        Reuses existing result handlers which now support frame write-back.

        Args:
            targets: List of AnalysisTarget objects
            operations: List of operation keys to run
        """
        if not targets or not operations:
            return

        operations = self._filter_available_analysis_operations(operations)
        if not operations:
            return

        for op_key in operations:
            self._reset_analysis_run_error(op_key)

        logger.info(
            f"Starting frame analysis: {operations} on {len(targets)} targets"
        )
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(
            f"Analyzing {len(targets)} frames..."
        )

        # Track running workers for completion
        self._frame_analysis_remaining = len(operations)
        self._frame_analysis_targets = targets
        self._frame_analysis_ops = list(operations)

        for op_key in operations:
            self._launch_frame_analysis_worker(op_key, targets)

    def _launch_frame_analysis_worker(self, op_key: str, targets: list):
        """Launch a worker for frame analysis using AnalysisTarget objects.

        Args:
            op_key: Operation key (e.g., "colors", "shots")
            targets: List of AnalysisTarget objects
        """
        if op_key == "colors":
            from ui.workers.color_worker import ColorAnalysisWorker
            worker = ColorAnalysisWorker(
                clips=[],
                analysis_targets=targets,
                parallelism=self.settings.color_analysis_parallelism,
            )
            worker.progress.connect(self._on_color_progress)
            worker.color_ready.connect(self._on_color_ready)
            worker.error.connect(self._on_color_error)
            worker.analysis_completed.connect(
                lambda: self._on_frame_analysis_op_finished("colors")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_color_worker = worker
            worker.start()

        elif op_key == "shots":
            from ui.workers.shot_type_worker import ShotTypeWorker
            worker = ShotTypeWorker(
                clips=[],
                sources_by_id={},
                analysis_targets=targets,
                parallelism=self.settings.local_model_parallelism,
            )
            worker.progress.connect(self._on_shot_type_progress)
            worker.shot_type_ready.connect(self._on_shot_type_ready)
            worker.error.connect(self._on_shot_type_error)
            worker.analysis_completed.connect(
                lambda: self._on_frame_analysis_op_finished("shots")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_shot_worker = worker
            worker.start()

        elif op_key == "classify":
            from ui.workers.classification_worker import ClassificationWorker
            worker = ClassificationWorker(
                clips=[],
                analysis_targets=targets,
                parallelism=self.settings.local_model_parallelism,
            )
            worker.progress.connect(self._on_classification_progress)
            worker.labels_ready.connect(self._on_classification_ready)
            worker.error.connect(self._on_classification_error)
            worker.classification_completed.connect(
                lambda: self._on_frame_analysis_op_finished("classify")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_classify_worker = worker
            worker.start()

        elif op_key == "detect_objects":
            from ui.workers.object_detection_worker import ObjectDetectionWorker
            worker = ObjectDetectionWorker(
                clips=[],
                analysis_targets=targets,
                parallelism=self.settings.local_model_parallelism,
            )
            worker.progress.connect(self._on_object_detection_progress)
            worker.objects_ready.connect(self._on_objects_ready)
            worker.error.connect(self._on_object_detection_error)
            worker.detection_completed.connect(
                lambda: self._on_frame_analysis_op_finished("detect_objects")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_detect_worker = worker
            worker.start()

        elif op_key == "extract_text":
            from ui.workers.text_extraction_worker import TextExtractionWorker
            worker = TextExtractionWorker(
                clips=[],
                sources_by_id={},
                analysis_targets=targets,
            )
            worker.progress.connect(self._on_text_extraction_progress)
            worker.clip_completed.connect(self._on_text_extraction_clip_ready)
            worker.finished.connect(
                lambda _: self._on_frame_analysis_op_finished("extract_text")
            )
            worker.error.connect(self._on_text_extraction_error)
            worker.finished.connect(worker.deleteLater)
            self._frame_text_worker = worker
            worker.start()

        elif op_key == "describe":
            from ui.workers.description_worker import DescriptionWorker
            worker = DescriptionWorker(
                clips=[],
                analysis_targets=targets,
                parallelism=self.settings.description_parallelism,
            )
            worker.progress.connect(self._on_description_progress)
            worker.description_ready.connect(self._on_description_ready)
            worker.error.connect(self._on_description_error)
            worker.description_completed.connect(
                lambda: self._on_frame_analysis_op_finished("describe")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_desc_worker = worker
            worker.start()

        elif op_key == "cinematography":
            from ui.workers.cinematography_worker import CinematographyWorker
            worker = CinematographyWorker(
                clips=[],
                sources_by_id={},
                analysis_targets=targets,
                parallelism=min(self.settings.description_parallelism, 2),
            )
            worker.progress.connect(self._on_cinematography_progress)
            worker.clip_completed.connect(self._on_cinematography_clip_ready)
            worker.error.connect(self._on_cinematography_error)
            worker.finished.connect(
                lambda _: self._on_frame_analysis_op_finished("cinematography")
            )
            worker.finished.connect(worker.deleteLater)
            self._frame_cine_worker = worker
            worker.start()

        else:
            logger.warning(f"Unknown frame analysis operation: {op_key}")
            self._on_frame_analysis_op_finished(op_key)

    def _on_frame_analysis_op_finished(self, op_key: str):
        """Handle completion of a single frame analysis operation."""
        self._frame_analysis_remaining -= 1
        logger.info(
            f"Frame analysis op '{op_key}' finished "
            f"({self._frame_analysis_remaining} remaining)"
        )
        if self._frame_analysis_remaining <= 0:
            self._on_frame_analysis_complete()

    def _on_frame_analysis_complete(self):
        """Handle completion of all frame analysis operations."""
        logger.info("Frame analysis complete")
        self.progress_bar.setVisible(False)
        completed_ops = getattr(self, "_frame_analysis_ops", [])
        error_labels = self._get_completed_analysis_error_labels(completed_ops)
        if error_labels:
            self.status_bar.showMessage(
                f"Frame analysis finished with errors ({', '.join(error_labels)})",
                5000,
            )
            self._show_completed_analysis_error_dialog(completed_ops)
        else:
            self.status_bar.showMessage("Frame analysis complete", 3000)

        # Mark analyzed frames
        targets = getattr(self, '_frame_analysis_targets', [])
        for target in targets:
            self.project.update_frame(target.id, analyzed=True)

        # Refresh the frame browser to show updated metadata
        self.frames_tab.update_frame_browser()
        self._mark_dirty()

        # Save project
        if self.project.path:
            self.project.save()

    def _on_add_frames_to_sequence(self, frame_ids: list):
        """Add selected frames to the active sequence."""
        if not frame_ids:
            return

        sequence = self.sequence_tab.get_sequence()
        self.project.add_frames_to_sequence(frame_ids, sequence)
        self.sequence_tab.timeline.refresh()
        self.status_bar.showMessage(f"Added {len(frame_ids)} frames to sequence")
        self._mark_dirty()

    def _on_frames_selection_changed(self, frame_ids: list):
        """Update GUI state when frame selection changes."""
        self._gui_state.selected_frame_ids = frame_ids

    def _default_clip_export_filename(
        self,
        clip: Clip,
        source: Source,
        ordinal: int | None = None,
    ) -> str:
        """Build a stable default filename for a clip export."""
        source_name = self._sanitize_filename(source.file_path.stem)
        if clip.name:
            clip_name = self._sanitize_filename(clip.name)
        else:
            clip_name = f"scene_{clip.start_frame:06d}_{clip.end_frame:06d}"

        parts = [source_name]
        if ordinal is not None:
            parts.append(f"{ordinal:03d}")
        parts.append(clip_name)
        return "_".join(parts) + ".mp4"

    def _export_clip_to_path(self, clip: Clip, source: Source, output_path: Path) -> bool:
        """Export a single clip to an explicit output path."""
        logger.info(
            "Manual clip export requested: clip=%s source=%s output=%s start_frame=%s end_frame=%s",
            clip.id,
            source.file_path,
            output_path,
            clip.start_frame,
            clip.end_frame,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor = FFmpegProcessor()
        success = processor.extract_clip(
            input_path=source.file_path,
            output_path=output_path,
            start_seconds=clip.start_time(source.fps),
            duration_seconds=clip.duration_seconds(source.fps),
            fps=source.fps,
        )
        if not success:
            logger.error(
                "Manual clip export failed: clip=%s source=%s output=%s",
                clip.id,
                source.file_path,
                output_path,
            )
        return success

    def _on_clip_export_requested(self, clip: Clip, source: Source):
        """Export the clicked clip only."""
        default_name = self._default_clip_export_filename(clip, source)
        default_path = str(self.settings.export_dir / default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Clip",
            default_path,
            "Video Files (*.mp4);;All Files (*)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        success = self._export_clip_to_path(clip, source, output_path)

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        if success:
            self.status_bar.showMessage(f"Exported clip to {output_path.name}", 5000)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))
        else:
            QMessageBox.critical(self, "Export Error", "Failed to export clip")

    def _export_clips(self, clips: list[Clip]):
        """Export clips to a folder."""
        if not clips:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Folder"
        )
        if not output_dir:
            return

        output_path = Path(output_dir)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(clips))

        exported = 0
        for i, clip in enumerate(clips):
            source = self.sources_by_id.get(clip.source_id)
            if source is None:
                self.progress_bar.setValue(i + 1)
                continue

            output_file = output_path / self._default_clip_export_filename(
                clip, source, ordinal=i + 1
            )
            success = self._export_clip_to_path(clip, source, output_file)
            if success:
                exported += 1

            self.progress_bar.setValue(i + 1)

        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Exported {exported}/{len(clips)} clips to {output_dir}")

        # Open the export folder in system file browser
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def _update_render_tab_sequence_info(self):
        """Update the Render tab with current sequence information."""
        sequence = self.sequence_tab.get_sequence()
        all_clips = sequence.get_all_clips()
        duration_seconds = sequence.duration_frames / sequence.fps if sequence.fps > 0 else 0
        self.render_tab.set_sequence_info(duration_seconds, len(all_clips))

    def _on_sequence_export_click(self):
        """Export the timeline sequence to a single video file."""
        # Check if export is already running
        if self.export_worker and self.export_worker.isRunning():
            QMessageBox.warning(
                self, "Export in Progress",
                "An export is already running. Please wait for it to complete."
            )
            return

        # Use the SequenceTab's timeline, not the legacy one
        sequence = self.sequence_tab.get_sequence()
        all_clips = sequence.get_all_clips()

        if not all_clips:
            QMessageBox.information(self, "Export Sequence", "No clips in timeline to export")
            return

        # Get output file path
        default_name = "sequence_export.mp4"
        if self.current_source:
            default_name = f"{self._sanitize_filename(self.current_source.file_path.stem)}_remix.mp4"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sequence",
            default_name,
            "Video Files (*.mp4);;All Files (*)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")

        if not self._validate_export_output_path(output_path):
            return

        logger.info(
            "Manual sequence export requested: output=%s clip_count=%d",
            output_path,
            len(all_clips),
        )

        # Build sources and clips dictionaries from the timeline's actual content
        # (not from self.current_source which may be different)
        sources = self.sequence_tab.timeline.get_sources_lookup()
        clips = self.sequence_tab.timeline.get_clips_lookup()

        # Fallback to project sources/clips if timeline lookups are empty
        if not sources and self.sources_by_id:
            sources = dict(self.sources_by_id)
        if not clips:
            for clip in self.clips:
                source = self.sources_by_id.get(clip.source_id)
                if source:
                    clips[clip.id] = (clip, source)

        # Get quality settings from RenderTab UI (not global settings)
        quality_setting = self.render_tab.get_quality_setting()
        resolution = self.render_tab.get_resolution_setting()
        target_fps = self.render_tab.get_fps_setting()

        # Map quality string to preset
        quality_presets = {
            "high": {"crf": 18, "preset": "slow", "bitrate": "8M"},
            "medium": {"crf": 23, "preset": "medium", "bitrate": "4M"},
            "low": {"crf": 28, "preset": "fast", "bitrate": "2M"},
        }
        quality_preset = quality_presets.get(quality_setting, quality_presets["medium"])

        # Resolve music_path if the sequence has one
        music_path = None
        raw_music = getattr(sequence, "music_path", None)
        if raw_music:
            p = Path(raw_music)
            if p.exists():
                music_path = p

        config = ExportConfig(
            output_path=output_path,
            fps=target_fps if target_fps else sequence.fps,
            width=resolution[0] if resolution else None,
            height=resolution[1] if resolution else None,
            crf=quality_preset["crf"],
            preset=quality_preset["preset"],
            video_bitrate=quality_preset["bitrate"],
            show_chromatic_color_bar=(
                bool(getattr(sequence, "show_chromatic_color_bar", False))
                and sequence.algorithm == "color"
            ),
            music_path=music_path,
        )

        # Start export in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.sequence_tab.timeline.export_btn.setEnabled(False)

        self.export_worker = SequenceExportWorker(sequence, sources, clips, config)
        self.export_worker.progress.connect(self._on_sequence_export_progress)
        self.export_worker.export_completed.connect(self._on_sequence_export_finished)
        self.export_worker.error.connect(self._on_sequence_export_error)
        # Clean up thread safely after it finishes
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_worker.finished.connect(lambda: setattr(self, 'export_worker', None))
        self.export_worker.start()

    def start_agent_export(self, sequence, sources: dict, clips: dict, config) -> bool:
        """Start a sequence export triggered by agent.

        Args:
            sequence: Sequence to export
            sources: Dict of source_id -> Source
            clips: Dict of clip_id -> (Clip, Source)
            config: ExportConfig

        Returns:
            True if export started, False if already in progress
        """
        # Check if export already running
        if self.export_worker and self.export_worker.isRunning():
            return False

        # Mark that agent is waiting
        self._pending_agent_export = True

        # Start export in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.sequence_tab.timeline.export_btn.setEnabled(False)

        self.export_worker = SequenceExportWorker(sequence, sources, clips, config)
        self.export_worker.progress.connect(self._on_sequence_export_progress)
        self.export_worker.export_completed.connect(self._on_sequence_export_finished)
        self.export_worker.error.connect(self._on_sequence_export_error)
        # Clean up thread safely after it finishes
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_worker.finished.connect(lambda: setattr(self, 'export_worker', None))
        self.export_worker.start()
        return True

    def start_agent_export_bundle(self, dest_dir: Path, include_videos: bool = True) -> bool:
        """Start a bundle export triggered by agent.

        Args:
            dest_dir: Destination directory for the bundle
            include_videos: Whether to include source video files

        Returns:
            True if export started, False if already in progress
        """
        # Check if bundle export already running
        if self.export_bundle_worker and self.export_bundle_worker.isRunning():
            return False

        # Mark that agent is waiting
        self._pending_agent_export_bundle = True

        # Start background export
        self.status_bar.showMessage("Exporting project bundle...")

        self.export_bundle_worker = ExportBundleWorker(
            project=self.project,
            dest_dir=dest_dir,
            include_videos=include_videos,
            parent=self,
        )
        self.export_bundle_worker.progress.connect(self._on_export_bundle_progress)
        self.export_bundle_worker.export_completed.connect(self._on_export_bundle_finished)
        self.export_bundle_worker.error.connect(self._on_export_bundle_error)
        self.export_bundle_worker.start()
        return True

    def _on_sequence_export_progress(self, progress: float, message: str):
        """Handle sequence export progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)

    def _on_sequence_export_finished(self, output_path: Path):
        """Handle sequence export completion."""
        self.progress_bar.setVisible(False)
        self.sequence_tab.timeline.export_btn.setEnabled(True)
        self.status_bar.showMessage(f"Sequence exported to {output_path.name}")

        # If agent was waiting for export, send result back
        if self._pending_agent_export and self._chat_worker:
            self._pending_agent_export = False
            sequence = self.sequence_tab.get_sequence()
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "output_path": str(output_path),
                    "clip_count": len(sequence.get_all_clips()) if sequence else 0,
                    "message": f"Exported to {output_path.name}"
                }
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent export result to agent: {output_path}")
        else:
            # Only open folder for manual exports
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))

    def _on_sequence_export_error(self, error: str):
        """Handle sequence export error."""
        self._gui_state.set_last_error(f"Export error: {error}")
        self.progress_bar.setVisible(False)
        self.sequence_tab.timeline.export_btn.setEnabled(True)

        # If agent was waiting for export, send error result
        if self._pending_agent_export and self._chat_worker:
            self._pending_agent_export = False
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": False,
                "error": error
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent export error to agent: {error}")
        else:
            # Only show dialog for manual exports
            QMessageBox.critical(self, "Export Error", f"Failed to export sequence: {error}")

    def _validate_export_output_path(self, output_path: Path) -> bool:
        """Validate that an export destination can be written."""
        parent = output_path.parent

        if not parent.exists():
            logger.error(
                "Manual export rejected: destination folder does not exist: output=%s parent=%s",
                output_path,
                parent,
            )
            QMessageBox.warning(
                self,
                "Export Sequence",
                f"Export folder does not exist:\n{parent}\n\nChoose a different location.",
            )
            return False

        if not parent.is_dir():
            logger.error(
                "Manual export rejected: destination parent is not a folder: output=%s parent=%s",
                output_path,
                parent,
            )
            QMessageBox.warning(
                self,
                "Export Sequence",
                f"Export folder is invalid:\n{parent}\n\nChoose a different location.",
            )
            return False

        if not os.access(parent, os.W_OK):
            logger.error(
                "Manual export rejected: destination folder is not writable: output=%s parent=%s",
                output_path,
                parent,
            )
            QMessageBox.warning(
                self,
                "Export Sequence",
                f"Cannot write to export folder:\n{parent}\n\nChoose a different location.",
            )
            return False

        return True

    def _validate_download_directory(self, download_dir: Path) -> Optional[Path]:
        """Validate download directory and prompt user to fix if invalid.

        If the configured directory doesn't exist and can't be created, shows a dialog
        allowing the user to select a new directory or use the default.

        Args:
            download_dir: The configured download directory path

        Returns:
            A valid Path if validation succeeds (possibly different from input),
            or None if the user cancels the operation.
        """
        # Try to validate the directory
        valid, error_msg = validate_download_dir(download_dir)
        if valid:
            return download_dir

        # Directory is invalid - check if it's from an environment variable
        from_env = is_download_dir_from_env()

        if from_env:
            # Show info dialog explaining the env var situation
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Download Directory Unavailable")
            msg_box.setText(
                f"The download directory set via SCENE_RIPPER_DOWNLOAD_DIR "
                f"environment variable is unavailable:\n\n{download_dir}\n\n{error_msg}"
            )
            msg_box.setInformativeText(
                "Would you like to use the default directory for this session? "
                "(To permanently fix this, update your environment variable.)"
            )
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

            result = msg_box.exec()
            if result == QMessageBox.StandardButton.Yes:
                # Use default temporarily (don't save)
                default_dir = get_default_download_dir()
                valid, default_error = validate_download_dir(default_dir)
                if valid:
                    logger.info(f"Using temporary default download dir: {default_dir}")
                    return default_dir
                else:
                    QMessageBox.critical(
                        self,
                        "Default Directory Also Unavailable",
                        f"Could not create default directory:\n{default_dir}\n\n{default_error}"
                    )
                    return None
            else:
                return None

        # Not from env var - show dialog with options
        while True:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Download Directory Unavailable")
            msg_box.setText(
                f"The configured download directory is unavailable:\n\n"
                f"{download_dir}\n\n{error_msg}"
            )
            msg_box.setInformativeText("Please select a new directory or use the default.")

            # Add custom buttons
            select_btn = msg_box.addButton("Select Directory...", QMessageBox.ButtonRole.ActionRole)
            default_btn = msg_box.addButton("Use Default", QMessageBox.ButtonRole.ActionRole)
            cancel_btn = msg_box.addButton(QMessageBox.StandardButton.Cancel)
            msg_box.setDefaultButton(select_btn)

            msg_box.exec()
            clicked = msg_box.clickedButton()

            if clicked == cancel_btn:
                return None

            elif clicked == default_btn:
                default_dir = get_default_download_dir()
                valid, default_error = validate_download_dir(default_dir)
                if valid:
                    # Save the new setting
                    self.settings.download_dir = default_dir
                    save_settings(self.settings)
                    logger.info(f"Updated download directory to default: {default_dir}")
                    return default_dir
                else:
                    QMessageBox.critical(
                        self,
                        "Default Directory Unavailable",
                        f"Could not create default directory:\n{default_dir}\n\n{default_error}"
                    )
                    # Continue the loop to let user try again

            elif clicked == select_btn:
                # Show directory picker
                new_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Download Directory",
                    str(Path.home()),
                    QFileDialog.Option.ShowDirsOnly
                )
                if not new_dir:
                    # User cancelled picker - return to dialog
                    continue

                new_path = Path(new_dir)
                valid, new_error = validate_download_dir(new_path)
                if valid:
                    # Save the new setting
                    self.settings.download_dir = new_path
                    save_settings(self.settings)
                    logger.info(f"Updated download directory to: {new_path}")
                    return new_path
                else:
                    QMessageBox.warning(
                        self,
                        "Directory Not Valid",
                        f"Cannot use selected directory:\n{new_path}\n\n{new_error}"
                    )
                    # Continue the loop to let user try again

    def start_agent_bulk_download(self, urls: list[str], download_dir: Path) -> bool:
        """Start bulk video downloads triggered by agent.

        Args:
            urls: List of video URLs to download
            download_dir: Directory to save downloads

        Returns:
            True if download started, False if already in progress or cancelled
        """
        # Check if download already running
        if self.url_bulk_download_worker and self.url_bulk_download_worker.isRunning():
            return False

        if not self._ensure_video_download_available():
            return False

        # Validate download directory
        validated_dir = self._validate_download_directory(download_dir)
        if validated_dir is None:
            # User cancelled - download not started
            return False

        # Mark that agent is waiting
        self._pending_agent_download = True
        self._agent_download_results = []

        # Start download in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(urls))
        self.status_bar.showMessage(f"Downloading {len(urls)} videos...")

        self.url_bulk_download_worker = URLBulkDownloadWorker(urls, validated_dir)
        self.url_bulk_download_worker.progress.connect(self._on_agent_download_progress)
        self.url_bulk_download_worker.video_finished.connect(self._on_agent_video_finished)
        self.url_bulk_download_worker.all_finished.connect(self._on_agent_bulk_download_finished)
        # Clean up thread safely after it finishes to prevent "QThread: Destroyed while running" crash
        self.url_bulk_download_worker.finished.connect(self.url_bulk_download_worker.deleteLater)
        self.url_bulk_download_worker.finished.connect(lambda: setattr(self, 'url_bulk_download_worker', None))
        self.url_bulk_download_worker.start()
        return True

    def _on_agent_download_progress(self, current: int, total: int, message: str):
        """Handle bulk download progress update."""
        self.progress_bar.setValue(current)
        self.status_bar.showMessage(message)

    def _on_agent_video_finished(self, url: str, result):
        """Handle individual video download completion."""
        # Add to project and source browser
        if result.success and result.file_path and hasattr(self, 'collect_tab'):
            from pathlib import Path
            file_path = Path(result.file_path)

            # Check if already in project
            for existing in self.project.sources:
                if existing.file_path == file_path:
                    logger.info(f"Source already in project: {file_path.name}")
                    return

            # Create source with metadata and add to project
            source = self._create_source_with_metadata(file_path)
            self.project.add_source(source)

            # Add to CollectTab grid
            self.collect_tab.add_source(source)

            # Generate thumbnail for the source
            self._generate_source_thumbnail(source)

            logger.info(f"Added downloaded source to project: {file_path.name} ({source.duration_seconds:.1f}s)")

    def _on_agent_bulk_download_finished(self, results: list):
        """Handle bulk download completion."""
        logger.info(f"Bulk download finished signal received: {len(results)} results, pending_agent_download={self._pending_agent_download}")
        self.progress_bar.setVisible(False)
        success_count = sum(1 for r in results if r.get("success"))
        self.status_bar.showMessage(f"Downloaded {success_count}/{len(results)} videos")

        # If agent was waiting, send result back
        if self._pending_agent_download and self._chat_worker:
            self._pending_agent_download = False
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": success_count > 0,
                "result": {
                    "success": success_count > 0,
                    "message": f"Downloaded {success_count} of {len(results)} videos",
                    "success_count": success_count,
                    "failed_count": len(results) - success_count,
                    "results": results,
                }
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent bulk download result to agent: {success_count}/{len(results)}")

    def _start_worker_for_tool(self, wait_type: str, tool_result: dict) -> bool:
        """Start the appropriate worker based on tool's _wait_for_worker type.

        Args:
            wait_type: Worker type from tool result
            tool_result: Full tool result dict with parameters

        Returns:
            True if worker started, False otherwise
        """
        if wait_type == "color_analysis":
            clip_ids = tool_result.get("clip_ids", [])
            return self.start_agent_color_analysis(clip_ids)

        elif wait_type == "shot_analysis":
            clip_ids = tool_result.get("clip_ids", [])
            return self.start_agent_shot_analysis(clip_ids)

        elif wait_type == "transcription":
            clip_ids = tool_result.get("clip_ids", [])
            return self.start_agent_transcription(clip_ids)

        elif wait_type == "export":
            # Export worker is started by the tool itself via start_agent_export
            # Just return True since the worker is already running
            return True

        elif wait_type == "export_bundle":
            # Bundle export worker is started by the tool itself via start_agent_export_bundle
            # Just return True since the worker is already running
            return True

        elif wait_type == "download":
            # Download worker is started by the tool itself via start_agent_bulk_download
            return True

        elif wait_type == "detection":
            # Detection was previously started eagerly in the tool; now handled here
            source_id = tool_result.get("source_id")
            mode = tool_result.get("mode", "adaptive")
            config = tool_result.get("config", {})

            # Find and select source
            source = self.project.sources_by_id.get(source_id)
            if not source:
                return False
            self._select_source(source)

            # Mark agent waiting
            self._pending_agent_detection = True

            # Start detection
            self._start_detection(mode, config)

            # Switch to Cut tab
            self._switch_to_tab("cut")
            return True

        elif wait_type == "classification":
            clip_ids = tool_result.get("clip_ids", [])
            top_k = tool_result.get("top_k", 5)
            return self.start_agent_classification(clip_ids, top_k)

        elif wait_type == "object_detection":
            clip_ids = tool_result.get("clip_ids", [])
            confidence = tool_result.get("confidence", 0.5)
            return self.start_agent_object_detection(clip_ids, confidence, detect_all=True)

        elif wait_type == "person_detection":
            clip_ids = tool_result.get("clip_ids", [])
            return self.start_agent_object_detection(clip_ids, 0.5, detect_all=False)

        elif wait_type == "description":
            clip_ids = tool_result.get("clip_ids", [])
            tier = tool_result.get("tier")
            prompt = tool_result.get("prompt")
            return self.start_agent_description(clip_ids, tier, prompt)

        elif wait_type == "analyze_all":
            # analyze_all pipeline — previously started eagerly in tool, now handled here
            clip_ids = tool_result.get("clip_ids", [])
            operations = tool_result.get("operations", [])

            # Resolve clips
            clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
            clips = [c for c in clips if c is not None]
            if not clips:
                return False

            # Mark agent waiting
            self._pending_agent_analyze_all = True

            # Add clips to Analyze tab and switch
            self.analyze_tab.add_clips(clip_ids)
            self._switch_to_tab("analyze")

            # Start the pipeline
            self._run_analysis_pipeline(clips, operations)
            return True

        else:
            logger.warning(f"Unknown worker type: {wait_type}")
            return False

    def start_agent_color_analysis(self, clip_ids: list[str]) -> bool:
        """Start color analysis for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to analyze

        Returns:
            True if started, False if already running
        """
        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        # Check if worker already running
        if self.color_worker and self.color_worker.isRunning():
            return False

        # Reset guard
        self._color_analysis_finished_handled = False

        # Mark that we're waiting for color analysis via agent
        self._pending_agent_color_analysis = True
        self._agent_color_clips = clips
        self._reset_analysis_run_error("colors")

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        self.analyze_tab.set_analyzing(True, "colors")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Extracting colors from {len(clips)} clips...")

        # Start worker
        from PySide6.QtCore import Qt
        self.color_worker = ColorAnalysisWorker(clips, parallelism=self.settings.color_analysis_parallelism, sources_by_id=self.project.sources_by_id)
        self.color_worker.progress.connect(self._on_color_progress)
        self.color_worker.color_ready.connect(self._on_color_ready)
        self.color_worker.error.connect(self._on_color_error)
        self.color_worker.analysis_completed.connect(self._on_agent_color_analysis_finished, Qt.UniqueConnection)
        # Clean up thread safely after it finishes
        self.color_worker.finished.connect(self.color_worker.deleteLater)
        self.color_worker.finished.connect(lambda: setattr(self, 'color_worker', None))
        self.color_worker.start()

        return True

    def start_agent_shot_analysis(self, clip_ids: list[str]) -> bool:
        """Start shot type classification for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to analyze

        Returns:
            True if started, False if already running
        """
        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        if not self._ensure_analysis_operation_available("shots"):
            return False

        # Check if worker already running
        if self.shot_type_worker and self.shot_type_worker.isRunning():
            return False

        # Reset guard
        self._shot_type_finished_handled = False
        self._shot_type_run_error = None

        # Mark that we're waiting for shot analysis via agent
        self._pending_agent_shot_analysis = True
        self._agent_shot_clips = clips

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        self.analyze_tab.set_analyzing(True, "shots")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Classifying shot types for {len(clips)} clips...")

        # Start worker
        from PySide6.QtCore import Qt
        self.shot_type_worker = ShotTypeWorker(clips, self.project.sources_by_id, parallelism=self.settings.local_model_parallelism)
        self.shot_type_worker.progress.connect(self._on_shot_type_progress)
        self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
        self.shot_type_worker.error.connect(self._on_shot_type_error)
        self.shot_type_worker.analysis_completed.connect(self._on_agent_shot_analysis_finished, Qt.UniqueConnection)
        # Clean up thread safely after it finishes
        self.shot_type_worker.finished.connect(self.shot_type_worker.deleteLater)
        self.shot_type_worker.finished.connect(lambda: setattr(self, 'shot_type_worker', None))
        self.shot_type_worker.start()

        return True

    def start_agent_transcription(self, clip_ids: list[str]) -> bool:
        """Start transcription for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to transcribe

        Returns:
            True if started, False if already running or unavailable
        """
        if not self._ensure_analysis_operation_available("transcribe"):
            return False

        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        # Check if worker already running
        if self.transcription_worker and self.transcription_worker.isRunning():
            return False

        # Group clips by source
        clips_by_source: dict = {}
        for clip in clips:
            if clip.source_id not in clips_by_source:
                clips_by_source[clip.source_id] = []
            clips_by_source[clip.source_id].append(clip)

        # Build queue of (source, clips) for multi-source transcription
        source_queue = []
        for source_id, source_clips in clips_by_source.items():
            source = self.sources_by_id.get(source_id)
            if source:
                source_queue.append((source, source_clips))

        if not source_queue:
            return False

        # Store queue and all clips for sequential processing
        self._agent_transcription_source_queue = source_queue[1:]  # Remaining after first
        self._agent_transcription_clips = clips  # All clips for final result
        self._pending_agent_transcription = True

        # Reset guard
        self._transcription_finished_handled = False

        # Start with first source
        first_source, first_clips = source_queue[0]

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        self.analyze_tab.set_analyzing(True, "transcribe")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        sources_info = f" (source 1/{len(source_queue)})" if len(source_queue) > 1 else ""
        self.status_bar.showMessage(f"Transcribing {len(first_clips)} clips{sources_info}...")

        # Safely stop any existing worker before creating new one
        self._stop_worker_safely(self.transcription_worker, "Transcription")

        # Start worker for first source
        from PySide6.QtCore import Qt
        self.transcription_worker = TranscriptionWorker(
            first_clips,
            first_source,
            self.settings.transcription_model,
            self.settings.transcription_language,
            parallelism=self.settings.transcription_parallelism,
            backend=self.settings.transcription_backend,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
        self.transcription_worker.transcription_completed.connect(self._on_agent_transcription_finished, Qt.UniqueConnection)
        self.transcription_worker.error.connect(self._on_transcription_error)
        # Clean up thread safely after it finishes
        self.transcription_worker.finished.connect(self.transcription_worker.deleteLater)
        self.transcription_worker.finished.connect(lambda: setattr(self, 'transcription_worker', None))
        self.transcription_worker.start()

        return True

    def start_agent_classification(self, clip_ids: list[str], top_k: int = 5) -> bool:
        """Start frame classification for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to classify
            top_k: Number of top labels to return per clip

        Returns:
            True if started, False if already running
        """
        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        if not self._ensure_analysis_operation_available("classify"):
            return False

        # Check if worker already running
        if self.classification_worker and self.classification_worker.isRunning():
            return False

        # Reset guard
        self._classification_finished_handled = False
        self._reset_analysis_run_error("classify")

        # Mark that we're waiting for classification via agent
        self._pending_agent_classification = True
        self._agent_classification_clips = clips

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        self.analyze_tab.set_analyzing(True, "classify")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Classifying content in {len(clips)} clips...")

        # Start worker
        from PySide6.QtCore import Qt
        self.classification_worker = ClassificationWorker(clips, top_k=top_k, parallelism=self.settings.local_model_parallelism)
        self.classification_worker.progress.connect(self._on_classification_progress)
        self.classification_worker.labels_ready.connect(self._on_classification_ready)
        self.classification_worker.error.connect(self._on_classification_error)
        self.classification_worker.classification_completed.connect(self._on_agent_classification_finished, Qt.UniqueConnection)
        # Clean up thread safely after it finishes
        self.classification_worker.finished.connect(self.classification_worker.deleteLater)
        self.classification_worker.finished.connect(lambda: setattr(self, 'classification_worker', None))
        self.classification_worker.start()

        return True

    def start_agent_object_detection(self, clip_ids: list[str], confidence: float = 0.5, detect_all: bool = True) -> bool:
        """Start object detection for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to analyze
            confidence: Detection confidence threshold
            detect_all: True for all objects, False for people only

        Returns:
            True if started, False if already running
        """
        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        if not self._ensure_analysis_operation_available("detect_objects"):
            return False

        # Check if worker already running
        if self.detection_worker_yolo and self.detection_worker_yolo.isRunning():
            return False

        # Reset guard
        self._object_detection_finished_handled = False
        self._reset_analysis_run_error("detect_objects")

        # Mark that we're waiting for object detection via agent
        self._pending_agent_object_detection = True
        self._agent_object_detection_clips = clips
        self._agent_object_detection_all = detect_all

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        task_name = "objects" if detect_all else "people"
        self.analyze_tab.set_analyzing(True, task_name)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        msg = f"Detecting {'objects' if detect_all else 'people'} in {len(clips)} clips..."
        self.status_bar.showMessage(msg)

        # Start worker
        from PySide6.QtCore import Qt
        self.detection_worker_yolo = ObjectDetectionWorker(clips, confidence=confidence, detect_all=detect_all, parallelism=self.settings.local_model_parallelism)
        self.detection_worker_yolo.progress.connect(self._on_object_detection_progress)
        self.detection_worker_yolo.objects_ready.connect(self._on_objects_ready)
        self.detection_worker_yolo.error.connect(self._on_object_detection_error)
        self.detection_worker_yolo.detection_completed.connect(self._on_agent_object_detection_finished, Qt.UniqueConnection)
        # Clean up thread safely after it finishes
        self.detection_worker_yolo.finished.connect(self.detection_worker_yolo.deleteLater)
        self.detection_worker_yolo.finished.connect(lambda: setattr(self, 'detection_worker_yolo', None))
        self.detection_worker_yolo.start()

        return True

    def start_agent_description(self, clip_ids: list[str], tier: Optional[str] = None, prompt: Optional[str] = None) -> bool:
        """Start description generation for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to analyze
            tier: Model tier ('cpu', 'gpu', 'cloud')
            prompt: Custom prompt for the model

        Returns:
            True if started, False if already running
        """
        # Resolve clips
        clips = [self.project.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            return False

        if not self._ensure_analysis_operation_available("describe", description_tier=tier):
            return False

        # Check if worker already running
        if self.description_worker and self.description_worker.isRunning():
            return False

        # Reset guard
        self._description_finished_handled = False
        self._reset_description_run_errors()

        # Mark that we're waiting for description via agent
        self._pending_agent_description = True
        self._agent_description_clips = clips

        # Add clips to Analyze tab and switch
        self.analyze_tab.add_clips([c.id for c in clips])
        self._switch_to_tab("analyze")

        # Update UI state
        self.analyze_tab.set_analyzing(True, "description")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(
            f"Generating descriptions for {len(clips)} clips... "
            "(First run may take several minutes to download models)"
        )

        # Start worker
        from PySide6.QtCore import Qt
        sources = self.project.sources_by_id
        self.description_worker = DescriptionWorker(clips, tier=tier, prompt=prompt, sources=sources, parallelism=self.settings.description_parallelism)
        self.description_worker.progress.connect(self._on_description_progress)
        self.description_worker.description_ready.connect(self._on_description_ready)
        self.description_worker.error.connect(self._on_description_error)
        self.description_worker.description_completed.connect(self._on_agent_description_finished, Qt.UniqueConnection)
        # Clean up thread safely after it finishes
        self.description_worker.finished.connect(self.description_worker.deleteLater)
        self.description_worker.finished.connect(lambda: setattr(self, 'description_worker', None))
        self.description_worker.start()

        return True

    @Slot(int, int)
    def _on_description_progress(self, current: int, total: int):
        """Handle description generation progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Describe: downloading vision model (first run)..."
                )
            else:
                self.status_bar.showMessage(f"Generating descriptions: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(str, str, str)
    def _on_description_ready(self, clip_id: str, description: str, model_name: str):
        """Handle description results for a single clip or frame."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            clip.description = description
            clip.description_model = model_name
            clip.description_frames = 1
            logger.debug(f"Description for {clip_id}: {description[:50]}...")
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            self.project.update_frame(
                clip_id,
                description=description,
                description_model=model_name,
            )
            logger.debug(f"Description for frame {clip_id}: {description[:50]}...")

    @Slot()
    def _on_agent_description_finished(self):
        """Handle description completion when triggered by agent."""
        logger.info("=== AGENT DESCRIPTION FINISHED ===")

        # Guard against duplicate calls
        if self._description_finished_handled:
            logger.warning("_on_agent_description_finished already handled, ignoring duplicate")
            return
        self._description_finished_handled = True

        # Reset UI state
        self.analyze_tab.set_analyzing(False)
        self.progress_bar.setVisible(False)

        # Get error info from worker
        error_count = 0
        last_error = None
        if hasattr(self, 'description_worker') and self.description_worker:
            error_count = self.description_worker.error_count
            last_error = self.description_worker.last_error

        if error_count > 0:
            self._gui_state.set_last_error(
                f"Description error: {self._description_run_error or last_error or 'Unknown'}"
            )
            self.status_bar.showMessage(
                f"Description complete with {error_count} errors. Last: {last_error[:80] if last_error else 'Unknown'}",
                5000
            )
        else:
            self.status_bar.showMessage("Description generation complete", 3000)

        # Save project if path is set
        if self.project.path:
            self.project.save()

        # Send result back to agent
        if hasattr(self, '_pending_agent_description') and self._pending_agent_description:
            self._pending_agent_description = False
            clips = getattr(self, '_agent_description_clips', [])

            # Build result summary
            described_count = sum(1 for c in clips if c.description)

            result = self._build_agent_analysis_result(
                clips,
                ["describe"],
                f"Generated descriptions for {described_count} of {len(clips)} clips",
                {
                    "described_clips": described_count,
                    "total_clips": len(clips),
                    "error_count": error_count,
                    "sample_descriptions": [],
                },
            )
            result["success"] = error_count == 0 or described_count > 0

            # Include error info if present
            if error_count > 0 and last_error:
                result["last_error"] = last_error

            # Include sample descriptions
            for clip in clips[:3]:
                if clip.description:
                    result["sample_descriptions"].append({
                        "clip_id": clip.id,
                        "description": clip.description,
                    })

            if self._chat_worker:
                result = {
                    "tool_call_id": self._pending_agent_tool_call_id,
                    "name": self._pending_agent_tool_name,
                    "success": result["success"],
                    "result": result
                }
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent description result to agent: {described_count}/{len(clips)} clips, {error_count} errors")

    @Slot(int, int)
    def _on_classification_progress(self, current: int, total: int):
        """Handle classification progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Classify Content: loading model..."
                )
            else:
                self.status_bar.showMessage(f"Classifying content: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(str, list)
    def _on_classification_ready(self, clip_id: str, results: list):
        """Handle classification results for a single clip or frame.

        Note: Frame model doesn't have object_labels, so frame results
        are stored as detected_objects for consistency.
        """
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            # Store labels (just the label strings, not confidences)
            clip.object_labels = [label for label, _ in results]
            logger.debug(f"Classification for {clip_id}: {clip.object_labels[:3]}")
            return
        # Try frame - store as detected_objects since Frame has no object_labels
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            objects = [
                {"label": label, "confidence": conf}
                for label, conf in results
            ]
            self.project.update_frame(clip_id, detected_objects=objects)
            logger.debug(
                f"Classification for frame {clip_id}: "
                f"{[label for label, _ in results[:3]]}"
            )

    @Slot()
    def _on_agent_classification_finished(self):
        """Handle classification completion when triggered by agent."""
        logger.info("=== AGENT CLASSIFICATION FINISHED ===")

        # Guard against duplicate calls
        if self._classification_finished_handled:
            logger.warning("_on_agent_classification_finished already handled, ignoring duplicate")
            return
        self._classification_finished_handled = True

        # Reset UI state
        self.analyze_tab.set_analyzing(False)
        self.progress_bar.setVisible(False)
        if self._classification_run_error:
            self.status_bar.showMessage("Classification finished with errors", 5000)
        else:
            self.status_bar.showMessage("Classification complete", 3000)

        # Save project if path is set
        if self.project.path:
            self.project.save()

        # Send result back to agent
        if hasattr(self, '_pending_agent_classification') and self._pending_agent_classification:
            self._pending_agent_classification = False
            clips = getattr(self, '_agent_classification_clips', [])

            # Build result summary
            classified_count = sum(1 for c in clips if c.object_labels)
            result = self._build_agent_analysis_result(
                clips,
                ["classify"],
                f"Classified {classified_count} of {len(clips)} clips",
                {
                    "classified_clips": classified_count,
                    "total_clips": len(clips),
                    "sample_labels": [],
                },
            )

            # Include sample labels from first few clips
            for clip in clips[:3]:
                if clip.object_labels:
                    result["sample_labels"].append({
                        "clip_id": clip.id,
                        "labels": clip.object_labels[:5],
                    })

            if self._chat_worker:
                result = {
                    "tool_call_id": self._pending_agent_tool_call_id,
                    "name": self._pending_agent_tool_name,
                    "success": True,
                    "result": result
                }
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent classification result to agent: {classified_count}/{len(clips)} clips")

    @Slot(int, int)
    def _on_face_detection_progress(self, current: int, total: int):
        """Handle face detection progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Face Detection: downloading model (first run, ~300 MB)..."
                )
            else:
                self.status_bar.showMessage(f"Detecting faces: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(int, int)
    def _on_gaze_progress(self, current: int, total: int):
        """Handle gaze analysis progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Gaze Detection: loading MediaPipe model..."
                )
            else:
                self.status_bar.showMessage(f"Detecting gaze: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(str, float, float, str)
    def _on_gaze_ready(self, clip_id: str, yaw: float, pitch: float, category: str):
        """Handle gaze analysis complete for a clip."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_gaze(clip_id, category)
            self.analyze_tab.update_clip_gaze(clip_id, category)
            # Refresh sidebar if it's showing this clip
            if hasattr(self, 'clip_details_sidebar'):
                self.clip_details_sidebar.refresh_gaze_if_showing(clip_id)
            self._mark_dirty()
            logger.debug(f"Clip {clip_id}: gaze={category} (yaw={yaw:.1f}, pitch={pitch:.1f})")

    @Slot(int, int)
    def _on_embeddings_progress(self, current: int, total: int):
        """Handle embedding extraction progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Generating Embeddings: loading DINOv2 model (first run, ~450 MB)..."
                )
            else:
                self.status_bar.showMessage(f"Generating embeddings: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(str)
    def _on_embedding_ready(self, clip_id: str):
        """Handle embedding attached to a clip. No per-clip UI indicator yet."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            self._mark_dirty()
            logger.debug(f"Clip {clip_id}: embedding populated")

    @Slot()
    def _on_pipeline_embeddings_finished(self):
        """Advance the pipeline once embeddings analysis completes."""
        if self._embeddings_finished_handled:
            return
        self._embeddings_finished_handled = True
        self._on_analysis_phase_worker_finished("embeddings")

    @Slot(str)
    def _on_embeddings_error(self, msg: str):
        """Handle embedding analysis errors — log and ensure pipeline advances."""
        logger.error("Embedding analysis error: %s", msg)
        self.statusBar().showMessage(f"Embedding analysis failed: {msg}", 5000)

    @Slot(int, int)
    def _on_object_detection_progress(self, current: int, total: int):
        """Handle object detection progress updates."""
        if total > 0:
            if current == 0:
                self.status_bar.showMessage(
                    "Object Detection: downloading YOLO model (first run)..."
                )
            else:
                detect_all = getattr(self, '_agent_object_detection_all', True)
                task = "objects" if detect_all else "people"
                self.status_bar.showMessage(f"Detecting {task}: {current}/{total} clips...")
            self.progress_bar.setValue(int(current / total * 100))

    @Slot(str, list, int)
    def _on_objects_ready(self, clip_id: str, detections: list, person_count: int):
        """Handle object detection results for a single clip or frame."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            clip.detected_objects = detections
            clip.person_count = person_count
            logger.debug(f"Detection for {clip_id}: {len(detections)} objects, {person_count} people")
            return
        # Try frame
        frame = self.project.frames_by_id.get(clip_id)
        if frame:
            self.project.update_frame(clip_id, detected_objects=detections)
            logger.debug(f"Detection for frame {clip_id}: {len(detections)} objects, {person_count} people")

    @Slot()
    def _on_agent_object_detection_finished(self):
        """Handle object detection completion when triggered by agent."""
        logger.info("=== AGENT OBJECT DETECTION FINISHED ===")

        # Guard against duplicate calls
        if self._object_detection_finished_handled:
            logger.warning("_on_agent_object_detection_finished already handled, ignoring duplicate")
            return
        self._object_detection_finished_handled = True

        # Reset UI state
        self.analyze_tab.set_analyzing(False)
        self.progress_bar.setVisible(False)
        if self._object_detection_run_error:
            self.status_bar.showMessage("Object detection finished with errors", 5000)
        else:
            self.status_bar.showMessage("Object detection complete", 3000)

        # Save project if path is set
        if self.project.path:
            self.project.save()

        # Send result back to agent
        if hasattr(self, '_pending_agent_object_detection') and self._pending_agent_object_detection:
            self._pending_agent_object_detection = False
            clips = getattr(self, '_agent_object_detection_clips', [])
            detect_all = getattr(self, '_agent_object_detection_all', True)

            # Build result summary
            detected_count = sum(1 for c in clips if c.detected_objects is not None or c.person_count is not None)
            total_people = sum(c.person_count or 0 for c in clips)

            result = {
                "analyzed_clips": detected_count,
                "total_clips": len(clips),
                "total_people_detected": total_people,
            }

            if detect_all:
                # Aggregate object counts across all clips
                all_labels: dict[str, int] = {}
                for clip in clips:
                    if clip.detected_objects:
                        for det in clip.detected_objects:
                            label = det["label"]
                            all_labels[label] = all_labels.get(label, 0) + 1
                result["object_counts"] = all_labels
            result = self._build_agent_analysis_result(
                clips,
                ["detect_objects"],
                f"Detected objects in {detected_count} of {len(clips)} clips",
                result,
            )

            if self._chat_worker:
                result = {
                    "tool_call_id": self._pending_agent_tool_call_id,
                    "name": self._pending_agent_tool_name,
                    "success": True,
                    "result": result
                }
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent object detection result to agent: {detected_count}/{len(clips)} clips")

    # ==================== Intention-First Workflow ====================

    def _on_intention_import_requested(self, algorithm: str, direction: object):
        """Handle intention import request from sequence tab.

        Called when user clicks a sequence card but no clips exist.
        Shows the import dialog and starts the workflow on confirmation.
        """
        logger.info(f"Intention import requested: algorithm={algorithm}, direction={direction}")

        # Store the algorithm for when import is confirmed
        self._intention_pending_algorithm = algorithm
        self._intention_pending_direction = direction

        # Create fresh dialog for this algorithm
        self.intention_import_dialog = IntentionImportDialog(algorithm, self)
        self.intention_import_dialog.import_requested.connect(
            self._on_intention_import_confirmed
        )
        self.intention_import_dialog.cancelled.connect(
            self._on_intention_import_cancelled
        )
        self.intention_import_dialog.show()

    def _on_intention_import_confirmed(
        self,
        local_files: list,
        urls: list,
        algorithm: str,
        direction: str = None,
        shot_type: str = None,
        poem_length: str = None,
        poem_form: str = None,
        storyteller_duration: str = None,
        storyteller_structure: str = None,
        storyteller_theme: str = None,
    ):
        """Handle confirmation of import in the intention workflow dialog.

        Starts the workflow coordinator to process the sources.
        """
        logger.info(
            f"Intention import confirmed: {len(local_files)} files, {len(urls)} URLs, "
            f"direction={direction}, shot_type={shot_type}, poem_length={poem_length}, "
            f"poem_form={poem_form}, storyteller_duration={storyteller_duration}"
        )

        # Use stored algorithm (from card click) if dialog didn't provide one
        algorithm_to_use = algorithm or self._intention_pending_algorithm
        # Use direction from dialog if provided, otherwise fallback to stored
        direction_to_use = direction or getattr(self, '_intention_pending_direction', None)
        # Store shot type for filtering when workflow completes
        self._intention_pending_shot_type = shot_type
        # Store poem length and form for exquisite_corpus
        self._intention_pending_poem_length = poem_length
        self._intention_pending_poem_form = poem_form
        # Store storyteller params
        self._intention_pending_storyteller_duration = storyteller_duration
        self._intention_pending_storyteller_structure = storyteller_structure
        self._intention_pending_storyteller_theme = storyteller_theme

        # Validate we have something to import
        if not local_files and not urls:
            QMessageBox.warning(
                self,
                "No Sources",
                "Please add local files or URLs to import."
            )
            return

        # Create coordinator if needed
        if not self.intention_workflow:
            self.intention_workflow = IntentionWorkflowCoordinator(self)
            self._connect_intention_workflow_signals()

        # Start the workflow
        success = self.intention_workflow.start(
            algorithm=algorithm_to_use,
            local_files=[Path(f) for f in local_files],
            urls=urls,
            direction=direction_to_use,
        )

        if success:
            # Switch dialog to progress view
            self.intention_import_dialog.show_progress()
            # Note: Work is triggered by _on_intention_step_started when
            # the coordinator emits step_started signal
        else:
            QMessageBox.warning(
                self,
                "Workflow Error",
                "Could not start the import workflow. Another workflow may be in progress."
            )

    def _on_intention_import_cancelled(self):
        """Handle cancellation of the import dialog."""
        logger.info("Intention import cancelled by user")

        # Cancel the workflow coordinator
        if self.intention_workflow and self.intention_workflow.is_running:
            self.intention_workflow.cancel()

        # Also cancel any running workers owned by MainWindow
        # The coordinator doesn't have references to these workers
        if hasattr(self, 'detection_worker') and self.detection_worker is not None:
            if self.detection_worker.isRunning():
                logger.info("Cancelling detection worker due to workflow cancellation")
                self.detection_worker.cancel()
                # Don't wait here - let the worker finish in background
                # Cleanup will happen in _start_intention_detection if needed

        if hasattr(self, 'thumbnail_worker') and self.thumbnail_worker is not None:
            if self.thumbnail_worker.isRunning():
                logger.info("Cancelling thumbnail worker due to workflow cancellation")
                # ThumbnailWorker may not have cancel() method, but we'll try
                if hasattr(self.thumbnail_worker, 'cancel'):
                    self.thumbnail_worker.cancel()

    def _connect_intention_workflow_signals(self):
        """Connect all signals from the intention workflow coordinator."""
        if not self.intention_workflow:
            return

        self.intention_workflow.step_started.connect(self._on_intention_step_started)
        self.intention_workflow.step_completed.connect(self._on_intention_step_completed)
        self.intention_workflow.step_skipped.connect(self._on_intention_step_skipped)
        self.intention_workflow.progress_updated.connect(self._on_intention_progress)
        self.intention_workflow.workflow_completed.connect(self._on_intention_workflow_completed)
        self.intention_workflow.workflow_cancelled.connect(self._on_intention_workflow_cancelled)
        self.intention_workflow.workflow_error.connect(self._on_intention_workflow_error)

    def _on_intention_step_started(self, step_name: str, current: int, total: int):
        """Handle workflow step starting."""
        logger.info(f"Intention workflow step started: {step_name} ({current}/{total})")
        if self.intention_import_dialog:
            from ui.dialogs.intention_import_dialog import WorkflowStep
            step_map = {
                "downloading": WorkflowStep.DOWNLOADING,
                "detecting": WorkflowStep.DETECTING,
                "thumbnails": WorkflowStep.THUMBNAILS,
                "analyzing": WorkflowStep.ANALYZING,
                "building": WorkflowStep.BUILDING,
            }
            step = step_map.get(step_name)
            if step:
                self.intention_import_dialog.set_step_active(step)

        # Start the work for the step that just started
        if self.intention_workflow:
            if step_name == "downloading":
                self._start_intention_downloads()
            elif step_name == "detecting":
                self._start_intention_detection()
            elif step_name == "thumbnails":
                self._start_intention_thumbnails()
            elif step_name == "analyzing":
                self._start_intention_analysis()
            elif step_name == "building":
                self._start_intention_building()

    def _on_intention_step_completed(self, step_name: str):
        """Handle workflow step completion."""
        logger.info(f"Intention workflow step completed: {step_name}")
        if self.intention_import_dialog:
            from ui.dialogs.intention_import_dialog import WorkflowStep
            step_map = {
                "downloading": WorkflowStep.DOWNLOADING,
                "detecting": WorkflowStep.DETECTING,
                "thumbnails": WorkflowStep.THUMBNAILS,
                "analyzing": WorkflowStep.ANALYZING,
                "building": WorkflowStep.BUILDING,
            }
            step = step_map.get(step_name)
            if step:
                self.intention_import_dialog.set_step_complete(step)
        # Note: Next step is triggered by _on_intention_step_started when
        # the coordinator emits step_started for the new phase

    def _on_intention_step_skipped(self, step_name: str):
        """Handle workflow step being skipped."""
        logger.info(f"Intention workflow step skipped: {step_name}")
        if self.intention_import_dialog:
            from ui.dialogs.intention_import_dialog import WorkflowStep
            step_map = {
                "downloading": WorkflowStep.DOWNLOADING,
                "detecting": WorkflowStep.DETECTING,
                "thumbnails": WorkflowStep.THUMBNAILS,
                "analyzing": WorkflowStep.ANALYZING,
                "building": WorkflowStep.BUILDING,
            }
            step = step_map.get(step_name)
            if step:
                self.intention_import_dialog.set_step_skipped(step)

    def _on_intention_progress(self, progress):
        """Handle workflow progress update."""
        if self.intention_import_dialog:
            from ui.dialogs.intention_import_dialog import WorkflowStep
            state_map = {
                WorkflowState.DOWNLOADING: WorkflowStep.DOWNLOADING,
                WorkflowState.DETECTING: WorkflowStep.DETECTING,
                WorkflowState.THUMBNAILS: WorkflowStep.THUMBNAILS,
                WorkflowState.ANALYZING: WorkflowStep.ANALYZING,
                WorkflowState.BUILDING: WorkflowStep.BUILDING,
            }
            step = state_map.get(progress.state)
            if step:
                self.intention_import_dialog.set_step_progress(
                    step, int(progress.step_progress * 100), progress.message
                )

    def _on_intention_workflow_completed(self, result):
        """Handle workflow completion."""
        logger.info(f"Intention workflow completed: {result.clips_created} clips, "
                    f"{result.sources_processed} sources")

        if self.intention_import_dialog:
            self.intention_import_dialog.set_complete(
                result.clips_created,
                result.sources_processed,
                result.sources_failed,
            )

        # Close dialog after brief delay
        QTimer.singleShot(1500, self._finalize_intention_workflow)

    def _finalize_intention_workflow(self):
        """Finalize the intention workflow and apply results."""
        if not self.intention_workflow:
            return

        # Close dialog
        if self.intention_import_dialog:
            self.intention_import_dialog.hide()

        # Get results from coordinator
        all_clips = self.intention_workflow.get_all_clips()
        all_sources = self.intention_workflow.get_all_sources()
        algorithm, direction = self.intention_workflow.get_algorithm_with_direction()

        if not all_clips:
            QMessageBox.warning(
                self,
                "Workflow Failed",
                "No clips were created. Please check your video sources and try again."
            )
            return

        # Add sources and clips to project
        for source in all_sources:
            if source.id not in self.sources_by_id:
                self.project.add_source(source)
                self.collect_tab.add_source(source)

        for clip in all_clips:
            if clip.id not in self.clips_by_id:
                self.project.add_clips([clip])

        # Sync UI state for intention workflow (Cut/Analyze tabs)
        self._sync_intention_workflow_ui(sources=all_sources)

        # Populate Analyze tab with clips that were analyzed during the workflow
        if algorithm == "exquisite_corpus":
            # Exquisite Corpus does OCR - add clips with extracted text
            analyzed_clip_ids = [c.id for c in all_clips if c.extracted_texts]
            if analyzed_clip_ids:
                logger.info(f"Adding {len(analyzed_clip_ids)} analyzed clips to Analyze tab")
                self.analyze_tab.add_clips(analyzed_clip_ids)
        elif algorithm == "color":
            # Color algorithm does color analysis
            analyzed_clip_ids = [c.id for c in all_clips if c.dominant_colors]
            if analyzed_clip_ids:
                logger.info(f"Adding {len(analyzed_clip_ids)} color-analyzed clips to Analyze tab")
                self.analyze_tab.add_clips(analyzed_clip_ids)
        elif algorithm == "shot_type":
            # Shot type algorithm does shot classification
            analyzed_clip_ids = [c.id for c in all_clips if c.shot_type]
            if analyzed_clip_ids:
                logger.info(f"Adding {len(analyzed_clip_ids)} shot-analyzed clips to Analyze tab")
                self.analyze_tab.add_clips(analyzed_clip_ids)

        # Dialog-based algorithms: show their dialog instead of auto-building
        from ui.algorithm_config import ALGORITHM_CONFIG
        algo_cfg = ALGORITHM_CONFIG.get(algorithm, {})
        if algo_cfg.get("is_dialog"):
            # Build clips_with_sources for the dialog
            clips_with_sources = []
            for clip in all_clips:
                source = self.sources_by_id.get(clip.source_id)
                if source:
                    clips_with_sources.append((clip, source))

            if clips_with_sources:
                # Set available clips so the sequence tab has them
                self.sequence_tab.set_available_clips(clips_with_sources)
                # Select all clip IDs so the dialog routing works
                all_clip_ids = [c.id for c in all_clips]
                if self._gui_state:
                    self._gui_state.analyze_selected_ids = all_clip_ids
                # Route to the dialog via the sequence tab's card click handler
                self.sequence_tab._on_card_clicked(algorithm)
            self._refresh_sequence_tab_clips()
            return

        # Non-dialog algorithms: auto-build sequence
        # Get shot type filter from pending state
        shot_type = getattr(self, '_intention_pending_shot_type', None)

        clips_with_sources = []
        for clip in all_clips:
            source = self.sources_by_id.get(clip.source_id)
            if source:
                # Apply shot type filter if specified
                if shot_type and clip.shot_type != shot_type:
                    continue
                clips_with_sources.append((clip, source))

        # Handle empty state after filtering
        if not clips_with_sources and shot_type:
            QMessageBox.information(
                self,
                "No Matching Clips",
                f"No clips match the selected shot type '{shot_type}'.\n"
                "Try selecting 'All' or analyzing clips for shot type first."
            )
            return

        result = self.sequence_tab.apply_intention_workflow_result(
            algorithm=algorithm,
            clips_with_sources=clips_with_sources,
            direction=direction,
        )

        if result.get("success"):
            self.status_bar.showMessage(
                f"Created {algorithm} sequence with {result.get('clip_count', 0)} clips"
            )
            self._mark_dirty()
        else:
            QMessageBox.warning(
                self,
                "Sequence Error",
                f"Failed to create sequence: {result.get('error', 'Unknown error')}"
            )

        # Refresh UI
        self._refresh_sequence_tab_clips()

    def _on_intention_workflow_cancelled(self):
        """Handle workflow cancellation."""
        logger.info("Intention workflow cancelled")
        if self.intention_import_dialog:
            self.intention_import_dialog.hide()
        self.status_bar.showMessage("Import cancelled")

    def _on_intention_workflow_error(self, error: str):
        """Handle workflow error."""
        logger.error(f"Intention workflow error: {error}")
        if self.intention_import_dialog:
            self.intention_import_dialog.set_error(error)

    # --- Intention workflow phase helpers ---

    def _start_intention_downloads(self, urls: list):
        """Start downloading URLs for the intention workflow."""
        if not urls:
            return

        if not self._ensure_video_download_available():
            self._on_intention_workflow_error("yt-dlp is required to download videos.")
            return

        download_dir = validate_download_dir(self.settings.download_dir)
        if not download_dir:
            download_dir = get_default_download_dir()

        self.url_bulk_download_worker = URLBulkDownloadWorker(urls, download_dir)

        # Connect to coordinator handlers
        self.url_bulk_download_worker.progress.connect(
            self.intention_workflow.on_download_progress
        )
        self.url_bulk_download_worker.video_finished.connect(
            self._on_intention_video_downloaded
        )
        self.url_bulk_download_worker.all_finished.connect(
            self.intention_workflow.on_download_all_finished
        )
        # Clean up thread safely
        self.url_bulk_download_worker.finished.connect(
            self.url_bulk_download_worker.deleteLater
        )
        self.url_bulk_download_worker.finished.connect(
            lambda: setattr(self, 'url_bulk_download_worker', None)
        )

        self.url_bulk_download_worker.start()

    def _on_intention_video_downloaded(self, url: str, result):
        """Handle individual video download completion during intention workflow."""
        # Notify coordinator
        if self.intention_workflow:
            self.intention_workflow.on_download_video_finished(url, result)

        # Also add to project and collect tab
        if result and result.success and result.file_path:
            from models.clip import Source
            file_path = Path(result.file_path)

            # Check if already in project
            for existing in self.project.sources:
                if existing.file_path == file_path:
                    return

            # Create source and add to project
            source = Source(
                file_path=file_path,
                duration_seconds=result.duration or 0,
                fps=result.fps or 30.0,
                width=result.width or 1920,
                height=result.height or 1080,
            )
            self.project.add_source(source)
            self.collect_tab.add_source(source)

    def _cleanup_worker(
        self,
        worker,
        worker_name: str,
        signal_names: list[str],
        wait_timeout: int = 2000,
        allow_terminate: bool = False,
    ) -> bool:
        """Clean up a QThread worker safely without blocking the GUI.

        This helper prevents "QThread: Destroyed while thread is still running" crashes
        by properly cancelling and cleaning up workers before they're replaced.

        Args:
            worker: The QThread worker to clean up (can be None)
            worker_name: Name for logging (e.g., "detection", "thumbnail")
            signal_names: List of signal attribute names to disconnect (e.g., ["progress", "finished"])
            wait_timeout: Max ms to wait for graceful shutdown (default 2000ms)
            allow_terminate: If True, terminate() as last resort (DANGEROUS - avoid if possible)

        Returns:
            True if worker was cleaned up or was None, False if cleanup is still in progress
        """
        if worker is None:
            return True

        if not worker.isRunning():
            # Worker finished - just clean up references
            for sig_name in signal_names:
                try:
                    sig = getattr(worker, sig_name, None)
                    if sig:
                        sig.disconnect()
                except (RuntimeError, TypeError):
                    pass  # Already disconnected
            worker.deleteLater()
            return True

        # Worker still running - request cancellation
        logger.warning(f"Previous {worker_name} worker still running, requesting cancellation")
        if hasattr(worker, 'cancel'):
            worker.cancel()

        # Brief non-blocking wait - if it doesn't stop quickly, let it finish in background
        if worker.wait(wait_timeout):
            logger.info(f"{worker_name} worker stopped gracefully")
            for sig_name in signal_names:
                try:
                    sig = getattr(worker, sig_name, None)
                    if sig:
                        sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            worker.deleteLater()
            return True

        # Worker didn't stop in time
        if allow_terminate:
            # SEVERE WARNING: terminate() is dangerous and can corrupt state
            logger.critical(
                f"SEVERE: {worker_name} worker did not stop in {wait_timeout}ms, "
                f"forcefully terminating. This may cause corruption or resource leaks!"
            )
            worker.terminate()
            worker.wait(500)  # Brief wait after terminate
            for sig_name in signal_names:
                try:
                    sig = getattr(worker, sig_name, None)
                    if sig:
                        sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            worker.deleteLater()
            return True
        else:
            # Don't terminate - let worker finish in background
            # Use generation ID pattern to ignore its signals
            logger.warning(
                f"{worker_name} worker still running after {wait_timeout}ms, "
                f"letting it finish in background (signals will be ignored via generation ID)"
            )
            # Don't disconnect signals - generation ID will handle stale signals
            # Don't deleteLater - let the finished signal handle cleanup
            return True  # Proceed anyway - generation ID protects against stale signals

    def _start_intention_detection(self):
        """Start scene detection for the intention workflow."""
        if not self.intention_workflow:
            return

        source_path = self.intention_workflow.get_current_source_path()
        if not source_path:
            return

        logger.info(f"Starting intention detection for: {source_path}")

        # Increment generation ID - signals from old workers will be ignored
        self._detection_generation += 1
        current_gen = self._detection_generation

        # Clean up any existing detection worker (non-blocking, no terminate)
        if hasattr(self, 'detection_worker') and self.detection_worker is not None:
            self._cleanup_worker(
                self.detection_worker,
                "detection",
                ["progress", "detection_completed", "error", "finished"],
                wait_timeout=2000,
                allow_terminate=False,  # Don't terminate - let it finish, ignore its signals
            )
            self.detection_worker = None

        # Determine detection mode based on algorithm
        algorithm, _ = self.intention_workflow.get_algorithm_with_direction()
        if algorithm == "exquisite_corpus":
            # Use karaoke (text-based) detection for Exquisite Corpus
            # Cuts scenes based on on-screen text changes, ideal for text-heavy content
            karaoke_config = KaraokeDetectionConfig(
                roi_top_percent=0.0,  # Full frame - let OCR find text anywhere
                text_similarity_threshold=60.0,
                confirm_frames=3,
                cut_offset=5,
            )
            logger.info("Using karaoke (text-based) detection for Exquisite Corpus")
            self.detection_worker = DetectionWorker(
                source_path,
                mode="karaoke",
                karaoke_config=karaoke_config,
            )
        else:
            config = DetectionConfig(
                threshold=self.settings.default_sensitivity,
                min_scene_length=15,
                use_adaptive=True,
            )
            self.detection_worker = DetectionWorker(source_path, config)

        # Capture generation for lambda closures - used to ignore stale signals
        gen = current_gen

        self.detection_worker.progress.connect(
            self.intention_workflow.on_detection_progress
        )
        # Use lambda with generation check to ignore signals from old workers
        self.detection_worker.detection_completed.connect(
            lambda src, clps, g=gen: self._on_intention_detection_completed(src, clps, g)
        )
        self.detection_worker.error.connect(
            lambda err, g=gen: self._on_intention_detection_error(err, g)
        )
        # Clean up - use generation check to avoid cleaning up wrong worker
        self.detection_worker.finished.connect(
            lambda g=gen: self._on_detection_worker_finished(g)
        )

        self._detection_finished_handled = False
        self.detection_worker.start()

    def _on_detection_worker_finished(self, generation: int):
        """Handle detection worker finished signal with generation check."""
        if generation != self._detection_generation:
            logger.debug(f"Ignoring finished signal from old detection worker (gen {generation} != {self._detection_generation})")
            return
        # Safe to clean up - this is the current worker
        if self.detection_worker:
            self.detection_worker.deleteLater()
            self.detection_worker = None

    def _on_intention_detection_completed(self, source, clips, generation: int = 0):
        """Handle detection completion during intention workflow.

        Args:
            source: The detected Source object
            clips: List of detected Clip objects
            generation: Worker generation ID - used to ignore stale signals from old workers
        """
        # Check generation ID - ignore signals from old/cancelled workers
        if generation != 0 and generation != self._detection_generation:
            logger.info(f"Ignoring detection_completed from old worker (gen {generation} != {self._detection_generation})")
            return

        if self._detection_finished_handled:
            return
        self._detection_finished_handled = True

        logger.info(f"Intention detection completed: {len(clips)} clips")

        # Add source and clips to project
        if source.id not in self.sources_by_id:
            self.project.add_source(source)
            self.collect_tab.add_source(source)

        # Set Cut tab source (needed for proper state display)
        self.cut_tab.set_source(source)

        for clip in clips:
            # Sync clip source_id to match the source we just added
            clip.source_id = source.id

        self.project.add_clips(clips)

        # Notify coordinator
        if self.intention_workflow:
            self.intention_workflow.on_detection_completed(source, clips)

            # Check if there are more sources to process
            next_source = self.intention_workflow.get_current_source_path()
            if next_source:
                # Start detection for next source
                self._start_intention_detection()

    def _on_intention_detection_error(self, error: str, generation: int = 0):
        """Handle detection error during intention workflow.

        Args:
            error: Error message
            generation: Worker generation ID - used to ignore stale signals from old workers
        """
        # Check generation ID - ignore signals from old/cancelled workers
        if generation != 0 and generation != self._detection_generation:
            logger.info(f"Ignoring detection_error from old worker (gen {generation} != {self._detection_generation})")
            return

        if self._detection_finished_handled:
            return
        self._detection_finished_handled = True

        logger.error(f"Intention detection error: {error}")

        # Notify coordinator
        if self.intention_workflow:
            self.intention_workflow.on_detection_error(error)

            # Check if there are more sources to process
            next_source = self.intention_workflow.get_current_source_path()
            if next_source:
                # Start detection for next source
                self._start_intention_detection()

    def _start_intention_thumbnails(self):
        """Start thumbnail generation for the intention workflow."""
        if not self.intention_workflow:
            return

        all_clips = self.intention_workflow.get_all_clips()
        all_sources = self.intention_workflow.get_all_sources()

        if not all_clips:
            # No clips - skip to next step
            if self.intention_workflow:
                self.intention_workflow.on_thumbnails_finished()
            return

        # Increment generation ID - signals from old workers will be ignored
        self._thumbnail_generation += 1
        current_gen = self._thumbnail_generation

        # Clean up any existing thumbnail worker (non-blocking, no terminate)
        if hasattr(self, 'thumbnail_worker') and self.thumbnail_worker is not None:
            self._cleanup_worker(
                self.thumbnail_worker,
                "thumbnail",
                ["progress", "thumbnail_ready", "finished"],
                wait_timeout=2000,
                allow_terminate=False,
            )
            self.thumbnail_worker = None

        # Build sources_by_id dict
        sources_by_id = {s.id: s for s in all_sources}
        default_source = all_sources[0] if all_sources else None

        self.thumbnail_worker = ThumbnailWorker(
            source=default_source,
            clips=all_clips,
            cache_dir=self.settings.thumbnail_cache_dir,
            sources_by_id=sources_by_id,
        )

        # Capture generation for lambda closures
        gen = current_gen

        self.thumbnail_worker.progress.connect(
            self.intention_workflow.on_thumbnail_progress
        )
        self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        # Use generation check for finished handler
        self.thumbnail_worker.finished.connect(
            lambda g=gen: self._on_intention_thumbnails_finished(g)
        )

        self._thumbnails_finished_handled = False
        self.thumbnail_worker.start()

    def _sync_intention_workflow_ui(self, sources: list = None):
        """Synchronize UI state for intention workflow (Cut/Analyze tabs).

        This helper consolidates state synchronization that's needed after
        detection and thumbnail generation in the intention workflow.

        Args:
            sources: Optional list of sources. If not provided, gets from intention_workflow.
        """
        # Get sources from intention workflow if not provided
        if sources is None and self.intention_workflow:
            sources = self.intention_workflow.get_all_sources()

        # Sync lookups for Analyze tab (same as normal flow)
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        # Ensure Cut tab has source set
        if sources:
            self.cut_tab.set_source(sources[0])

        # Sync all clips to Cut tab
        self.cut_tab.set_clips(self.clips)

    def _on_intention_thumbnails_finished(self, generation: int = 0):
        """Handle thumbnail generation completion during intention workflow.

        Args:
            generation: Worker generation ID - used to ignore stale signals from old workers
        """
        # Check generation ID - ignore signals from old/cancelled workers
        if generation != 0 and generation != self._thumbnail_generation:
            logger.info(f"Ignoring thumbnails_finished from old worker (gen {generation} != {self._thumbnail_generation})")
            return

        if self._thumbnails_finished_handled:
            return
        self._thumbnails_finished_handled = True

        # Clean up worker reference
        if self.thumbnail_worker:
            self.thumbnail_worker.deleteLater()
            self.thumbnail_worker = None

        logger.info("Intention thumbnails finished")

        # Sync UI state for intention workflow
        self._sync_intention_workflow_ui()

        if self.intention_workflow:
            self.intention_workflow.on_thumbnails_finished()

    def _start_intention_analysis(self):
        """Start analysis for the intention workflow (if needed)."""
        if not self.intention_workflow:
            return

        all_clips = self.intention_workflow.get_all_clips()
        algorithm, _ = self.intention_workflow.get_algorithm_with_direction()

        if algorithm == "color":
            # Start color analysis
            clips_needing_colors = [c for c in all_clips if not c.dominant_colors]

            if not clips_needing_colors:
                # All clips already have colors - skip
                self.intention_workflow.on_analysis_finished()
                return

            self._reset_analysis_run_error("colors")
            self.color_worker = ColorAnalysisWorker(clips_needing_colors, parallelism=self.settings.color_analysis_parallelism, sources_by_id=self.project.sources_by_id)
            self.color_worker.progress.connect(
                self.intention_workflow.on_analysis_progress
            )
            self.color_worker.color_ready.connect(self._on_color_ready)
            self.color_worker.error.connect(self._on_color_error)
            self.color_worker.analysis_completed.connect(
                self._on_intention_analysis_finished
            )
            # Clean up
            self.color_worker.finished.connect(self.color_worker.deleteLater)
            self.color_worker.finished.connect(
                lambda: setattr(self, 'color_worker', None)
            )

            self._color_analysis_finished_handled = False
            self.color_worker.start()

        elif algorithm == "shot_type":
            # Start shot type analysis
            clips_needing_shots = [c for c in all_clips if not c.shot_type]

            if not clips_needing_shots:
                # All clips already have shot types - skip
                self.intention_workflow.on_analysis_finished()
                return

            if not self._ensure_analysis_operation_available("shots"):
                self.intention_workflow.on_analysis_finished()
                return

            self._shot_type_finished_handled = False
            self._shot_type_run_error = None
            self.shot_type_worker = ShotTypeWorker(clips_needing_shots, self.project.sources_by_id, parallelism=self.settings.local_model_parallelism)
            self.shot_type_worker.progress.connect(
                self.intention_workflow.on_analysis_progress
            )
            self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
            self.shot_type_worker.error.connect(self._on_shot_type_error)
            self.shot_type_worker.analysis_completed.connect(
                self._on_intention_shot_analysis_finished, Qt.UniqueConnection
            )
            # Clean up
            self.shot_type_worker.finished.connect(self.shot_type_worker.deleteLater)
            self.shot_type_worker.finished.connect(
                lambda: setattr(self, 'shot_type_worker', None)
            )

            self.shot_type_worker.start()

        elif algorithm == "storyteller":
            # Start description analysis - Storyteller needs clip descriptions
            clips_needing_descriptions = [c for c in all_clips if not c.description]

            if not clips_needing_descriptions:
                # All clips already have descriptions - skip
                self.intention_workflow.on_analysis_finished()
                return

            # Get description settings
            tier = self.settings.description_model_tier
            sources = self.project.sources_by_id

            if not self._ensure_analysis_operation_available("describe", description_tier=tier):
                self.intention_workflow.on_analysis_finished()
                return

            self._description_finished_handled = False
            self._reset_description_run_errors()
            logger.info(f"Creating DescriptionWorker (intention) for {len(clips_needing_descriptions)} clips with tier={tier}")
            self.description_worker = DescriptionWorker(clips_needing_descriptions, tier=tier, sources=sources, parallelism=self.settings.description_parallelism)
            self.description_worker.progress.connect(
                self.intention_workflow.on_analysis_progress
            )
            self.description_worker.description_ready.connect(self._on_description_ready)
            self.description_worker.error.connect(self._on_description_error)
            self.description_worker.description_completed.connect(
                self._on_intention_description_analysis_finished, Qt.UniqueConnection
            )
            # Clean up
            self.description_worker.finished.connect(self.description_worker.deleteLater)
            self.description_worker.finished.connect(
                lambda: setattr(self, 'description_worker', None)
            )

            self.description_worker.start()

        else:
            # No analysis needed for this algorithm
            self.intention_workflow.on_analysis_finished()

    def _on_intention_analysis_finished(self):
        """Handle analysis completion during intention workflow."""
        if self._color_analysis_finished_handled:
            return
        self._color_analysis_finished_handled = True

        logger.info("Intention analysis finished")
        if self._color_run_error:
            self.status_bar.showMessage(
                "Color extraction finished with errors",
                5000,
            )

        if self.intention_workflow:
            self.intention_workflow.on_analysis_finished()

    def _on_intention_shot_analysis_finished(self):
        """Handle shot type analysis completion during intention workflow."""
        if self._shot_type_finished_handled:
            return
        self._shot_type_finished_handled = True

        logger.info("Intention shot type analysis finished")
        if self._shot_type_run_error:
            self.status_bar.showMessage(
                "Shot type classification finished with errors",
                5000,
            )

        if self.intention_workflow:
            self.intention_workflow.on_analysis_finished()

    def _on_intention_description_analysis_finished(self):
        """Handle description analysis completion during intention workflow (for Storyteller)."""
        if self._description_finished_handled:
            return
        self._description_finished_handled = True

        logger.info("Intention description analysis finished")
        if self._description_run_error:
            self.status_bar.showMessage(
                "Description generation finished with errors",
                5000,
            )

        if self.intention_workflow:
            self.intention_workflow.on_analysis_finished()

    def _start_intention_building(self):
        """Start building the sequence for the intention workflow."""
        if not self.intention_workflow:
            return

        algorithm, direction = self.intention_workflow.get_algorithm_with_direction()
        all_clips = self.intention_workflow.get_all_clips()

        if algorithm == "exquisite_corpus":
            # Show the Exquisite Corpus dialog (handles prompt, OCR, poem generation)
            self._show_exquisite_corpus_dialog_for_intention(all_clips)
        elif algorithm == "storyteller":
            # Show the Storyteller dialog (handles theme, structure, LLM narrative generation)
            self._show_storyteller_dialog_for_intention(all_clips)
        else:
            # Other algorithms complete immediately
            self.intention_workflow.on_building_complete(all_clips)

    def _show_exquisite_corpus_dialog_for_intention(self, clips: list):
        """Show Exquisite Corpus dialog during intention workflow building phase.

        Args:
            clips: List of Clip objects to process
        """
        from ui.dialogs.exquisite_corpus_dialog import ExquisiteCorpusDialog
        from PySide6.QtWidgets import QDialog

        # Build sources lookup from workflow
        all_sources = self.intention_workflow.get_all_sources() if self.intention_workflow else []
        sources_by_id = {s.id: s for s in all_sources}

        # Get poem length and form from import dialog if set
        poem_length = getattr(self, '_intention_pending_poem_length', None)
        poem_form = getattr(self, '_intention_pending_poem_form', None)

        dialog = ExquisiteCorpusDialog(
            clips=clips,
            sources_by_id=sources_by_id,
            project=self.project,
            parent=self,
            initial_poem_length=poem_length,
            initial_form=poem_form,
        )

        # Connect to sequence_ready signal - this is how the dialog returns results
        dialog.sequence_ready.connect(self._on_exquisite_corpus_sequence_ready)

        result = dialog.exec()

        if result != QDialog.Accepted:
            # User cancelled - cancel the workflow
            if self.intention_workflow:
                self.intention_workflow.cancel()

    @Slot(list)
    def _on_exquisite_corpus_sequence_ready(self, sequence_clips: list):
        """Handle sequence ready from Exquisite Corpus dialog during intention workflow.

        Args:
            sequence_clips: List of (Clip, Source) tuples in poem order
        """
        logger.info(f"Exquisite Corpus sequence ready: {len(sequence_clips)} clips")

        # Apply to sequence tab using existing method
        self.sequence_tab._apply_exquisite_corpus_sequence(sequence_clips)

        # Complete the workflow with just the clips (not tuples)
        if self.intention_workflow:
            # Extract clips from tuples for the workflow completion
            clips_only = [clip for clip, source in sequence_clips]
            self.intention_workflow.on_building_complete(clips_only)

    def _show_storyteller_dialog_for_intention(self, clips: list):
        """Show Storyteller dialog during intention workflow building phase.

        Args:
            clips: List of Clip objects to process
        """
        from ui.dialogs.storyteller_dialog import StorytellerDialog
        from PySide6.QtWidgets import QDialog

        # Build sources lookup from workflow
        all_sources = self.intention_workflow.get_all_sources() if self.intention_workflow else []
        sources_by_id = {s.id: s for s in all_sources}

        # Get storyteller params from import dialog if set
        duration = getattr(self, '_intention_pending_storyteller_duration', None)
        structure = getattr(self, '_intention_pending_storyteller_structure', None)
        theme = getattr(self, '_intention_pending_storyteller_theme', None)

        dialog = StorytellerDialog(
            clips=clips,
            sources_by_id=sources_by_id,
            project=self.project,
            parent=self,
            initial_duration=duration,
            initial_structure=structure,
            initial_theme=theme,
        )

        # Connect to sequence_ready signal
        dialog.sequence_ready.connect(self._on_storyteller_sequence_ready)

        result = dialog.exec()

        if result != QDialog.Accepted:
            # User cancelled - cancel the workflow
            if self.intention_workflow:
                self.intention_workflow.cancel()

    @Slot(list)
    def _on_storyteller_sequence_ready(self, sequence_clips: list):
        """Handle sequence ready from Storyteller dialog during intention workflow.

        Args:
            sequence_clips: List of (Clip, Source) tuples in narrative order
        """
        logger.info(f"Storyteller sequence ready: {len(sequence_clips)} clips")

        # Apply to sequence tab using existing method
        self.sequence_tab._apply_storyteller_sequence(sequence_clips)

        # Complete the workflow with just the clips (not tuples)
        if self.intention_workflow:
            clips_only = [clip for clip, source in sequence_clips]
            self.intention_workflow.on_building_complete(clips_only)

    def _get_edl_sources(self) -> dict:
        """Return project-wide sources for EDL export."""
        sources_by_id = getattr(self, "sources_by_id", {})
        if sources_by_id:
            return dict(sources_by_id)
        current_source = getattr(self, "current_source", None)
        if current_source:
            return {current_source.id: current_source}
        return {}

    def _get_edl_frames(self) -> dict:
        """Return project-wide frames for EDL export."""
        if getattr(self, "project", None):
            return dict(self.project.frames_by_id)
        return {}

    def _default_sequence_edl_filename(self, sequence, index: int | None = None) -> str:
        """Build a safe default filename for a sequence EDL."""
        name = sequence.name or "Sequence"
        if name == "Untitled Sequence" and getattr(self, "project_metadata", None):
            name = self.project_metadata.name
        if index is not None:
            name = f"{index + 1:02d}_{name}"
        return f"{self._sanitize_filename(name)}.edl"

    def _persist_sequence_tab_state_for_export(self) -> None:
        """Ensure the active timeline is written back before exporting project sequences."""
        if hasattr(self.sequence_tab, "_persist_current_sequence"):
            self.sequence_tab._persist_current_sequence()

    def _normalize_edl_output_path(self, output_path: Path) -> Path:
        """Ensure an EDL export path uses the extension the exporter writes."""
        if output_path.suffix.lower() != ".edl":
            return output_path.with_suffix(".edl")
        return output_path

    def _export_sequence_edl_to_path(self, sequence, output_path: Path) -> bool:
        """Write one sequence EDL to an explicit path."""
        output_path = self._normalize_edl_output_path(output_path)

        config = EDLExportConfig(
            output_path=output_path,
            title=sequence.name or "Scene Ripper Export",
        )
        return export_edl(
            sequence,
            self._get_edl_sources(),
            config,
            frames=self._get_edl_frames(),
        )

    def _unique_edl_output_path(self, output_dir: Path, filename: str) -> Path:
        """Avoid overwriting duplicate sequence names during batch EDL export."""
        output_path = output_dir / filename
        if not output_path.exists():
            return output_path

        stem = output_path.stem
        suffix = output_path.suffix or ".edl"
        counter = 2
        while True:
            candidate = output_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _get_populated_edl_sequences(self) -> list[tuple[int, object]]:
        """Return project sequences that contain timeline entries."""
        if not self.project:
            return []
        return [
            (index, sequence)
            for index, sequence in enumerate(self.project.sequences)
            if sequence.get_all_clips()
        ]

    def _format_sequence_edl_choice_label(self, index: int, sequence) -> str:
        """Build a user-facing sequence choice label for File > Export EDL."""
        clip_count = len(sequence.get_all_clips())
        clip_word = "clip" if clip_count == 1 else "clips"
        return f"{index + 1}. {sequence.name or 'Untitled Sequence'} ({clip_count} {clip_word})"

    def _prompt_for_edl_sequence_index(self) -> int | None:
        """Ask which populated sequence should be exported from the File menu."""
        self._persist_sequence_tab_state_for_export()
        populated = self._get_populated_edl_sequences()
        if not populated:
            QMessageBox.information(
                self, "Export EDL", "No populated sequences to export"
            )
            return None

        if len(populated) == 1:
            return populated[0][0]

        labels = [
            self._format_sequence_edl_choice_label(index, sequence)
            for index, sequence in populated
        ]
        active_index = self.project.active_sequence_index
        default_choice = 0
        for position, (index, _sequence) in enumerate(populated):
            if index == active_index:
                default_choice = position
                break

        selected, accepted = QInputDialog.getItem(
            self,
            "Export EDL",
            "Sequence:",
            labels,
            default_choice,
            False,
        )
        if not accepted or not selected:
            return None

        return populated[labels.index(selected)][0]

    @Slot(int)
    def _on_sequence_edl_export_requested(self, sequence_index: int):
        """Export a selected project sequence as an EDL file."""
        if not self.project or sequence_index < 0 or sequence_index >= len(self.project.sequences):
            QMessageBox.warning(self, "Export EDL", "Sequence not found")
            return

        self._persist_sequence_tab_state_for_export()
        sequence = self.project.sequences[sequence_index]
        all_clips = sequence.get_all_clips()

        if not all_clips:
            QMessageBox.information(
                self, "Export EDL", "No clips in this sequence to export"
            )
            return

        # Get output file path
        default_name = self._default_sequence_edl_filename(sequence)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export EDL",
            str(Path.home() / default_name),
            "Edit Decision List (*.edl);;All Files (*)",
        )
        if not file_path:
            return

        output_path = self._normalize_edl_output_path(Path(file_path))

        self.status_bar.showMessage("Exporting EDL...")
        success = self._export_sequence_edl_to_path(sequence, output_path)

        if success:
            self.status_bar.showMessage(f"EDL exported to {output_path.name}", 5000)
            # Open containing folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))
        else:
            QMessageBox.warning(self, "Export EDL", "Failed to export EDL file")

    def _on_all_sequence_edl_export_requested(self):
        """Export every populated project sequence as its own EDL file."""
        if not self.project:
            QMessageBox.warning(self, "Export EDL", "No project is open")
            return

        self._persist_sequence_tab_state_for_export()
        populated = [
            (index, sequence)
            for index, sequence in enumerate(self.project.sequences)
            if sequence.get_all_clips()
        ]
        if not populated:
            QMessageBox.information(
                self, "Export EDL", "No populated sequences to export"
            )
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Export Sequence EDLs",
            str(getattr(getattr(self, "settings", None), "export_dir", Path.home())),
        )
        if not output_dir:
            return

        output_dir_path = Path(output_dir)
        exported: list[Path] = []
        failed: list[str] = []

        self.status_bar.showMessage("Exporting sequence EDLs...")
        for index, sequence in populated:
            output_path = self._unique_edl_output_path(
                output_dir_path,
                self._default_sequence_edl_filename(sequence, index=index),
            )
            if self._export_sequence_edl_to_path(sequence, output_path):
                exported.append(output_path)
            else:
                failed.append(sequence.name or f"Sequence {index + 1}")

        if failed:
            QMessageBox.warning(
                self,
                "Export EDL",
                "Some sequences could not be exported:\n" + "\n".join(failed),
            )

        if exported:
            self.status_bar.showMessage(
                f"Exported {len(exported)} sequence EDL(s)", 5000
            )
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir_path)))
        else:
            self.status_bar.showMessage("No EDL files exported", 5000)

    def _on_export_edl_click(self):
        """Export a user-selected sequence as an EDL file from the File menu."""
        index = self._prompt_for_edl_sequence_index()
        if index is not None:
            self._on_sequence_edl_export_requested(index)

    def _on_export_bundle_click(self):
        """Export the project as a self-contained bundle folder."""
        import shutil
        from ui.dialogs.export_bundle_dialog import ExportBundleDialog

        dialog = ExportBundleDialog(self.project, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        dest_dir = dialog.dest_dir
        include_videos = dialog.include_videos

        if not dest_dir or str(dest_dir).strip() == "":
            return

        # Overwrite confirmation
        if dest_dir.exists():
            reply = QMessageBox.question(
                self,
                "Export Project Bundle",
                f"Destination already exists:\n{dest_dir}\n\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            shutil.rmtree(dest_dir)

        # Start background export
        self.export_bundle_worker = ExportBundleWorker(
            project=self.project,
            dest_dir=dest_dir,
            include_videos=include_videos,
            parent=self,
        )
        self.export_bundle_worker.progress.connect(self._on_export_bundle_progress)
        self.export_bundle_worker.export_completed.connect(self._on_export_bundle_finished)
        self.export_bundle_worker.error.connect(self._on_export_bundle_error)

        self.status_bar.showMessage("Exporting project bundle...")
        self.export_bundle_worker.start()

    def _on_export_bundle_progress(self, current: int, total: int, filename: str):
        """Update status bar with export progress."""
        if total > 0:
            self.status_bar.showMessage(
                f"Exporting bundle: {current}/{total} files — {filename}"
            )

    def _on_export_bundle_finished(self, result):
        """Handle successful bundle export."""
        self.export_bundle_worker = None

        parts = []
        if result.sources_copied:
            parts.append(f"{result.sources_copied} source(s)")
        if result.clips_exported:
            parts.append(f"{result.clips_exported} trimmed clip(s)")
        if result.frames_copied:
            parts.append(f"{result.frames_copied} frame(s)")

        summary = ", ".join(parts) if parts else "project file only"

        # Include warning about skipped files
        warnings = []
        if result.sources_skipped:
            warnings.append(f"{len(result.sources_skipped)} source(s) missing")
        if result.clips_skipped:
            warnings.append(f"{result.clips_skipped} clip(s) failed")
        if result.frames_skipped:
            warnings.append(f"{len(result.frames_skipped)} frame(s) missing")

        msg = f"Bundle exported: {summary}"
        if warnings:
            msg += f"\n\nSkipped: {', '.join(warnings)}"

        self.status_bar.showMessage(f"Bundle exported to {result.dest_dir.name}", 5000)

        # If agent was waiting for bundle export, send result back
        if self._pending_agent_export_bundle and self._chat_worker:
            self._pending_agent_export_bundle = False
            agent_result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "output_path": str(result.dest_dir),
                    "sources_copied": result.sources_copied,
                    "clips_exported": result.clips_exported,
                    "frames_copied": result.frames_copied,
                    "message": msg,
                }
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(agent_result)
            logger.info(f"Sent bundle export result to agent: {result.dest_dir}")
        else:
            QMessageBox.information(self, "Export Project Bundle", msg)

            # Open containing folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.dest_dir.parent)))

    def _on_export_bundle_error(self, error_msg: str):
        """Handle bundle export error."""
        self.export_bundle_worker = None
        self.status_bar.showMessage("Bundle export failed", 5000)

        # If agent was waiting for bundle export, send error result
        if self._pending_agent_export_bundle and self._chat_worker:
            self._pending_agent_export_bundle = False
            agent_result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": False,
                "error": error_msg
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._chat_worker.set_gui_tool_result(agent_result)
            logger.info(f"Sent bundle export error to agent: {error_msg}")
        else:
            QMessageBox.warning(
                self,
                "Export Project Bundle",
                f"Export failed:\n{error_msg}",
            )

    # ==================== Project Save/Load ====================

    def _mark_dirty(self):
        """Mark the project as having unsaved changes."""
        if not self.project.is_dirty:
            self.project.mark_dirty()
            self._update_window_title()

    def _mark_clean(self):
        """Mark the project as having no unsaved changes."""
        self.project.mark_clean()
        self._update_window_title()

    def _update_window_title(self):
        """Update window title to reflect project state."""
        base_title = "Scene Ripper"
        if self.current_project_path:
            title = f"{base_title} - {self.current_project_path.name}"
        elif self.current_source:
            title = f"{base_title} - {self.current_source.filename}"
        else:
            title = base_title

        if self._is_dirty:
            title += "*"

        self.setWindowTitle(title)

    def _get_recent_projects(self) -> list[str]:
        """Get list of recent project paths from settings."""
        from PySide6.QtCore import QSettings
        qsettings = QSettings()
        recent = qsettings.value("recent_projects", [])
        if isinstance(recent, str):
            return [recent] if recent else []
        return list(recent) if recent else []

    def _add_recent_project(self, path: Path):
        """Add a project to the recent projects list."""
        from PySide6.QtCore import QSettings
        qsettings = QSettings()
        recent = self._get_recent_projects()

        path_str = str(path.resolve())
        # Remove if already exists (to move to top)
        if path_str in recent:
            recent.remove(path_str)
        # Add to front
        recent.insert(0, path_str)
        # Keep only last 10
        recent = recent[:10]

        qsettings.setValue("recent_projects", recent)
        qsettings.sync()
        self._update_recent_projects_menu()

    def _update_recent_projects_menu(self):
        """Update the Recent Projects submenu."""
        self.recent_projects_menu.clear()
        recent = self._get_recent_projects()

        if not recent:
            action = QAction("(No recent projects)", self)
            action.setEnabled(False)
            self.recent_projects_menu.addAction(action)
            return

        for path_str in recent:
            path = Path(path_str)
            action = QAction(path.name, self)
            action.setToolTip(path_str)
            action.triggered.connect(lambda checked, p=path: self._open_recent_project(p))
            self.recent_projects_menu.addAction(action)

        self.recent_projects_menu.addSeparator()
        clear_action = QAction("Clear Recent", self)
        clear_action.triggered.connect(self._clear_recent_projects)
        self.recent_projects_menu.addAction(clear_action)

    def _clear_recent_projects(self):
        """Clear the recent projects list."""
        from PySide6.QtCore import QSettings
        qsettings = QSettings()
        qsettings.setValue("recent_projects", [])
        qsettings.sync()
        self._update_recent_projects_menu()

    def _open_recent_project(self, path: Path):
        """Open a recent project file."""
        if not path.exists():
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"The project file no longer exists:\n{path}"
            )
            # Remove from recent list
            from PySide6.QtCore import QSettings
            qsettings = QSettings()
            recent = self._get_recent_projects()
            if str(path) in recent:
                recent.remove(str(path))
                qsettings.setValue("recent_projects", recent)
                qsettings.sync()
                self._update_recent_projects_menu()
            return

        self._load_project_file(path)

    def _on_new_project(self):
        """Handle New Project action - reset to initial state."""
        if not self._check_unsaved_changes():
            return

        # Clear all project state
        self._clear_project_state()

        # Reset project tracking
        self.current_project_path = None
        self.project_metadata = None
        self._mark_clean()

        # Update window title to default
        self._update_window_title()

        self.status_bar.showMessage("New project created", 3000)
        logger.info("Created new project")

    def _on_select_all(self):
        """Handle Select All action (Cmd+A) - select all items in the active tab."""
        current_tab = self.tab_widget.currentWidget()

        if current_tab == self.collect_tab:
            self.collect_tab.source_browser.select_all()
            count = len(self.collect_tab.source_browser.selected_source_ids)
            self.status_bar.showMessage(f"Selected {count} videos", 2000)
        elif current_tab == self.cut_tab:
            self.cut_tab.clip_browser.select_all()
            count = len(self.cut_tab.clip_browser.selected_clips)
            self.status_bar.showMessage(f"Selected {count} clips", 2000)
            self.cut_tab._update_selection_ui()
        elif current_tab == self.analyze_tab:
            self.analyze_tab.clip_browser.select_all()
            count = len(self.analyze_tab.clip_browser.selected_clips)
            self.status_bar.showMessage(f"Selected {count} clips", 2000)
            self.analyze_tab._update_selection_ui()

    def _on_deselect_all(self):
        """Handle Deselect All action (Cmd+Shift+A) - clear selection in the active tab."""
        current_tab = self.tab_widget.currentWidget()

        if current_tab == self.collect_tab:
            self.collect_tab.source_browser.clear_selection()
            self.status_bar.showMessage("Selection cleared", 2000)
        elif current_tab == self.cut_tab:
            self.cut_tab.clip_browser.clear_selection()
            self.status_bar.showMessage("Selection cleared", 2000)
            self.cut_tab._update_selection_ui()
        elif current_tab == self.analyze_tab:
            self.analyze_tab.clip_browser.clear_selection()
            self.status_bar.showMessage("Selection cleared", 2000)
            self.analyze_tab._update_selection_ui()

    def _on_open_project(self):
        """Handle Open Project action."""
        if not self._check_unsaved_changes():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.home()),
            "Scene Ripper Projects (*.sceneripper);;All Files (*)",
        )

        if file_path:
            self._load_project_file(Path(file_path))

    def _on_save_project(self):
        """Handle Save Project action."""
        if self.current_project_path:
            self._save_project_to_file(self.current_project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        """Handle Save Project As action."""
        default_name = "project.sceneripper"
        if self.current_project_path:
            # Keep existing path but ensure .sceneripper extension for new saves
            default_path = str(self.current_project_path.with_suffix(".sceneripper"))
        elif self.current_source:
            default_path = str(self.current_source.file_path.parent / f"{self.current_source.file_path.stem}.sceneripper")
        else:
            default_path = str(Path.home() / default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            default_path,
            "Scene Ripper Projects (*.sceneripper);;All Files (*)",
        )

        if file_path:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".sceneripper")
            self._save_project_to_file(path)

    def _save_project_to_file(self, filepath: Path):
        """Save project to the specified file."""
        self.status_bar.showMessage("Saving project...")

        # Persist current timeline state to the project's active sequence
        self.sequence_tab._persist_current_sequence()

        # Get UI state and update project
        self.project.ui_state = {
            "analyze_clip_ids": self.analyze_tab.get_clip_ids(),
        }

        # Update metadata name to match filename
        self.project.metadata.name = filepath.stem

        # Save using Project class
        success = self.project.save(filepath)

        if success:
            self._add_recent_project(filepath)
            self._update_window_title()
            self.status_bar.showMessage(f"Project saved: {filepath.name}")
        else:
            QMessageBox.warning(self, "Save Project", "Failed to save project")
            self.status_bar.showMessage("Save failed")

    def _load_project_file(self, filepath: Path):
        """Load project from the specified file."""
        self.status_bar.showMessage("Loading project...")

        def handle_missing_source(missing_path: Path, source_id: str) -> Optional[Path]:
            """Callback to handle missing source files."""
            result = QMessageBox.question(
                self,
                "Missing Source Video",
                f"Source video not found:\n{missing_path}\n\nWould you like to locate it?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )

            if result == QMessageBox.Yes:
                new_path, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Locate {missing_path.name}",
                    str(missing_path.parent) if missing_path.parent.exists() else str(Path.home()),
                    "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)",
                )
                if new_path:
                    return Path(new_path)
                return None
            elif result == QMessageBox.No:
                return None  # Skip this source
            else:
                raise ProjectLoadError("Load cancelled by user")

        try:
            loaded_project = Project.load(
                filepath,
                missing_source_callback=handle_missing_source,
            )
        except ProjectLoadError as e:
            QMessageBox.warning(self, "Load Project", f"Failed to load project:\n{e}")
            self.status_bar.showMessage("Load failed")
            return

        if not loaded_project.sources:
            QMessageBox.warning(
                self,
                "Load Project",
                "No valid sources found in project."
            )
            self.status_bar.showMessage("Load failed - no sources")
            return

        # Clear existing UI state
        self._clear_project_state()

        # Set the new project
        self.project = loaded_project
        self._project_adapter.set_project(self.project)

        # Set project on Frames tab
        self.frames_tab.set_project(self.project)

        # Add all sources to CollectTab
        for source in self.sources:
            self.collect_tab.add_source(source)
            # Generate source thumbnails if missing
            if not source.thumbnail_path or not source.thumbnail_path.exists():
                self._generate_source_thumbnail(source)

        # Restore audio sources in the Collect tab
        self.collect_tab.set_audio_sources(self.project.audio_sources)

        # Set first source as current (for backwards compatibility)
        self.current_source = self.sources[0]

        # Set lookups for Analyze tab
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        # Update Cut tab with ALL clips from ALL sources
        self.cut_tab.set_source(self.current_source)
        self.cut_tab.clear_clips()
        self.cut_tab.set_clips(self.clips)
        self._populate_cut_tab_clip_browser_incrementally(
            filepath,
            on_finished=self._regenerate_missing_thumbnails,
        )

        # Set project reference on sequence tab and restore timeline state
        self.sequence_tab.set_project(self.project)
        self._refresh_timeline_from_project()

        # Restore UI state
        ui_state = self.project.ui_state

        # Restore Analyze tab clips
        if "analyze_clip_ids" in ui_state:
            # Validate clip IDs exist before restoring
            valid_clip_ids = [cid for cid in ui_state["analyze_clip_ids"]
                              if cid in self.clips_by_id]
            if valid_clip_ids:
                self.analyze_tab.add_clips(valid_clip_ids)
                logger.info(f"Restored {len(valid_clip_ids)} clips to Analyze tab")

        self._add_recent_project(filepath)

        # Update UI
        self._on_sequence_changed()  # Updates Export EDL menu state
        self._update_window_title()
        self._update_chat_project_state()  # Context-aware chat prompts
        self.status_bar.showMessage(
            f"Project loaded: {filepath.name} ({len(self.clips)} clips)"
        )

    def _populate_cut_tab_clip_browser_incrementally(
        self,
        filepath: Path,
        *,
        on_finished=None,
    ) -> None:
        """Populate the Cut tab clip grid in small UI-thread batches."""
        clip_source_pairs = []
        for clip in self.clips:
            clip_source = self.sources_by_id.get(clip.source_id)
            if not clip_source:
                logger.warning(
                    "Clip %s references unknown source %s",
                    clip.id,
                    clip.source_id,
                )
                continue
            clip_source_pairs.append((clip, clip_source))

        total = len(clip_source_pairs)
        if total == 0:
            if on_finished:
                on_finished()
            return

        generation = getattr(self, "_project_load_generation", 0) + 1
        self._project_load_generation = generation
        batch_size = 75

        def add_batch(start: int = 0) -> None:
            if getattr(self, "_project_load_generation", None) != generation:
                return

            batch = clip_source_pairs[start:start + batch_size]
            if batch:
                self.cut_tab.clip_browser.add_clips(batch)

            loaded = min(start + len(batch), total)
            if loaded < total:
                self.status_bar.showMessage(
                    f"Loading clip browser: {loaded}/{total} clips..."
                )
                QTimer.singleShot(0, lambda: add_batch(loaded))
                return

            self.status_bar.showMessage(
                f"Project loaded: {filepath.name} ({len(self.clips)} clips)"
            )
            if on_finished:
                on_finished()

        self.status_bar.showMessage(f"Loading clip browser: 0/{total} clips...")
        QTimer.singleShot(0, add_batch)

    def _clear_project_state(self):
        """Clear current project state.

        MAINTAINABILITY NOTE: When adding new workers, state flags, or clip lists
        to MainWindow, you MUST also update this method and _stop_all_workers()
        to ensure proper cleanup on New Project. Failure to do so will cause
        state leakage between projects.
        """
        # Stop playback and timers first
        self._stop_playback()
        self._auto_save_timer.stop()
        self._project_load_generation = getattr(self, "_project_load_generation", 0) + 1

        # Stop any running workers
        self._stop_all_workers()

        # Clear project data
        self.project.clear()

        # Clear UI state
        self.current_source = None
        self._analyze_queue = deque()
        self._analyze_queue_total = 0
        self._detection_start_time = None
        self._rendered_sequence_preview_path = None
        self._rendered_sequence_preview_signature = None
        self._sequence_preview_play_after_render_frame = None
        self._using_rendered_sequence_preview = False
        self._detection_current_progress = 0.0
        self.queue_label.setVisible(False)
        self.collect_tab.clear()
        self.cut_tab.clear_clips()
        self.cut_tab.set_source(None)
        self.analyze_tab.clear_clips()
        self.frames_tab.frame_browser.clear()
        self.sequence_tab.clear()  # Clear all state including _clips and _available_clips
        self.sequence_tab.set_project(self.project)  # Re-sync dropdown after clear
        self.render_tab.clear()  # Clear render tab state

        # Clear progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Clear YouTube search panel (results and search input)
        self.collect_tab.youtube_search_panel.clear()

        # Clear chat panel and history
        self.chat_panel.clear_messages()
        self._chat_history.clear()
        self._last_user_message = ""

        # Clear all agent pending flags
        self._pending_agent_detection = False
        self._pending_agent_color_analysis = False
        self._pending_agent_shot_analysis = False
        self._pending_agent_transcription = False
        self._pending_agent_classification = False
        self._pending_agent_object_detection = False
        self._pending_agent_description = False
        self._pending_agent_analyze_all = False
        self._pending_agent_export = False
        self._pending_agent_download = False
        self._pending_agent_tool_call_id = None
        self._pending_agent_tool_name = None

        # Clear agent clip tracking lists
        self._agent_color_clips = []
        self._agent_shot_clips = []
        self._agent_transcription_clips = []
        self._agent_classification_clips = []
        self._agent_object_detection_clips = []
        self._agent_description_clips = []
        self._agent_download_results = []
        if hasattr(self, '_agent_transcription_source_queue'):
            self._agent_transcription_source_queue = []

        # Clear intention workflow pending state
        if hasattr(self, '_intention_pending_algorithm'):
            self._intention_pending_algorithm = None
        if hasattr(self, '_intention_pending_direction'):
            self._intention_pending_direction = None
        if hasattr(self, '_intention_pending_shot_type'):
            self._intention_pending_shot_type = None
        if hasattr(self, '_intention_pending_poem_length'):
            self._intention_pending_poem_length = None
        if hasattr(self, '_intention_pending_poem_form'):
            self._intention_pending_poem_form = None
        if hasattr(self, '_intention_pending_storyteller_duration'):
            self._intention_pending_storyteller_duration = None
        if hasattr(self, '_intention_pending_storyteller_structure'):
            self._intention_pending_storyteller_structure = None
        if hasattr(self, '_intention_pending_storyteller_theme'):
            self._intention_pending_storyteller_theme = None

        # Clear GUI state (for agent context)
        self._gui_state.clear()

    def _refresh_ui_from_project(self):
        """Refresh all UI components after project load.

        Called after setting a new project to update all UI elements.
        """
        # Add all sources to CollectTab
        for source in self.sources:
            self.collect_tab.add_source(source)
            # Generate source thumbnails if missing
            if not source.thumbnail_path or not source.thumbnail_path.exists():
                self._generate_source_thumbnail(source)

        # Restore audio sources in the Collect tab
        self.collect_tab.set_audio_sources(self.project.audio_sources)

        # Set first source as current
        if self.sources:
            self.current_source = self.sources[0]

            # Set lookups for Analyze tab
            self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

            # Update Cut tab with ALL clips from ALL sources
            self.cut_tab.set_source(self.current_source)
            self.cut_tab.clear_clips()
            self.cut_tab.set_clips(self.clips)

            # Add clips to Cut tab browser with their correct source
            for clip in self.clips:
                clip_source = self.sources_by_id.get(clip.source_id)
                if not clip_source:
                    logger.warning(f"Clip {clip.id} references unknown source {clip.source_id}")
                    continue
                self.cut_tab.add_clip(clip, clip_source)
                # Update colors and shot type if present
                if clip.dominant_colors:
                    self.cut_tab.update_clip_colors(clip.id, clip.dominant_colors)
                if clip.shot_type:
                    self.cut_tab.update_clip_shot_type(clip.id, clip.shot_type)
                if clip.gaze_category:
                    self.cut_tab.update_clip_gaze(clip.id, clip.gaze_category)
                if clip.transcript:
                    self.cut_tab.update_clip_transcript(clip.id, clip.transcript)

        # Set project reference on sequence tab and restore timeline state
        self.sequence_tab.set_project(self.project)
        self._refresh_timeline_from_project()

        # Restore UI state
        ui_state = self.project.ui_state

        # Restore Analyze tab clips
        if "analyze_clip_ids" in ui_state:
            valid_clip_ids = [cid for cid in ui_state["analyze_clip_ids"]
                              if cid in self.clips_by_id]
            if valid_clip_ids:
                self.analyze_tab.add_clips(valid_clip_ids)

        # Update UI
        self._on_sequence_changed()
        self._update_window_title()
        self._update_chat_project_state()
        self.status_bar.showMessage(
            f"Project loaded: {self.project.metadata.name} ({len(self.clips)} clips)"
        )

        # Regenerate missing thumbnails
        self._regenerate_missing_thumbnails()

    def _regenerate_missing_thumbnails(self):
        """Regenerate thumbnails for clips that don't have them."""
        logger.info("_regenerate_missing_thumbnails called")
        logger.info(f"  self.clips count: {len(self.clips) if self.clips else 0}")
        logger.info(f"  self.sources count: {len(self.sources) if self.sources else 0}")
        logger.info(f"  self.current_source: {self.current_source.id if self.current_source else None}")
        logger.info(f"  sources_by_id keys: {list(self.sources_by_id.keys()) if self.sources_by_id else []}")

        if not self.clips:
            logger.warning("  No clips, returning early")
            return

        # Check which clips need thumbnails
        clips_needing_thumbnails = [
            clip for clip in self.clips
            if not clip.thumbnail_path or not clip.thumbnail_path.exists()
        ]

        if not clips_needing_thumbnails:
            logger.info("  All clips have thumbnails, returning")
            return

        # Log clip source IDs to verify they match
        for clip in clips_needing_thumbnails[:3]:  # Log first 3
            logger.info(f"  Clip {clip.id[:8]} source_id: {clip.source_id}")

        logger.info(f"Regenerating thumbnails for {len(clips_needing_thumbnails)} clips")
        self.status_bar.showMessage(f"Regenerating {len(clips_needing_thumbnails)} thumbnails...")

        try:
            # Safely stop any running thumbnail worker first
            self._stop_worker_safely(self.thumbnail_worker, "thumbnail")

            # Use existing ThumbnailWorker with project-load-specific handlers
            # Pass sources_by_id so each clip uses its correct source
            self.thumbnail_worker = ThumbnailWorker(
                self.current_source,
                clips_needing_thumbnails,
                self.settings.thumbnail_cache_dir,
                sources_by_id=self.sources_by_id,
            )
            # Use handlers that update existing clips instead of adding new ones
            self.thumbnail_worker.thumbnail_ready.connect(self._on_project_thumbnail_ready)
            self.thumbnail_worker.finished.connect(self._on_project_thumbnails_finished, Qt.UniqueConnection)
            logger.info("Starting ThumbnailWorker for project load...")
            self.thumbnail_worker.start()
            logger.info(f"ThumbnailWorker started, isRunning: {self.thumbnail_worker.isRunning()}")
        except Exception as e:
            logger.error(f"Failed to start thumbnail worker: {e}", exc_info=True)

    def _on_project_thumbnail_ready(self, clip_id: str, thumb_path: str):
        """Handle individual thumbnail during project load (update, don't add)."""
        logger.info(f"_on_project_thumbnail_ready: clip_id={clip_id}, thumb_path={thumb_path}")
        clip = self.clips_by_id.get(clip_id)
        if clip:
            thumb_path_obj = Path(thumb_path)
            logger.info(f"  thumbnail exists: {thumb_path_obj.exists()}")
            clip.thumbnail_path = thumb_path_obj
            self.cut_tab.update_clip_thumbnail(clip_id, thumb_path_obj)
            self.analyze_tab.update_clip_thumbnail(clip_id, thumb_path_obj)
        else:
            logger.warning("  clip not found in clips_by_id!")

    def _on_project_thumbnails_finished(self):
        """Handle thumbnails completed during project load (no analysis restart)."""
        logger.info("Project thumbnail regeneration finished")
        self.status_bar.showMessage("Project loaded - thumbnails regenerated", 3000)

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user.

        Returns:
            True if safe to proceed, False if user cancelled.
        """
        if not self._is_dirty:
            return True

        project_name = self.current_project_path.name if self.current_project_path else "Untitled"
        result = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Save changes to '{project_name}'?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if result == QMessageBox.Save:
            self._on_save_project()
            return not self._is_dirty  # True if save succeeded
        elif result == QMessageBox.Discard:
            return True
        else:  # Cancel
            return False

    def closeEvent(self, event):
        """Clean up workers and timers before closing."""
        logger.info("=== CLOSE EVENT ===")

        # Check for unsaved changes
        if not self._check_unsaved_changes():
            event.ignore()
            return

        if os.environ.get("SCENE_RIPPER_STARTUP_SMOKE_TEST") == "1":
            logger.info("Skipping native shutdown during startup smoke test")
            event.accept()
            return

        # Stop playback timer
        if self._playback_timer.isActive():
            logger.info("Stopping playback timer")
            self._playback_timer.stop()

        # Shutdown MPV players (must happen from main thread before exit)
        logger.info("Shutting down video players")
        self.sequence_tab.video_player.shutdown()
        self.clip_details_sidebar.video_player.shutdown()

        workers = [
            ("detection", self.detection_worker),
            ("thumbnail", self.thumbnail_worker),
            ("download", self.download_worker),
            ("export", self.export_worker),
            ("color", self.color_worker),
            ("shot_type", self.shot_type_worker),
            ("transcription", self.transcription_worker),
            ("classification", self.classification_worker),
            ("object_detection", self.detection_worker_yolo),
            ("gaze", getattr(self, '_gaze_worker', None)),
        ]

        for name, worker in workers:
            if worker:
                logger.info(f"{name} worker running: {worker.isRunning()}")
                if worker.isRunning():
                    logger.info(f"Stopping {name} worker...")
                    # Try graceful cancellation if supported
                    if hasattr(worker, 'cancel'):
                        worker.cancel()
                    # Wait up to 3 seconds for graceful shutdown
                    if not worker.wait(3000):
                        # SEVERE WARNING: terminate() is dangerous but acceptable during shutdown
                        # At app exit, we have no choice but to stop threads
                        logger.critical(
                            f"SEVERE: {name} worker did not stop gracefully after 3s, "
                            f"forcefully terminating. This may cause corruption!"
                        )
                        worker.terminate()
                        worker.wait(1000)
                    logger.info(f"{name} worker stopped")

        if sys.platform == "win32":
            from core.windows_updater import shutdown as shutdown_windows_updater

            shutdown_windows_updater()

        event.accept()
