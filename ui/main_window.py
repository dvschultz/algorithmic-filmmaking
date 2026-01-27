"""Main application window."""

import logging
import re
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
    QLineEdit,
    QTabWidget,
    QLabel,
    QDockWidget,
)
from PySide6.QtCore import Qt, Signal, QThread, QUrl, QTimer, Slot
from PySide6.QtGui import QDesktopServices, QKeySequence, QAction, QDragEnterEvent, QDropEvent

from models.clip import Source, Clip
from core.scene_detect import SceneDetector, DetectionConfig
from core.thumbnail import ThumbnailGenerator
from core.downloader import VideoDownloader
from core.sequence_export import SequenceExporter, ExportConfig
from core.dataset_export import export_dataset, DatasetExportConfig
from core.edl_export import export_edl, EDLExportConfig
from core.analysis.color import extract_dominant_colors
from core.analysis.shots import classify_shot_type
from core.settings import load_settings, save_settings, migrate_from_qsettings
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
from ui.tabs import CollectTab, CutTab, AnalyzeTab, GenerateTab, SequenceTab, RenderTab
from ui.theme import theme
from ui.chat_panel import ChatPanel
from ui.chat_worker import ChatAgentWorker
from ui.clip_details_sidebar import ClipDetailsSidebar
from core.gui_state import GUIState

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


class DetectionWorker(QThread):
    """Background worker for scene detection."""

    progress = Signal(float, str)  # progress (0-1), status message
    finished = Signal(object, list)  # source, clips
    error = Signal(str)

    def __init__(self, video_path: Path, config: DetectionConfig):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self._cancelled = False
        logger.debug("DetectionWorker created")

    def cancel(self):
        """Request cancellation of the detection."""
        logger.info("DetectionWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("DetectionWorker.run() STARTING")
        try:
            if self._cancelled:
                logger.info("DetectionWorker cancelled before start")
                return
            detector = SceneDetector(self.config)
            source, clips = detector.detect_scenes_with_progress(
                self.video_path,
                lambda p, m: self.progress.emit(p, m),
            )
            if self._cancelled:
                logger.info("DetectionWorker cancelled after detection")
                return
            logger.info("DetectionWorker.run() emitting finished signal")
            self.finished.emit(source, clips)
            logger.info("DetectionWorker.run() COMPLETED")
        except Exception as e:
            if not self._cancelled:
                logger.error(f"DetectionWorker.run() ERROR: {e}")
                self.error.emit(str(e))


class ThumbnailWorker(QThread):
    """Background worker for thumbnail generation."""

    progress = Signal(int, int)  # current, total
    thumbnail_ready = Signal(str, str)  # clip_id, thumbnail_path
    finished = Signal()

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

        logger.info("ThumbnailWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("ThumbnailWorker.run() COMPLETED")


class DownloadWorker(QThread):
    """Background worker for video downloads."""

    progress = Signal(float, str)  # progress (0-100), status message
    finished = Signal(object)  # DownloadResult
    error = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the download."""
        self._cancelled = True

    def run(self):
        try:
            downloader = VideoDownloader()
            result = downloader.download(
                self.url,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
            )
            if result.success:
                self.finished.emit(result)
            else:
                self.error.emit(result.error or "Download failed")
        except Exception as e:
            self.error.emit(str(e))


class URLBulkDownloadWorker(QThread):
    """Background worker for downloading multiple videos from URLs in parallel."""

    progress = Signal(int, int, str)  # current, total, message
    video_finished = Signal(str, object)  # url, DownloadResult
    all_finished = Signal(list)  # list of result dicts

    MAX_WORKERS = 3  # Parallel download limit

    def __init__(self, urls: list[str], download_dir: Path):
        super().__init__()
        self.urls = urls
        self.download_dir = download_dir
        self._cancelled = False
        self._results = []
        self._completed_count = 0
        self._lock = None  # Initialized in run()

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def _download_single(self, url: str) -> dict:
        """Download a single URL (called from thread pool)."""
        downloader = VideoDownloader(download_dir=self.download_dir)

        try:
            valid, error = downloader.is_valid_url(url)
            if not valid:
                return {"url": url, "success": False, "error": error, "result": None}

            result = downloader.download(url)

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

        total = len(self.urls)
        self._lock = threading.Lock()
        self._completed_count = 0

        self.progress.emit(0, total, f"Starting {total} downloads...")

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_url = {executor.submit(self._download_single, url): url for url in self.urls}

            # Process results as they complete
            for future in as_completed(future_to_url):
                if self._cancelled:
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

        self.progress.emit(total, total, "Downloads complete")
        self.all_finished.emit(self._results)


class SequenceExportWorker(QThread):
    """Background worker for sequence export."""

    progress = Signal(float, str)  # progress (0-1), status message
    finished = Signal(object)  # output path (Path)
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
                self.finished.emit(self.config.output_path)
            else:
                self.error.emit("Export failed")
        except Exception as e:
            self.error.emit(str(e))


class ColorAnalysisWorker(QThread):
    """Background worker for color extraction from thumbnails."""

    progress = Signal(int, int)  # current, total
    color_ready = Signal(str, list)  # clip_id, colors (list of RGB tuples)
    finished = Signal()

    def __init__(self, clips: list[Clip]):
        super().__init__()
        self.clips = clips
        self._cancelled = False
        logger.debug("ColorAnalysisWorker created")

    def cancel(self):
        """Request cancellation of the color analysis."""
        logger.info("ColorAnalysisWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("ColorAnalysisWorker.run() STARTING")
        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("ColorAnalysisWorker cancelled")
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    colors = extract_dominant_colors(clip.thumbnail_path)
                    self.color_ready.emit(clip.id, colors)
            except Exception as e:
                logger.warning(f"Failed to extract colors for clip {clip.id}: {e}")
            self.progress.emit(i + 1, total)
        logger.info("ColorAnalysisWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("ColorAnalysisWorker.run() COMPLETED")


class ShotTypeWorker(QThread):
    """Background worker for shot type classification using CLIP."""

    progress = Signal(int, int)  # current, total
    shot_type_ready = Signal(str, str, float)  # clip_id, shot_type, confidence
    finished = Signal()

    def __init__(self, clips: list[Clip]):
        super().__init__()
        self.clips = clips
        self._cancelled = False
        logger.debug("ShotTypeWorker created")

    def cancel(self):
        """Request cancellation of the shot type classification."""
        logger.info("ShotTypeWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("ShotTypeWorker.run() STARTING")
        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("ShotTypeWorker cancelled")
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    shot_type, confidence = classify_shot_type(clip.thumbnail_path)
                    self.shot_type_ready.emit(clip.id, shot_type, confidence)
            except Exception as e:
                logger.warning(f"Shot type classification failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)
        logger.info("ShotTypeWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("ShotTypeWorker.run() COMPLETED")


class TranscriptionWorker(QThread):
    """Background worker for transcribing clips using faster-whisper."""

    progress = Signal(int, int)  # current, total
    transcript_ready = Signal(str, list)  # clip_id, segments (list of TranscriptSegment)
    finished = Signal()
    error = Signal(str)  # error message

    def __init__(self, clips: list[Clip], source: Source, model_name: str = "small.en", language: str = "en"):
        super().__init__()
        self.clips = clips
        self.source = source
        self.model_name = model_name
        self.language = language
        self._cancelled = False
        logger.debug("TranscriptionWorker created")

    def cancel(self):
        """Request cancellation of the transcription."""
        logger.info("TranscriptionWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("TranscriptionWorker.run() STARTING")
        from core.transcription import (
            transcribe_clip,
            FasterWhisperNotInstalledError,
            ModelDownloadError,
            TranscriptionError,
        )

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("TranscriptionWorker cancelled")
                break
            try:
                segments = transcribe_clip(
                    self.source.file_path,
                    clip.start_time(self.source.fps),
                    clip.end_time(self.source.fps),
                    self.model_name,
                    self.language,
                )
                self.transcript_ready.emit(clip.id, segments)
            except FasterWhisperNotInstalledError as e:
                logger.error(f"faster-whisper not installed: {e}")
                self.error.emit(str(e))
                break  # Stop processing - critical error
            except ModelDownloadError as e:
                logger.error(f"Model download failed: {e}")
                self.error.emit(str(e))
                break  # Stop processing - critical error
            except TranscriptionError as e:
                logger.warning(f"Transcription error for {clip.id}: {e}")
                # Continue processing other clips for non-critical errors
            except Exception as e:
                logger.warning(f"Transcription failed for {clip.id}: {e}")
                # Continue processing other clips
            self.progress.emit(i + 1, total)
        logger.info("TranscriptionWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("TranscriptionWorker.run() COMPLETED")


class ClassificationWorker(QThread):
    """Background worker for frame classification using MobileNet."""

    progress = Signal(int, int)  # current, total
    labels_ready = Signal(str, list)  # clip_id, [(label, confidence), ...]
    finished = Signal()

    def __init__(self, clips: list[Clip], top_k: int = 5, threshold: float = 0.1):
        super().__init__()
        self.clips = clips
        self.top_k = top_k
        self.threshold = threshold
        self._cancelled = False
        logger.debug("ClassificationWorker created")

    def cancel(self):
        """Request cancellation of the classification."""
        logger.info("ClassificationWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("ClassificationWorker.run() STARTING")
        from core.analysis.classification import classify_frame

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("ClassificationWorker cancelled")
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    results = classify_frame(
                        clip.thumbnail_path,
                        top_k=self.top_k,
                        threshold=self.threshold,
                    )
                    self.labels_ready.emit(clip.id, results)
            except Exception as e:
                logger.warning(f"Classification failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)
        logger.info("ClassificationWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("ClassificationWorker.run() COMPLETED")


class ObjectDetectionWorker(QThread):
    """Background worker for object detection using YOLOv8."""

    progress = Signal(int, int)  # current, total
    objects_ready = Signal(str, list, int)  # clip_id, detections, person_count
    finished = Signal()

    def __init__(self, clips: list[Clip], confidence: float = 0.5, detect_all: bool = True):
        super().__init__()
        self.clips = clips
        self.confidence = confidence
        self.detect_all = detect_all  # False = persons only (faster)
        self._cancelled = False
        logger.debug("ObjectDetectionWorker created")

    def cancel(self):
        """Request cancellation of the object detection."""
        logger.info("ObjectDetectionWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("ObjectDetectionWorker.run() STARTING")
        from core.analysis.detection import detect_objects, count_people

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("ObjectDetectionWorker cancelled")
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    if self.detect_all:
                        detections = detect_objects(
                            clip.thumbnail_path,
                            confidence_threshold=self.confidence,
                        )
                        person_count = sum(1 for d in detections if d["label"] == "person")
                    else:
                        detections = []
                        person_count = count_people(
                            clip.thumbnail_path,
                            confidence_threshold=self.confidence,
                        )
                    self.objects_ready.emit(clip.id, detections, person_count)
            except Exception as e:
                logger.warning(f"Object detection failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)
        logger.info("ObjectDetectionWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("ObjectDetectionWorker.run() COMPLETED")


class DescriptionWorker(QThread):
    """Background worker for generating video descriptions."""

    progress = Signal(int, int)  # current, total
    description_ready = Signal(str, str, str)  # clip_id, description, model_name
    finished = Signal()

    def __init__(self, clips: list[Clip], tier: Optional[str] = None, prompt: Optional[str] = None):
        super().__init__()
        self.clips = clips
        self.tier = tier
        self.prompt = prompt
        self._cancelled = False
        logger.debug("DescriptionWorker created")

    def cancel(self):
        """Request cancellation of the description generation."""
        logger.info("DescriptionWorker.cancel() called")
        self._cancelled = True

    def run(self):
        logger.info("DescriptionWorker.run() STARTING")
        from core.analysis.description import describe_frame

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                logger.info("DescriptionWorker cancelled")
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    description, model = describe_frame(
                        clip.thumbnail_path,
                        tier=self.tier,
                        prompt=self.prompt or "Describe this video frame in detail."
                    )
                    self.description_ready.emit(clip.id, description, model)
            except Exception as e:
                logger.warning(f"Description failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)
        logger.info("DescriptionWorker.run() emitting finished signal")
        self.finished.emit()
        logger.info("DescriptionWorker.run() COMPLETED")


class YouTubeSearchWorker(QThread):
    """Background worker for YouTube search."""

    finished = Signal(object)  # YouTubeSearchResult
    error = Signal(str)

    def __init__(self, client: YouTubeSearchClient, query: str, max_results: int = 25):
        super().__init__()
        self.client = client
        self.query = query
        self.max_results = max_results

    def run(self):
        try:
            result = self.client.search(self.query, self.max_results)
            self.finished.emit(result)
        except QuotaExceededError as e:
            self.error.emit(str(e))
        except InvalidAPIKeyError as e:
            self.error.emit(str(e))
        except YouTubeAPIError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Search failed: {e}")


class BulkDownloadWorker(QThread):
    """Background worker for parallel bulk downloads."""

    progress = Signal(int, int, str)  # current, total, message
    video_finished = Signal(object)  # DownloadResult
    video_error = Signal(str, str)  # video_id, error message
    all_finished = Signal()

    def __init__(self, videos: list[YouTubeVideo], download_dir: Path, max_parallel: int = 2):
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

    def _download_one(self, video: YouTubeVideo):
        """Download a single video."""
        logger.debug(f"Starting download: {video.title} ({video.video_id}) to {self.download_dir}")
        downloader = VideoDownloader(download_dir=self.download_dir)
        result = downloader.download(
            video.youtube_url,
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

        # Apply theme preference from settings
        theme().set_preference(self.settings.theme_preference)

        # Project state - single source of truth
        self.project = Project.new()
        self._project_adapter = ProjectSignalAdapter(self.project, self)

        # Connect project adapter signals for view synchronization
        self._project_adapter.clips_updated.connect(self._on_clips_updated)

        # UI state (not part of Project - these are GUI-specific selections)
        self.current_source: Optional[Source] = None  # Currently active/selected source
        self._analyze_queue: deque[Source] = deque()  # Queue for batch analysis (O(1) popleft)
        self._analyze_queue_total: int = 0  # Total count for progress display
        self.detection_worker: Optional[DetectionWorker] = None
        self.thumbnail_worker: Optional[ThumbnailWorker] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.url_bulk_download_worker: Optional[URLBulkDownloadWorker] = None
        self.export_worker: Optional[SequenceExportWorker] = None
        self.color_worker: Optional[ColorAnalysisWorker] = None
        self.shot_type_worker: Optional[ShotTypeWorker] = None
        self.transcription_worker: Optional[TranscriptionWorker] = None
        self.classification_worker: Optional[ClassificationWorker] = None
        self.detection_worker_yolo: Optional[ObjectDetectionWorker] = None
        self.description_worker: Optional[DescriptionWorker] = None
        self.youtube_search_worker: Optional[YouTubeSearchWorker] = None
        self.bulk_download_worker: Optional[BulkDownloadWorker] = None
        self.youtube_client: Optional[YouTubeSearchClient] = None

        # Guards to prevent duplicate signal handling
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._classification_finished_handled = False
        self._object_detection_finished_handled = False
        self._description_finished_handled = False
        self._shot_type_finished_handled = False
        self._transcription_finished_handled = False

        # State for "Analyze All" sequential processing
        self._analyze_all_pending: list[str] = []  # Pending steps: "colors", "shots", "transcribe"
        self._analyze_all_clips: list = []  # Clips being analyzed
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

        # Project path tracking (for backwards compatibility - delegates to project)
        # Note: self.project.path is the actual storage

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

        # Playback state (must be after _setup_ui so tabs are initialized)
        logger.info("Setting up playback state...")
        self._is_playing = False
        self._current_playback_clip = None  # Currently playing SequenceClip
        self._playback_timer = QTimer(self)  # Parent to self for proper lifecycle
        self._playback_timer.setInterval(33)  # ~30fps update rate
        self._playback_timer.timeout.connect(self._on_playback_tick)
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

    def _setup_ui(self):
        """Set up the user interface."""
        # Create menu bar
        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget for workflow pages
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Create tabs
        self.collect_tab = CollectTab()
        self.cut_tab = CutTab()
        self.analyze_tab = AnalyzeTab()
        self.generate_tab = GenerateTab()
        self.sequence_tab = SequenceTab()
        self.render_tab = RenderTab()

        # Add tabs
        self.tab_widget.addTab(self.collect_tab, "Collect")
        self.tab_widget.addTab(self.cut_tab, "Cut")
        self.tab_widget.addTab(self.analyze_tab, "Analyze")
        self.tab_widget.addTab(self.generate_tab, "Generate")
        self.tab_widget.addTab(self.sequence_tab, "Sequence")
        self.tab_widget.addTab(self.render_tab, "Render")

        # Set up Analyze tab lookups (it uses references, not copies)
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

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
        self.queue_label.setStyleSheet(f"color: {theme().text_secondary}; padding-right: 10px;")
        self.queue_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.queue_label)

    def _on_tab_changed(self, index: int):
        """Handle tab switching."""
        # Get all tabs
        tabs = [
            self.collect_tab,
            self.cut_tab,
            self.analyze_tab,
            self.generate_tab,
            self.sequence_tab,
            self.render_tab,
        ]

        # Track active tab for agent context
        tab_names = ["collect", "cut", "analyze", "generate", "sequence", "render"]
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

        # Notify tabs of activation/deactivation
        for i, tab in enumerate(tabs):
            if i == index:
                tab.on_tab_activated()
            else:
                tab.on_tab_deactivated()

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

        tab_names = ["&Collect", "&Analyze", "&Generate", "&Sequence", "&Render"]
        for i, name in enumerate(tab_names):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            action.triggered.connect(lambda checked, idx=i: self.tab_widget.setCurrentIndex(idx))
            view_menu.addAction(action)

        view_menu.addSeparator()

        # Store reference to view menu for chat panel toggle
        self._view_menu = view_menu

    def _is_any_worker_running(self) -> bool:
        """Check if any background worker is currently running."""
        workers = [
            self.detection_worker,
            self.thumbnail_worker,
            self.download_worker,
            self.export_worker,
            self.color_worker,
            self.shot_type_worker,
            self.transcription_worker,
            self.classification_worker,
            self.detection_worker_yolo,
            self.youtube_search_worker,
            self.bulk_download_worker,
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
        # Update sensitivity in Cut tab
        self.cut_tab.set_sensitivity(self.settings.default_sensitivity)

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
        self.collect_tab.analyze_requested.connect(self._on_analyze_requested)
        self.collect_tab.source_selected.connect(self._on_source_selected)
        self.collect_tab.download_requested.connect(self._on_download_requested_from_tab)

        # YouTube search panel signals
        self.collect_tab.youtube_search_panel.search_requested.connect(
            self._on_youtube_search
        )
        self.collect_tab.youtube_search_panel.download_requested.connect(
            self._on_bulk_download
        )

        # Cut tab signals
        self.cut_tab.detect_requested.connect(self._on_detect_from_tab)
        self.cut_tab.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)
        self.cut_tab.clips_sent_to_analyze.connect(self._on_clips_sent_to_analyze)
        self.cut_tab.selection_changed.connect(self._on_cut_selection_changed)
        self.cut_tab.clip_browser.filters_changed.connect(self._on_cut_filters_changed)
        self.cut_tab.clip_browser.view_details_requested.connect(self.show_clip_details)

        # Analyze tab signals
        self.analyze_tab.transcribe_requested.connect(self._on_transcribe_from_tab)
        self.analyze_tab.analyze_colors_requested.connect(self._on_analyze_colors_from_tab)
        self.analyze_tab.analyze_shots_requested.connect(self._on_analyze_shots_from_tab)
        self.analyze_tab.analyze_all_requested.connect(self._on_analyze_all_from_tab)
        self.analyze_tab.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)
        self.analyze_tab.selection_changed.connect(self._on_analyze_selection_changed)
        self.analyze_tab.clips_changed.connect(self._on_analyze_clips_changed)
        self.analyze_tab.clip_browser.filters_changed.connect(self._on_analyze_filters_changed)
        self.analyze_tab.clip_browser.view_details_requested.connect(self.show_clip_details)

        # Sequence tab signals
        self.sequence_tab.playback_requested.connect(self._on_playback_requested)
        self.sequence_tab.stop_requested.connect(self._on_stop_requested)
        self.sequence_tab.export_requested.connect(self._on_sequence_export_click)
        # Update Render tab when sequence changes (clips added/removed/generated)
        self.sequence_tab.timeline.sequence_changed.connect(self._update_render_tab_sequence_info)
        # Update EDL export menu item when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._on_sequence_changed)
        # Update agent context when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._on_sequence_ids_changed)
        # Mark project dirty when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._mark_dirty)

        # Render tab signals
        self.render_tab.export_sequence_requested.connect(self._on_sequence_export_click)
        self.render_tab.export_clips_requested.connect(self._on_export_click)
        self.render_tab.export_all_clips_requested.connect(self._on_export_all_click)
        self.render_tab.export_dataset_requested.connect(self._on_export_dataset_click)

        # Sequence tab timeline signals for playback sync
        self.sequence_tab.timeline.playhead_changed.connect(self._on_timeline_playhead_changed)

        # Sequence tab video player signals for playback sync
        self.sequence_tab.video_player.position_updated.connect(self._on_video_position_updated)
        self.sequence_tab.video_player.player.playbackStateChanged.connect(self._on_video_state_changed)

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

        Args:
            clips: List of updated clips
        """
        # Update clip browsers in both tabs
        if hasattr(self, 'cut_tab') and hasattr(self.cut_tab, 'clip_browser'):
            self.cut_tab.clip_browser.update_clips(clips)
        if hasattr(self, 'analyze_tab') and hasattr(self.analyze_tab, 'clip_browser'):
            self.analyze_tab.clip_browser.update_clips(clips)

    def _on_chat_message(self, message: str):
        """Handle user message from chat panel."""
        from core.llm_client import ProviderConfig, ProviderType
        from core.settings import (
            get_anthropic_api_key, get_openai_api_key,
            get_gemini_api_key, get_openrouter_api_key
        )

        # Store message for history
        self._last_user_message = message

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
                "analyze_colors": "color_worker",
                "analyze_shots": "shot_type_worker",
                "transcribe": "transcription_worker",
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
            gui_state_context=self._gui_state.to_context_string(),
        )

        # Connect signals
        bubble = self.chat_panel.start_streaming_response()
        self._current_chat_bubble = bubble

        self._chat_worker.text_chunk.connect(self.chat_panel.on_stream_chunk)
        self._chat_worker.clear_current_bubble.connect(self.chat_panel.on_clear_bubble)
        self._chat_worker.tool_called.connect(self._on_chat_tool_called)
        self._chat_worker.tool_result.connect(self._on_chat_tool_result)
        self._chat_worker.tool_result_formatted.connect(self.chat_panel.on_tool_result_formatted)
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
            "analyze_colors_live": "color_analysis_worker",
            "analyze_shots_live": "shot_type_worker",
            "transcribe_live": "transcription_worker",
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
                        # Worker couldn't start - send error immediately
                        result = {
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "success": False,
                            "error": f"Failed to start {wait_type} worker"
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
            tab_names = ["collect", "cut", "analyze", "sequence", "generate", "render"]
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

        elif tool_name == "add_to_sequence":
            # Refresh the timeline to show the newly added clips
            self._refresh_timeline_from_project()

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
            return

        # Build sources dict from project
        sources = {s.id: s for s in self.project.sources}

        # Load sequence into timeline
        self.sequence_tab.timeline.load_sequence(
            self.project.sequence,
            sources,
            self.project.clips
        )

        # Zoom to fit the content
        self.sequence_tab.timeline._on_zoom_fit()

        logger.info(f"Refreshed timeline with {len(self.project.sequence.tracks[0].clips)} clips")

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

        summary = f"**Plan stopped**\n\n"
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
            tab_name: Tab name ("collect", "cut", "analyze", "sequence", "generate", "render")
        """
        tab_map = {
            "collect": 0,
            "cut": 1,
            "analyze": 2,
            "sequence": 3,
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

    def _on_download_requested_from_tab(self, url: str):
        """Handle download request from Collect tab."""
        self._download_video(url)

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
        self._start_next_analysis()

    def _start_next_analysis(self):
        """Start analyzing the next source in the queue."""
        if not self._analyze_queue:
            self.status_bar.showMessage("Batch analysis complete")
            self.queue_label.setVisible(False)
            self._analyze_queue_total = 0
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
        self._start_detection(self.settings.default_sensitivity)
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

    def _add_video_to_library(self, path: Path):
        """Add a video file to the library without making it active."""
        # Check if already in library
        for source in self.sources:
            if source.file_path == path:
                self.status_bar.showMessage(f"Video already in library: {path.name}")
                return

        # Create new source and add to project
        source = Source(file_path=path)
        self.project.add_source(source)

        # Add to CollectTab grid
        self.collect_tab.add_source(source)

        # Generate thumbnail for the source
        self._generate_source_thumbnail(source)

        # Update chat panel with project state (new source added)
        self._update_chat_project_state()

        self.status_bar.showMessage(f"Added to library: {path.name}")

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

    def _on_detect_from_tab(self, threshold: float):
        """Handle detection request from Cut tab."""
        if not self.current_source:
            return
        # Start detection with the provided threshold
        self._start_detection(threshold)

    def _on_cut_selection_changed(self, clip_ids: list[str]):
        """Handle selection change in Cut tab."""
        self._gui_state.selected_clip_ids = clip_ids
        logger.debug(f"GUI State updated: {len(clip_ids)} clips selected")

    def _on_analyze_selection_changed(self, clip_ids: list[str]):
        """Handle selection change in Analyze tab."""
        self._gui_state.selected_clip_ids = clip_ids
        logger.debug(f"GUI State updated (Analyze): {len(clip_ids)} clips selected")

    def _on_cut_filters_changed(self):
        """Handle filter change in Cut tab."""
        filters = self.cut_tab.get_active_filters()
        self._gui_state.update_active_filters(filters)
        logger.debug(f"GUI State updated: Cut tab filters changed")

    def _on_analyze_filters_changed(self):
        """Handle filter change in Analyze tab."""
        filters = self.analyze_tab.get_active_filters()
        self._gui_state.update_active_filters(filters)
        logger.debug(f"GUI State updated: Analyze tab filters changed")

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

    def _on_clips_sent_to_analyze(self, clip_ids: list[str]):
        """Handle clips being sent from Cut to Analyze tab."""
        self.analyze_tab.add_clips(clip_ids)
        self.tab_widget.setCurrentWidget(self.analyze_tab)
        self.status_bar.showMessage(f"Sent {len(clip_ids)} clips to Analyze")

        # Automatically start "Analyze All" on the sent clips
        self._on_analyze_all_from_tab()

    def _on_analyze_colors_from_tab(self):
        """Handle color extraction request from Analyze tab."""
        clips = self.analyze_tab.get_clips()
        if not clips:
            return

        # Reset guard
        self._color_analysis_finished_handled = False

        # Update UI state
        self.analyze_tab.set_analyzing(True, "colors")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Extracting colors from {len(clips)} clips...")

        logger.info("Creating ColorAnalysisWorker (manual)...")
        self.color_worker = ColorAnalysisWorker(clips)
        self.color_worker.progress.connect(self._on_color_progress)
        self.color_worker.color_ready.connect(self._on_color_ready)
        self.color_worker.finished.connect(self._on_manual_color_analysis_finished, Qt.UniqueConnection)
        logger.info("Starting ColorAnalysisWorker (manual)...")
        self.color_worker.start()

    @Slot()
    def _on_manual_color_analysis_finished(self):
        """Handle manual color analysis completion."""
        logger.info("=== MANUAL COLOR ANALYSIS FINISHED ===")

        # Guard against duplicate calls
        if self._color_analysis_finished_handled:
            logger.warning("_on_manual_color_analysis_finished already handled, ignoring duplicate call")
            return
        self._color_analysis_finished_handled = True

        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)
        self.status_bar.showMessage(f"Color extraction complete - {len(self.analyze_tab.get_clips())} clips")

    def _on_analyze_shots_from_tab(self):
        """Handle shot type classification request from Analyze tab."""
        clips = self.analyze_tab.get_clips()
        if not clips:
            return

        # Reset guard
        self._shot_type_finished_handled = False

        # Update UI state
        self.analyze_tab.set_analyzing(True, "shots")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Classifying shot types for {len(clips)} clips...")

        logger.info("Creating ShotTypeWorker (manual)...")
        self.shot_type_worker = ShotTypeWorker(clips)
        self.shot_type_worker.progress.connect(self._on_shot_type_progress)
        self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
        self.shot_type_worker.finished.connect(self._on_manual_shot_type_finished, Qt.UniqueConnection)
        logger.info("Starting ShotTypeWorker (manual)...")
        self.shot_type_worker.start()

    @Slot()
    def _on_manual_shot_type_finished(self):
        """Handle manual shot type classification completion."""
        logger.info("=== MANUAL SHOT TYPE FINISHED ===")

        # Guard against duplicate calls
        if self._shot_type_finished_handled:
            logger.warning("_on_manual_shot_type_finished already handled, ignoring duplicate call")
            return
        self._shot_type_finished_handled = True

        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)
        self.status_bar.showMessage(f"Shot type classification complete - {len(self.analyze_tab.get_clips())} clips")

        # Update chat panel with project state (clips now have shot types)
        self._update_chat_project_state()

    def _on_transcribe_from_tab(self):
        """Handle manual transcription request from Analyze tab."""
        clips = self.analyze_tab.get_clips()
        if not clips:
            return

        # Check if faster-whisper is available
        from core.transcription import is_faster_whisper_available, WHISPER_MODELS
        if not is_faster_whisper_available():
            QMessageBox.critical(
                self,
                "Transcription Unavailable",
                "The faster-whisper package is not installed.\n\n"
                "To enable transcription, install it with:\n"
                "pip install faster-whisper\n\n"
                "Then restart the application."
            )
            return

        # Get source for the first clip (for audio extraction)
        # Note: All clips should have the same source for transcription to work
        first_clip = clips[0]
        source = self.sources_by_id.get(first_clip.source_id)
        if not source:
            logger.error(f"Source not found for clip {first_clip.id}")
            return

        # Reset transcription guard
        self._transcription_finished_handled = False

        # Update UI state
        self.analyze_tab.set_analyzing(True, "transcribe")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        # Show model info in status
        model_info = WHISPER_MODELS.get(self.settings.transcription_model, {})
        model_size = model_info.get("size", "unknown")
        self.status_bar.showMessage(
            f"Transcribing {len(clips)} clips using {self.settings.transcription_model} model ({model_size})..."
        )

        logger.info("Creating TranscriptionWorker (manual)...")
        self.transcription_worker = TranscriptionWorker(
            clips,
            source,
            self.settings.transcription_model,
            self.settings.transcription_language,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
        self.transcription_worker.finished.connect(self._on_manual_transcription_finished, Qt.UniqueConnection)
        self.transcription_worker.error.connect(self._on_transcription_error)
        logger.info("Starting TranscriptionWorker (manual)...")
        self.transcription_worker.start()

    def _on_transcription_error(self, error: str):
        """Handle transcription error."""
        logger.error(f"Transcription error: {error}")
        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)

        if "not installed" in error.lower():
            QMessageBox.critical(
                self,
                "Transcription Error",
                "The faster-whisper package is not installed.\n\n"
                "Install it with: pip install faster-whisper"
            )
        elif "download" in error.lower() or "model" in error.lower():
            QMessageBox.warning(
                self,
                "Model Download Error",
                f"Failed to download or load the Whisper model:\n\n{error}\n\n"
                "Check your internet connection and try again."
            )
        else:
            QMessageBox.warning(
                self,
                "Transcription Error",
                f"An error occurred during transcription:\n\n{error}"
            )

    @Slot()
    def _on_manual_transcription_finished(self):
        """Handle manual transcription completion."""
        logger.info("=== MANUAL TRANSCRIPTION FINISHED ===")

        # Guard against duplicate calls
        if self._transcription_finished_handled:
            logger.warning("_on_manual_transcription_finished already handled, ignoring duplicate call")
            return
        self._transcription_finished_handled = True

        self.progress_bar.setVisible(False)
        self.analyze_tab.set_analyzing(False)
        self.status_bar.showMessage(f"Transcription complete - {len(self.analyze_tab.get_clips())} clips")

        # Update chat panel with project state (clips now have transcripts)
        self._update_chat_project_state()

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
        self.status_bar.showMessage(f"Color extraction complete - {clip_count} clips")

        # Send result back to agent
        if self._pending_agent_color_analysis and self._chat_worker:
            self._pending_agent_color_analysis = False
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "clip_count": clip_count,
                    "clip_ids": [c.id for c in clips],
                    "message": f"Extracted colors from {clip_count} clips"
                }
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
        self.status_bar.showMessage(f"Shot type classification complete - {clip_count} clips")

        # Build shot type summary
        shot_types = {}
        for clip in clips:
            st = clip.shot_type or "unknown"
            shot_types[st] = shot_types.get(st, 0) + 1

        # Send result back to agent
        if self._pending_agent_shot_analysis and self._chat_worker:
            self._pending_agent_shot_analysis = False
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "clip_count": clip_count,
                    "clip_ids": [c.id for c in clips],
                    "shot_type_summary": shot_types,
                    "message": f"Classified shot types for {clip_count} clips"
                }
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

            # Start worker for next source
            self.transcription_worker = TranscriptionWorker(
                next_clips,
                next_source,
                self.settings.transcription_model,
                self.settings.transcription_language,
            )
            self.transcription_worker.progress.connect(self._on_transcription_progress)
            self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
            self.transcription_worker.finished.connect(self._on_agent_transcription_finished, Qt.UniqueConnection)
            self.transcription_worker.error.connect(self._on_transcription_error)
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
            result = {
                "tool_call_id": self._pending_agent_tool_call_id,
                "name": self._pending_agent_tool_name,
                "success": True,
                "result": {
                    "success": True,
                    "clip_count": clip_count,
                    "transcribed_count": transcribed_count,
                    "clip_ids": [c.id for c in clips],
                    "message": f"Transcribed {transcribed_count} of {clip_count} clips"
                }
            }
            self._pending_agent_tool_call_id = None
            self._pending_agent_tool_name = None
            self._agent_transcription_clips = []
            self._agent_transcription_source_queue = []
            self._chat_worker.set_gui_tool_result(result)
            logger.info(f"Sent transcription result to agent: {transcribed_count}/{clip_count} clips")

        # Update chat panel with project state
        self._update_chat_project_state()

    # "Analyze All" handlers - sequential colors → shots → transcribe

    def _on_analyze_all_from_tab(self):
        """Handle 'Analyze All' request from Analyze tab.

        Runs colors, shots, and transcribe sequentially on all clips in the tab.
        """
        clips = self.analyze_tab.get_clips()
        if not clips:
            return

        logger.info(f"Starting 'Analyze All' for {len(clips)} clips")

        # Store state for sequential processing
        self._analyze_all_pending = ["colors", "shots", "transcribe"]
        self._analyze_all_clips = clips

        # Update UI state
        self.analyze_tab.set_analyzing(True, "all")

        # Start with colors
        self._start_next_analyze_all_step()

    def _start_next_analyze_all_step(self):
        """Start the next step in the 'Analyze All' sequence."""
        if not self._analyze_all_pending:
            # All done
            logger.info("'Analyze All' complete")
            self.analyze_tab.set_analyzing(False)
            self.progress_bar.setVisible(False)

            clips = self._analyze_all_clips
            clip_count = len(clips)
            self.status_bar.showMessage(f"Analysis complete - {clip_count} clips")

            # If agent was waiting for analyze_all, send result back
            if self._pending_agent_analyze_all and self._chat_worker:
                self._pending_agent_analyze_all = False

                # Build summary
                shot_types = {}
                transcribed_count = 0
                for clip in clips:
                    if clip.shot_type:
                        shot_types[clip.shot_type] = shot_types.get(clip.shot_type, 0) + 1
                    if clip.transcript:
                        transcribed_count += 1

                result = {
                    "tool_call_id": self._pending_agent_tool_call_id,
                    "name": self._pending_agent_tool_name,
                    "success": True,
                    "result": {
                        "success": True,
                        "clip_count": clip_count,
                        "clip_ids": [c.id for c in clips],
                        "shot_type_summary": shot_types,
                        "transcribed_count": transcribed_count,
                        "message": f"Analyzed {clip_count} clips (colors, shots, transcription)"
                    }
                }
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent analyze_all result to agent: {clip_count} clips")

            self._analyze_all_clips = []
            return

        next_step = self._analyze_all_pending.pop(0)
        logger.info(f"'Analyze All' starting step: {next_step}")

        if next_step == "colors":
            self._start_color_analysis_for_analyze_all()
        elif next_step == "shots":
            self._start_shot_analysis_for_analyze_all()
        elif next_step == "transcribe":
            self._start_transcription_for_analyze_all()

    def _start_color_analysis_for_analyze_all(self):
        """Start color analysis as part of 'Analyze All' sequence."""
        clips = self._analyze_all_clips
        if not clips:
            self._start_next_analyze_all_step()
            return

        # Reset guard
        self._color_analysis_finished_handled = False

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Extracting colors from {len(clips)} clips...")

        logger.info("Creating ColorAnalysisWorker (Analyze All)...")
        self.color_worker = ColorAnalysisWorker(clips)
        self.color_worker.progress.connect(self._on_color_progress)
        self.color_worker.color_ready.connect(self._on_color_ready)
        self.color_worker.finished.connect(self._on_analyze_all_color_finished, Qt.UniqueConnection)
        self.color_worker.start()

    @Slot()
    def _on_analyze_all_color_finished(self):
        """Handle color analysis completion in 'Analyze All' flow."""
        logger.info("=== ANALYZE ALL: COLOR ANALYSIS FINISHED ===")

        # Guard against duplicate calls
        if self._color_analysis_finished_handled:
            logger.warning("_on_analyze_all_color_finished already handled, ignoring duplicate")
            return
        self._color_analysis_finished_handled = True

        # Continue to next step
        self._start_next_analyze_all_step()

    def _start_shot_analysis_for_analyze_all(self):
        """Start shot type classification as part of 'Analyze All' sequence."""
        clips = self._analyze_all_clips
        if not clips:
            self._start_next_analyze_all_step()
            return

        # Reset guard
        self._shot_type_finished_handled = False

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(f"Classifying shot types for {len(clips)} clips...")

        logger.info("Creating ShotTypeWorker (Analyze All)...")
        self.shot_type_worker = ShotTypeWorker(clips)
        self.shot_type_worker.progress.connect(self._on_shot_type_progress)
        self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
        self.shot_type_worker.finished.connect(self._on_analyze_all_shot_finished, Qt.UniqueConnection)
        self.shot_type_worker.start()

    @Slot()
    def _on_analyze_all_shot_finished(self):
        """Handle shot type classification completion in 'Analyze All' flow."""
        logger.info("=== ANALYZE ALL: SHOT TYPE FINISHED ===")

        # Guard against duplicate calls
        if self._shot_type_finished_handled:
            logger.warning("_on_analyze_all_shot_finished already handled, ignoring duplicate")
            return
        self._shot_type_finished_handled = True

        # Continue to next step
        self._start_next_analyze_all_step()

    def _start_transcription_for_analyze_all(self):
        """Start transcription as part of 'Analyze All' sequence.

        Handles clips from multiple sources by grouping and processing sequentially.
        """
        clips = self._analyze_all_clips
        if not clips:
            self._start_next_analyze_all_step()
            return

        # Check if faster-whisper is available
        from core.transcription import is_faster_whisper_available
        if not is_faster_whisper_available():
            logger.warning("Transcription skipped in 'Analyze All': faster-whisper not installed")
            self.status_bar.showMessage(
                "Analysis complete (transcription unavailable - install faster-whisper)"
            )
            # Skip transcription, continue to next step (which will complete the flow)
            self._start_next_analyze_all_step()
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
        self._start_next_source_transcription()

    def _start_next_source_transcription(self):
        """Start transcription for the next source in the multi-source queue."""
        if not self._transcription_source_queue:
            # All sources done
            logger.info("All source transcriptions complete")
            self._start_next_analyze_all_step()
            return

        source_id, clips = self._transcription_source_queue.pop(0)
        source = self.sources_by_id.get(source_id)

        if not source:
            logger.warning(f"Source {source_id} not found, skipping transcription for {len(clips)} clips")
            # Try next source
            self._start_next_source_transcription()
            return

        # Reset guard
        self._transcription_finished_handled = False

        remaining_sources = len(self._transcription_source_queue)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_bar.showMessage(
            f"Transcribing {len(clips)} clips from source ({remaining_sources + 1} sources remaining)..."
        )

        logger.info(f"Creating TranscriptionWorker for source {source_id} ({len(clips)} clips)")
        self.transcription_worker = TranscriptionWorker(
            clips,
            source,
            self.settings.transcription_model,
            self.settings.transcription_language,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
        self.transcription_worker.finished.connect(self._on_analyze_all_source_transcription_finished, Qt.UniqueConnection)
        self.transcription_worker.error.connect(self._on_transcription_error)
        self.transcription_worker.start()

    @Slot()
    def _on_analyze_all_source_transcription_finished(self):
        """Handle transcription completion for one source in 'Analyze All' flow."""
        logger.info("=== ANALYZE ALL: SOURCE TRANSCRIPTION FINISHED ===")

        # Guard against duplicate calls
        if self._transcription_finished_handled:
            logger.warning("_on_analyze_all_source_transcription_finished already handled, ignoring duplicate")
            return
        self._transcription_finished_handled = True

        # Continue to next source (or finish if all done)
        self._start_next_source_transcription()

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
            # Add to library
            source = Source(file_path=path)
            self.project.add_source(source)
            self.collect_tab.add_source(source)

            # Generate thumbnail
            self._generate_source_thumbnail(source)

            # Select it
            self._select_source(source)

        # Set sensitivity on Cut tab
        self.cut_tab.set_sensitivity(self.settings.default_sensitivity)

    def _on_import_url_click(self):
        """Handle import URL button click."""
        url, ok = QInputDialog.getText(
            self,
            "Import from URL",
            "Enter YouTube or Vimeo URL:",
            QLineEdit.Normal,
            "",
        )
        if ok and url.strip():
            self._download_video(url.strip())

    def _download_video(self, url: str):
        """Start downloading a video from URL."""
        # Update UI state
        self.collect_tab.set_downloading(True)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        self.download_worker = DownloadWorker(url)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.error.connect(self._on_download_error)
        self.download_worker.start()

    def _on_download_progress(self, progress: float, message: str):
        """Handle download progress update."""
        self.progress_bar.setValue(int(progress))
        self.status_bar.showMessage(message)

    def _on_download_finished(self, result):
        """Handle download completion."""
        self.progress_bar.setVisible(False)
        self.collect_tab.set_downloading(False)

        if result.file_path and result.file_path.exists():
            self._load_video(result.file_path)
            self.status_bar.showMessage(f"Downloaded: {result.title}")
            # Switch to Analyze tab after successful download
            self.tab_widget.setCurrentIndex(1)  # Analyze tab
        else:
            QMessageBox.warning(self, "Download Error", "Download completed but file not found")

    def _on_download_error(self, error: str):
        """Handle download error."""
        self.progress_bar.setVisible(False)
        self.collect_tab.set_downloading(False)
        QMessageBox.critical(self, "Download Error", error)

    # YouTube search handlers
    @Slot(str)
    def _on_youtube_search(self, query: str):
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
            self.youtube_client, query, self.settings.youtube_results_count
        )
        self.youtube_search_worker.finished.connect(self._on_youtube_search_finished)
        self.youtube_search_worker.error.connect(self._on_youtube_search_error)
        self.youtube_search_worker.start()

    @Slot(object)
    def _on_youtube_search_finished(self, result: YouTubeSearchResult):
        """Handle search completion."""
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
        """Handle search error."""
        self.collect_tab.youtube_search_panel.set_searching(False)
        QMessageBox.critical(self, "Search Failed", error)

    @Slot(list)
    def _on_bulk_download(self, videos: list):
        """Start bulk download of selected videos."""
        if not videos:
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

            # Show detailed error dialog
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

    def _start_detection(self, threshold: float):
        """Start scene detection with given threshold."""
        logger.info("=== START DETECTION ===")
        if not self.current_source:
            return

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

        # Update Cut tab state
        self.cut_tab.set_detecting(True)

        config = DetectionConfig(threshold=threshold)

        # Start detection in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        logger.info("Creating DetectionWorker...")
        self.detection_worker = DetectionWorker(
            self.current_source.file_path, config
        )
        self.detection_worker.progress.connect(self._on_detection_progress)
        self.detection_worker.finished.connect(self._on_detection_finished, Qt.UniqueConnection)
        self.detection_worker.error.connect(self._on_detection_error)
        logger.info("Starting DetectionWorker...")
        self.detection_worker.start()
        logger.info("DetectionWorker started")

    def _on_detection_progress(self, progress: float, message: str):
        """Handle detection progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)

    @Slot(object, list)
    def _on_detection_finished(self, source: Source, clips: list[Clip]):
        """Handle detection completion."""
        logger.info("=== DETECTION FINISHED ===")

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
        self.project.replace_source_clips(self.current_source.id, clips)
        self._update_window_title()

        # Remove old clips for this source from the Cut tab UI (handles re-analysis case)
        self.cut_tab.remove_clips_for_source(self.current_source.id)

        # Remove orphaned clips from Analyze tab (clips that no longer exist)
        valid_clip_ids = set(self.clips_by_id.keys())
        removed_count = self.analyze_tab.remove_orphaned_clips(valid_clip_ids)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} orphaned clips from Analyze tab")

        self.status_bar.showMessage(f"Found {len(clips)} scenes. Generating thumbnails...")

        # Start thumbnail generation
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
        self.progress_bar.setVisible(False)
        self.cut_tab.set_detecting(False)

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
            logger.info(f"Found clip {clip_id}, thumbnail_path={clip.thumbnail_path}, adding to cut_tab")
            # Add to Cut tab (primary clip browser for detection)
            self.cut_tab.add_clip(clip, self.current_source)
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

        # Update Cut tab with all clips for this source
        current_source_clips = self.clips_by_source.get(self.current_source.id, []) if self.current_source else []
        logger.info(f"_on_thumbnails_finished: found {len(current_source_clips)} clips for source {self.current_source.id if self.current_source else 'None'}")
        self.cut_tab.set_clips(current_source_clips)
        self.cut_tab.set_detecting(False)

        # Make all clips available for timeline remix (via Sequence tab)
        # Pass clips for the current source being analyzed
        current_source_clips = self.clips_by_source.get(self.current_source.id, []) if self.current_source else []
        if self.current_source and current_source_clips:
            self.sequence_tab.set_clips_available(current_source_clips, self.current_source)
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
        """Handle color extraction complete for a clip."""
        # Update the clip model
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.dominant_colors = colors
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_colors(clip_id, colors)
            self.analyze_tab.update_clip_colors(clip_id, colors)
            self._mark_dirty()

    def _on_shot_type_progress(self, current: int, total: int):
        """Handle shot type classification progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_shot_type_ready(self, clip_id: str, shot_type: str, confidence: float):
        """Handle shot type classification complete for a clip."""
        # Update the clip model
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.shot_type = shot_type
            # Update both tabs' clip browsers
            self.cut_tab.update_clip_shot_type(clip_id, shot_type)
            self.analyze_tab.update_clip_shot_type(clip_id, shot_type)
            self._mark_dirty()
            logger.debug(f"Clip {clip_id}: {shot_type} ({confidence:.2f})")

    def _on_transcription_progress(self, current: int, total: int):
        """Handle transcription progress."""
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
        # Don't seek during playback - playhead is driven by video position
        if not self._is_playing:
            self.sequence_tab.video_player.seek_to(time_seconds)

    def _on_sequence_changed(self):
        """Handle sequence modification."""
        # Update Export EDL menu item state
        sequence = self.sequence_tab.timeline.get_sequence()
        has_clips = sequence.duration_frames > 0
        self.export_edl_action.setEnabled(has_clips)

        # Update chat panel with project state (sequence may have changed)
        self._update_chat_project_state()

    # --- Playback methods ---

    def _on_playback_requested(self, start_frame: int):
        """Start sequence playback from given frame."""
        if self._is_playing:
            # Toggle pause
            self._pause_playback()
            return

        sequence = self.sequence_tab.timeline.get_sequence()
        if sequence.duration_frames == 0:
            return  # Nothing to play

        self._is_playing = True
        self.sequence_tab.timeline.set_playing(True)

        # Start playback from current position
        self._play_clip_at_frame(start_frame)

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
            self.sequence_tab.video_player.player.stop()  # Shows black
            self._playback_timer.start()
            return

        self._current_playback_clip = seq_clip

        # Calculate source position
        # frame_in_clip = where we are relative to clip start on timeline
        frame_in_clip = frame - seq_clip.start_frame
        # source_frame = in_point + offset into clip
        source_frame = seq_clip.in_point + frame_in_clip
        source_seconds = source_frame / source.fps

        # Calculate end of this clip in source time
        end_seconds = seq_clip.out_point / source.fps

        # Load source and play range
        self.sequence_tab.video_player.load_video(source.file_path)
        self.sequence_tab.video_player.play_range(source_seconds, end_seconds)

        # Start timer to monitor for clip transitions
        self._playback_timer.start()

    def _on_playback_tick(self):
        """Called during playback to check for clip transitions and advance playhead in gaps."""
        if not self._is_playing:
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
        if not self._is_playing or not self._current_playback_clip:
            return

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

    def _on_video_state_changed(self, state):
        """Handle video player state changes."""
        from PySide6.QtMultimedia import QMediaPlayer

        logger.debug(f"Video state changed: {state}, is_playing: {self._is_playing}")

        if not self._is_playing:
            return

        if state == QMediaPlayer.StoppedState:
            # Clip ended naturally - check if we should continue to next
            if self._current_playback_clip:
                next_frame = self._current_playback_clip.end_frame()
                self.sequence_tab.timeline.set_playhead_time(
                    next_frame / self.sequence_tab.timeline.sequence.fps
                )
                self._play_clip_at_frame(next_frame)

    def _pause_playback(self):
        """Pause playback."""
        self._is_playing = False
        self._playback_timer.stop()
        self.sequence_tab.video_player.player.pause()
        self.sequence_tab.timeline.set_playing(False)

    def _on_stop_requested(self):
        """Handle stop request from timeline."""
        self._stop_playback()

    def _stop_playback(self):
        """Stop playback and reset state."""
        self._is_playing = False
        self._playback_timer.stop()
        self._current_playback_clip = None
        self.sequence_tab.video_player.player.stop()
        self.sequence_tab.timeline.set_playing(False)

    def _on_export_click(self):
        """Export selected clips."""
        selected = self.cut_tab.clip_browser.get_selected_clips()
        if not selected:
            QMessageBox.information(self, "Export", "No clips selected")
            return
        self._export_clips(selected)

    def _on_export_all_click(self):
        """Export all clips."""
        if not self.clips:
            return
        self._export_clips(self.clips)

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

        success = export_dataset(self.current_source, self.clips, config, on_progress)

        self.progress_bar.setVisible(False)

        if success:
            self.status_bar.showMessage(f"Dataset exported to {output_path.name}")
            # Open containing folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))
        else:
            QMessageBox.critical(self, "Export Error", "Failed to export dataset")

    def _export_clips(self, clips: list[Clip]):
        """Export clips to a folder."""
        if not self.current_source:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Folder"
        )
        if not output_dir:
            return

        from core.ffmpeg import FFmpegProcessor
        processor = FFmpegProcessor()

        output_path = Path(output_dir)
        source_name = self._sanitize_filename(self.current_source.file_path.stem)
        fps = self.current_source.fps

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(clips))

        exported = 0
        for i, clip in enumerate(clips):
            start = clip.start_time(fps)
            duration = clip.duration_seconds(fps)
            output_file = output_path / f"{source_name}_scene_{i + 1:03d}.mp4"

            success = processor.extract_clip(
                input_path=self.current_source.file_path,
                output_path=output_file,
                start_seconds=start,
                duration_seconds=duration,
                fps=fps,
            )
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

        # Build sources and clips dictionaries
        sources = {}
        clips = {}
        if self.current_source:
            sources[self.current_source.id] = self.current_source
            for clip in self.clips:
                clips[clip.id] = (clip, self.current_source)

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

        config = ExportConfig(
            output_path=output_path,
            fps=target_fps if target_fps else sequence.fps,
            width=resolution[0] if resolution else None,
            height=resolution[1] if resolution else None,
            crf=quality_preset["crf"],
            preset=quality_preset["preset"],
            video_bitrate=quality_preset["bitrate"],
        )

        # Start export in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.sequence_tab.timeline.export_btn.setEnabled(False)

        self.export_worker = SequenceExportWorker(sequence, sources, clips, config)
        self.export_worker.progress.connect(self._on_sequence_export_progress)
        self.export_worker.finished.connect(self._on_sequence_export_finished)
        self.export_worker.error.connect(self._on_sequence_export_error)
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
        self.export_worker.finished.connect(self._on_sequence_export_finished)
        self.export_worker.error.connect(self._on_sequence_export_error)
        self.export_worker.start()
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

    def start_agent_bulk_download(self, urls: list[str], download_dir: Path) -> bool:
        """Start bulk video downloads triggered by agent.

        Args:
            urls: List of video URLs to download
            download_dir: Directory to save downloads

        Returns:
            True if download started, False if already in progress
        """
        # Check if download already running
        if self.url_bulk_download_worker and self.url_bulk_download_worker.isRunning():
            return False

        # Mark that agent is waiting
        self._pending_agent_download = True
        self._agent_download_results = []

        # Start download in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(urls))
        self.status_bar.showMessage(f"Downloading {len(urls)} videos...")

        self.url_bulk_download_worker = URLBulkDownloadWorker(urls, download_dir)
        self.url_bulk_download_worker.progress.connect(self._on_agent_download_progress)
        self.url_bulk_download_worker.video_finished.connect(self._on_agent_video_finished)
        self.url_bulk_download_worker.all_finished.connect(self._on_agent_bulk_download_finished)
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
            from models.clip import Source

            file_path = Path(result.file_path)

            # Check if already in project
            for existing in self.project.sources:
                if existing.file_path == file_path:
                    logger.info(f"Source already in project: {file_path.name}")
                    return

            # Create source and add to project
            source = Source(file_path=file_path)
            self.project.add_source(source)

            # Add to CollectTab grid
            self.collect_tab.add_source(source)

            # Generate thumbnail for the source
            self._generate_source_thumbnail(source)

            logger.info(f"Added downloaded source to project: {file_path.name}")

    def _on_agent_bulk_download_finished(self, results: list):
        """Handle bulk download completion."""
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

        elif wait_type == "download":
            # Download worker is started by the tool itself via start_agent_bulk_download
            return True

        elif wait_type == "detection":
            # Detection worker is started by the tool itself via _start_detection
            # Just return True since the worker is already running
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
        self.color_worker = ColorAnalysisWorker(clips)
        self.color_worker.progress.connect(self._on_color_progress)
        self.color_worker.color_ready.connect(self._on_color_ready)
        self.color_worker.finished.connect(self._on_agent_color_analysis_finished, Qt.UniqueConnection)
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

        # Check if worker already running
        if self.shot_type_worker and self.shot_type_worker.isRunning():
            return False

        # Reset guard
        self._shot_type_finished_handled = False

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
        self.shot_type_worker = ShotTypeWorker(clips)
        self.shot_type_worker.progress.connect(self._on_shot_type_progress)
        self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
        self.shot_type_worker.finished.connect(self._on_agent_shot_analysis_finished, Qt.UniqueConnection)
        self.shot_type_worker.start()

        return True

    def start_agent_transcription(self, clip_ids: list[str]) -> bool:
        """Start transcription for clips triggered by agent.

        Args:
            clip_ids: List of clip IDs to transcribe

        Returns:
            True if started, False if already running or unavailable
        """
        # Check if faster-whisper is available
        from core.transcription import is_faster_whisper_available
        if not is_faster_whisper_available():
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

        # Start worker for first source
        from PySide6.QtCore import Qt
        self.transcription_worker = TranscriptionWorker(
            first_clips,
            first_source,
            self.settings.transcription_model,
            self.settings.transcription_language,
        )
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.transcript_ready.connect(self._on_transcript_ready)
        self.transcription_worker.finished.connect(self._on_agent_transcription_finished, Qt.UniqueConnection)
        self.transcription_worker.error.connect(self._on_transcription_error)
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

        # Check if worker already running
        if self.classification_worker and self.classification_worker.isRunning():
            return False

        # Reset guard
        self._classification_finished_handled = False

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
        self.classification_worker = ClassificationWorker(clips, top_k=top_k)
        self.classification_worker.progress.connect(self._on_classification_progress)
        self.classification_worker.labels_ready.connect(self._on_classification_ready)
        self.classification_worker.finished.connect(self._on_agent_classification_finished, Qt.UniqueConnection)
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

        # Check if worker already running
        if self.detection_worker_yolo and self.detection_worker_yolo.isRunning():
            return False

        # Reset guard
        self._object_detection_finished_handled = False

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
        self.detection_worker_yolo = ObjectDetectionWorker(clips, confidence=confidence, detect_all=detect_all)
        self.detection_worker_yolo.progress.connect(self._on_object_detection_progress)
        self.detection_worker_yolo.objects_ready.connect(self._on_objects_ready)
        self.detection_worker_yolo.finished.connect(self._on_agent_object_detection_finished, Qt.UniqueConnection)
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

        # Check if worker already running
        if self.description_worker and self.description_worker.isRunning():
            return False

        # Reset guard
        self._description_finished_handled = False

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
        self.status_bar.showMessage(f"Generating descriptions for {len(clips)} clips...")

        # Start worker
        from PySide6.QtCore import Qt
        self.description_worker = DescriptionWorker(clips, tier=tier, prompt=prompt)
        self.description_worker.progress.connect(self._on_description_progress)
        self.description_worker.description_ready.connect(self._on_description_ready)
        self.description_worker.finished.connect(self._on_agent_description_finished, Qt.UniqueConnection)
        self.description_worker.start()

        return True

    @Slot(int, int)
    def _on_description_progress(self, current: int, total: int):
        """Handle description generation progress updates."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.status_bar.showMessage(f"Generating descriptions: {current}/{total} clips...")

    @Slot(str, str, str)
    def _on_description_ready(self, clip_id: str, description: str, model_name: str):
        """Handle description results for a single clip."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            clip.description = description
            clip.description_model = model_name
            clip.description_frames = 1
            logger.debug(f"Description for {clip_id}: {description[:50]}...")

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
        self.status_bar.showMessage("Description generation complete", 3000)

        # Save project
        self.project.save()

        # Send result back to agent
        if hasattr(self, '_pending_agent_description') and self._pending_agent_description:
            self._pending_agent_description = False
            clips = getattr(self, '_agent_description_clips', [])

            # Build result summary
            described_count = sum(1 for c in clips if c.description)
            
            result = {
                "success": True,
                "described_clips": described_count,
                "total_clips": len(clips),
                "sample_descriptions": [],
            }
            
            # Include sample descriptions
            for clip in clips[:3]:
                if clip.description:
                    result["sample_descriptions"].append({
                        "clip_id": clip.id,
                        "description": clip.description,
                    })

            if self._chat_worker:
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent description result to agent: {described_count}/{len(clips)} clips")

    @Slot(int, int)
    def _on_classification_progress(self, current: int, total: int):
        """Handle classification progress updates."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.status_bar.showMessage(f"Classifying content: {current}/{total} clips...")

    @Slot(str, list)
    def _on_classification_ready(self, clip_id: str, results: list):
        """Handle classification results for a single clip."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            # Store labels (just the label strings, not confidences)
            clip.object_labels = [label for label, _ in results]
            logger.debug(f"Classification for {clip_id}: {clip.object_labels[:3]}")

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
        self.status_bar.showMessage("Classification complete", 3000)

        # Save project
        self.project.save()

        # Send result back to agent
        if hasattr(self, '_pending_agent_classification') and self._pending_agent_classification:
            self._pending_agent_classification = False
            clips = getattr(self, '_agent_classification_clips', [])

            # Build result summary
            classified_count = sum(1 for c in clips if c.object_labels)
            result = {
                "success": True,
                "classified_clips": classified_count,
                "total_clips": len(clips),
                "sample_labels": [],
            }

            # Include sample labels from first few clips
            for clip in clips[:3]:
                if clip.object_labels:
                    result["sample_labels"].append({
                        "clip_id": clip.id,
                        "labels": clip.object_labels[:5],
                    })

            if self._chat_worker:
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent classification result to agent: {classified_count}/{len(clips)} clips")

    @Slot(int, int)
    def _on_object_detection_progress(self, current: int, total: int):
        """Handle object detection progress updates."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            detect_all = getattr(self, '_agent_object_detection_all', True)
            task = "objects" if detect_all else "people"
            self.status_bar.showMessage(f"Detecting {task}: {current}/{total} clips...")

    @Slot(str, list, int)
    def _on_objects_ready(self, clip_id: str, detections: list, person_count: int):
        """Handle object detection results for a single clip."""
        clip = self.project.clips_by_id.get(clip_id)
        if clip:
            clip.detected_objects = detections
            clip.person_count = person_count
            logger.debug(f"Detection for {clip_id}: {len(detections)} objects, {person_count} people")

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
        self.status_bar.showMessage("Object detection complete", 3000)

        # Save project
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
                "success": True,
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

            if self._chat_worker:
                self._pending_agent_tool_call_id = None
                self._pending_agent_tool_name = None
                self._chat_worker.set_gui_tool_result(result)
                logger.info(f"Sent object detection result to agent: {detected_count}/{len(clips)} clips")

    def _on_export_edl_click(self):
        """Export the timeline sequence as an EDL file."""
        sequence = self.sequence_tab.timeline.get_sequence()
        all_clips = sequence.get_all_clips()

        if not all_clips:
            QMessageBox.information(
                self, "Export EDL", "No clips in timeline to export"
            )
            return

        # Get output file path
        default_name = "sequence.edl"
        if self.current_source:
            default_name = f"{self._sanitize_filename(self.current_source.file_path.stem)}_timeline.edl"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export EDL",
            str(Path.home() / default_name),
            "Edit Decision List (*.edl);;All Files (*)",
        )
        if not file_path:
            return

        output_path = Path(file_path)

        # Build sources dictionary
        sources = {}
        if self.current_source:
            sources[self.current_source.id] = self.current_source

        config = EDLExportConfig(
            output_path=output_path,
            title=sequence.name or "Scene Ripper Export",
        )

        self.status_bar.showMessage("Exporting EDL...")
        success = export_edl(sequence, sources, config)

        if success:
            self.status_bar.showMessage(f"EDL exported to {output_path.name}", 5000)
            # Open containing folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))
        else:
            QMessageBox.warning(self, "Export EDL", "Failed to export EDL file")

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
            "Project Files (*.json);;All Files (*)",
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
        default_name = "project.json"
        if self.current_project_path:
            default_path = str(self.current_project_path)
        elif self.current_source:
            default_path = str(self.current_source.file_path.parent / f"{self.current_source.file_path.stem}.json")
        else:
            default_path = str(Path.home() / default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            default_path,
            "Project Files (*.json);;All Files (*)",
        )

        if file_path:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")
            self._save_project_to_file(path)

    def _save_project_to_file(self, filepath: Path):
        """Save project to the specified file."""
        if not self.sources:
            QMessageBox.warning(
                self,
                "Save Project",
                "No videos in library. Import a video first."
            )
            return

        self.status_bar.showMessage("Saving project...")

        # Get sequence from timeline and update project
        self.project.sequence = self.sequence_tab.timeline.get_sequence()

        # Get UI state and update project
        self.project.ui_state = {
            "sensitivity": self.cut_tab.sensitivity_slider.value() / 10.0,
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

        # Add all sources to CollectTab
        for source in self.sources:
            self.collect_tab.add_source(source)
            # Generate source thumbnails if missing
            if not source.thumbnail_path or not source.thumbnail_path.exists():
                self._generate_source_thumbnail(source)

        # Set first source as current (for backwards compatibility)
        self.current_source = self.sources[0]

        # Set lookups for Analyze tab
        self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

        # Update Cut tab with clips for current source
        self.cut_tab.set_source(self.current_source)
        self.cut_tab.clear_clips()
        current_source_clips = self.clips_by_source.get(self.current_source.id, [])
        self.cut_tab.set_clips(current_source_clips)

        # Add clips to Cut tab browser with their correct source
        for clip in self.clips:
            clip_source = self.sources_by_id.get(clip.source_id)
            if not clip_source:
                logging.warning(f"Clip {clip.id} references unknown source {clip.source_id}")
                continue
            self.cut_tab.add_clip(clip, clip_source)
            # Update colors and shot type if present
            if clip.dominant_colors:
                self.cut_tab.update_clip_colors(clip.id, clip.dominant_colors)
            if clip.shot_type:
                self.cut_tab.update_clip_shot_type(clip.id, clip.shot_type)
            if clip.transcript:
                self.cut_tab.update_clip_transcript(clip.id, clip.transcript)

        # Restore sequence (pass all sources for multi-source playback)
        if self.project.sequence:
            self.sequence_tab.timeline.load_sequence(
                self.project.sequence, self.sources_by_id, self.clips
            )

        # Restore UI state
        ui_state = self.project.ui_state
        if "sensitivity" in ui_state:
            self.cut_tab.set_sensitivity(ui_state["sensitivity"])

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

        # Regenerate missing thumbnails
        self._regenerate_missing_thumbnails()

    def _clear_project_state(self):
        """Clear current project state."""
        # Clear project data
        self.project.clear()

        # Clear UI state
        self.current_source = None
        self._analyze_queue = deque()
        self._analyze_queue_total = 0
        self.queue_label.setVisible(False)
        self.collect_tab.clear()
        self.cut_tab.clear_clips()
        self.cut_tab.set_source(None)
        self.analyze_tab.clear_clips()
        self.sequence_tab.timeline.clear()

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

        # Set first source as current
        if self.sources:
            self.current_source = self.sources[0]

            # Set lookups for Analyze tab
            self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

            # Update Cut tab with clips for current source
            self.cut_tab.set_source(self.current_source)
            self.cut_tab.clear_clips()
            current_source_clips = self.clips_by_source.get(self.current_source.id, [])
            self.cut_tab.set_clips(current_source_clips)

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
                if clip.transcript:
                    self.cut_tab.update_clip_transcript(clip.id, clip.transcript)

        # Restore sequence
        if self.project.sequence:
            self.sequence_tab.timeline.load_sequence(
                self.project.sequence, self.sources_by_id, self.clips
            )

        # Restore UI state
        ui_state = self.project.ui_state
        if "sensitivity" in ui_state:
            self.cut_tab.set_sensitivity(ui_state["sensitivity"])

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
            logger.warning(f"  clip not found in clips_by_id!")

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

        # Stop playback timer
        if self._playback_timer.isActive():
            logger.info("Stopping playback timer")
            self._playback_timer.stop()

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
                        logger.warning(f"{name} worker did not stop gracefully, terminating...")
                        worker.terminate()
                        worker.wait(1000)
                    logger.info(f"{name} worker stopped")

        event.accept()
