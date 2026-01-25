"""Main application window."""

import logging
import re
from collections import deque
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
from PySide6.QtCore import Qt, Signal, QThread, QMimeData, QUrl, QTimer, Slot
from PySide6.QtGui import QDesktopServices, QKeySequence, QAction
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from models.clip import Source, Clip
from core.scene_detect import SceneDetector, DetectionConfig
from core.thumbnail import ThumbnailGenerator
from core.downloader import VideoDownloader
from core.sequence_export import SequenceExporter, ExportConfig
from core.dataset_export import export_dataset, DatasetExportConfig
from core.edl_export import export_edl, EDLExportConfig
from core.analysis.color import extract_dominant_colors
from core.analysis.shots import classify_shot_type
from core.settings import Settings, load_settings, save_settings, migrate_from_qsettings
from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeSearchResult,
    YouTubeVideo,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)
from core.project import (
    save_project,
    load_project,
    Project,
    ProjectMetadata,
    ProjectLoadError,
    MissingSourceError,
)
from ui.project_adapter import ProjectSignalAdapter
from ui.settings_dialog import SettingsDialog
from ui.tabs import CollectTab, CutTab, AnalyzeTab, GenerateTab, SequenceTab, RenderTab
from ui.theme import theme
from ui.chat_panel import ChatPanel
from ui.chat_worker import ChatAgentWorker


class DetectionWorker(QThread):
    """Background worker for scene detection."""

    progress = Signal(float, str)  # progress (0-1), status message
    finished = Signal(object, list)  # source, clips
    error = Signal(str)

    def __init__(self, video_path: Path, config: DetectionConfig):
        super().__init__()
        self.video_path = video_path
        self.config = config
        logger.debug("DetectionWorker created")

    def run(self):
        logger.info("DetectionWorker.run() STARTING")
        try:
            detector = SceneDetector(self.config)
            source, clips = detector.detect_scenes_with_progress(
                self.video_path,
                lambda p, m: self.progress.emit(p, m),
            )
            logger.info("DetectionWorker.run() emitting finished signal")
            self.finished.emit(source, clips)
            logger.info("DetectionWorker.run() COMPLETED")
        except Exception as e:
            logger.error(f"DetectionWorker.run() ERROR: {e}")
            self.error.emit(str(e))


class ThumbnailWorker(QThread):
    """Background worker for thumbnail generation."""

    progress = Signal(int, int)  # current, total
    thumbnail_ready = Signal(str, str)  # clip_id, thumbnail_path
    finished = Signal()

    def __init__(self, source: Source, clips: list[Clip], cache_dir: Path = None):
        super().__init__()
        self.source = source
        self.clips = clips
        self.cache_dir = cache_dir
        logger.debug("ThumbnailWorker created")

    def run(self):
        logger.info("ThumbnailWorker.run() STARTING")
        logger.info(f"ThumbnailWorker: {len(self.clips)} clips to process")
        generator = ThumbnailGenerator(cache_dir=self.cache_dir)
        total = len(self.clips)

        for i, clip in enumerate(self.clips):
            try:
                logger.info(f"ThumbnailWorker: generating thumbnail for clip {clip.id}")
                thumb_path = generator.generate_clip_thumbnail(
                    video_path=self.source.file_path,
                    start_seconds=clip.start_time(self.source.fps),
                    end_seconds=clip.end_time(self.source.fps),
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

        # UI state (not part of Project - these are GUI-specific selections)
        self.current_source: Optional[Source] = None  # Currently active/selected source
        self._analyze_queue: deque[Source] = deque()  # Queue for batch analysis (O(1) popleft)
        self._analyze_queue_total: int = 0  # Total count for progress display
        self.detection_worker: Optional[DetectionWorker] = None
        self.thumbnail_worker: Optional[ThumbnailWorker] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.export_worker: Optional[SequenceExportWorker] = None
        self.color_worker: Optional[ColorAnalysisWorker] = None
        self.shot_type_worker: Optional[ShotTypeWorker] = None
        self.transcription_worker: Optional[TranscriptionWorker] = None
        self.youtube_search_worker: Optional[YouTubeSearchWorker] = None
        self.bulk_download_worker: Optional[BulkDownloadWorker] = None
        self.youtube_client: Optional[YouTubeSearchClient] = None

        # Guards to prevent duplicate signal handling
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._shot_type_finished_handled = False
        self._transcription_finished_handled = False

        # State for "Analyze All" sequential processing
        self._analyze_all_pending: list[str] = []  # Pending steps: "colors", "shots", "transcribe"
        self._analyze_all_clips: list = []  # Clips being analyzed
        self._transcription_source_queue: list = []  # Queue for multi-source transcription

        # Project path tracking (for backwards compatibility - delegates to project)
        # Note: self.project.path is the actual storage

        logger.info("Setting up UI...")
        self._setup_ui()
        logger.info("Connecting signals...")
        self._connect_signals()

        # Set up chat panel
        logger.info("Setting up chat panel...")
        self._setup_chat_panel()

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

        # View menu with tab shortcuts
        view_menu = menu_bar.addMenu("&View")

        tab_names = ["&Collect", "&Analyze", "&Generate", "&Sequence", "&Render"]
        for i, name in enumerate(tab_names):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
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

        # Analyze tab signals
        self.analyze_tab.transcribe_requested.connect(self._on_transcribe_from_tab)
        self.analyze_tab.analyze_colors_requested.connect(self._on_analyze_colors_from_tab)
        self.analyze_tab.analyze_shots_requested.connect(self._on_analyze_shots_from_tab)
        self.analyze_tab.analyze_all_requested.connect(self._on_analyze_all_from_tab)
        self.analyze_tab.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)

        # Sequence tab signals
        self.sequence_tab.playback_requested.connect(self._on_playback_requested)
        self.sequence_tab.stop_requested.connect(self._on_stop_requested)
        self.sequence_tab.export_requested.connect(self._on_sequence_export_click)
        # Update Render tab when sequence changes (clips added/removed/generated)
        self.sequence_tab.timeline.sequence_changed.connect(self._update_render_tab_sequence_info)
        # Update EDL export menu item when sequence changes
        self.sequence_tab.timeline.sequence_changed.connect(self._on_sequence_changed)
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
        from core.llm_client import ProviderConfig, ProviderType, create_provider_config_from_settings
        from core.settings import get_llm_api_key

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

    def _on_chat_message(self, message: str):
        """Handle user message from chat panel."""
        from core.llm_client import ProviderConfig, ProviderType, create_provider_config_from_settings
        from core.settings import get_llm_api_key

        # Store message for history
        self._last_user_message = message

        # Cancel any existing worker
        if self._chat_worker and self._chat_worker.isRunning():
            self._chat_worker.stop()
            self._chat_worker.wait(1000)

        # Get current provider config
        provider_key = self.chat_panel.get_provider()

        # Build provider config
        config = ProviderConfig(
            provider=ProviderType(provider_key),
            model=self.settings.llm_model,
            api_key=get_llm_api_key() or None,
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

        # Start worker
        self._chat_worker = ChatAgentWorker(
            config=config,
            messages=messages,
            project=self.project,
            busy_check=check_busy,
        )

        # Connect signals
        bubble = self.chat_panel.start_streaming_response()
        self._current_chat_bubble = bubble

        self._chat_worker.text_chunk.connect(self.chat_panel.on_stream_chunk)
        self._chat_worker.tool_called.connect(self._on_chat_tool_called)
        self._chat_worker.tool_result.connect(self._on_chat_tool_result)
        self._chat_worker.gui_tool_requested.connect(self._on_gui_tool_requested)
        self._chat_worker.complete.connect(self._on_chat_complete)
        self._chat_worker.error.connect(self._on_chat_error)

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

    @Slot(str, dict, str)
    def _on_gui_tool_requested(self, tool_name: str, args: dict, tool_call_id: str):
        """Execute a GUI-modifying tool on the main thread.

        This slot is called by ChatAgentWorker when a tool that modifies
        GUI state needs to be executed. The tool runs on the main thread
        to ensure thread safety with Qt.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            tool_call_id: ID for tracking the tool call
        """
        from core.chat_tools import tools as tool_registry

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
                # Inject project and execute
                args["project"] = self.project
                tool_result = tool.func(**args)
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

    def _on_chat_cancel(self):
        """Handle chat cancellation."""
        if self._chat_worker and self._chat_worker.isRunning():
            logger.info("Cancelling chat worker")
            self._chat_worker.stop()

    def _on_chat_provider_changed(self, provider: str):
        """Handle provider selection change."""
        logger.info(f"Chat provider changed to: {provider}")
        # Update settings
        self.settings.llm_provider = provider
        # Note: Not auto-saving to allow temporary changes during session

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
            self.status_bar.showMessage(
                f"Analysis complete - {len(self._analyze_all_clips)} clips"
            )
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
                f"Analysis complete (transcription unavailable - install faster-whisper)"
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
            self.status_bar.showMessage(f"Added clip to timeline")
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
            output_file = output_path / f"{source_name}_scene_{i+1:03d}.mp4"

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

    def _on_sequence_export_progress(self, progress: float, message: str):
        """Handle sequence export progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)

    def _on_sequence_export_finished(self, output_path: Path):
        """Handle sequence export completion."""
        self.progress_bar.setVisible(False)
        self.sequence_tab.timeline.export_btn.setEnabled(True)
        self.status_bar.showMessage(f"Sequence exported to {output_path.name}")

        # Open containing folder
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))

    def _on_sequence_export_error(self, error: str):
        """Handle sequence export error."""
        self.progress_bar.setVisible(False)
        self.sequence_tab.timeline.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Export Error", f"Failed to export sequence: {error}")

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

    def _regenerate_missing_thumbnails(self):
        """Regenerate thumbnails for clips that don't have them."""
        if not self.current_source or not self.clips:
            return

        # Check which clips need thumbnails
        clips_needing_thumbnails = [
            clip for clip in self.clips
            if not clip.thumbnail_path or not clip.thumbnail_path.exists()
        ]

        if not clips_needing_thumbnails:
            return

        logger.info(f"Regenerating thumbnails for {len(clips_needing_thumbnails)} clips")
        self.status_bar.showMessage(f"Regenerating {len(clips_needing_thumbnails)} thumbnails...")

        # Use existing ThumbnailWorker with project-load-specific handlers
        self.thumbnail_worker = ThumbnailWorker(
            self.current_source,
            clips_needing_thumbnails,
            self.settings.thumbnail_cache_dir,
        )
        # Use handlers that update existing clips instead of adding new ones
        self.thumbnail_worker.thumbnail_ready.connect(self._on_project_thumbnail_ready)
        self.thumbnail_worker.finished.connect(self._on_project_thumbnails_finished, Qt.UniqueConnection)
        self.thumbnail_worker.start()

    def _on_project_thumbnail_ready(self, clip_id: str, thumb_path: str):
        """Handle individual thumbnail during project load (update, don't add)."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.thumbnail_path = Path(thumb_path)
            self.cut_tab.update_clip_thumbnail(clip_id, Path(thumb_path))
            self.analyze_tab.update_clip_thumbnail(clip_id, Path(thumb_path))

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
