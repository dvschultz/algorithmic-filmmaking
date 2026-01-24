"""Main application window."""

import logging
import re
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
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QMessageBox,
    QStatusBar,
    QInputDialog,
    QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QThread, QMimeData, QUrl, QTimer, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from models.clip import Source, Clip
from core.scene_detect import SceneDetector, DetectionConfig
from core.thumbnail import ThumbnailGenerator
from core.downloader import VideoDownloader
from core.sequence_export import SequenceExporter, ExportConfig
from core.analysis.color import extract_dominant_colors
from core.analysis.shots import classify_shot_type
from ui.clip_browser import ClipBrowser
from ui.video_player import VideoPlayer
from ui.timeline import TimelineWidget


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

    def __init__(self, source: Source, clips: list[Clip]):
        super().__init__()
        self.source = source
        self.clips = clips
        logger.debug("ThumbnailWorker created")

    def run(self):
        logger.info("ThumbnailWorker.run() STARTING")
        generator = ThumbnailGenerator()
        total = len(self.clips)

        for i, clip in enumerate(self.clips):
            try:
                thumb_path = generator.generate_clip_thumbnail(
                    video_path=self.source.file_path,
                    start_seconds=clip.start_time(self.source.fps),
                    end_seconds=clip.end_time(self.source.fps),
                )
                clip.thumbnail_path = thumb_path
                self.thumbnail_ready.emit(clip.id, str(thumb_path))
            except Exception:
                pass  # Skip failed thumbnails

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
            except Exception:
                pass  # Skip failed color extraction
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
        self.setAcceptDrops(True)

        # State
        self.current_source: Optional[Source] = None
        self.clips: list[Clip] = []
        self.clips_by_id: dict[str, Clip] = {}  # For fast lookup
        self.detection_worker: Optional[DetectionWorker] = None
        self.thumbnail_worker: Optional[ThumbnailWorker] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.export_worker: Optional[SequenceExportWorker] = None
        self.color_worker: Optional[ColorAnalysisWorker] = None
        self.shot_type_worker: Optional[ShotTypeWorker] = None

        # Guards to prevent duplicate signal handling
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._shot_type_finished_handled = False

        logger.info("Setting up UI...")
        self._setup_ui()
        logger.info("Connecting signals...")
        self._connect_signals()

        # Playback state (must be after _setup_ui so self.timeline exists)
        logger.info("Setting up playback state...")
        self._is_playing = False
        self._current_playback_clip = None  # Currently playing SequenceClip
        self._playback_timer = QTimer(self)  # Parent to self for proper lifecycle
        self._playback_timer.setInterval(33)  # ~30fps update rate
        self._playback_timer.timeout.connect(self._on_playback_tick)
        logger.info(f"=== MAINWINDOW INIT COMPLETE (instance #{self._instance_id}) ===")

    def _setup_ui(self):
        """Set up the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top toolbar
        toolbar = self._create_toolbar()
        layout.addLayout(toolbar)

        # Main vertical splitter (content area + timeline)
        main_splitter = QSplitter(Qt.Vertical)

        # Top content area (horizontal splitter)
        content_splitter = QSplitter(Qt.Horizontal)

        # Left: Clip browser
        self.clip_browser = ClipBrowser()
        self.clip_browser.set_drag_enabled(True)  # Enable drag-drop
        content_splitter.addWidget(self.clip_browser)

        # Right: Video player
        self.video_player = VideoPlayer()
        content_splitter.addWidget(self.video_player)

        content_splitter.setSizes([400, 600])
        main_splitter.addWidget(content_splitter)

        # Bottom: Timeline
        self.timeline = TimelineWidget()
        main_splitter.addWidget(self.timeline)

        main_splitter.setSizes([500, 250])
        layout.addWidget(main_splitter)

        # Bottom: Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Drop a video file to begin")

    def _create_toolbar(self) -> QHBoxLayout:
        """Create the top toolbar."""
        toolbar = QHBoxLayout()

        # Import button
        self.import_btn = QPushButton("Import Video")
        self.import_btn.clicked.connect(self._on_import_click)
        toolbar.addWidget(self.import_btn)

        # Import URL button
        self.import_url_btn = QPushButton("Import URL")
        self.import_url_btn.setToolTip("Download from YouTube or Vimeo")
        self.import_url_btn.clicked.connect(self._on_import_url_click)
        toolbar.addWidget(self.import_url_btn)

        # Sensitivity slider
        toolbar.addWidget(QLabel("Sensitivity:"))

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)  # 1.0 to 10.0
        self.sensitivity_slider.setValue(30)  # Default 3.0
        self.sensitivity_slider.setMaximumWidth(150)
        self.sensitivity_slider.setToolTip("Lower = more scenes detected")
        toolbar.addWidget(self.sensitivity_slider)

        self.sensitivity_label = QLabel("3.0")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v/10:.1f}")
        )
        toolbar.addWidget(self.sensitivity_label)

        # Detect button
        self.detect_btn = QPushButton("Detect Scenes")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self._on_detect_click)
        toolbar.addWidget(self.detect_btn)

        toolbar.addStretch()

        # Export button
        self.export_btn = QPushButton("Export Selected")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_click)
        toolbar.addWidget(self.export_btn)

        # Export all button
        self.export_all_btn = QPushButton("Export All")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self._on_export_all_click)
        toolbar.addWidget(self.export_all_btn)

        return toolbar

    def _connect_signals(self):
        """Connect UI signals."""
        self.clip_browser.clip_selected.connect(self._on_clip_selected)
        self.clip_browser.clip_double_clicked.connect(self._on_clip_double_clicked)
        self.clip_browser.clip_dragged_to_timeline.connect(self._on_clip_dragged_to_timeline)

        # Timeline signals
        self.timeline.playhead_changed.connect(self._on_timeline_playhead_changed)
        self.timeline.export_requested.connect(self._on_sequence_export_click)
        self.timeline.playback_requested.connect(self._on_playback_requested)
        self.timeline.stop_requested.connect(self._on_stop_requested)

        # Video player signals for playback sync
        self.video_player.position_updated.connect(self._on_video_position_updated)
        self.video_player.player.playbackStateChanged.connect(self._on_video_state_changed)

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
        """Load a video file."""
        self.current_source = Source(file_path=path)
        self.clips = []
        self.clip_browser.clear()
        self.video_player.load_video(path)
        self.detect_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)
        self.status_bar.showMessage(f"Loaded: {path.name}")
        self.setWindowTitle(f"Scene Ripper - {path.name}")

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
        # Disable buttons during download
        self.import_btn.setEnabled(False)
        self.import_url_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)

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
        self.import_btn.setEnabled(True)
        self.import_url_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)

        if result.file_path and result.file_path.exists():
            self._load_video(result.file_path)
            self.status_bar.showMessage(f"Downloaded: {result.title}")
        else:
            QMessageBox.warning(self, "Download Error", "Download completed but file not found")

    def _on_download_error(self, error: str):
        """Handle download error."""
        self.progress_bar.setVisible(False)
        self.import_btn.setEnabled(True)
        self.import_url_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        QMessageBox.critical(self, "Download Error", error)

    def _on_detect_click(self):
        """Handle detect button click."""
        logger.info("=== DETECT CLICK ===")
        if not self.current_source:
            return

        # Reset guards for new detection run
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._color_analysis_finished_handled = False
        self._shot_type_finished_handled = False

        # Disable buttons during detection
        self.detect_btn.setEnabled(False)
        self.import_btn.setEnabled(False)

        # Get sensitivity from slider
        threshold = self.sensitivity_slider.value() / 10.0
        config = DetectionConfig(threshold=threshold)

        # Start detection in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        logger.info("Creating DetectionWorker...")
        self.detection_worker = DetectionWorker(
            self.current_source.file_path, config
        )
        self.detection_worker.progress.connect(self._on_detection_progress)
        self.detection_worker.finished.connect(self._on_detection_finished)
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

        self.current_source = source
        self.clips = clips
        self.clips_by_id = {clip.id: clip for clip in clips}

        self.status_bar.showMessage(f"Found {len(clips)} scenes. Generating thumbnails...")

        # Start thumbnail generation
        logger.info("Creating ThumbnailWorker...")
        self.thumbnail_worker = ThumbnailWorker(source, clips)
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
        self.detect_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", error)

    def _on_thumbnail_progress(self, current: int, total: int):
        """Handle thumbnail generation progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
        """Handle individual thumbnail completion."""
        clip = self.clips_by_id.get(clip_id)
        if clip:
            self.clip_browser.add_clip(clip, self.current_source)

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

        self.detect_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)

        # Make clips available for timeline remix
        if self.current_source and self.clips:
            self.timeline.set_fps(self.current_source.fps)
            self.timeline.set_available_clips(self.clips, self.current_source)

        # Start color analysis
        if self.clips:
            self.status_bar.showMessage(f"Analyzing colors for {len(self.clips)} scenes...")
            logger.info("Creating ColorAnalysisWorker...")
            self.color_worker = ColorAnalysisWorker(self.clips)
            self.color_worker.progress.connect(self._on_color_progress)
            self.color_worker.color_ready.connect(self._on_color_ready)
            self.color_worker.finished.connect(self._on_color_analysis_finished, Qt.UniqueConnection)
            logger.info("Starting ColorAnalysisWorker...")
            self.color_worker.start()
            logger.info("ColorAnalysisWorker started")
        else:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"Ready - {len(self.clips)} scenes detected")

    def _on_color_progress(self, current: int, total: int):
        """Handle color analysis progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_color_ready(self, clip_id: str, colors: list):
        """Handle color extraction complete for a clip."""
        # Update the clip model
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.dominant_colors = colors
            # Update the browser thumbnail with colors
            self.clip_browser.update_clip_colors(clip_id, colors)

    @Slot()
    def _on_color_analysis_finished(self):
        """Handle all color analysis completed."""
        logger.info("=== COLOR ANALYSIS FINISHED ===")

        # Guard against duplicate calls
        if self._color_analysis_finished_handled:
            logger.warning("_on_color_analysis_finished already handled, ignoring duplicate call")
            return
        self._color_analysis_finished_handled = True

        logger.info(f"Color worker running: {self.color_worker.isRunning() if self.color_worker else 'None'}")

        # Start shot type classification
        if self.clips:
            self.status_bar.showMessage(f"Classifying shot types for {len(self.clips)} scenes...")
            logger.info("Creating ShotTypeWorker...")
            self.shot_type_worker = ShotTypeWorker(self.clips)
            self.shot_type_worker.progress.connect(self._on_shot_type_progress)
            self.shot_type_worker.shot_type_ready.connect(self._on_shot_type_ready)
            self.shot_type_worker.finished.connect(self._on_shot_type_finished, Qt.UniqueConnection)
            logger.info("Starting ShotTypeWorker...")
            self.shot_type_worker.start()
            logger.info("ShotTypeWorker started")
        else:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"Ready - {len(self.clips)} scenes detected")

    def _on_shot_type_progress(self, current: int, total: int):
        """Handle shot type classification progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_shot_type_ready(self, clip_id: str, shot_type: str, confidence: float):
        """Handle shot type classification complete for a clip."""
        # Update the clip model
        clip = self.clips_by_id.get(clip_id)
        if clip:
            clip.shot_type = shot_type
            # Update the browser thumbnail with shot type
            self.clip_browser.update_clip_shot_type(clip_id, shot_type)
            logger.debug(f"Clip {clip_id}: {shot_type} ({confidence:.2f})")

    @Slot()
    def _on_shot_type_finished(self):
        """Handle all shot type classification completed."""
        logger.info("=== SHOT TYPE CLASSIFICATION FINISHED ===")

        # Guard against duplicate calls
        if self._shot_type_finished_handled:
            logger.warning("_on_shot_type_finished already handled, ignoring duplicate call")
            return
        self._shot_type_finished_handled = True

        logger.info(f"Shot type worker running: {self.shot_type_worker.isRunning() if self.shot_type_worker else 'None'}")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Ready - {len(self.clips)} scenes detected")

    def _on_clip_selected(self, clip: Clip):
        """Handle clip selection in browser."""
        if self.current_source:
            start_time = clip.start_time(self.current_source.fps)
            self.video_player.seek_to(start_time)

    def _on_clip_double_clicked(self, clip: Clip):
        """Handle clip double-click (play from clip start)."""
        if self.current_source:
            start_time = clip.start_time(self.current_source.fps)
            end_time = clip.end_time(self.current_source.fps)
            self.video_player.play_range(start_time, end_time)

    def _on_clip_dragged_to_timeline(self, clip: Clip):
        """Handle clip dragged from browser to timeline."""
        if self.current_source:
            self.timeline.set_fps(self.current_source.fps)
            self.timeline.add_clip(clip, self.current_source)
            self.status_bar.showMessage(f"Added clip to timeline")

    def _on_timeline_playhead_changed(self, time_seconds: float):
        """Handle timeline playhead position change."""
        # Don't seek during playback - playhead is driven by video position
        if not self._is_playing:
            self.video_player.seek_to(time_seconds)

    # --- Playback methods ---

    def _on_playback_requested(self, start_frame: int):
        """Start sequence playback from given frame."""
        if self._is_playing:
            # Toggle pause
            self._pause_playback()
            return

        sequence = self.timeline.get_sequence()
        if sequence.duration_frames == 0:
            return  # Nothing to play

        self._is_playing = True
        self.timeline.set_playing(True)

        # Start playback from current position
        self._play_clip_at_frame(start_frame)

    def _play_clip_at_frame(self, frame: int):
        """Load and play the clip at given timeline frame."""
        sequence = self.timeline.get_sequence()

        # Check if we're past the end of sequence
        if frame >= sequence.duration_frames:
            self._stop_playback()
            # Reset playhead to beginning
            self.timeline.set_playhead_time(0)
            return

        # Get clip at current position
        seq_clip, clip, source = self.timeline.get_clip_at_playhead()

        if not seq_clip:
            # No clip at this position (gap) - show black and advance via timer
            self._current_playback_clip = None
            self.video_player.player.stop()  # Shows black
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
        self.video_player.load_video(source.file_path)
        self.video_player.play_range(source_seconds, end_seconds)

        # Start timer to monitor for clip transitions
        self._playback_timer.start()

    def _on_playback_tick(self):
        """Called during playback to check for clip transitions and advance playhead in gaps."""
        if not self._is_playing:
            self._playback_timer.stop()
            return

        sequence = self.timeline.get_sequence()
        current_time = self.timeline.get_playhead_time()
        current_frame = int(current_time * sequence.fps)

        # Check if we're past the end of sequence
        if current_frame >= sequence.duration_frames:
            self._stop_playback()
            self.timeline.set_playhead_time(0)
            return

        if self._current_playback_clip:
            # Playing a clip - check if we've moved past it
            if current_frame >= self._current_playback_clip.end_frame():
                # Move to next position
                next_frame = self._current_playback_clip.end_frame()
                self.timeline.set_playhead_time(next_frame / sequence.fps)
                self._play_clip_at_frame(next_frame)
        else:
            # In a gap - advance playhead manually
            # Advance by ~1 frame worth of time (33ms at 30fps)
            new_time = current_time + (self._playback_timer.interval() / 1000.0)
            new_frame = int(new_time * sequence.fps)

            # Check if we've reached a clip
            self.timeline.set_playhead_time(new_time)
            seq_clip, _, _ = self.timeline.get_clip_at_playhead()

            if seq_clip:
                # Found a clip - start playing it
                self._play_clip_at_frame(new_frame)

    def _on_video_position_updated(self, position_ms: int):
        """Sync timeline playhead to video position during playback."""
        if not self._is_playing or not self._current_playback_clip:
            return

        seq_clip = self._current_playback_clip
        clip_data = self.timeline._clip_lookup.get(seq_clip.source_clip_id)
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
        timeline_seconds = timeline_frame / self.timeline.sequence.fps

        # Update playhead position
        self.timeline.set_playhead_time(timeline_seconds)

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
                self.timeline.set_playhead_time(next_frame / self.timeline.sequence.fps)
                self._play_clip_at_frame(next_frame)

    def _pause_playback(self):
        """Pause playback."""
        self._is_playing = False
        self._playback_timer.stop()
        self.video_player.player.pause()
        self.timeline.set_playing(False)

    def _on_stop_requested(self):
        """Handle stop request from timeline."""
        self._stop_playback()

    def _stop_playback(self):
        """Stop playback and reset state."""
        self._is_playing = False
        self._playback_timer.stop()
        self._current_playback_clip = None
        self.video_player.player.stop()
        self.timeline.set_playing(False)

    def _on_export_click(self):
        """Export selected clips."""
        selected = self.clip_browser.get_selected_clips()
        if not selected:
            QMessageBox.information(self, "Export", "No clips selected")
            return
        self._export_clips(selected)

    def _on_export_all_click(self):
        """Export all clips."""
        if not self.clips:
            return
        self._export_clips(self.clips)

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

    def _on_sequence_export_click(self):
        """Export the timeline sequence to a single video file."""
        sequence = self.timeline.get_sequence()
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

        config = ExportConfig(
            output_path=output_path,
            fps=sequence.fps,
        )

        # Start export in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.timeline.export_btn.setEnabled(False)

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
        self.timeline.export_btn.setEnabled(True)
        self.status_bar.showMessage(f"Sequence exported to {output_path.name}")

        # Open containing folder
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.parent)))

    def _on_sequence_export_error(self, error: str):
        """Handle sequence export error."""
        self.progress_bar.setVisible(False)
        self.timeline.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Export Error", f"Failed to export sequence: {error}")

    def closeEvent(self, event):
        """Clean up workers and timers before closing."""
        logger.info("=== CLOSE EVENT ===")

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
