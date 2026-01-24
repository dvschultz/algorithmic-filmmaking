"""Main application window."""

from pathlib import Path
from typing import Optional

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
from PySide6.QtCore import Qt, Signal, QThread, QMimeData, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from models.clip import Source, Clip
from core.scene_detect import SceneDetector, DetectionConfig
from core.thumbnail import ThumbnailGenerator
from core.downloader import VideoDownloader
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

    def run(self):
        try:
            detector = SceneDetector(self.config)
            source, clips = detector.detect_scenes_with_progress(
                self.video_path,
                lambda p, m: self.progress.emit(p, m),
            )
            self.finished.emit(source, clips)
        except Exception as e:
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

    def run(self):
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

        self.finished.emit()


class DownloadWorker(QThread):
    """Background worker for video downloads."""

    progress = Signal(float, str)  # progress (0-100), status message
    finished = Signal(object)  # DownloadResult
    error = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            downloader = VideoDownloader()
            result = downloader.download(
                self.url,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            if result.success:
                self.finished.emit(result)
            else:
                self.error.emit(result.error or "Download failed")
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window with drag-drop, detection, and preview."""

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scene Ripper - Algorithmic Filmmaking")
        self.setMinimumSize(1200, 800)
        self.setAcceptDrops(True)

        # State
        self.current_source: Optional[Source] = None
        self.clips: list[Clip] = []
        self.detection_worker: Optional[DetectionWorker] = None
        self.thumbnail_worker: Optional[ThumbnailWorker] = None
        self.download_worker: Optional[DownloadWorker] = None

        self._setup_ui()
        self._connect_signals()

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
        if not self.current_source:
            return

        # Disable buttons during detection
        self.detect_btn.setEnabled(False)
        self.import_btn.setEnabled(False)

        # Get sensitivity from slider
        threshold = self.sensitivity_slider.value() / 10.0
        config = DetectionConfig(threshold=threshold)

        # Start detection in background
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        self.detection_worker = DetectionWorker(
            self.current_source.file_path, config
        )
        self.detection_worker.progress.connect(self._on_detection_progress)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.start()

    def _on_detection_progress(self, progress: float, message: str):
        """Handle detection progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)

    def _on_detection_finished(self, source: Source, clips: list[Clip]):
        """Handle detection completion."""
        self.current_source = source
        self.clips = clips

        self.status_bar.showMessage(f"Found {len(clips)} scenes. Generating thumbnails...")

        # Start thumbnail generation
        self.thumbnail_worker = ThumbnailWorker(source, clips)
        self.thumbnail_worker.progress.connect(self._on_thumbnail_progress)
        self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_worker.finished.connect(self._on_thumbnails_finished)
        self.thumbnail_worker.start()

    def _on_detection_error(self, error: str):
        """Handle detection error."""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", error)

    def _on_thumbnail_progress(self, current: int, total: int):
        """Handle thumbnail generation progress."""
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
        """Handle individual thumbnail completion."""
        # Find clip and add to browser
        for clip in self.clips:
            if clip.id == clip_id:
                self.clip_browser.add_clip(clip, self.current_source)
                break

    def _on_thumbnails_finished(self):
        """Handle all thumbnails completed."""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
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
        self.video_player.seek_to(time_seconds)

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
        source_name = self.current_source.file_path.stem
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
