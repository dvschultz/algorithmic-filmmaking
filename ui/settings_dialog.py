"""Settings dialog with tabbed interface."""

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

from core.settings import (
    Settings,
    get_default_settings,
    get_cache_size,
    format_size,
    QUALITY_PRESETS,
    RESOLUTION_PRESETS,
    FPS_PRESETS,
)

logger = logging.getLogger(__name__)


def is_network_or_cloud_path(path: Path) -> tuple[bool, str]:
    """
    Check if a path appears to be on network or cloud storage.
    Returns (is_network_or_cloud, warning_message).
    """
    path_str = str(path).lower()

    # Common cloud storage paths
    cloud_indicators = [
        ("dropbox", "Dropbox"),
        ("onedrive", "OneDrive"),
        ("google drive", "Google Drive"),
        ("icloud", "iCloud Drive"),
        ("box sync", "Box"),
        ("pcloud", "pCloud"),
    ]

    for indicator, name in cloud_indicators:
        if indicator in path_str:
            return True, f"This path appears to be on {name}. Cloud storage may be slower and could cause sync issues."

    # Network paths (macOS/Linux)
    if path_str.startswith("/volumes/") and not path_str.startswith("/volumes/macintosh"):
        return True, "This path appears to be on an external or network volume. Performance may vary."

    # Check if path is on a network mount (Linux)
    if path_str.startswith("/mnt/") or path_str.startswith("/media/"):
        return True, "This path appears to be on a mounted volume. Performance may vary."

    # Windows network paths
    if path_str.startswith("\\\\") or path_str.startswith("//"):
        return True, "This path appears to be a network share. Performance may vary."

    return False, ""


class PathSelector(QWidget):
    """Widget for selecting a directory path with browse button."""

    def __init__(self, label: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._base_tooltip = tooltip
        self._setup_ui(label, tooltip)

    def _setup_ui(self, label: str, tooltip: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        layout.addWidget(self.label)

        self.path_edit = QLineEdit()
        self.path_edit.setToolTip(tooltip)
        layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip("Browse for folder")
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

        # Check for network paths when text changes
        self.path_edit.textChanged.connect(self._check_path_warning)

    def _on_browse(self):
        current = self.path_edit.text()
        start_dir = current if Path(current).exists() else str(Path.home())

        folder = QFileDialog.getExistingDirectory(
            self,
            f"Select {self.label.text().rstrip(':')}",
            start_dir,
        )
        if folder:
            self.path_edit.setText(folder)
            self._check_path_warning()

    def get_path(self) -> Path:
        return Path(self.path_edit.text())

    def set_path(self, path: Path):
        self.path_edit.setText(str(path))

    def set_enabled(self, enabled: bool):
        """Enable or disable the path selector."""
        self.path_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        if not enabled:
            self.path_edit.setToolTip(
                f"{self._base_tooltip}\n\n⚠️ Cannot change while background operations are running."
            )

    def _check_path_warning(self):
        """Check if path is on network/cloud and update tooltip with warning."""
        path = self.get_path()
        is_network, warning = is_network_or_cloud_path(path)

        if is_network:
            # Show warning style and update tooltip
            self.path_edit.setStyleSheet("border: 1px solid #f0ad4e;")  # Warning orange
            self.path_edit.setToolTip(f"{self._base_tooltip}\n\n⚠️ {warning}")
        else:
            # Reset to normal
            self.path_edit.setStyleSheet("")
            self.path_edit.setToolTip(self._base_tooltip)

    def validate(self) -> tuple[bool, str]:
        """Validate the path. Returns (is_valid, error_message)."""
        path = self.get_path()
        if not path.parent.exists():
            return False, f"Parent directory does not exist: {path.parent}"
        return True, ""


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings."""

    def __init__(self, settings: Settings, paths_disabled: bool = False, parent=None):
        """
        Initialize the settings dialog.

        Args:
            settings: Current settings object
            paths_disabled: If True, disable path settings (when operations are running)
            parent: Parent widget
        """
        super().__init__(parent)
        self.settings = settings
        self._paths_disabled = paths_disabled
        self.original_settings = Settings(
            thumbnail_cache_dir=settings.thumbnail_cache_dir,
            download_dir=settings.download_dir,
            export_dir=settings.export_dir,
            default_sensitivity=settings.default_sensitivity,
            min_scene_length_seconds=settings.min_scene_length_seconds,
            auto_analyze_colors=settings.auto_analyze_colors,
            auto_classify_shots=settings.auto_classify_shots,
            export_quality=settings.export_quality,
            export_resolution=settings.export_resolution,
            export_fps=settings.export_fps,
            transcription_model=settings.transcription_model,
            transcription_language=settings.transcription_language,
            auto_transcribe=settings.auto_transcribe,
        )

        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        self._setup_ui()
        self._load_settings()

        # Disable paths if operations are running
        if self._paths_disabled:
            self._disable_path_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_paths_tab(), "Paths")
        self.tabs.addTab(self._create_detection_tab(), "Detection")
        self.tabs.addTab(self._create_export_tab(), "Export")
        layout.addWidget(self.tabs)

        # Button row
        button_layout = QHBoxLayout()

        self.restore_btn = QPushButton("Restore Defaults")
        self.restore_btn.clicked.connect(self._on_restore_defaults)
        button_layout.addWidget(self.restore_btn)

        button_layout.addStretch()

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)

        layout.addLayout(button_layout)

    def _create_paths_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Paths group
        paths_group = QGroupBox("Storage Locations")
        paths_layout = QVBoxLayout(paths_group)

        self.cache_path = PathSelector(
            "Thumbnail Cache:",
            "Directory where thumbnail images are cached",
        )
        paths_layout.addWidget(self.cache_path)

        self.download_path = PathSelector(
            "Download Folder:",
            "Directory for downloaded videos from YouTube/Vimeo",
        )
        paths_layout.addWidget(self.download_path)

        self.export_path = PathSelector(
            "Export Folder:",
            "Default directory for exported video clips",
        )
        paths_layout.addWidget(self.export_path)

        layout.addWidget(paths_group)

        # Cache management group
        cache_group = QGroupBox("Cache Management")
        cache_layout = QHBoxLayout(cache_group)

        self.cache_size_label = QLabel("Calculating...")
        cache_layout.addWidget(self.cache_size_label)

        cache_layout.addStretch()

        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.setToolTip("Delete all cached thumbnails")
        self.clear_cache_btn.clicked.connect(self._on_clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)

        layout.addWidget(cache_group)

        layout.addStretch()
        return tab

    def _create_detection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Detection defaults group
        detection_group = QGroupBox("Detection Defaults")
        detection_layout = QVBoxLayout(detection_group)

        # Sensitivity slider
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Default Sensitivity:"))

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)  # 1.0 to 10.0
        self.sensitivity_slider.setToolTip(
            "Lower values detect more scenes (more sensitive to cuts).\n"
            "Higher values detect fewer, larger scenes."
        )
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_layout.addWidget(self.sensitivity_slider)

        self.sensitivity_label = QLabel("3.0")
        self.sensitivity_label.setFixedWidth(40)
        sens_layout.addWidget(self.sensitivity_label)

        detection_layout.addLayout(sens_layout)

        # Min scene length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Min Scene Length:"))

        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.1, 10.0)
        self.min_length_spin.setSingleStep(0.1)
        self.min_length_spin.setSuffix(" seconds")
        self.min_length_spin.setToolTip(
            "Minimum duration for a detected scene.\n"
            "Scenes shorter than this will be merged."
        )
        length_layout.addWidget(self.min_length_spin)

        length_layout.addStretch()
        detection_layout.addLayout(length_layout)

        layout.addWidget(detection_group)

        # Auto-analysis group
        analysis_group = QGroupBox("Automatic Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.auto_colors_check = QCheckBox("Analyze colors after detection")
        self.auto_colors_check.setToolTip(
            "Automatically extract dominant colors from each clip thumbnail"
        )
        analysis_layout.addWidget(self.auto_colors_check)

        self.auto_shots_check = QCheckBox("Classify shot types after detection")
        self.auto_shots_check.setToolTip(
            "Automatically classify each clip as wide shot, medium shot, close-up, etc."
        )
        analysis_layout.addWidget(self.auto_shots_check)

        note_label = QLabel("Note: Changes apply to future detections only")
        note_label.setStyleSheet("color: #666; font-style: italic;")
        analysis_layout.addWidget(note_label)

        layout.addWidget(analysis_group)

        # Transcription group
        transcription_group = QGroupBox("Transcription")
        transcription_layout = QVBoxLayout(transcription_group)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Whisper Model:"))

        self.transcription_model_combo = QComboBox()
        self.transcription_model_combo.addItems([
            "tiny.en - Fast, basic accuracy (39MB)",
            "small.en - Good balance (244MB)",
            "medium.en - Better accuracy (769MB)",
            "large-v3 - Best accuracy (1.5GB)",
        ])
        self.transcription_model_combo.setToolTip(
            "Larger models are more accurate but slower.\n"
            "Models are downloaded on first use."
        )
        model_layout.addWidget(self.transcription_model_combo)

        transcription_layout.addLayout(model_layout)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))

        self.transcription_lang_combo = QComboBox()
        self.transcription_lang_combo.addItems([
            "English",
            "Auto-detect",
        ])
        self.transcription_lang_combo.setToolTip(
            "Select 'Auto-detect' for multi-language content.\n"
            "English is faster for English-only content."
        )
        lang_layout.addWidget(self.transcription_lang_combo)

        lang_layout.addStretch()
        transcription_layout.addLayout(lang_layout)

        # Auto-transcribe checkbox
        self.auto_transcribe_check = QCheckBox("Auto-transcribe after detection")
        self.auto_transcribe_check.setToolTip(
            "Automatically transcribe speech after scene detection completes.\n"
            "Disable to manually trigger transcription from the Analyze tab."
        )
        transcription_layout.addWidget(self.auto_transcribe_check)

        layout.addWidget(transcription_group)

        layout.addStretch()
        return tab

    def _create_export_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export defaults group
        export_group = QGroupBox("Export Defaults")
        export_layout = QVBoxLayout(export_group)

        # Quality preset
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))

        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "High - Best quality, larger files",
            "Medium - Balanced (recommended)",
            "Low - Smaller files, faster encoding",
        ])
        self.quality_combo.setToolTip(
            "High: CRF 18, slow preset (~10 Mbps)\n"
            "Medium: CRF 23, medium preset (~5 Mbps)\n"
            "Low: CRF 28, fast preset (~2 Mbps)"
        )
        quality_layout.addWidget(self.quality_combo)

        export_layout.addLayout(quality_layout)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Original",
            "1080p (1920x1080)",
            "720p (1280x720)",
            "480p (854x480)",
        ])
        self.resolution_combo.setToolTip(
            "Maximum output resolution.\n"
            "Aspect ratio is preserved; video is scaled to fit."
        )
        res_layout.addWidget(self.resolution_combo)

        export_layout.addLayout(res_layout)

        # Frame rate
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate:"))

        self.fps_combo = QComboBox()
        self.fps_combo.addItems([
            "Original",
            "24 fps",
            "30 fps",
            "60 fps",
        ])
        self.fps_combo.setToolTip(
            "Output frame rate.\n"
            "'Original' uses the source video's frame rate."
        )
        fps_layout.addWidget(self.fps_combo)

        export_layout.addLayout(fps_layout)

        layout.addWidget(export_group)

        layout.addStretch()
        return tab

    def _disable_path_settings(self):
        """Disable path-related settings when operations are running."""
        self.cache_path.set_enabled(False)
        self.download_path.set_enabled(False)
        self.export_path.set_enabled(False)
        self.clear_cache_btn.setEnabled(False)
        self.clear_cache_btn.setToolTip(
            "Cannot clear cache while background operations are running."
        )

    def _load_settings(self):
        """Load current settings into UI controls."""
        # Paths
        self.cache_path.set_path(self.settings.thumbnail_cache_dir)
        self.download_path.set_path(self.settings.download_dir)
        self.export_path.set_path(self.settings.export_dir)

        # Update cache size display
        self._update_cache_size()

        # Detection
        self.sensitivity_slider.setValue(int(self.settings.default_sensitivity * 10))
        self.min_length_spin.setValue(self.settings.min_scene_length_seconds)
        self.auto_colors_check.setChecked(self.settings.auto_analyze_colors)
        self.auto_shots_check.setChecked(self.settings.auto_classify_shots)

        # Export
        quality_map = {"high": 0, "medium": 1, "low": 2}
        self.quality_combo.setCurrentIndex(
            quality_map.get(self.settings.export_quality, 1)
        )

        res_map = {"original": 0, "1080p": 1, "720p": 2, "480p": 3}
        self.resolution_combo.setCurrentIndex(
            res_map.get(self.settings.export_resolution, 0)
        )

        fps_map = {"original": 0, "24": 1, "30": 2, "60": 3}
        self.fps_combo.setCurrentIndex(fps_map.get(self.settings.export_fps, 0))

        # Transcription
        model_map = {"tiny.en": 0, "small.en": 1, "medium.en": 2, "large-v3": 3}
        self.transcription_model_combo.setCurrentIndex(
            model_map.get(self.settings.transcription_model, 1)
        )

        lang_map = {"en": 0, "auto": 1}
        self.transcription_lang_combo.setCurrentIndex(
            lang_map.get(self.settings.transcription_language, 0)
        )

        self.auto_transcribe_check.setChecked(self.settings.auto_transcribe)

    def _save_to_settings(self):
        """Save UI values to settings object."""
        # Paths
        self.settings.thumbnail_cache_dir = self.cache_path.get_path()
        self.settings.download_dir = self.download_path.get_path()
        self.settings.export_dir = self.export_path.get_path()

        # Detection
        self.settings.default_sensitivity = self.sensitivity_slider.value() / 10.0
        self.settings.min_scene_length_seconds = self.min_length_spin.value()
        self.settings.auto_analyze_colors = self.auto_colors_check.isChecked()
        self.settings.auto_classify_shots = self.auto_shots_check.isChecked()

        # Export
        quality_values = ["high", "medium", "low"]
        self.settings.export_quality = quality_values[self.quality_combo.currentIndex()]

        res_values = ["original", "1080p", "720p", "480p"]
        self.settings.export_resolution = res_values[self.resolution_combo.currentIndex()]

        fps_values = ["original", "24", "30", "60"]
        self.settings.export_fps = fps_values[self.fps_combo.currentIndex()]

        # Transcription
        model_values = ["tiny.en", "small.en", "medium.en", "large-v3"]
        self.settings.transcription_model = model_values[self.transcription_model_combo.currentIndex()]

        lang_values = ["en", "auto"]
        self.settings.transcription_language = lang_values[self.transcription_lang_combo.currentIndex()]

        self.settings.auto_transcribe = self.auto_transcribe_check.isChecked()

    def _validate(self) -> tuple[bool, str]:
        """Validate all settings. Returns (is_valid, error_message)."""
        # Validate paths
        for path_widget, name in [
            (self.cache_path, "Thumbnail Cache"),
            (self.download_path, "Download Folder"),
            (self.export_path, "Export Folder"),
        ]:
            valid, error = path_widget.validate()
            if not valid:
                return False, f"{name}: {error}"

        return True, ""

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        for path in [
            self.cache_path.get_path(),
            self.download_path.get_path(),
            self.export_path.get_path(),
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not create directory {path}: {e}")

    def _on_sensitivity_changed(self, value: int):
        self.sensitivity_label.setText(f"{value / 10:.1f}")

    def _on_clear_cache(self):
        """Clear the thumbnail cache."""
        cache_path = self.cache_path.get_path()
        size = get_cache_size(cache_path)

        if size == 0:
            QMessageBox.information(self, "Clear Cache", "Cache is already empty.")
            return

        reply = QMessageBox.question(
            self,
            "Clear Cache",
            f"Delete all cached thumbnails ({format_size(size)})?\n\n"
            "Thumbnails will be regenerated when needed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                import shutil
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    cache_path.mkdir(parents=True, exist_ok=True)
                self._update_cache_size()
                QMessageBox.information(self, "Clear Cache", "Cache cleared successfully.")
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache: {e}")

    def _update_cache_size(self):
        """Update the cache size display."""
        size = get_cache_size(self.cache_path.get_path())
        self.cache_size_label.setText(f"Cache size: {format_size(size)}")

    def _on_restore_defaults(self):
        """Restore all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Reset all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            defaults = get_default_settings()
            self.settings = defaults
            self._load_settings()

    def _on_accept(self):
        """Handle OK button click."""
        valid, error = self._validate()
        if not valid:
            QMessageBox.warning(self, "Invalid Settings", error)
            return

        self._ensure_directories()
        self._save_to_settings()
        self.accept()

    def get_settings(self) -> Settings:
        """Get the current settings (after dialog closes)."""
        return self.settings

    def has_changes(self) -> bool:
        """Check if settings have been modified."""
        self._save_to_settings()
        return (
            self.settings.thumbnail_cache_dir != self.original_settings.thumbnail_cache_dir
            or self.settings.download_dir != self.original_settings.download_dir
            or self.settings.export_dir != self.original_settings.export_dir
            or self.settings.default_sensitivity != self.original_settings.default_sensitivity
            or self.settings.min_scene_length_seconds != self.original_settings.min_scene_length_seconds
            or self.settings.auto_analyze_colors != self.original_settings.auto_analyze_colors
            or self.settings.auto_classify_shots != self.original_settings.auto_classify_shots
            or self.settings.export_quality != self.original_settings.export_quality
            or self.settings.export_resolution != self.original_settings.export_resolution
            or self.settings.export_fps != self.original_settings.export_fps
            or self.settings.transcription_model != self.original_settings.transcription_model
            or self.settings.transcription_language != self.original_settings.transcription_language
            or self.settings.auto_transcribe != self.original_settings.auto_transcribe
        )
