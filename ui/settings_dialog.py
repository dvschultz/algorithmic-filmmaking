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
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

from ui.widgets.styled_slider import StyledSlider

from core.settings import (
    Settings,
    get_default_settings,
    get_cache_size,
    format_size,
    is_from_environment,
    get_env_overridden_settings,
    QUALITY_PRESETS,
    RESOLUTION_PRESETS,
    FPS_PRESETS,
    ENV_YOUTUBE_API_KEY,
    ENV_CACHE_DIR,
    ENV_DOWNLOAD_DIR,
    ENV_EXPORT_DIR,
    ENV_SENSITIVITY,
    ENV_WHISPER_MODEL,
    # LLM API key functions and constants
    ENV_ANTHROPIC_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_GEMINI_API_KEY,
    ENV_OPENROUTER_API_KEY,
    get_anthropic_api_key,
    set_anthropic_api_key,
    get_openai_api_key,
    set_openai_api_key,
    get_gemini_api_key,
    set_gemini_api_key,
    get_openrouter_api_key,
    set_openrouter_api_key,
    is_api_key_from_env,
)
from core.llm_client import get_provider_models
from ui.theme import theme

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
        self._is_env_override = False
        self._env_var_name = ""
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
        self.browse_btn.setFixedWidth(44)  # Minimum touch target size
        self.browse_btn.setAccessibleName("Browse")
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
            self.path_edit.setStyleSheet(f"border: 1px solid {theme().accent_orange};")
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

    def set_from_environment(self, env_var_name: str):
        """Mark this path as set from an environment variable.

        This disables editing and shows an indicator.

        Args:
            env_var_name: Name of the environment variable (e.g., "SCENE_RIPPER_CACHE_DIR")
        """
        self._is_env_override = True
        self._env_var_name = env_var_name

        # Disable editing
        self.path_edit.setReadOnly(True)
        self.browse_btn.setEnabled(False)

        # Update label to show "(from env)"
        current_text = self.label.text()
        if not current_text.endswith("(env)"):
            self.label.setText(current_text.replace(":", " (env):"))
            self.label.setStyleSheet(f"color: {theme().accent_blue};")

        # Update tooltip
        self.path_edit.setToolTip(
            f"{self._base_tooltip}\n\n"
            f"🔒 This value is set by environment variable:\n"
            f"   {env_var_name}\n\n"
            "To change this, unset the environment variable or modify it."
        )
        self.path_edit.setStyleSheet(f"background-color: {theme().surface_highlight};")


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
            export_quality=settings.export_quality,
            export_resolution=settings.export_resolution,
            export_fps=settings.export_fps,
            transcription_model=settings.transcription_model,
            transcription_language=settings.transcription_language,
            theme_preference=settings.theme_preference,
            youtube_api_key=settings.youtube_api_key,
            youtube_results_count=settings.youtube_results_count,
            youtube_parallel_downloads=settings.youtube_parallel_downloads,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            ollama_model=settings.ollama_model,
            openai_model=settings.openai_model,
            anthropic_model=settings.anthropic_model,
            gemini_model=settings.gemini_model,
            openrouter_model=settings.openrouter_model,
            description_model_tier=settings.description_model_tier,
            description_model_cpu=settings.description_model_cpu,
            description_model_gpu=settings.description_model_gpu,
            description_model_cloud=settings.description_model_cloud,
            description_temporal_frames=settings.description_temporal_frames,
        )

        self.setWindowTitle("Settings")
        self.setMinimumSize(1000, 400)
        self.setModal(True)

        self._setup_ui()
        self._load_settings()

        # Disable paths if operations are running
        if self._paths_disabled:
            self._disable_path_settings()

        # Connect to theme changes to refresh dialog styles
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_paths_tab(), "Paths")
        self.tabs.addTab(self._create_detection_tab(), "Models")
        self.tabs.addTab(self._create_export_tab(), "Export")
        self.tabs.addTab(self._create_api_keys_tab(), "API Keys")
        self.tabs.addTab(self._create_appearance_tab(), "Appearance")
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

        # Shot Detection group
        detection_group = QGroupBox("Shot Detection")
        detection_layout = QVBoxLayout(detection_group)

        # Sensitivity slider
        sens_layout = QHBoxLayout()
        self.sensitivity_lbl = QLabel("Default Sensitivity:")
        sens_layout.addWidget(self.sensitivity_lbl)

        self.sensitivity_slider = StyledSlider(Qt.Horizontal)
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

        # Transcription group
        transcription_group = QGroupBox("Transcription")
        transcription_layout = QVBoxLayout(transcription_group)

        # Model selection
        model_layout = QHBoxLayout()
        self.whisper_model_lbl = QLabel("Whisper Model:")
        model_layout.addWidget(self.whisper_model_lbl)

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

        layout.addWidget(transcription_group)

        # Vision Description group
        vision_group = QGroupBox("Vision Description")
        vision_layout = QVBoxLayout(vision_group)

        # Tier selection
        tier_layout = QHBoxLayout()
        tier_layout.addWidget(QLabel("Processing Tier:"))
        self.vision_tier_combo = QComboBox()
        self.vision_tier_combo.addItems([
            "CPU (Local) - Free, runs on device (Moondream)",
            "Cloud API - Higher quality, costs money (GPT-4o/Claude)",
        ])
        self.vision_tier_combo.setToolTip(
            "Choose where to run video analysis.\n"
            "CPU: Downloads ~1.6GB model once, runs locally.\n"
            "Cloud: Sends frames to LLM provider (requires API key)."
        )
        tier_layout.addWidget(self.vision_tier_combo)
        vision_layout.addLayout(tier_layout)

        # CPU Model
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("Local Model:"))
        self.vision_cpu_combo = QComboBox()
        self.vision_cpu_combo.addItems([
            "vikhyatk/moondream2",
        ])
        self.vision_cpu_combo.setEditable(True)  # Allow custom model IDs
        self.vision_cpu_combo.setToolTip("HuggingFace model ID for local analysis")
        cpu_layout.addWidget(self.vision_cpu_combo)
        vision_layout.addLayout(cpu_layout)

        # Cloud Model
        cloud_layout = QHBoxLayout()
        cloud_layout.addWidget(QLabel("Cloud Model:"))
        self.vision_cloud_combo = QComboBox()
        self.vision_cloud_combo.addItems([
            "gpt-5.2",
            "gpt-5",
            "gpt-5-mini",
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ])
        self.vision_cloud_combo.setEditable(True)
        self.vision_cloud_combo.setToolTip("Model ID for cloud analysis (via LiteLLM)")
        cloud_layout.addWidget(self.vision_cloud_combo)
        vision_layout.addLayout(cloud_layout)

        # Temporal Frames
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Frames per Clip:"))
        self.vision_frames_spin = QSpinBox()
        self.vision_frames_spin.setRange(1, 10)
        self.vision_frames_spin.setValue(4)
        self.vision_frames_spin.setToolTip(
            "Number of frames to analyze per clip for descriptions.\n"
            "Higher = better understanding of action, but slower."
        )
        frames_layout.addWidget(self.vision_frames_spin)
        frames_layout.addStretch()
        vision_layout.addLayout(frames_layout)

        # Connect tier change to enable/disable appropriate model fields
        self.vision_tier_combo.currentIndexChanged.connect(self._on_vision_tier_changed)

        layout.addWidget(vision_group)

        # Chat Agent group
        chat_group = QGroupBox("Chat Agent")
        chat_layout = QVBoxLayout(chat_group)

        # Default provider dropdown
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Default Provider:"))
        self.chat_provider_combo = QComboBox()
        self.chat_provider_combo.addItems([
            "Local (Ollama)",
            "OpenAI",
            "Anthropic",
            "Gemini",
            "OpenRouter"
        ])
        self.chat_provider_combo.setToolTip(
            "Select the default LLM provider for the chat agent.\n"
            "Local (Ollama) runs models on your machine."
        )
        provider_layout.addWidget(self.chat_provider_combo)
        provider_layout.addStretch()
        chat_layout.addLayout(provider_layout)

        # Per-provider model dropdowns
        self._create_provider_model_section(chat_layout, "Local (Ollama)", "local")
        self._create_provider_model_section(chat_layout, "OpenAI", "openai")
        self._create_provider_model_section(chat_layout, "Anthropic", "anthropic")
        self._create_provider_model_section(chat_layout, "Gemini", "gemini")
        self._create_provider_model_section(chat_layout, "OpenRouter", "openrouter")

        layout.addWidget(chat_group)

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

    def _create_api_keys_tab(self) -> QWidget:
        """Create the API Keys settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # LLM API Keys group
        llm_group = QGroupBox("LLM Provider API Keys")
        llm_layout = QVBoxLayout(llm_group)

        # Anthropic API Key
        anthropic_layout = QHBoxLayout()
        self.anthropic_api_key_lbl = QLabel("Anthropic:")
        self.anthropic_api_key_lbl.setFixedWidth(100)
        anthropic_layout.addWidget(self.anthropic_api_key_lbl)

        self.anthropic_api_key_edit = QLineEdit()
        self.anthropic_api_key_edit.setEchoMode(QLineEdit.Password)
        self.anthropic_api_key_edit.setPlaceholderText("sk-ant-...")
        anthropic_layout.addWidget(self.anthropic_api_key_edit)

        self.anthropic_show_btn = QPushButton("Show")
        self.anthropic_show_btn.setCheckable(True)
        self.anthropic_show_btn.toggled.connect(
            lambda show: self._toggle_llm_key_visibility(self.anthropic_api_key_edit, self.anthropic_show_btn, show)
        )
        anthropic_layout.addWidget(self.anthropic_show_btn)

        llm_layout.addLayout(anthropic_layout)

        # OpenAI API Key
        openai_layout = QHBoxLayout()
        self.openai_api_key_lbl = QLabel("OpenAI:")
        self.openai_api_key_lbl.setFixedWidth(100)
        openai_layout.addWidget(self.openai_api_key_lbl)

        self.openai_api_key_edit = QLineEdit()
        self.openai_api_key_edit.setEchoMode(QLineEdit.Password)
        self.openai_api_key_edit.setPlaceholderText("sk-...")
        openai_layout.addWidget(self.openai_api_key_edit)

        self.openai_show_btn = QPushButton("Show")
        self.openai_show_btn.setCheckable(True)
        self.openai_show_btn.toggled.connect(
            lambda show: self._toggle_llm_key_visibility(self.openai_api_key_edit, self.openai_show_btn, show)
        )
        openai_layout.addWidget(self.openai_show_btn)

        llm_layout.addLayout(openai_layout)

        # Gemini API Key
        gemini_layout = QHBoxLayout()
        self.gemini_api_key_lbl = QLabel("Gemini:")
        self.gemini_api_key_lbl.setFixedWidth(100)
        gemini_layout.addWidget(self.gemini_api_key_lbl)

        self.gemini_api_key_edit = QLineEdit()
        self.gemini_api_key_edit.setEchoMode(QLineEdit.Password)
        self.gemini_api_key_edit.setPlaceholderText("AIza...")
        gemini_layout.addWidget(self.gemini_api_key_edit)

        self.gemini_show_btn = QPushButton("Show")
        self.gemini_show_btn.setCheckable(True)
        self.gemini_show_btn.toggled.connect(
            lambda show: self._toggle_llm_key_visibility(self.gemini_api_key_edit, self.gemini_show_btn, show)
        )
        gemini_layout.addWidget(self.gemini_show_btn)

        llm_layout.addLayout(gemini_layout)

        # OpenRouter API Key
        openrouter_layout = QHBoxLayout()
        self.openrouter_api_key_lbl = QLabel("OpenRouter:")
        self.openrouter_api_key_lbl.setFixedWidth(100)
        openrouter_layout.addWidget(self.openrouter_api_key_lbl)

        self.openrouter_api_key_edit = QLineEdit()
        self.openrouter_api_key_edit.setEchoMode(QLineEdit.Password)
        self.openrouter_api_key_edit.setPlaceholderText("sk-or-...")
        openrouter_layout.addWidget(self.openrouter_api_key_edit)

        self.openrouter_show_btn = QPushButton("Show")
        self.openrouter_show_btn.setCheckable(True)
        self.openrouter_show_btn.toggled.connect(
            lambda show: self._toggle_llm_key_visibility(self.openrouter_api_key_edit, self.openrouter_show_btn, show)
        )
        openrouter_layout.addWidget(self.openrouter_show_btn)

        llm_layout.addLayout(openrouter_layout)

        # LLM Help text
        llm_help_label = QLabel(
            "API keys are stored securely in your system keyring. "
            "Environment variables (e.g., ANTHROPIC_API_KEY) take priority."
        )
        llm_help_label.setWordWrap(True)
        llm_help_label.setStyleSheet(f"color: {theme().text_secondary};")
        llm_layout.addWidget(llm_help_label)

        layout.addWidget(llm_group)

        # YouTube API group
        youtube_group = QGroupBox("YouTube Data API")
        youtube_layout = QVBoxLayout(youtube_group)

        # API Key input
        key_layout = QHBoxLayout()
        self.youtube_api_key_lbl = QLabel("API Key:")
        key_layout.addWidget(self.youtube_api_key_lbl)

        self.youtube_api_key_edit = QLineEdit()
        self.youtube_api_key_edit.setEchoMode(QLineEdit.Password)
        self.youtube_api_key_edit.setPlaceholderText("Enter your YouTube Data API v3 key")
        key_layout.addWidget(self.youtube_api_key_edit)

        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_api_key_visibility)
        key_layout.addWidget(self.show_key_btn)

        youtube_layout.addLayout(key_layout)

        # Help text (hidden when API key is already set)
        self.youtube_help_label = QLabel(
            '<a href="https://console.cloud.google.com/apis/credentials">'
            "Get an API key from Google Cloud Console</a>"
        )
        self.youtube_help_label.setOpenExternalLinks(True)
        self.youtube_help_label.setStyleSheet(f"color: {theme().text_secondary};")
        youtube_layout.addWidget(self.youtube_help_label)

        # Update help label visibility when API key changes
        self.youtube_api_key_edit.textChanged.connect(self._update_youtube_help_visibility)

        # Results count and parallel downloads on same line (two columns)
        options_layout = QHBoxLayout()

        # Left column: Search results
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Search results:"))
        self.youtube_results_spin = QSpinBox()
        self.youtube_results_spin.setRange(10, 50)
        self.youtube_results_spin.setValue(25)
        self.youtube_results_spin.setToolTip("Number of results per search (affects API quota)")
        results_layout.addWidget(self.youtube_results_spin)
        results_layout.addStretch()

        # Right column: Parallel downloads
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("Parallel downloads:"))
        self.youtube_parallel_spin = QSpinBox()
        self.youtube_parallel_spin.setRange(1, 3)
        self.youtube_parallel_spin.setValue(2)
        self.youtube_parallel_spin.setToolTip("Number of simultaneous downloads")
        parallel_layout.addWidget(self.youtube_parallel_spin)
        parallel_layout.addStretch()

        # Add both columns with equal stretch
        left_widget = QWidget()
        left_widget.setLayout(results_layout)
        right_widget = QWidget()
        right_widget.setLayout(parallel_layout)

        options_layout.addWidget(left_widget, 1)
        options_layout.addWidget(right_widget, 1)

        youtube_layout.addLayout(options_layout)

        layout.addWidget(youtube_group)
        layout.addStretch()

        return tab

    def _toggle_llm_key_visibility(self, edit: QLineEdit, btn: QPushButton, show: bool):
        """Toggle visibility for an LLM API key field."""
        if show:
            edit.setEchoMode(QLineEdit.Normal)
            btn.setText("Hide")
        else:
            edit.setEchoMode(QLineEdit.Password)
            btn.setText("Show")

    def _toggle_api_key_visibility(self, show: bool):
        """Toggle API key visibility."""
        if show:
            self.youtube_api_key_edit.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.youtube_api_key_edit.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Show")

    def _update_youtube_help_visibility(self):
        """Show help link only when no API key is entered."""
        has_key = bool(self.youtube_api_key_edit.text().strip())
        self.youtube_help_label.setVisible(not has_key)

    def _create_provider_model_section(self, parent_layout, label: str, provider_key: str):
        """Create a model selection subsection for a provider."""
        section_layout = QHBoxLayout()

        model_label = QLabel(f"  {label} Model:")
        section_layout.addWidget(model_label)

        combo = QComboBox()
        for model_id, display_name in get_provider_models(provider_key):
            combo.addItem(display_name, model_id)  # userData = model_id
        combo.setToolTip(f"Select the model to use with {label}")

        section_layout.addWidget(combo)

        # Add "Add API key to enable" label (hidden by default, shown when disabled)
        hint_label = QLabel("Add API key to enable")
        hint_label.setStyleSheet(f"color: {theme().text_secondary}; font-style: italic;")
        hint_label.setVisible(False)
        section_layout.addWidget(hint_label)

        section_layout.addStretch()
        parent_layout.addLayout(section_layout)

        # Store references
        setattr(self, f"{provider_key}_model_combo", combo)
        setattr(self, f"{provider_key}_model_label", model_label)
        setattr(self, f"{provider_key}_hint_label", hint_label)

    def _update_chat_model_availability(self):
        """Disable model dropdowns for providers without API keys."""
        api_key_checks = {
            "openai": get_openai_api_key,
            "anthropic": get_anthropic_api_key,
            "gemini": get_gemini_api_key,
            "openrouter": get_openrouter_api_key,
        }

        for provider, get_key in api_key_checks.items():
            combo = getattr(self, f"{provider}_model_combo", None)
            label = getattr(self, f"{provider}_model_label", None)
            hint = getattr(self, f"{provider}_hint_label", None)
            if combo:
                has_key = bool(get_key())
                combo.setEnabled(has_key)
                # Gray out the label when disabled
                if label:
                    if has_key:
                        label.setStyleSheet("")  # Reset to default
                    else:
                        label.setStyleSheet(f"color: {theme().text_muted};")
                if hint:
                    hint.setVisible(not has_key)

        # Local (Ollama) is always enabled - no API key needed

    def _set_model_combo(self, provider: str, model_id: str):
        """Set a model combo box to the specified model ID.

        Args:
            provider: Provider key (local, openai, etc.)
            model_id: Model identifier to select
        """
        combo = getattr(self, f"{provider}_model_combo", None)
        if combo:
            # Find the index by userData
            for i in range(combo.count()):
                if combo.itemData(i) == model_id:
                    combo.setCurrentIndex(i)
                    return
            # If not found, default to first item
            combo.setCurrentIndex(0)

    def _create_appearance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Theme group
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout(theme_group)

        # Theme selector
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Appearance:"))

        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "System (follow OS setting)",
            "Light",
            "Dark",
        ])
        self.theme_combo.setToolTip(
            "Choose the app's color theme:\n"
            "• System: Follows your operating system's light/dark mode\n"
            "• Light: Always use light colors\n"
            "• Dark: Always use dark colors"
        )
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()

        theme_layout.addLayout(theme_row)

        note_label = QLabel("Note: Theme changes take effect immediately")
        note_label.setStyleSheet(f"color: {theme().text_secondary}; font-style: italic;")
        theme_layout.addWidget(note_label)

        layout.addWidget(theme_group)

        layout.addStretch()
        return tab

    def _on_vision_tier_changed(self, index: int):
        """Enable/disable model fields based on selected tier."""
        is_cpu = (index == 0)
        self.vision_cpu_combo.setEnabled(is_cpu)
        self.vision_cloud_combo.setEnabled(not is_cpu)

    def _disable_path_settings(self):
        """Disable path-related settings when operations are running."""
        self.cache_path.set_enabled(False)
        self.download_path.set_enabled(False)
        self.export_path.set_enabled(False)
        self.clear_cache_btn.setEnabled(False)
        self.clear_cache_btn.setToolTip(
            "Cannot clear cache while background operations are running."
        )

    def _apply_env_indicator_to_widget(
        self, widget: QWidget, label: QLabel, setting_name: str, env_var_name: str
    ):
        """Apply environment override indicator to a widget.

        Args:
            widget: The input widget (QLineEdit, QSlider, QComboBox, etc.)
            label: The associated label widget
            setting_name: Name of the setting field
            env_var_name: Name of the environment variable
        """
        if not is_from_environment(setting_name):
            return

        # Update label
        current_text = label.text()
        if not current_text.endswith("(env)"):
            label.setText(current_text.replace(":", " (env):"))
            label.setStyleSheet(f"color: {theme().accent_blue};")

        # Disable the widget
        widget.setEnabled(False)

        # Set tooltip explaining the environment override
        widget.setToolTip(
            f"🔒 This value is set by environment variable:\n"
            f"   {env_var_name}\n\n"
            "To change this, unset the environment variable or modify it."
        )

    def _load_settings(self):
        """Load current settings into UI controls."""
        # Paths
        self.cache_path.set_path(self.settings.thumbnail_cache_dir)
        self.download_path.set_path(self.settings.download_dir)
        self.export_path.set_path(self.settings.export_dir)

        # Apply environment override indicators for paths
        if is_from_environment("thumbnail_cache_dir"):
            self.cache_path.set_from_environment(ENV_CACHE_DIR)
        if is_from_environment("download_dir"):
            self.download_path.set_from_environment(ENV_DOWNLOAD_DIR)
        if is_from_environment("export_dir"):
            self.export_path.set_from_environment(ENV_EXPORT_DIR)

        # Update cache size display
        self._update_cache_size()

        # Detection
        self.sensitivity_slider.setValue(int(self.settings.default_sensitivity * 10))
        self.min_length_spin.setValue(self.settings.min_scene_length_seconds)

        # Apply environment override indicator for sensitivity
        self._apply_env_indicator_to_widget(
            self.sensitivity_slider, self.sensitivity_lbl,
            "default_sensitivity", ENV_SENSITIVITY
        )

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

        # Apply environment override indicator for whisper model
        self._apply_env_indicator_to_widget(
            self.transcription_model_combo, self.whisper_model_lbl,
            "transcription_model", ENV_WHISPER_MODEL
        )

        # Vision Description
        tier_idx = 0 if self.settings.description_model_tier == "cpu" else 1
        self.vision_tier_combo.setCurrentIndex(tier_idx)
        
        self.vision_cpu_combo.setCurrentText(self.settings.description_model_cpu)
        self.vision_cloud_combo.setCurrentText(self.settings.description_model_cloud)
        self.vision_frames_spin.setValue(self.settings.description_temporal_frames)
        
        # Trigger enable/disable state
        self._on_vision_tier_changed(tier_idx)

        # Appearance
        theme_map = {"system": 0, "light": 1, "dark": 2}
        self.theme_combo.setCurrentIndex(
            theme_map.get(self.settings.theme_preference, 0)
        )

        # YouTube
        self.youtube_api_key_edit.setText(self.settings.youtube_api_key)
        self.youtube_results_spin.setValue(self.settings.youtube_results_count)
        self.youtube_parallel_spin.setValue(self.settings.youtube_parallel_downloads)

        # Apply environment override indicator for YouTube API key
        self._apply_env_indicator_to_widget(
            self.youtube_api_key_edit, self.youtube_api_key_lbl,
            "youtube_api_key", ENV_YOUTUBE_API_KEY
        )
        # Also disable the show/hide button if from env
        if is_from_environment("youtube_api_key"):
            self.show_key_btn.setEnabled(False)

        # LLM API Keys - load from keyring
        self.anthropic_api_key_edit.setText(get_anthropic_api_key())
        self.openai_api_key_edit.setText(get_openai_api_key())
        self.gemini_api_key_edit.setText(get_gemini_api_key())
        self.openrouter_api_key_edit.setText(get_openrouter_api_key())

        # Apply environment override indicators for LLM API keys
        self._apply_llm_env_override("anthropic", self.anthropic_api_key_edit,
                                      self.anthropic_api_key_lbl, self.anthropic_show_btn,
                                      ENV_ANTHROPIC_API_KEY)
        self._apply_llm_env_override("openai", self.openai_api_key_edit,
                                      self.openai_api_key_lbl, self.openai_show_btn,
                                      ENV_OPENAI_API_KEY)
        self._apply_llm_env_override("gemini", self.gemini_api_key_edit,
                                      self.gemini_api_key_lbl, self.gemini_show_btn,
                                      ENV_GEMINI_API_KEY)
        self._apply_llm_env_override("openrouter", self.openrouter_api_key_edit,
                                      self.openrouter_api_key_lbl, self.openrouter_show_btn,
                                      ENV_OPENROUTER_API_KEY)

        # Chat Agent settings
        provider_map = {"local": 0, "openai": 1, "anthropic": 2, "gemini": 3, "openrouter": 4}
        self.chat_provider_combo.setCurrentIndex(provider_map.get(self.settings.llm_provider, 0))

        # Per-provider models
        self._set_model_combo("local", self.settings.ollama_model)
        self._set_model_combo("openai", self.settings.openai_model)
        self._set_model_combo("anthropic", self.settings.anthropic_model)
        self._set_model_combo("gemini", self.settings.gemini_model)
        self._set_model_combo("openrouter", self.settings.openrouter_model)

        # Update model dropdown availability based on API keys
        self._update_chat_model_availability()

    def _apply_llm_env_override(self, provider: str, edit: QLineEdit, label: QLabel,
                                 show_btn: QPushButton, env_var: str):
        """Apply environment override indicator for an LLM API key field."""
        if not is_api_key_from_env(provider):
            return

        # Update label to show "(env)"
        current_text = label.text()
        if not current_text.endswith("(env)"):
            label.setText(f"{current_text} (env)")
            label.setStyleSheet(f"color: {theme().accent_blue};")

        # Disable the field
        edit.setEnabled(False)
        edit.setToolTip(
            f"🔒 This value is set by environment variable:\n"
            f"   {env_var}\n\n"
            "To change this, unset the environment variable or modify it."
        )
        edit.setStyleSheet(f"background-color: {theme().surface_highlight};")

        # Disable show/hide button
        show_btn.setEnabled(False)

    def _save_to_settings(self):
        """Save UI values to settings object."""
        # Paths
        self.settings.thumbnail_cache_dir = self.cache_path.get_path()
        self.settings.download_dir = self.download_path.get_path()
        self.settings.export_dir = self.export_path.get_path()

        # Detection
        self.settings.default_sensitivity = self.sensitivity_slider.value() / 10.0
        self.settings.min_scene_length_seconds = self.min_length_spin.value()

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

        # Vision Description
        self.settings.description_model_tier = "cpu" if self.vision_tier_combo.currentIndex() == 0 else "cloud"
        self.settings.description_model_cpu = self.vision_cpu_combo.currentText()
        self.settings.description_model_cloud = self.vision_cloud_combo.currentText()
        self.settings.description_temporal_frames = self.vision_frames_spin.value()

        # Appearance
        theme_values = ["system", "light", "dark"]
        self.settings.theme_preference = theme_values[self.theme_combo.currentIndex()]

        # YouTube
        self.settings.youtube_api_key = self.youtube_api_key_edit.text()
        self.settings.youtube_results_count = self.youtube_results_spin.value()
        self.settings.youtube_parallel_downloads = self.youtube_parallel_spin.value()

        # Chat Agent
        provider_values = ["local", "openai", "anthropic", "gemini", "openrouter"]
        self.settings.llm_provider = provider_values[self.chat_provider_combo.currentIndex()]

        # Per-provider models
        self.settings.ollama_model = self.local_model_combo.currentData()
        self.settings.openai_model = self.openai_model_combo.currentData()
        self.settings.anthropic_model = self.anthropic_model_combo.currentData()
        self.settings.gemini_model = self.gemini_model_combo.currentData()
        self.settings.openrouter_model = self.openrouter_model_combo.currentData()

        # Update llm_model to match current provider's selection
        self.settings.llm_model = self.settings.get_model_for_provider(self.settings.llm_provider)

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

    def _on_theme_changed(self, index: int):
        """Apply theme change immediately."""
        theme_values = ["system", "light", "dark"]
        preference = theme_values[index]
        theme().set_preference(preference)
        # Force refresh even if is_dark didn't change (in case of system -> explicit same)
        theme().refresh()

    def _refresh_theme(self):
        """Refresh themed elements in this dialog."""
        # Update note labels with theme colors
        for widget in self.findChildren(QLabel):
            if "font-style: italic" in (widget.styleSheet() or ""):
                widget.setStyleSheet(f"color: {theme().text_secondary}; font-style: italic;")

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
        self._save_llm_api_keys()
        self.accept()

    def _save_llm_api_keys(self):
        """Save LLM API keys to keyring (only if not from environment)."""
        # Only save if not overridden by environment variable
        if not is_api_key_from_env("anthropic"):
            set_anthropic_api_key(self.anthropic_api_key_edit.text())
        if not is_api_key_from_env("openai"):
            set_openai_api_key(self.openai_api_key_edit.text())
        if not is_api_key_from_env("gemini"):
            set_gemini_api_key(self.gemini_api_key_edit.text())
        if not is_api_key_from_env("openrouter"):
            set_openrouter_api_key(self.openrouter_api_key_edit.text())

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
            or self.settings.export_quality != self.original_settings.export_quality
            or self.settings.export_resolution != self.original_settings.export_resolution
            or self.settings.export_fps != self.original_settings.export_fps
            or self.settings.transcription_model != self.original_settings.transcription_model
            or self.settings.transcription_language != self.original_settings.transcription_language
            or self.settings.description_model_tier != self.original_settings.description_model_tier
            or self.settings.description_model_cpu != self.original_settings.description_model_cpu
            or self.settings.description_model_cloud != self.original_settings.description_model_cloud
            or self.settings.description_temporal_frames != self.original_settings.description_temporal_frames
            or self.settings.theme_preference != self.original_settings.theme_preference
            or self.settings.youtube_api_key != self.original_settings.youtube_api_key
            or self.settings.youtube_results_count != self.original_settings.youtube_results_count
            or self.settings.youtube_parallel_downloads != self.original_settings.youtube_parallel_downloads
            or self.settings.llm_provider != self.original_settings.llm_provider
            or self.settings.llm_model != self.original_settings.llm_model
            or self.settings.ollama_model != self.original_settings.ollama_model
            or self.settings.openai_model != self.original_settings.openai_model
            or self.settings.anthropic_model != self.original_settings.anthropic_model
            or self.settings.gemini_model != self.original_settings.gemini_model
            or self.settings.openrouter_model != self.original_settings.openrouter_model
        )
