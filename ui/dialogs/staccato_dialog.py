"""Staccato dialog — beat-driven sequencing with onset-strength visual contrast.

User selects a music file, previews the waveform with beat markers,
then generates a sequence where onset strength drives visual contrast
between consecutive clips via DINOv2 embedding distance.
"""

import logging
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QSignalBlocker, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

from core.analysis.audio import (
    AudioAnalysis,
    OnsetDetectionConfig,
    analyze_music_file,
    make_onset_detection_config,
)
from core.audio_formats import AUDIO_FILE_DIALOG_FILTER as _AUDIO_FORMATS
from core.remix.staccato import generate_staccato_sequence
from ui.theme import theme, Spacing, TypeScale, UISizes
from ui.widgets.waveform_widget import WaveformWidget
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class StaccatoAnalyzeWorker(CancellableWorker):
    """Background worker for analyzing a music file or separated stem."""

    audio_ready = Signal(object, object)  # AudioAnalysis, np.ndarray (samples)
    progress_message = Signal(str)

    def __init__(
        self,
        music_path: Path,
        stem_name: str | None = None,
        stems_cache_dir: Path | None = None,
        onset_config: OnsetDetectionConfig | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._music_path = music_path
        self._stem_name = stem_name
        self._stems_cache_dir = stems_cache_dir
        self._onset_config = onset_config

    def run(self):
        self._log_start()
        try:
            # Ensure librosa is installed before attempting audio analysis
            from core.feature_registry import check_feature_ready, install_for_feature

            available, _missing = check_feature_ready("audio_analysis")
            if not available:
                self.progress_message.emit("Installing audio analysis dependencies...")
                if not install_for_feature("audio_analysis"):
                    self.error.emit("Failed to install audio analysis dependencies (librosa)")
                    return

            audio_path = self._music_path

            # If stem separation requested, ensure deps are installed then separate
            if self._stem_name and self._stems_cache_dir:
                stem_available, _missing = check_feature_ready("stem_separation")
                if not stem_available:
                    self.progress_message.emit("Installing stem separation dependencies...")
                    if not install_for_feature("stem_separation"):
                        self.error.emit("Failed to install stem separation dependencies (demucs)")
                        return

                audio_path = self._get_or_separate_stem()
                if audio_path is None:
                    return  # cancelled or error already emitted

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.progress_message.emit(
                f"Analyzing {self._stem_name} track..."
                if self._stem_name else "Analyzing audio..."
            )
            analysis = analyze_music_file(
                audio_path,
                include_onsets=True,
                onset_config=self._onset_config,
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Load raw samples for waveform display
            from core.analysis.audio import _get_librosa
            librosa = _get_librosa()
            y, _sr = librosa.load(str(audio_path), sr=22050)

            self.audio_ready.emit(analysis, y)
        except Exception as e:
            if not self.is_cancelled():
                logger.error("Staccato audio analysis failed: %s", e)
                self.error.emit(str(e))
        self._log_complete()

    def _get_or_separate_stem(self) -> Path | None:
        """Get the stem audio path, running Demucs if not cached."""
        from core.analysis.stem_separation import (
            get_cached_stems,
            get_stem_cache_key,
            separate_stems,
        )

        cache_key = get_stem_cache_key(self._music_path)
        stem_dir = self._stems_cache_dir / cache_key

        # Check cache first
        cached = get_cached_stems(self._music_path, self._stems_cache_dir)
        if cached and self._stem_name in cached:
            logger.info(f"Using cached {self._stem_name} stem")
            return cached[self._stem_name]

        # Run separation
        self.progress_message.emit("Separating stems (this may take a minute)...")

        try:
            stems = separate_stems(
                self._music_path,
                stem_dir,
                progress_cb=lambda msg: self.progress_message.emit(msg),
            )
        except ImportError as e:
            self.error.emit(str(e))
            return None
        except RuntimeError as e:
            self.error.emit(str(e))
            return None

        if self.is_cancelled():
            self._log_cancelled()
            return None

        if self._stem_name in stems:
            return stems[self._stem_name]

        self.error.emit(f"Stem '{self._stem_name}' not found in separation output")
        return None


class StaccatoGenerateWorker(CancellableWorker):
    """Background worker for generating the beat-driven sequence."""

    progress_update = Signal(int, int)
    progress_message = Signal(str)
    finished_sequence = Signal(object)  # StaccatoResult (list-like)

    def __init__(
        self,
        clips: list,
        audio_analysis: AudioAnalysis,
        strategy: str,
        cut_times: list[float] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._clips = clips
        self._audio_analysis = audio_analysis
        self._strategy = strategy
        self._cut_times = cut_times

    def run(self):
        self._log_start()
        try:
            # Step 1: Auto-compute missing embeddings
            self.progress_message.emit("Computing clip embeddings...")
            self._auto_compute_embeddings()

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Step 2: Generate beat-driven sequence
            self.progress_message.emit("Matching clips to beats...")
            result = generate_staccato_sequence(
                clips=self._clips,
                audio_analysis=self._audio_analysis,
                strategy=self._strategy,
                cut_times=self._cut_times,
                progress_cb=self._on_progress,
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.finished_sequence.emit(result)
        except Exception as e:
            if not self.is_cancelled():
                logger.error("Staccato generation failed: %s", e)
                self.error.emit(str(e))
        self._log_complete()

    def _auto_compute_embeddings(self):
        """Compute DINOv2 embeddings for clips that don't have them."""
        needs_embedding = [
            (clip, source) for clip, source in self._clips
            if clip.embedding is None and clip.thumbnail_path
        ]
        if not needs_embedding:
            return

        from core.feature_registry import check_feature

        available, missing = check_feature("embeddings")
        if not available:
            raise RuntimeError(
                "DINOv2 embeddings require torch and transformers. "
                f"Missing: {', '.join(missing)}. "
                "Run embedding analysis first or install dependencies via Settings."
            )

        from core.analysis.embeddings import extract_clip_embeddings_batch, _EMBEDDING_MODEL_TAG

        self.progress_message.emit(
            f"Computing embeddings for {len(needs_embedding)} clips..."
        )
        thumbnail_paths = [clip.thumbnail_path for clip, _ in needs_embedding]
        try:
            embeddings = extract_clip_embeddings_batch(thumbnail_paths)
            for (clip, _), emb in zip(needs_embedding, embeddings):
                clip.embedding = emb
                clip.embedding_model = _EMBEDDING_MODEL_TAG
        except Exception as e:
            raise RuntimeError(f"Failed to compute clip embeddings: {e}") from e

        still_missing = []
        for clip, _source in self._clips:
            if getattr(clip, "embedding", None) is None:
                still_missing.append(getattr(clip, "id", "<unknown>"))
        if still_missing:
            raise RuntimeError(
                "Missing DINOv2 embeddings for "
                f"{len(still_missing)} clips. Run embedding analysis first or "
                "ensure thumbnails exist before generating Staccato."
            )

    def _on_progress(self, current: int, total: int):
        self.progress_update.emit(current, total)


class StaccatoDialog(QDialog):
    """Dialog for Staccato beat-driven sequencing.

    Page 0: Config — music file picker, waveform, sensitivity, strategy, generate
    Page 1: Progress — progress bar during generation
    """

    sequence_ready = Signal(object)  # list of (Clip, Source)

    # Sentinel value stored in the combo box for the "Import new…" item.
    _IMPORT_NEW_SENTINEL = "__import_new__"

    def __init__(self, clips: list, project=None, parent=None):
        """Create the Staccato dialog.

        Args:
            clips: List of (Clip, Source) tuples to process.
            project: Project instance — used to populate the audio-source
                picker and to register newly imported audio sources. May be
                None for legacy callers that pre-date the picker (the dialog
                falls back to a single Import-new item in that case).
            parent: Qt parent widget.
        """
        super().__init__(parent)
        self._clips = clips
        self._project = project
        self._analyze_worker = None
        self._generate_worker = None
        self._import_worker = None  # In-flight AudioImportWorker, if any
        self._audio_analysis: AudioAnalysis | None = None
        self._audio_samples: np.ndarray | None = None
        self._music_path: Path | None = None
        self._handler_executed = False
        self._sequence_data = None
        self._debug_info = None
        self._onset_analysis_dirty = False
        self._detection_change_timer = QTimer(self)
        self._detection_change_timer.setSingleShot(True)
        self._detection_change_timer.setInterval(450)
        self._detection_change_timer.timeout.connect(self._reanalyze_after_detection_change)

        self.setWindowTitle("Staccato")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        self._setup_ui()
        self._populate_audio_combo()

    @property
    def music_path(self) -> Path | None:
        """The music file used for this staccato sequence."""
        return self._music_path

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._config_page = self._create_config_page()
        self._stack.addWidget(self._config_page)

        self._progress_page = self._create_progress_page()
        self._stack.addWidget(self._progress_page)

        self._results_page = self._create_results_page()
        self._stack.addWidget(self._results_page)

        self._stack.setCurrentIndex(0)

    def _create_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Cut clips to the rhythm of a music track.\n"
            "Stronger beats trigger bigger visual jumps between clips."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(desc)

        layout.addSpacing(Spacing.SM)

        # Music source picker (project audio sources + "Import new…")
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Audio source:"))

        self._audio_combo = QComboBox()
        self._audio_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._audio_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        self._audio_combo.currentIndexChanged.connect(self._on_audio_combo_changed)
        file_row.addWidget(self._audio_combo, 1)
        layout.addLayout(file_row)

        # Stem separation controls
        stem_row = QHBoxLayout()
        self._stem_checkbox = QCheckBox("Separate stems")
        self._stem_checkbox.setToolTip(
            "Use Demucs to isolate a specific instrument track for beat detection"
        )
        self._stem_checkbox.toggled.connect(self._on_stem_toggled)
        stem_row.addWidget(self._stem_checkbox)

        self._stem_combo = QComboBox()
        self._stem_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._stem_combo.addItems(["Drums", "Bass", "Vocals", "Other"])
        self._stem_combo.setVisible(False)
        self._stem_combo.currentTextChanged.connect(self._on_stem_changed)
        stem_row.addWidget(self._stem_combo)
        stem_row.addStretch()
        layout.addLayout(stem_row)

        # Waveform
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(Spacing.LG)

        # Sensitivity slider
        sens_layout = QVBoxLayout()
        sens_label = QLabel("Cut Density")
        sens_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        sens_layout.addWidget(sens_label)

        sens_row = QHBoxLayout()
        fewer_label = QLabel("Fewer Cuts")
        fewer_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;")
        sens_row.addWidget(fewer_label)

        self._sensitivity_slider = QSlider(Qt.Horizontal)
        self._sensitivity_slider.setRange(1, 10)
        self._sensitivity_slider.setValue(7)
        self._sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self._sensitivity_slider.setTickInterval(1)
        self._sensitivity_slider.valueChanged.connect(self._on_cut_density_changed)
        sens_row.addWidget(self._sensitivity_slider)

        more_label = QLabel("More Cuts")
        more_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;")
        sens_row.addWidget(more_label)

        sens_layout.addLayout(sens_row)
        controls.addLayout(sens_layout, 2)

        # Strategy dropdown
        strat_layout = QVBoxLayout()
        strat_label = QLabel("Beat Strategy")
        strat_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        strat_layout.addWidget(strat_label)

        self._strategy_combo = QComboBox()
        self._strategy_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._strategy_combo.addItems(["Onsets", "Beats", "Downbeats"])
        self._strategy_combo.currentTextChanged.connect(self._on_sensitivity_or_strategy_changed)
        strat_layout.addWidget(self._strategy_combo)
        controls.addLayout(strat_layout, 1)

        layout.addLayout(controls)

        self._advanced_toggle = QCheckBox("Advanced Onset Detection")
        self._advanced_toggle.setToolTip(
            "Tune onset detection for drums, ghost notes, and dense transients."
        )
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self._advanced_toggle)

        self._advanced_panel = self._create_advanced_onset_panel()
        self._advanced_panel.setVisible(False)
        layout.addWidget(self._advanced_panel)

        # Info labels
        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.SM}px;")
        layout.addWidget(self._info_label)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setEnabled(False)
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-weight: bold;
            }}
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._generate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    def _create_advanced_onset_panel(self) -> QWidget:
        """Create editor-facing onset detector controls."""
        panel = QWidget()
        panel.setStyleSheet(f"""
            QWidget {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_primary};
                border-radius: 6px;
            }}
            QLabel, QComboBox, QCheckBox, QDoubleSpinBox {{
                border: none;
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        profile_row = QHBoxLayout()
        profile_label = QLabel("Onset Profile")
        profile_label.setStyleSheet(f"color: {theme().text_secondary};")
        profile_row.addWidget(profile_label)

        self._onset_profile_combo = QComboBox()
        self._onset_profile_combo.addItem("Balanced", "balanced")
        self._onset_profile_combo.addItem("Drums / Percussion", "drums")
        self._onset_profile_combo.addItem("Dense / Ghost Notes", "dense")
        self._onset_profile_combo.addItem("Sparse / Strong Hits", "sparse")
        self._onset_profile_combo.addItem("Custom", "custom")
        self._onset_profile_combo.currentIndexChanged.connect(
            self._on_detection_controls_changed
        )
        profile_row.addWidget(self._onset_profile_combo, 1)
        layout.addLayout(profile_row)

        options_row = QHBoxLayout()

        min_gap_col = QVBoxLayout()
        min_gap_label = QLabel("Minimum Gap")
        min_gap_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        min_gap_col.addWidget(min_gap_label)
        self._min_gap_combo = QComboBox()
        for label, value in (
            ("30 ms", 30),
            ("60 ms", 60),
            ("100 ms", 100),
            ("160 ms", 160),
        ):
            self._min_gap_combo.addItem(label, value)
        self._min_gap_combo.setCurrentIndex(1)
        self._min_gap_combo.currentIndexChanged.connect(self._on_detection_controls_changed)
        min_gap_col.addWidget(self._min_gap_combo)
        options_row.addLayout(min_gap_col)

        timing_col = QVBoxLayout()
        timing_label = QLabel("Timing")
        timing_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        timing_col.addWidget(timing_label)
        self._timing_combo = QComboBox()
        self._timing_combo.addItem("Peak", False)
        self._timing_combo.addItem("Transient Start", True)
        self._timing_combo.setCurrentIndex(1)
        self._timing_combo.currentIndexChanged.connect(self._on_detection_controls_changed)
        timing_col.addWidget(self._timing_combo)
        options_row.addLayout(timing_col)

        resolution_col = QVBoxLayout()
        resolution_label = QLabel("Resolution")
        resolution_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        resolution_col.addWidget(resolution_label)
        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("Standard", 512)
        self._resolution_combo.addItem("High Precision", 256)
        self._resolution_combo.setCurrentIndex(1)
        self._resolution_combo.currentIndexChanged.connect(self._on_detection_controls_changed)
        resolution_col.addWidget(self._resolution_combo)
        options_row.addLayout(resolution_col)

        layout.addLayout(options_row)

        custom_row = QHBoxLayout()
        self._custom_delta_label = QLabel("Delta Threshold")
        self._custom_delta_label.setStyleSheet(f"color: {theme().text_secondary};")
        custom_row.addWidget(self._custom_delta_label)
        self._custom_delta_spin = QDoubleSpinBox()
        self._custom_delta_spin.setRange(0.005, 0.3)
        self._custom_delta_spin.setSingleStep(0.005)
        self._custom_delta_spin.setDecimals(3)
        self._custom_delta_spin.setValue(0.04)
        self._custom_delta_spin.valueChanged.connect(self._on_detection_controls_changed)
        custom_row.addWidget(self._custom_delta_spin)

        self._custom_superflux_checkbox = QCheckBox("SuperFlux")
        self._custom_superflux_checkbox.setChecked(True)
        self._custom_superflux_checkbox.toggled.connect(self._on_detection_controls_changed)
        custom_row.addWidget(self._custom_superflux_checkbox)
        custom_row.addStretch()
        layout.addLayout(custom_row)

        self._custom_delta_label.setVisible(False)
        self._custom_delta_spin.setVisible(False)
        self._custom_superflux_checkbox.setVisible(False)

        hint = QLabel(
            "Use Drums / Percussion or Dense / Ghost Notes when snares, fills, "
            "or ghost notes are missing."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;")
        layout.addWidget(hint)

        return panel

    def _create_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        self._progress_label = QLabel("Preparing...")
        self._progress_label.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    def _populate_audio_combo(
        self,
        *,
        selected_audio_source_id: str | None = None,
        analyze_selection: bool = True,
    ) -> None:
        """Populate the audio-source combo from the project + the Import-new item."""
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()

        audio_sources = list(self._project.audio_sources) if self._project is not None else []

        if not audio_sources:
            self._audio_combo.addItem("No audio sources — import one below", None)
            # Disable that placeholder so it's never picked as a real source
            model = self._audio_combo.model()
            item = model.item(0) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(False)
        else:
            for audio in audio_sources:
                label = f"{audio.filename} ({audio.duration_str})"
                self._audio_combo.addItem(label, audio.id)

        self._audio_combo.insertSeparator(self._audio_combo.count())
        self._audio_combo.addItem("Import new…", self._IMPORT_NEW_SENTINEL)

        # Default selection: requested source, first real source, or placeholder.
        selected_index = 0
        if audio_sources:
            for index, audio in enumerate(audio_sources):
                if audio.id == selected_audio_source_id:
                    selected_index = index
                    break
        self._audio_combo.setCurrentIndex(selected_index)

        self._audio_combo.blockSignals(False)

        # Trigger analysis for the default selection (if any real source picked)
        if audio_sources and analyze_selection:
            self._select_audio_source_by_id(audio_sources[selected_index].id)

    def _select_audio_source_by_id(self, audio_source_id: str) -> None:
        """Switch the dialog to use the given audio source and re-analyze."""
        if self._project is None:
            return
        audio = self._project.get_audio_source(audio_source_id)
        if audio is None:
            return

        if not audio.file_path.exists():
            self._info_label.setText(
                f"Audio file is missing on disk: {audio.filename}"
            )
            self._music_path = None
            self._generate_btn.setEnabled(False)
            return

        self._music_path = audio.file_path
        self._generate_btn.setEnabled(False)
        self._info_label.setText("Analyzing audio...")
        self._waveform.clear()
        self._analyze_audio()

    @Slot(int)
    def _on_audio_combo_changed(self, index: int) -> None:
        """Handle selection change in the audio source combo."""
        if index < 0:
            return
        data = self._audio_combo.itemData(index)
        if data is None:
            # Placeholder ("No audio sources"); nothing to do.
            return
        if data == self._IMPORT_NEW_SENTINEL:
            self._on_import_new_audio()
            return
        # Real audio source id
        self._select_audio_source_by_id(data)

    def _on_import_new_audio(self) -> None:
        """Open a file dialog to pick an audio file; spawn an import worker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Audio File", "", _AUDIO_FORMATS,
        )
        if not path:
            self._reset_combo_to_current_source()
            return

        if self._project is None:
            # Legacy fallback: just use the path directly
            self._music_path = Path(path)
            self._info_label.setText("Analyzing audio...")
            self._waveform.clear()
            self._analyze_audio()
            return

        # Spawn AudioImportWorker; on success, add to project and select it.
        from ui.workers.audio_import_worker import AudioImportWorker

        self._info_label.setText(f"Importing {Path(path).name}…")
        self._generate_btn.setEnabled(False)

        self._import_worker = AudioImportWorker(Path(path), parent=self)
        self._import_worker.audio_ready.connect(self._on_imported_audio_ready)
        self._import_worker.error.connect(self._on_imported_audio_error)
        self._import_worker.finished_signal.connect(self._on_import_worker_finished)
        self._import_worker.start()

    @Slot(object)
    def _on_imported_audio_ready(self, audio) -> None:
        """Add the freshly imported audio source and select it."""
        if self._project is not None:
            self._project.add_audio_source(audio)
        # Refresh the combo so the new source appears, then select it.
        self._populate_audio_combo(selected_audio_source_id=audio.id)
        if self._project is not None:
            return
        # Project-less fallback (shouldn't really happen): use path directly
        self._music_path = audio.file_path
        self._info_label.setText("Analyzing audio...")
        self._analyze_audio()

    @Slot(str)
    def _on_imported_audio_error(self, message: str) -> None:
        """Surface the import error and reset the combo."""
        self._info_label.setText(f"Audio import failed: {message}")
        self._reset_combo_to_current_source()

    def _on_import_worker_finished(self) -> None:
        self._import_worker = None

    def _reset_combo_to_current_source(self) -> None:
        """After a cancelled / failed Import-new, restore the combo to the
        currently-selected audio source (or first real one) without triggering
        analysis.
        """
        self._audio_combo.blockSignals(True)
        target_id: str | None = None
        if self._music_path is not None and self._project is not None:
            for audio in self._project.audio_sources:
                if audio.file_path == self._music_path:
                    target_id = audio.id
                    break
        if target_id is None and self._project is not None and self._project.audio_sources:
            target_id = self._project.audio_sources[0].id

        if target_id is not None:
            for i in range(self._audio_combo.count()):
                if self._audio_combo.itemData(i) == target_id:
                    self._audio_combo.setCurrentIndex(i)
                    break
        else:
            self._audio_combo.setCurrentIndex(0)
        self._audio_combo.blockSignals(False)

    @Slot(bool)
    def _on_stem_toggled(self, checked: bool):
        """Show/hide stem dropdown and check dependency availability."""
        self._stem_combo.setVisible(checked)
        if checked:
            # Check if demucs-infer is available — the worker thread will
            # install on demand if needed, but warn the user upfront about
            # the download size so they can opt out before waiting.
            from core.feature_registry import check_feature_ready
            available, _missing = check_feature_ready("stem_separation")
            if not available:
                reply = QMessageBox.question(
                    self,
                    "Stem Separation",
                    "Stem separation requires demucs + torch (~2 GB download).\n\n"
                    "Dependencies will be installed automatically when you "
                    "generate the sequence. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    blocker = QSignalBlocker(self._stem_checkbox)
                    self._stem_checkbox.setChecked(False)
                    del blocker
                    self._stem_combo.setVisible(False)
                    return
            self._select_drums_profile_if_needed()

        # Re-analyze if we already have a file loaded
        if self._music_path:
            self._info_label.setText("Re-analyzing...")
            self._generate_btn.setEnabled(False)
            self._waveform.clear()
            self._analyze_audio()

    @Slot(str)
    def _on_stem_changed(self, _stem_text: str):
        """Re-analyze when the user changes the stem selection."""
        self._select_drums_profile_if_needed()
        if self._music_path and self._stem_checkbox.isChecked():
            self._info_label.setText("Re-analyzing...")
            self._generate_btn.setEnabled(False)
            self._waveform.clear()
            self._analyze_audio()

    def _get_stem_name(self) -> str | None:
        """Return the selected stem name if separation is enabled."""
        if self._stem_checkbox.isChecked():
            return self._stem_combo.currentText().lower()
        return None

    def _select_drums_profile_if_needed(self):
        """Default drum-stem workflows to the percussion-tuned onset profile."""
        if (
            self._stem_checkbox.isChecked()
            and self._stem_combo.currentText().lower() == "drums"
            and self._onset_profile_combo.currentData() == "balanced"
        ):
            blocker = QSignalBlocker(self._onset_profile_combo)
            self._onset_profile_combo.setCurrentIndex(
                self._onset_profile_combo.findData("drums")
            )
            del blocker

    def _build_onset_config(self) -> OnsetDetectionConfig:
        """Build onset detector settings from the current Staccato controls."""
        profile = self._active_onset_profile()
        if not self._advanced_toggle.isChecked():
            return make_onset_detection_config(
                profile,
                cut_density=self._sensitivity_slider.value(),
            )

        min_gap_ms = self._min_gap_combo.currentData()
        backtrack = self._timing_combo.currentData()
        hop_length = self._resolution_combo.currentData()

        delta = None
        superflux = None
        if profile == "custom":
            delta = self._custom_delta_spin.value()
            superflux = self._custom_superflux_checkbox.isChecked()

        return make_onset_detection_config(
            profile,
            cut_density=self._sensitivity_slider.value(),
            min_gap_ms=min_gap_ms,
            backtrack=backtrack,
            hop_length=hop_length,
            delta=delta,
            superflux=superflux,
        )

    def _active_onset_profile(self) -> str:
        """Return the visible/effective onset profile."""
        if self._advanced_toggle.isChecked():
            return self._onset_profile_combo.currentData() or "balanced"
        if (
            self._stem_checkbox.isChecked()
            and self._stem_combo.currentText().lower() == "drums"
        ):
            return "drums"
        return "balanced"

    @Slot(bool)
    def _on_advanced_toggled(self, checked: bool):
        self._advanced_panel.setVisible(checked)
        self._schedule_reanalysis_for_detection_change()

    def _on_detection_controls_changed(self, *_args):
        """Handle changes that require re-running onset detection."""
        is_custom = self._onset_profile_combo.currentData() == "custom"
        self._custom_delta_label.setVisible(is_custom)
        self._custom_delta_spin.setVisible(is_custom)
        self._custom_superflux_checkbox.setVisible(is_custom)
        self._schedule_reanalysis_for_detection_change()

    def _on_cut_density_changed(self, *_args):
        """Update current preview and re-run onset detection after a short pause."""
        self._on_sensitivity_or_strategy_changed()
        self._schedule_reanalysis_for_detection_change()

    def _schedule_reanalysis_for_detection_change(self):
        if not self._music_path:
            return
        self._onset_analysis_dirty = True
        if self._strategy_combo.currentText().lower() != "onsets":
            return
        self._queue_onset_reanalysis()

    def _queue_onset_reanalysis(self):
        self._info_label.setText("Updating onset detection...")
        self._generate_btn.setEnabled(False)
        self._detection_change_timer.start()

    @Slot()
    def _reanalyze_after_detection_change(self):
        if self._music_path:
            self._analyze_audio()

    def _analyze_audio(self):
        """Start background audio analysis."""
        self._detection_change_timer.stop()
        self._onset_analysis_dirty = False
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.cancel()
            self._analyze_worker.wait(2000)

        self._handler_executed = False

        stem_name = self._get_stem_name()
        stems_cache_dir = None
        if stem_name:
            from core.settings import load_settings
            settings = load_settings()
            stems_cache_dir = settings.stems_cache_dir

        self._analyze_worker = StaccatoAnalyzeWorker(
            self._music_path,
            stem_name=stem_name,
            stems_cache_dir=stems_cache_dir,
            onset_config=self._build_onset_config(),
            parent=self,
        )
        self._analyze_worker.audio_ready.connect(
            self._on_audio_ready, Qt.UniqueConnection,
        )
        self._analyze_worker.error.connect(
            self._on_analyze_error, Qt.UniqueConnection,
        )
        self._analyze_worker.progress_message.connect(
            self._on_analyze_progress, Qt.UniqueConnection,
        )
        self._analyze_worker.start()

    @Slot(str)
    def _on_analyze_progress(self, message: str):
        """Update info label with progress from the analyze worker."""
        self._info_label.setText(message)

    @Slot(object, object)
    def _on_audio_ready(self, analysis: AudioAnalysis, samples: np.ndarray):
        if self._handler_executed:
            return
        self._handler_executed = True

        self._audio_analysis = analysis
        self._audio_samples = samples

        # Update waveform and info using current sensitivity/strategy
        self._on_sensitivity_or_strategy_changed()
        self._generate_btn.setEnabled(True)

    def _get_filtered_markers(self) -> list[float]:
        """Get cut-point markers filtered by current strategy and sensitivity.

        Cut density 1 = only the strongest onsets, 10 = all onsets.
        For beats/downbeats, sensitivity has no effect (all are used).
        """
        analysis = self._audio_analysis
        if not analysis:
            return []

        strategy = self._strategy_combo.currentText().lower()
        if strategy == "downbeats":
            return analysis.downbeat_times
        elif strategy == "beats":
            return analysis.beat_times

        # Onsets: filter by strength threshold based on cut density.
        sensitivity = self._sensitivity_slider.value()
        if sensitivity >= 10 or not analysis.onset_strengths:
            return analysis.onset_times

        # Map slider 1-10 to threshold 0.9-0.0
        # Slider 1 (fewer cuts) = threshold 0.9 (only strongest)
        # Slider 10 (more cuts) = threshold 0.0 (all onsets)
        threshold = (10 - sensitivity) / 10.0

        filtered = [
            t for t, s in zip(analysis.onset_times, analysis.onset_strengths)
            if s >= threshold
        ]
        return filtered if filtered else analysis.onset_times[:1]

    @Slot()
    def _on_sensitivity_or_strategy_changed(self):
        """Update waveform and info when sensitivity or strategy changes."""
        if not self._audio_analysis or self._audio_samples is None:
            return

        strategy = self._strategy_combo.currentText().lower()
        if strategy == "onsets" and self._onset_analysis_dirty and self._music_path:
            self._queue_onset_reanalysis()
            return

        markers = self._get_filtered_markers()
        self._waveform.set_audio_data(
            samples=self._audio_samples,
            duration=self._audio_analysis.duration_seconds,
            beat_times=self._audio_analysis.beat_times,
            onset_times=markers,
        )

        # Update info label
        duration_str = f"{self._audio_analysis.duration_seconds:.1f}s"
        stem_label = ""
        stem_name = self._get_stem_name()
        if stem_name:
            stem_label = f" ({stem_name} stem)"
        if self._active_onset_profile() == "balanced":
            profile_label = "Balanced"
        else:
            profile_label = self._onset_profile_combo.currentText()
        self._info_label.setText(
            f"{self._audio_analysis.tempo_bpm:.0f} BPM · {len(markers)} cut points · "
            f"{duration_str}{stem_label} · {profile_label} · "
            f"{len(self._clips)} clips available"
        )

    @Slot(str)
    def _on_analyze_error(self, error_msg: str):
        self._info_label.setText(f"Analysis failed: {error_msg}")
        self._generate_btn.setEnabled(False)

    @Slot()
    def _on_generate(self):
        if not self._audio_analysis:
            return

        self._handler_executed = False
        self._stack.setCurrentIndex(1)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Preparing...")

        strategy_text = self._strategy_combo.currentText().lower()
        cut_times = self._get_filtered_markers()

        self._generate_worker = StaccatoGenerateWorker(
            clips=self._clips,
            audio_analysis=self._audio_analysis,
            strategy=strategy_text,
            cut_times=cut_times,
            parent=self,
        )
        self._generate_worker.progress_update.connect(self._on_progress_update)
        self._generate_worker.progress_message.connect(self._on_progress_message)
        self._generate_worker.finished_sequence.connect(
            self._on_finished, Qt.UniqueConnection,
        )
        self._generate_worker.error.connect(
            self._on_generate_error, Qt.UniqueConnection,
        )
        self._generate_worker.start()

    @Slot(int, int)
    def _on_progress_update(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)

    @Slot(str)
    def _on_progress_message(self, message: str):
        self._progress_label.setText(message)

    def _create_results_page(self) -> QWidget:
        """Create the results page shown after generation completes."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        self._results_summary = QLabel("")
        self._results_summary.setWordWrap(True)
        self._results_summary.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._results_summary)

        layout.addStretch()

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_report_btn = QPushButton("Save Debug Report")
        save_report_btn.setToolTip(
            "Export an interactive HTML report with onset/distance analysis"
        )
        save_report_btn.clicked.connect(self._on_save_debug_report)
        btn_layout.addWidget(save_report_btn)

        use_btn = QPushButton("Use Sequence")
        use_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-weight: bold;
            }}
        """)
        use_btn.clicked.connect(self._on_use_sequence)
        btn_layout.addWidget(use_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    @Slot(object)
    def _on_finished(self, sequence_data):
        if self._handler_executed:
            return
        self._handler_executed = True

        self._sequence_data = sequence_data

        # Extract debug info from StaccatoResult or from worker
        if hasattr(sequence_data, 'debug'):
            self._debug_info = sequence_data.debug
        elif self._generate_worker and hasattr(self._generate_worker, '_debug_info'):
            self._debug_info = self._generate_worker._debug_info

        # Update results summary
        n_slots = len(sequence_data)
        unique_clips = len({
            getattr(entry[0], 'id', i) for i, entry in enumerate(sequence_data)
        })
        summary = f"Generated {n_slots} slots using {unique_clips} unique clips."
        if self._debug_info:
            summary += f"\nStrategy: {self._debug_info.strategy}"
        self._results_summary.setText(summary)

        self._stack.setCurrentIndex(2)

    @Slot()
    def _on_save_debug_report(self):
        """Save the debug report as an interactive HTML file."""
        if not self._debug_info:
            QMessageBox.warning(self, "No Debug Data", "No debug data available.")
            return

        from core.settings import load_settings
        settings = load_settings()
        default_dir = str(settings.export_dir) if settings.export_dir else ""
        default_path = Path(default_dir) / "staccato_debug_report.html"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Debug Report", str(default_path),
            "HTML Files (*.html);;All Files (*)",
        )
        if not path:
            return

        try:
            from core.remix.staccato_report import save_staccato_report
            save_staccato_report(self._debug_info, Path(path))
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception as e:
            logger.error(f"Failed to save debug report: {e}")
            QMessageBox.critical(
                self, "Save Failed", f"Could not save report:\n{e}",
            )

    @Slot()
    def _on_use_sequence(self):
        """Emit the sequence and close the dialog."""
        if self._sequence_data is not None:
            self.sequence_ready.emit(self._sequence_data)
        self.accept()

    @Slot(str)
    def _on_generate_error(self, error_msg: str):
        logger.error("Staccato generation error: %s", error_msg)
        self._stack.setCurrentIndex(0)
        QMessageBox.critical(
            self, "Staccato Error", f"Generation failed:\n{error_msg}",
        )

    @Slot()
    def _on_cancel(self):
        for worker in (self._analyze_worker, self._generate_worker):
            if worker and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        self._analyze_worker = None
        self._generate_worker = None
        self.reject()

    def closeEvent(self, event):
        for worker in (self._analyze_worker, self._generate_worker):
            if worker and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)
